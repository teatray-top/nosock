import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
import warnings
from datetime import date

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import msma_v1 as M1
import ms_role_v2 as R
import ms_alloc_v1 as A1
import ms_alloc_v3 as A3
import ms_alloc_v6 as A6

COV_WINDOW = 60
SHRINK = 0.1
SEEDS = 20
DEPOSIT = R.DEPOSIT
VAULT_SHARE = R.VAULT_SHARE


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def returns_table(inst):
    days = sorted(set(d for ins in inst.values() for d in ins["days"]))
    pos = {d: i for i, d in enumerate(days)}
    codes = list(inst)
    Rm = np.full((len(days), len(codes)), np.nan)
    for j, c in enumerate(codes):
        ins = inst[c]
        lr = np.r_[np.nan, np.diff(np.log(ins["C"]))]
        for i, d in enumerate(ins["days"]):
            Rm[pos[d], j] = lr[i]
    return days, pos, codes, Rm


def joint_weights(mu, S, w_held, S_ch):
    idx = list(range(len(mu)))
    w = np.zeros(len(mu))
    while idx:
        Sc = S[np.ix_(idx, idx)]
        rhs = mu[idx] - (S_ch[idx] @ w_held if len(w_held) else 0.0)
        x = np.linalg.solve(Sc, rhs)
        if (x >= 0).all():
            w[idx] = x
            break
        idx = [i for i, xi in zip(idx, x) if xi >= 0]
    return w


def cov(Rm, rows, cols):
    X = Rm[np.ix_(rows, cols)]
    ok = ~np.isnan(X).any(axis=1)
    X = X[ok]
    if len(X) < 20:
        return None
    S = np.cov(X, rowvar=False)
    S = np.atleast_2d(S)
    return (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S)) + 1e-12 * np.eye(len(cols))


def engine(dec, days, rt, pick=None, corr=True):
    rdays, rpos, rcodes, Rm = rt
    col = {c: j for j, c in enumerate(rcodes)}
    pool, vault, dep = 0.0, 0.0, 0.0
    cum, hwm = 0.0, 0.0
    opn = {}
    events, losses = [], []
    use = {}
    trades, skipped_price = 0, 0
    vault_year = {}
    npos, expo, shrink_ratio = [], [], []

    def release(c):
        nonlocal pool, vault, cum, hwm
        size, Rr, exday = opn.pop(c)
        prof = size * math.expm1(Rr)
        cum += prof
        v = VAULT_SHARE * (cum - hwm) if cum > hwm else 0.0
        hwm = max(hwm, cum)
        if v > 0:
            vault += v
            vault_year[exday[:4]] = vault_year.get(exday[:4], 0.0) + v
            events.append((exday, v))
        pool += size + prof - v
        if prof < 0:
            losses.append(prof)

    for D in days:
        for c in [c for c, x in opn.items() if x[2] <= D]:
            release(c)
        pool += DEPOSIT
        dep += DEPOSIT
        free = [(c, r) for c, r in dec[D].items() if c not in opn and "R" in r and r.get("ref_won")]
        cands = pick(free) if pick else sorted([(c, r) for c, r in free if r["v"] > 0], key=lambda z: -z[1]["v"])
        equity = pool + sum(x[0] for x in opn.values())
        wmap = {c: r["f"] for c, r in cands}
        if corr and cands and equity > 0:
            i1 = rpos[D]
            rows = list(range(max(0, i1 - COV_WINDOW + 1), i1 + 1))
            held = list(opn)
            allc = [c for c, _ in cands] + held
            S = cov(Rm, rows, [col[c] for c in allc])
            if S is not None:
                k = len(cands)
                var = np.diag(S)[:k]
                mu = np.array([r["f"] for _, r in cands]) * var
                w_h = np.array([opn[c][0] / equity for c in held])
                w = joint_weights(mu, S[:k, :k], w_h, S[:k, k:])
                wmap = {c: float(min(w[i], r["f"])) for i, (c, r) in enumerate(cands)}
                tot_f = sum(r["f"] for _, r in cands)
                shrink_ratio.append(sum(wmap.values()) / tot_f if tot_f > 0 else 1.0)
        for c, r in cands:
            f = wmap[c]
            budget = min(pool, f * equity)
            if budget <= 0:
                continue
            shares = math.floor(budget / r["ref_won"])
            if shares < 1:
                skipped_price += 1
                continue
            px = r["entry_won"] if r["filled"] else r["ref_won"]
            size = shares * px
            while size > pool and shares > 0:
                shares -= 1
                size = shares * px
            if shares < 1:
                skipped_price += 1
                continue
            opn[c] = (size, r["R"], r["exit_day"])
            pool -= size
            trades += 1
            u = use.setdefault(c, dict(money=0.0, trades=0, f_sum=0.0, cycles={}))
            u["money"] += size
            u["trades"] += 1
            u["f_sum"] += f
            key = f"hold{r['hz']}" if r["hold"] else str(r["hz"])
            u["cycles"][key] = u["cycles"].get(key, 0) + 1
        npos.append(len(opn))
        eq = pool + sum(x[0] for x in opn.values())
        expo.append(sum(x[0] for x in opn.values()) / eq if eq > 0 else 0.0)
    for c in list(opn):
        release(c)
    cash_days = sorted(set(e[0] for e in events))
    gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(cash_days[:-1], cash_days[1:])]
    months = len(days) / 21.0
    return dict(net_realized_profit_won=cum, deposits_won=dep, vault_won=vault, acct_won=pool, total_won=pool + vault, gain_over_deposits_pct=100 * ((pool + vault) / dep - 1), trades=trades, skipped_for_share_price=skipped_price, losing_trades=len(losses), loss_sum_won=float(sum(losses)), worst_loss_won=float(min(losses)) if losses else 0.0, open_positions_mean=float(np.mean(npos)), invested_share_mean=float(np.mean(expo)), size_after_correlation_vs_model_mean=float(np.mean(shrink_ratio)) if shrink_ratio else None, cash_out_days=len(cash_days), median_days_between_cash_outs=float(np.median(gaps)) if gaps else None, vault_per_month_avg_won=vault / months, vault_last_12_months_won=sum(v for d, v in events if d >= "2025-11-06"), vault_by_year_won=vault_year, use=use)


def main(out_path, spec_path, spec_sha):
    if sha(spec_path) != spec_sha:
        raise SystemExit("spec hash differs")
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec["code_sha256"] != sha(os.path.abspath(__file__)):
        raise SystemExit("code hash differs from the spec")
    for key, pth in spec["inputs"]["files"].items():
        if sha(os.path.join(os.path.dirname(HERE), pth)) != spec["inputs"]["sha256"][key]:
            raise SystemExit(f"input hash differs: {key}")
    recs = R.load()
    inst = {c: R.instrument(recs[c], c) for c in M1.CLASS}
    raw = A3.load_raw()
    samp = A1.samples(inst)
    print("samples done", flush=True)
    dec = A6.decisions_f(inst, samp)
    A3.attach_prices(dec, inst, raw)
    days = sorted(dec)
    rt = returns_table(inst)
    names = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"]
    ai = engine(dec, days, rt)
    base = engine(dec, days, rt, corr=False)
    rnd = []
    for s in range(SEEDS):
        g = np.random.default_rng(4500 + s)

        def pick(free, g=g):
            k = sum(1 for _, r in free if r["v"] > 0)
            if k == 0:
                return []
            idx = g.choice(len(free), size=k, replace=False)
            return [free[i] for i in idx]

        rnd.append(engine(dec, days, rt, pick))
    ctl = dict(v6_reproduced_without_correlation=dict(total_won=base["total_won"], vault_won=base["vault_won"]), random_instruments_same_count=dict(total_won_mean=float(np.mean([x["total_won"] for x in rnd])), total_won_range=[float(min(x["total_won"] for x in rnd)), float(max(x["total_won"] for x in rnd))], ai_beats=int(sum(ai["total_won"] > x["total_won"] for x in rnd))))
    print(f"KR correlation-aware sizes (shrink only): deposits {ai['deposits_won'] / 1e4:.0f}만 vault {ai['vault_won'] / 1e4:.1f}만 acct {ai['acct_won'] / 1e4:.1f}만 total {ai['total_won'] / 1e4:.1f}만 ({ai['gain_over_deposits_pct']:+.1f}%) trades {ai['trades']} positions mean {ai['open_positions_mean']:.1f} invested {100 * ai['invested_share_mean']:.0f}% size after correlation / model size {ai['size_after_correlation_vs_model_mean']:.2f} losing {ai['losing_trades']} worst {ai['worst_loss_won'] / 1e4:.1f}만 cash-out days {ai['cash_out_days']} median gap {ai['median_days_between_cash_outs']} d, vault last 12m {ai['vault_last_12_months_won'] / 1e4:.1f}만", flush=True)
    print(f"  without correlation (= v6): total {base['total_won'] / 1e4:.1f}만 vault {base['vault_won'] / 1e4:.1f}만; random instruments: total {ctl['random_instruments_same_count']['total_won_mean'] / 1e4:.1f}만, AI beats {ctl['random_instruments_same_count']['ai_beats']}/{SEEDS}", flush=True)
    tot = sum(u["money"] for u in ai["use"].values())
    for c, u in sorted(ai["use"].items(), key=lambda z: -z[1]["money"]):
        print(f"    {c} {names[c][:16]:16s} money share {100 * u['money'] / tot:5.1f}% trades {u['trades']:4d} mean size {100 * u['f_sum'] / u['trades']:4.1f}% cycles {u['cycles']}", flush=True)
    print("    vault by year", {y: round(v / 1e4) for y, v in sorted(ai["vault_by_year_won"].items())}, flush=True)
    res = dict(AI=ai, controls=ctl, cov_window=COV_WINDOW, shrink=SHRINK)
    rec = dict(version="ms-alloc-v8b", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
