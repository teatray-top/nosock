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
import lightgbm as lgb
import learn_inverse as LI
import msma_v1 as M1
import ms_role_v2 as R
import ms_alloc_v1 as A1
import ms_alloc_v3 as A3

F_GRID = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
SEEDS = 20
DEPOSIT = R.DEPOSIT
VAULT_SHARE = R.VAULT_SHARE


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rows_f(Fv, acts, v20, fs):
    return np.hstack([R.rows(Fv, acts, v20), fs[:, None]])


def full_rows(Fv, v20):
    A = np.repeat(R.GRID, len(F_GRID), axis=0)
    fs = np.tile(F_GRID, len(R.GRID))
    return rows_f(Fv, A, v20, fs), np.repeat(np.arange(len(R.GRID)), len(F_GRID)), fs


def decisions_f(inst, samp):
    rng = np.random.default_rng(77)
    fsamp = {c: [rng.choice(F_GRID, size=len(idx)) for _, idx, _, _ in samp[c]] for c in inst}
    dec = {}
    for y in R.TEST_YEARS:
        cut = f"{y}-01-01"
        X, Y = [], []
        for c, ins in inst.items():
            for (t, idx, Rr, ex), fs in zip(samp[c], fsamp[c]):
                if ins["days"][t] < cut and ex.max() < len(ins["days"]) and ins["days"][int(ex.max())] < cut and not np.isnan(Rr).any():
                    X.append(rows_f(ins["F"][t], R.GRID[idx], ins["v20"][t], fs))
                    Y.append(np.log1p(fs * np.expm1(Rr)))
        m = lgb.train(LI.PARAMS, lgb.Dataset(np.vstack(X), label=np.concatenate(Y)), num_boost_round=LI.ROUNDS)
        for c, ins in inst.items():
            n = len(ins["C"])
            for t, dday in enumerate(ins["days"]):
                if dday[:4] != str(y) or t < 260 or t + 1 >= n or np.isnan(ins["F"][t]).any():
                    continue
                Xa, ai, fi = full_rows(ins["F"][t], ins["v20"][t])
                p = m.predict(Xa)
                k = int(np.argmax(p))
                a = int(ai[k])
                rec = dict(t=t, a=a, f=float(fi[k]), v=float(p[k]))
                Rr, ex = R.outcome(ins, t, R.GRID[a])
                if not np.isnan(Rr):
                    rec.update(R=float(Rr), exit_day=ins["days"][int(ex)], hz=int(R.GRID[a][3]), hold=bool(R.GRID[a][1] == 0))
                dec.setdefault(dday, {})[c] = rec
        print("year", y, "model fitted, sample-days", len(Y), flush=True)
    return dec


def engine(dec, days, pick=None, fixed_f=None):
    pool, vault, dep = 0.0, 0.0, 0.0
    cum, hwm = 0.0, 0.0
    opn = {}
    events, losses = [], []
    use = {}
    trades, skipped_price = 0, 0
    vault_year = {}
    npos, expo = [], []

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
        for c, r in cands:
            f = fixed_f if fixed_f is not None else r["f"]
            budget = min(pool, f * equity)
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
    return dict(net_realized_profit_won=cum, deposits_won=dep, vault_won=vault, acct_won=pool, total_won=pool + vault, gain_over_deposits_pct=100 * ((pool + vault) / dep - 1), trades=trades, skipped_for_share_price=skipped_price, losing_trades=len(losses), loss_sum_won=float(sum(losses)), worst_loss_won=float(min(losses)) if losses else 0.0, open_positions_mean=float(np.mean(npos)), open_positions_max=int(max(npos)), invested_share_mean=float(np.mean(expo)), cash_out_days=len(cash_days), median_days_between_cash_outs=float(np.median(gaps)) if gaps else None, vault_per_month_avg_won=vault / months, vault_last_12_months_won=sum(v for d, v in events if d >= "2025-11-06"), vault_by_year_won=vault_year, use=use)


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
    dec = decisions_f(inst, samp)
    A3.attach_prices(dec, inst, raw)
    days = sorted(dec)
    names = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"]
    ai = engine(dec, days)
    rnd = []
    for s in range(SEEDS):
        g = np.random.default_rng(2500 + s)

        def pick(free, g=g):
            k = sum(1 for _, r in free if r["v"] > 0)
            if k == 0:
                return []
            idx = g.choice(len(free), size=k, replace=False)
            return [free[i] for i in idx]

        rnd.append(engine(dec, days, pick))
    fs = [r["f"] for row in dec.values() for r in row.values() if r["v"] > 0]
    f_mean = float(np.mean(fs)) if fs else 0.2
    flat = engine(dec, days, fixed_f=f_mean)
    ctl = dict(random_instruments_same_count=dict(total_won_mean=float(np.mean([x["total_won"] for x in rnd])), total_won_range=[float(min(x["total_won"] for x in rnd)), float(max(x["total_won"] for x in rnd))], ai_beats=int(sum(ai["total_won"] > x["total_won"] for x in rnd))), same_selection_flat_size=dict(f=f_mean, total_won=flat["total_won"], vault_won=flat["vault_won"]))
    print(f"AI select+size: deposits {ai['deposits_won'] / 1e4:.0f}만 vault {ai['vault_won'] / 1e4:.1f}만 acct {ai['acct_won'] / 1e4:.1f}만 total {ai['total_won'] / 1e4:.1f}만 ({ai['gain_over_deposits_pct']:+.1f}%) net realized {ai['net_realized_profit_won'] / 1e4:.1f}만 trades {ai['trades']} skipped-for-price {ai['skipped_for_share_price']} positions mean {ai['open_positions_mean']:.1f} max {ai['open_positions_max']} invested share mean {100 * ai['invested_share_mean']:.0f}% losing {ai['losing_trades']} worst {ai['worst_loss_won'] / 1e4:.1f}만 cash-out days {ai['cash_out_days']} median gap {ai['median_days_between_cash_outs']} d, vault last 12m {ai['vault_last_12_months_won'] / 1e4:.1f}만", flush=True)
    print(f"  control random instruments (same count, their own size): total {ctl['random_instruments_same_count']['total_won_mean'] / 1e4:.1f}만 range {[round(x / 1e4) for x in ctl['random_instruments_same_count']['total_won_range']]}, AI beats {ctl['random_instruments_same_count']['ai_beats']}/{SEEDS}", flush=True)
    print(f"  control same selection, flat size f={f_mean:.3f}: total {flat['total_won'] / 1e4:.1f}만 vault {flat['vault_won'] / 1e4:.1f}만", flush=True)
    tot = sum(u["money"] for u in ai["use"].values())
    v20_mean = {c: float(np.nanmean(inst[c]["v20"])) for c in inst}
    for c, u in sorted(ai["use"].items(), key=lambda z: -z[1]["money"]):
        print(f"    {c} {names[c][:16]:16s} money share {100 * u['money'] / tot:5.1f}% trades {u['trades']:4d} mean size {100 * u['f_sum'] / u['trades']:4.1f}% of equity, daily vol {100 * v20_mean[c]:.2f}% cycles {u['cycles']}", flush=True)
    print("    vault by year", {y: round(v / 1e4) for y, v in sorted(ai["vault_by_year_won"].items())}, flush=True)
    res = dict(AI=ai, controls=ctl, f_grid=F_GRID.tolist())
    rec = dict(version="ms-alloc-v6", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
