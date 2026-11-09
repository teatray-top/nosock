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

SLOTS = (1, 2, 4)
SEEDS = 20
MIN_TRADE = 1000.0
DEPOSIT = R.DEPOSIT
VAULT_SHARE = R.VAULT_SHARE


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def samples(inst):
    rng = np.random.default_rng(31)
    hold_idx = np.nonzero(R.GRID[:, 1] == 0)[0]
    samp = {}
    for c, ins in inst.items():
        n = len(ins["C"])
        rows_ = []
        for t in range(260, n - 1):
            if np.isnan(ins["F"][t]).any():
                continue
            idx = np.r_[hold_idx, rng.choice(len(R.GRID) - len(hold_idx), size=R.SAMPLES - len(hold_idx), replace=False)]
            out = [R.outcome(ins, t, R.GRID[a]) for a in idx]
            rows_.append((t, idx, np.array([o[0] for o in out]), np.array([o[1] for o in out])))
        samp[c] = rows_
    return samp


def decisions(inst, samp):
    dec = {}
    for y in R.TEST_YEARS:
        cut = f"{y}-01-01"
        X, Y = [], []
        for c, ins in inst.items():
            for t, idx, Rr, ex in samp[c]:
                if ins["days"][t] < cut and ex.max() < len(ins["days"]) and ins["days"][int(ex.max())] < cut and not np.isnan(Rr).any():
                    X.append(R.rows(ins["F"][t], R.GRID[idx], ins["v20"][t]))
                    Y.append(R.reward(Rr, R.KAPPA))
        m = lgb.train(LI.PARAMS, lgb.Dataset(np.vstack(X), label=np.concatenate(Y)), num_boost_round=LI.ROUNDS)
        for c, ins in inst.items():
            n = len(ins["C"])
            for t, dday in enumerate(ins["days"]):
                if dday[:4] != str(y) or t < 260 or t + 1 >= n or np.isnan(ins["F"][t]).any():
                    continue
                p = m.predict(R.rows(ins["F"][t], R.GRID, ins["v20"][t]))
                a = int(np.argmax(p))
                v = float(p[a])
                rec = dict(t=t, a=a, v=v)
                if v > 0:
                    Rr, ex = R.outcome(ins, t, R.GRID[a])
                    if not np.isnan(Rr):
                        rec.update(R=float(Rr), exit_day=ins["days"][int(ex)], hz=int(R.GRID[a][3]), hold=bool(R.GRID[a][1] == 0))
                dec.setdefault(dday, {})[c] = rec
        print("year", y, "model fitted, rows", len(Y), flush=True)
    return dec


def engine(dec, days, slots, order):
    pool, vault, dep = 0.0, 0.0, 0.0
    opn = {}
    events, losses = [], []
    use = {}
    trades = 0
    vault_year = {}

    def release(c):
        nonlocal pool, vault
        size, Rr, exday = opn.pop(c)
        prof = size * math.expm1(Rr)
        if prof > 0:
            v = VAULT_SHARE * prof
            vault += v
            vault_year[exday[:4]] = vault_year.get(exday[:4], 0.0) + v
            events.append((exday, v))
            pool += size + prof - v
        else:
            pool += size + prof
            if prof < 0:
                losses.append(prof)

    for D in days:
        for c in [c for c, x in opn.items() if x[2] <= D]:
            release(c)
        pool += DEPOSIT
        dep += DEPOSIT
        cands = [(c, r) for c, r in dec[D].items() if c not in opn and r["v"] > 0 and "R" in r]
        cands = order(cands)
        equity = pool + sum(x[0] for x in opn.values())
        for c, r in cands:
            if len(opn) >= slots:
                break
            size = min(pool, equity / slots)
            if size < MIN_TRADE:
                break
            opn[c] = (size, r["R"], r["exit_day"])
            pool -= size
            trades += 1
            u = use.setdefault(c, dict(money=0.0, trades=0, cycles={}))
            u["money"] += size
            u["trades"] += 1
            key = f"hold{r['hz']}" if r["hold"] else str(r["hz"])
            u["cycles"][key] = u["cycles"].get(key, 0) + 1
    for c in list(opn):
        release(c)
    cash_days = sorted(set(e[0] for e in events))
    gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(cash_days[:-1], cash_days[1:])]
    months = len(days) / 21.0
    return dict(deposits_won=dep, vault_won=vault, acct_won=pool, total_won=pool + vault, gain_over_deposits_pct=100 * ((pool + vault) / dep - 1), trades=trades, losing_trades=len(losses), loss_sum_won=float(sum(losses)), worst_loss_won=float(min(losses)) if losses else 0.0, cash_out_days=len(cash_days), median_days_between_cash_outs=float(np.median(gaps)) if gaps else None, vault_per_month_avg_won=vault / months, vault_last_12_months_won=sum(v for d, v in events if d >= "2025-11-06"), vault_by_year_won=vault_year, use=use)


def hold_one(ins, days):
    pos = {d: i for i, d in enumerate(ins["days"])}
    units, dep = 0.0, 0.0
    for D in days:
        if D in pos and pos[D] + 1 < len(ins["O"]):
            units += DEPOSIT / (ins["O"][pos[D] + 1] * (1 + LI.FEE))
            dep += DEPOSIT
    return dict(deposits_won=dep, total_won=units * ins["C"][-1])


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
    samp = samples(inst)
    print("samples done", flush=True)
    dec = decisions(inst, samp)
    days = sorted(dec)
    names = json.load(open(os.path.join(R.ETF_DIR, "manifest.json"), encoding="utf-8"))["names"]
    res = {}
    by_value = lambda cs: sorted(cs, key=lambda z: -z[1]["v"])
    for k in SLOTS:
        ai = engine(dec, days, k, by_value)
        rnd = []
        for s in range(SEEDS):
            g = np.random.default_rng(900 + 10 * k + s)
            rnd.append(engine(dec, days, k, lambda cs, g=g: [cs[i] for i in g.permutation(len(cs))]))
        res[f"slots{k}"] = dict(AI_ranked=ai, random_order_among_positive=dict(total_won_mean=float(np.mean([x["total_won"] for x in rnd])), vault_won_mean=float(np.mean([x["vault_won"] for x in rnd])), total_won_range=[float(min(x["total_won"] for x in rnd)), float(max(x["total_won"] for x in rnd))], ai_rank_beats=int(sum(ai["total_won"] > x["total_won"] for x in rnd))))
        print(f"slots {k}: deposits {ai['deposits_won'] / 1e4:.0f}만 vault {ai['vault_won'] / 1e4:.1f}만 acct {ai['acct_won'] / 1e4:.1f}만 total {ai['total_won'] / 1e4:.1f}만 ({ai['gain_over_deposits_pct']:+.1f}%) trades {ai['trades']} losing {ai['losing_trades']} worst {ai['worst_loss_won'] / 1e4:.1f}만 cash-out days {ai['cash_out_days']} median gap {ai['median_days_between_cash_outs']} d, vault last 12m {ai['vault_last_12_months_won'] / 1e4:.1f}만 | random order: total {res[f'slots{k}']['random_order_among_positive']['total_won_mean'] / 1e4:.1f}만, AI beats {res[f'slots{k}']['random_order_among_positive']['ai_rank_beats']}/{SEEDS}", flush=True)
        tot = sum(u["money"] for u in ai["use"].values())
        for c, u in sorted(ai["use"].items(), key=lambda z: -z[1]["money"]):
            print(f"    {c} {names[c][:16]:16s} money share {100 * u['money'] / tot:5.1f}% trades {u['trades']:4d} cycles {u['cycles']}", flush=True)
    res["hold_069500_same_deposits"] = hold_one(inst["069500"], days)
    print("hold 069500 with the same deposits:", round(res["hold_069500_same_deposits"]["total_won"] / 1e4, 1), "만", flush=True)
    rec = dict(version="ms-alloc-v1", spec=dict(file=os.path.relpath(spec_path, HERE).replace("\\", "/"), sha256=spec_sha), code_sha256=sha(os.path.abspath(__file__)), inputs=spec["inputs"], results=res)
    with open(out_path, "x", encoding="utf-8") as fo:
        json.dump(rec, fo, ensure_ascii=False, default=float)
    print("saved", out_path, sha(out_path))


if __name__ == "__main__":
    main(*sys.argv[1:4])
