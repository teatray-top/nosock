import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np
import ms_alloc_v3 as A3
import mom_shadow as MS
import dd_study_v1 as DD
from nosock import allocator as AL
from nosock import engine as E
from nosock import run_backtest as RB
from nosock import selector as SEL

OUT = os.path.join(HERE, "results", "alloc_policy_study_v2.json")
LAMS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def growth(pol, rows):
    g = []
    for r in rows:
        w, _, _, _ = pol.weights(dict(codes=r["codes"], Z=r["Z"], s=r["s"]))
        port = sum(w.get(c, 0.0) * R * (1 - k) for c, R, k in zip(r["codes"], r["R"], r["cost"])) + (1 - sum(w.values()))
        g.append(math.log(port))
    return float(np.mean(g)) if g else -1e9


def pick_lambda(tr, y):
    cut = f"{y - 3}-01-01"
    fit_rows = [r for r in tr if r["t_end"] < cut]
    val = [r for r in tr if r["t"] >= cut]
    if len(fit_rows) < 100 or len(val) < 30:
        return AL.LAMBDA, {}
    scores = {lam: growth(AL.fit(fit_rows, lam), val) for lam in LAMS}
    return max(scores, key=scores.get), scores
TEST_FROM = 2010


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    t0 = time.time()
    inst, groups = DD.market_all()
    m = E.Market(inst, groups["U29"], {})
    m.raw = A3.load_raw()
    mom = RB.strategy_for(inst, groups["U29"], "dual", m)
    st = SEL.State(m, mom)
    feats = AL.Features(m, st)
    log("features built")
    rows = AL.dataset(m, feats, "2006-01-01", MS.ANCHOR)
    log(f"dataset {len(rows)} rows {rows[0]['t']}..{rows[-1]['t']}")
    pols, oos = {}, {}
    for y in range(TEST_FROM, int(MS.ANCHOR[:4]) + 1):
        tr = [r for r in rows if r["t_end"] < f"{y}-01-01"]
        te = [r for r in rows if r["t"][:4] == str(y)]
        lam, lam_scores = pick_lambda(tr, y)
        pol = AL.fit(tr, lam)
        pols[y] = pol
        g_pol, g_eq = [], []
        for r in te:
            f = dict(codes=r["codes"], Z=r["Z"], s=r["s"], raw=r["Z"])
            w, cash, _, _ = pol.weights(f)
            port = sum(w.get(c, 0.0) * R * (1 - k) for c, R, k in zip(r["codes"], r["R"], r["cost"])) + (1 - sum(w.values()))
            g_pol.append(math.log(port))
            g_eq.append(math.log(float(np.mean(r["R"] * (1 - r["cost"])))))
        oos[y] = dict(lam=lam, lam_scores={str(k): v for k, v in lam_scores.items()}, n=len(te), policy=float(np.mean(g_pol)) if te else None, equal=float(np.mean(g_eq)) if te else None, beta=dict(zip(AL.ASSET_F, map(float, pol.beta))), gamma=dict(zip(AL.STATE_F, map(float, pol.gamma))), c0=pol.c0)
        log(f"{y} lam={lam:g} n={len(te)} policy {oos[y]['policy']:+.4f} equal {oos[y]['equal']:+.4f} beta " + " ".join(f"{k}{v:+.2f}" for k, v in oos[y]["beta"].items()) + f" c0 {pol.c0:+.2f}")
    pooled = dict(policy=float(np.mean([v["policy"] for v in oos.values() if v["policy"] is not None])), equal=float(np.mean([v["equal"] for v in oos.values() if v["equal"] is not None])))
    wins = sum(1 for v in oos.values() if v["policy"] is not None and v["policy"] > v["equal"])
    log(f"pooled per-20d log growth policy {pooled['policy']:+.4f} equal {pooled['equal']:+.4f}; policy beats equal in {wins}/{len(oos)} years")
    years = list(range(2006, int(MS.ANCHOR[:4]) + 1))
    kinfo = E.choose_k(m, mom, years)
    k_of = lambda D: kinfo[int(D[:4])]["k"]
    strategies = [
        ("A1", "후보 → 비율 모델", AL.Allocator(m, feats, pols), k_of),
        ("Q0", "29종 균등", RB.strategy_for(inst, groups["U29"], "equal", m), lambda D: 1),
        ("M0", "모멘텀 집중 (듀얼)", mom, k_of),
    ]
    report = dict(anchor=MS.ANCHOR, test_from=TEST_FROM, asset_features=AL.ASSET_F, state_features=AL.STATE_F, lam=AL.LAMBDA, oos=oos, pooled=pooled, wins=wins, runs=[])
    for code, label, strat, kf in strategies:
        row = dict(code=code, label=label)
        for tag, start in (("long", f"{TEST_FROM}-01-04"), ("short", "2016-01-04")):
            r = E.run(m, strat, start, MS.ANCHOR, kf, whole=True, tax=True, record=True)
            row[tag] = DD.metrics(r)
            if code == "A1" and tag == "short":
                row["last_decision"] = r["decisions"][-1] if r["decisions"] else None
                row["avg_names"] = float(np.mean([len(d["targets"]) for d in r["decisions"]])) if r["decisions"] else None
                row["avg_cash"] = float(np.mean([d.get("cash_weight", 0.0) for d in r["decisions"]])) if r["decisions"] else None
        report["runs"].append(row)
        L, Sh = row["long"], row["short"]
        log(f"{code} {label}: 2016~ 합계 {Sh['total'] / 1e4:,.0f}만 금고 {Sh['vault'] / 1e4:,.0f}만 순이익낙폭 {Sh['worst_profit_drawdown'] / 1e4:,.0f}만 최대낙폭 {100 * Sh['twr_max_dd']:.1f}% 비용 {Sh['costs'] / 1e4:,.0f}만 세금 {Sh['taxes'] / 1e4:,.0f}만 | {TEST_FROM}~ 합계 {L['total'] / 1e4:,.0f}만 TWR연 {100 * L['twr_cagr']:.1f}% 최대낙폭 {100 * L['twr_max_dd']:.1f}% | " + " ".join(f"{k} {100 * v:.0f}%" for k, v in L["episodes"].items()) + (f" | 평균 {row.get('avg_names', 0):.1f}종목 현금 {100 * row.get('avg_cash', 0):.0f}%" if code == "A1" else ""))
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)
    log(f"done {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
