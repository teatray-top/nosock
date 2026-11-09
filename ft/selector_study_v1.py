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
from nosock import engine as E
from nosock import run_backtest as RB
from nosock import selector as SEL

OUT = os.path.join(HERE, "results", "selector_study_v1.json")
TEST_FROM = 2010


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    t0 = time.time()
    inst, groups = DD.market_all()
    m = E.Market(inst, groups["U29"], {})
    m.raw = A3.load_raw()
    mom = RB.strategy_for(inst, groups["U29"], "dual", m)
    years = list(range(2006, int(MS.ANCHOR[:4]) + 1))
    kinfo = E.choose_k(m, mom, years)
    kpath = {y: v["k"] for y, v in kinfo.items()}
    k_of = lambda D: kpath[int(D[:4])]
    log(f"K path {kpath}")
    ex = SEL.Experts(m, mom, k_of)
    st = SEL.State(m, mom)
    log("state built")
    rows = SEL.dataset(m, ex, st, "2006-01-01", MS.ANCHOR)
    log(f"dataset {len(rows)} rows {rows[0]['t']}..{rows[-1]['t']}")
    cond, stat, oos = {}, {}, {}
    for y in range(TEST_FROM, int(MS.ANCHOR[:4]) + 1):
        tr = [r for r in rows if r["t_end"] < f"{y}-01-01"]
        te = [r for r in rows if r["t"][:4] == str(y)]
        cond[y] = SEL.fit(tr, True)
        stat[y] = SEL.fit(tr, False)
        g = {}
        for name, pol in (("conditional", cond[y]), ("static", stat[y])):
            v = []
            for r in te:
                a, _ = pol.alpha(r["f"])
                v.append(math.log(float(np.dot(a, np.exp(r["y"])))))
            g[name] = float(np.mean(v)) if v else None
        for i, e in enumerate(SEL.EXPERTS):
            g["always_" + e] = float(np.mean([r["y"][i] for r in te])) if te else None
        g["half_mom_eq"] = float(np.mean([math.log(0.5 * math.exp(r["y"][0]) + 0.5 * math.exp(r["y"][1])) for r in te])) if te else None
        g["n"] = len(te)
        g["train_n"] = len(tr)
        g["static_alpha"] = dict(zip(SEL.EXPERTS, [float(x) for x in stat[y].alpha({k: 0.0 for k in SEL.FEATURES})[0]]))
        oos[y] = g
        log(f"{y} n={len(te)} cond {g['conditional']:+.4f} static {g['static']:+.4f} mom {g['always_mom']:+.4f} eq {g['always_eq']:+.4f} def {g['always_def']:+.4f} half {g['half_mom_eq']:+.4f}")
    keys = ["conditional", "static", "always_mom", "always_eq", "always_def", "half_mom_eq"]
    pooled = {k: float(np.mean([r for y in oos for r in [oos[y][k]] if r is not None])) for k in keys}
    wins = {k: sum(1 for y in oos if oos[y]["conditional"] is not None and oos[y][k] is not None and oos[y]["conditional"] > oos[y][k]) for k in keys if k != "conditional"}
    log(f"pooled per-20d mean log growth: " + ", ".join(f"{k} {v:+.4f}" for k, v in pooled.items()))
    log(f"conditional beats (years of {len(oos)}): {wins}")

    fixed = lambda a: {y: SEL.Policy(np.zeros(len(SEL.FEATURES)), np.ones(len(SEL.FEATURES)), np.zeros((3, len(SEL.FEATURES))), np.log(np.array(a) + 1e-12), False) for y in oos}
    strategies = [
        ("S1", "선택 모델 (상태 보고 비중 결정)", SEL.Selector(ex, st, cond, rows)),
        ("S0", "고정 비중 (학습한 평균 비중)", SEL.Selector(ex, st, stat, rows)),
        ("H0", "고정 반반 (모멘텀 50 + 균등 50)", SEL.Selector(ex, st, fixed([0.5, 0.5, 0.0]), rows)),
        ("M0", "늘 모멘텀 집중", SEL.Selector(ex, st, fixed([1.0, 0.0, 0.0]), rows)),
        ("Q0", "늘 29종 균등", SEL.Selector(ex, st, fixed([0.0, 1.0, 0.0]), rows)),
    ]
    report = dict(anchor=MS.ANCHOR, k_path=kpath, test_from=TEST_FROM, features=SEL.FEATURES, experts=SEL.EXPERTS, lam=SEL.LAMBDA, oos=oos, pooled=pooled, cond_wins=wins, runs=[])
    for code, label, strat in strategies:
        row = dict(code=code, label=label)
        for tag, start in (("long", f"{TEST_FROM}-01-04"), ("short", "2016-01-04")):
            r = E.run(m, strat, start, MS.ANCHOR, k_of, whole=True, tax=True, record=True)
            row[tag] = DD.metrics(r)
            if code == "S1" and tag == "short":
                row["alpha_path"] = [dict(day=d["day"], alpha=d.get("alpha")) for d in r["decisions"]]
        report["runs"].append(row)
        L, Sh = row["long"], row["short"]
        log(f"{code} {label}: 2016~ 합계 {Sh['total'] / 1e4:,.0f}만 금고 {Sh['vault'] / 1e4:,.0f}만 순이익낙폭 {Sh['worst_profit_drawdown'] / 1e4:,.0f}만 최대낙폭 {100 * Sh['twr_max_dd']:.1f}% | {TEST_FROM}~ 합계 {L['total'] / 1e4:,.0f}만 TWR연 {100 * L['twr_cagr']:.1f}% 최대낙폭 {100 * L['twr_max_dd']:.1f}% | " + " ".join(f"{k} {100 * v:.0f}%" for k, v in L["episodes"].items()))
    final = SEL.fit(rows, True)
    report["final_policy"] = dict(mu=final.mu.tolist(), sd=final.sd.tolist(), B=final.B.tolist(), b=final.b.tolist(), trained_rows=len(rows), last_label_end=rows[-1]["t_end"])
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)
    log(f"done {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
