import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np
import regime_compare as RC
import ms_alloc_v3 as A3
from nosock import engine as E
from nosock import run_backtest as RB

KR_RISK = {"069500", "229200", "091160", "091170"}
inst, groups, names, classes = RB.load_market()
codes = groups["U29"]
m = E.Market(inst, codes, names)
m.raw = A3.load_raw()
model = RB.model_strategy(inst, codes, m)
ks = RC.Series(RC.yf("IDX_KS11"))


def kr_state(n_ma, lo, hi):
    st, out = 1.0, {}
    for D in m.days:
        w = ks.window(D, n_ma)
        if len(w) >= n_ma:
            ma = sum(w) / n_ma
            if st == 1.0 and w[-1] < ma * lo:
                st = 0.0
            elif st == 0.0 and w[-1] > ma * hi:
                st = 1.0
        out[D] = st
    return out


def make_xform(state, to_inverse):
    def f(D, te):
        if state.get(D, 1.0) >= 1.0:
            return te
        out = dict(te)
        freed = sum(out.pop(c, 0.0) for c in KR_RISK if c in out)
        if freed <= 0:
            return out
        inv = freed * to_inverse
        out["114800"] = out.get("114800", 0.0) + inv
        out["153130"] = out.get("153130", 0.0) + freed - inv
        return out
    return f


runs = [("R1", None)]
for n_ma, lo, hi in ((200, 0.98, 1.01), (120, 0.97, 1.01), (60, 0.96, 1.01)):
    st = kr_state(n_ma, lo, hi)
    runs.append((f"K{n_ma} 국내 주식 → 단기채", make_xform(st, 0.0)))
    runs.append((f"K{n_ma} 국내 주식 → 절반 인버스", make_xform(st, 0.5)))
res = {}
for start in ("2009-05-01", "2017-01-02"):
    for name, xf in runs:
        marks, info = RC.run_x(m, model, start, max(m.days), decide="h20", band=0.02, xform=xf)
        r = RC.stats(marks, info)
        ix = RC.twr(marks)
        d = {k: v for k, v in ix}
        a = max(k for k in d if k <= "2026-06-19")
        r["since_2026_06"] = d[max(d)] / d[a] - 1
        res[f"{start} {name}"] = r
        print(f"{start} {name}: CAGR {100 * r['cagr']:+.2f}% MDD {100 * r['mdd']:.1f}% ({r['mdd_from']}~{r['mdd_to']}) worst20 {100 * r['worst20']:.1f}% 2026-06이후 {100 * r['since_2026_06']:+.1f}% trades {r['trades']} cost {r['cost']:,.0f}", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(HERE), "ft", "results", "regime_kr_trend_v1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
