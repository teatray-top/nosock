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

inst, groups, names, classes = RB.load_market()
codes = groups["U29"]
m = E.Market(inst, codes, names)
m.raw = A3.load_raw()
model = RB.model_strategy(inst, codes, m)
eq = RB.strategy_for(inst, codes, "equal", m)
for label, strat in (("R1 model", model), ("E0 equal", eq)):
    marks, info = RC.run_x(m, strat, "2025-06-02", max(m.days), decide="h20", band=0.02)
    ix = RC.twr(marks)
    by = {}
    for d, v in ix:
        by[d[:7]] = v
    prev = None
    line = []
    for mo in sorted(by):
        if prev:
            line.append(f"{mo[2:]}:{100 * (by[mo] / prev - 1):+.1f}")
        prev = by[mo]
    print(label, " ".join(line))
for D in ("2026-06-01", "2026-07-01", "2026-08-03", "2026-09-01", "2026-10-01", "2026-11-02"):
    tw, why = model.decide(D, 1)
    top = sorted(tw.items(), key=lambda kv: -kv[1])[:6]
    print(D, "cash %.0f%%" % (100 * why.get("cash_weight", 0)), [(c, round(100 * w)) for c, w in top])
ret = {}
for c in codes:
    a, _ = m.close_at(c, "2026-06-19")
    b, _ = m.close_at(c, max(m.days))
    if a and b:
        ret[c] = round(100 * (b / a - 1), 1)
print("KRW-terms return 2026-06-19 -> end:", sorted(ret.items(), key=lambda kv: kv[1]))
