import json
import math
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
start, end = "2007-02-01", max(m.days)
dials = RC.build_dials(m.days)
marks, info = RC.run_x(m, model, start, end, decide="h20", band=0.02)
ix = RC.twr(marks)
days = [d for d, _ in ix]
v = np.array([x for _, x in ix])
r = np.r_[0.0, np.log(v[1:] / v[:-1])]
for name in ("vts", "credit", "trend", "event", "pre", "combo"):
    s = np.array([dials[name].get(d, 1.0) for d in days])
    off = np.r_[False, s[:-1] < 1.0]
    eps, k = [], 0
    while k < len(days):
        if off[k]:
            j = k
            while j < len(days) and off[j]:
                j += 1
            seg = r[k:j].sum()
            eps.append((days[k], days[j - 1], j - k, seg))
            k = j
        else:
            k += 1
    loss_eps = [e for e in eps if e[3] < -0.03]
    gain_eps = [e for e in eps if e[3] > 0.0]
    print(f"[{name}] off {100 * off.mean():.0f}% of days, {len(eps)} alarms; R1 log-return while off {100 * r[off].sum():+.0f}% (on {100 * r[~off].sum():+.0f}%); alarms before a >3% loss {len(loss_eps)}, alarms during gains {len(gain_eps)}")
    top = sorted(eps, key=lambda e: e[3])[:3]
    worst = sorted(eps, key=lambda e: -e[3])[:3]
    print("   best saves:", [(a, n, round(100 * x, 1)) for a, b, n, x in top])
    print("   worst misses:", [(a, n, round(100 * x, 1)) for a, b, n, x in worst])
