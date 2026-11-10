import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import regime_compare as RC
import ms_alloc_v3 as A3
from nosock import engine as E
from nosock import run_backtest as RB

inst, groups, names, classes = RB.load_market()
codes = groups["U29"]
m = E.Market(inst, codes, names)
m.raw = A3.load_raw()
model = RB.model_strategy(inst, codes, m)
start, end = "2016-01-04", max(m.days)
E.VAULT_SHARE = 0.0
t = time.time()
r = E.run(m, model, start, end, lambda D: 1, whole=True, tax=False, record=True, deposit=E.DEPOSIT)
mk = [(x["day"], x["total"], x["deposits"]) for x in r["marks"]]
a = RC.stats(mk, dict(total=r["total"], deposits=r["deposits"], trades=len(r["fills"]), cost=sum(f.get("cost", 0) for f in r["fills"])))
print(f"engine  E.run: CAGR {100 * a['cagr']:+.2f}% MDD {100 * a['mdd']:.1f}% profit {a['profit']:,.0f} trades {a['trades']} ({time.time() - t:.0f}s)", flush=True)
t = time.time()
mk2, info = RC.run_x(m, model, start, end, decide="h20")
b = RC.stats(mk2, info)
print(f"new   run_x R0: CAGR {100 * b['cagr']:+.2f}% MDD {100 * b['mdd']:.1f}% profit {b['profit']:,.0f} trades {b['trades']} ({time.time() - t:.0f}s)", flush=True)
