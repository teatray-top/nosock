import cProfile
import os
import pstats
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
t = time.time()
intr = RC.load_minute(0.02)
print("load_minute", round(time.time() - t, 1), "s", type(intr["C"]), flush=True)
pr = cProfile.Profile()
pr.enable()
t = time.time()
import faulthandler
faulthandler.dump_traceback_later(60, exit=True)
for y0, y1 in (("2021-11-05", "2022-12-31"), ("2021-11-05", "2023-12-31"), ("2021-11-05", "2024-12-31"), ("2021-11-05", "2025-12-31"), ("2021-11-05", max(m.days))):
    t1 = time.time()
    intr = RC.load_minute(0.02)
    marks, info = RC.run_x(m, model, y0, y1, decide="h20", band=0.02, intraday=intr)
    print("upto", y1, round(time.time() - t1, 1), "s", flush=True)
pr.disable()
print("run_x 5 months", round(time.time() - t, 1), "s hits", intr["hits"], flush=True)
pstats.Stats(pr).sort_stats("cumulative").print_stats(12)
