import json
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
start, end = "2007-02-01", max(m.days)
dials = RC.build_dials(m.days)
half = {k: {D: (0.5 if s < 1.0 else 1.0) for D, s in v.items()} for k, v in dials.items()}
runs = [
    ("R1 기준 (모델 20일 + 2%p 매도)", dict(decide="h20", band=0.02)),
    ("T1 R1 + 추세", dict(decide="h20", band=0.02, dial=dials["trend"], safe=["SHY"])),
    ("T2 R1 + 변동성 구조", dict(decide="h20", band=0.02, dial=dials["vts"], safe=["SHY"])),
    ("T3 R1 + 신용", dict(decide="h20", band=0.02, dial=dials["credit"], safe=["SHY"])),
    ("T4 R1 + 사전 합", dict(decide="h20", band=0.02, dial=dials["pre"], safe=["SHY"])),
    ("T5 R1 + 전부", dict(decide="h20", band=0.02, dial=dials["combo"], safe=["SHY"])),
    ("T6 R1 + 사전 합 (절반만 줄임)", dict(decide="h20", band=0.02, dial=half["pre"], safe=["SHY"])),
    ("T7 R1 + 전부 (절반만 줄임)", dict(decide="h20", band=0.02, dial=half["combo"], safe=["SHY"])),
    ("T8 R1 + 사전 합, 현금으로", dict(decide="h20", band=0.02, dial=dials["pre"], safe=None)),
    ("T9 R1 + 사전 합, 국내 단기채로", dict(decide="h20", band=0.02, dial=dials["pre"], safe=["153130"])),
    ("T10 R1 + 사전 합, 밴드 5%p", dict(decide="h20", band=0.05, dial=dials["pre"], safe=["SHY"])),
]
res = {}
for name, kw in runs:
    t1 = time.time()
    marks, info = RC.run_x(m, model, start, end, **kw)
    r = RC.stats(marks, info)
    res[name] = r
    print(f"{name}: CAGR {100 * r['cagr']:+.2f}% MDD {100 * r['mdd']:.1f}% ({r['mdd_from']}~{r['mdd_to']}) worst20 {100 * r['worst20']:.1f}% trades {r['trades']} cost {r['cost']:,.0f}", flush=True)
json.dump(dict(start=start, end=end, windows=RC.WINDOWS, runs=res), open(os.path.join(os.path.dirname(HERE), "ft", "results", "regime_compare_r1dial_v1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
W = [w[0] for w in RC.WINDOWS]
print("창 | " + " | ".join(n.split()[0] for n in res))
for w in W:
    print(w, "|", " | ".join(("%+.1f/%.1f" % (100 * x["ret"], 100 * x["mdd"])) if (x := res[n]["windows"].get(w)) else "-" for n in res))
