import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import ms_alloc_v3 as A3
import mom_shadow as MS
from nosock import allocator as AL
from nosock import engine as E
from nosock import run_backtest as RB
from nosock import selector as SEL

OUT = os.path.join(ROOT, "nosock", "policies_v1.json")


def main():
    t0 = time.time()
    inst, groups, names, classes = RB.load_market()
    codes = groups["U29"]
    m = E.Market(inst, codes, names)
    m.raw = A3.load_raw()
    st = SEL.State(m, RB.strategy_for(inst, codes, "dual", m))
    feats = AL.Features(m, st)
    rows = AL.dataset(m, feats, "2006-01-01", MS.ANCHOR)
    print(f"dataset {len(rows)} rows {rows[0]['t']}..{rows[-1]['t']}", flush=True)
    out = {}
    for y in range(2008, int(MS.ANCHOR[:4]) + 2):
        tr = [r for r in rows if r["t_end"] < f"{y}-01-01"]
        if len(tr) < 100:
            print(y, "skip", len(tr), flush=True)
            continue
        pol, lam, scores = AL.fit_auto(tr, f"{y}-01-01")
        out[str(y)] = dict(beta=[float(x) for x in pol.beta], gamma=[float(x) for x in pol.gamma], c0=float(pol.c0), smu=[float(x) for x in pol.smu], ssd=[float(x) for x in pol.ssd], lam=lam, rows=len(tr), last_label_end=tr[-1]["t_end"])
        print(y, f"lam={lam:g} rows={len(tr)} beta=" + " ".join(f"{k}{v:+.2f}" for k, v in zip(AL.ASSET_F, pol.beta)), f"{time.time() - t0:.0f}s", flush=True)
    rep = dict(anchor=MS.ANCHOR, asset_features=AL.ASSET_F, state_features=AL.STATE_F, years=out, note="year y policy is trained only on 20-day windows that ended before y-01-01")
    tmp = OUT + ".tmp"
    json.dump(rep, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)
    print("done", f"{time.time() - t0:.0f}s", OUT, flush=True)


if __name__ == "__main__":
    main()
