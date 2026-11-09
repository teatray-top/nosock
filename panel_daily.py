import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from panel_massive import PLAIN, cs_tickers

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "massive", "flat")


def build(out, min_dv=5e6, min_px=5.0, lookback=60, every=21):
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(ROOT, "day", "*.csv.gz")))
    days = [os.path.basename(f)[:10] for f in files]
    cs = cs_tickers()
    frames = []
    for f in files:
        x = pd.read_csv(f, usecols=["ticker", "volume", "open", "close", "high", "low", "transactions"], dtype={"ticker": str}).dropna(subset=["ticker"])
        x = x[x["ticker"].map(lambda t: bool(PLAIN.match(t)) and t in cs)]
        frames.append(x.set_index("ticker"))
    syms = sorted(set().union(*[set(x.index) for x in frames]))
    jx = {s: j for j, s in enumerate(syms)}
    D, N = len(days), len(syms)
    O, H, L, C, V, T = [np.full((D, N), np.nan) for _ in range(6)]
    for i, x in enumerate(frames):
        j = x.index.map(jx).values
        O[i, j], H[i, j], L[i, j], C[i, j], V[i, j], T[i, j] = x["open"].values, x["high"].values, x["low"].values, x["close"].values, x["volume"].values, x["transactions"].values
    DV = C * V
    sp = os.path.join(ROOT, "..", "splits.json")
    n_adj = 0
    ADJ = np.ones((D, N))
    if os.path.exists(sp):
        di = {d: i for i, d in enumerate(days)}
        for s in json.load(open(sp)):
            t, ex = s.get("ticker"), s.get("execution_date", "")
            if t not in jx or not s.get("split_from") or not s.get("split_to"):
                continue
            i = next((di[d] for d in days if d >= ex), None)
            if i is None or i == 0:
                continue
            ratio = float(s["split_to"]) / float(s["split_from"])
            j = jx[t]
            for a in (O, H, L, C):
                a[:i, j] /= ratio
            V[:i, j] *= ratio
            ADJ[:i, j] /= ratio
            n_adj += 1
    print(f"splits applied: {n_adj}", flush=True)
    dv = DV
    elig = np.zeros((D, N), dtype=bool)
    for i in range(lookback, D, every):
        w = dv[i - lookback:i]
        px = C[i - lookback:i] / ADJ[i - lookback:i]
        ok = (np.sum(~np.isnan(w), 0) >= 0.8 * lookback) & (np.nanmedian(w, 0) >= min_dv) & (np.nanmedian(px, 0) >= min_px)
        elig[i:i + every] = ok
    keep = elig.any(0)
    syms = [s for s, k in zip(syms, keep) if k]
    O, H, L, C, V, T, elig, ADJ, DV = (a[:, keep] for a in (O, H, L, C, V, T, elig, ADJ, DV))
    np.savez_compressed(out, O=O, H=H, L=L, C=C, V=V, T=T, adj=ADJ, dv=DV, elig=elig, days=np.array(days), syms=np.array(syms))
    json.dump(dict(days=D, syms=len(syms), first=days[0], last=days[-1], min_dv=min_dv, min_px=min_px, lookback=lookback, every=every, elig_per_day=float(elig.sum(1)[lookback:].mean())), open(out[:-4] + ".json", "w"), indent=1)
    print(f"daily panel {out}: {D} days x {len(syms)} syms ({days[0]}..{days[-1]}), eligible/day {elig.sum(1)[lookback:].mean():.0f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "panel_daily_v1.npz"))
