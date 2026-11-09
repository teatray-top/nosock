import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "GLD", "SLV", "TLT", "IEF", "LQD", "HYG", "EEM", "EFA", "VNQ", "USO", "UNG"]


def main(out, start="2004-01-01"):
    t0 = time.time()
    raw = yf.download(ETFS, start=start, auto_adjust=False, progress=False, group_by="ticker", threads=True)
    days = [d.strftime("%Y-%m-%d") for d in raw.index]
    syms = [s for s in ETFS if s in raw.columns.get_level_values(0)]
    D, N = len(days), len(syms)
    O, H, L, C, V, ADJ, DV = [np.full((D, N), np.nan) for _ in range(7)]
    split_log = {}
    dd = np.array(days)
    for j, s in enumerate(syms):
        x = raw[s]
        div = (x["Adj Close"] / x["Close"]).values
        S = np.ones(D)
        try:
            sp = yf.Ticker(s).splits
        except Exception:
            sp = None
        if sp is not None and len(sp):
            for date, ratio in sp.items():
                ds = date.strftime("%Y-%m-%d")
                if ratio and ratio > 0:
                    S[dd < ds] /= float(ratio)
                    split_log.setdefault(s, []).append((ds, float(ratio)))
        adj = div * S
        O[:, j], H[:, j], L[:, j], C[:, j], V[:, j], ADJ[:, j] = x["Open"].values * div, x["High"].values * div, x["Low"].values * div, x["Adj Close"].values, x["Volume"].values, adj
        DV[:, j] = x["Close"].values * x["Volume"].values
    elig = ~np.isnan(C)
    T = np.zeros_like(V)
    np.savez_compressed(out, O=O, H=H, L=L, C=C, V=V, T=T, adj=np.nan_to_num(ADJ, nan=1.0), dv=DV, elig=elig, days=np.array(days), syms=np.array(syms))
    print("splits (raw = adjusted / adj):", split_log, flush=True)
    massive_day = os.path.join(HERE, "data", "massive", "flat", "day")
    for s, lst in split_log.items():
        for ds, ratio in lst:
            prev = [d for d in days if d < ds][-1:]
            f = os.path.join(massive_day, f"{prev[0]}.csv.gz") if prev else ""
            if prev and os.path.exists(f):
                m = pd.read_csv(f, usecols=["ticker", "close"])
                mv = m.loc[m["ticker"] == s, "close"]
                i, j = days.index(prev[0]), syms.index(s)
                print(f"  verify {s} {prev[0]} (day before {ds} split x{ratio}): recovered raw {C[i, j] / ADJ[i, j]:.3f} vs Massive raw {float(mv.iloc[0]) if len(mv) else float('nan'):.3f}", flush=True)
    first = {s: days[int(np.argmax(~np.isnan(C[:, j])))] for j, s in enumerate(syms)}
    json.dump(dict(days=D, syms=syms, first=first, adjusted="dividends+splits via Adj Close ratio"), open(out[:-4] + ".json", "w"), indent=1)
    print(f"etf panel {out}: {D} days x {N} etfs ({days[0]}..{days[-1]}); first dates {first}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "panel_etf_v1.npz"))
