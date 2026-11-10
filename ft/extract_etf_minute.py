import glob
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "massive", "flat", "minute")
OUT = os.path.join(ROOT, "data", "massive", "etf_minute_v1.npz")
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "SMH", "XLF", "XLE", "XLV", "XLU", "XLP", "EFA", "EEM", "TLT", "IEF", "SHY", "GLD", "SLV", "USO", "SH", "UUP"]
NY = ZoneInfo("America/New_York")


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.csv.gz")))
    want = {t: k for k, t in enumerate(TICKERS)}
    days = []
    C = np.full((len(files), len(TICKERS), 390), np.nan, dtype=np.float32)
    O = np.full_like(C, np.nan)
    t0 = time.time()
    for di, f in enumerate(files):
        day = os.path.basename(f)[:10]
        days.append(day)
        y, m, d = map(int, day.split("-"))
        open_ns = int(datetime(y, m, d, 9, 30, tzinfo=NY).timestamp()) * 1_000_000_000
        with gzip.open(f, "rt") as fh:
            fh.readline()
            for line in fh:
                p = line.split(",")
                k = want.get(p[0])
                if k is None:
                    continue
                mi = (int(p[6]) - open_ns) // 60_000_000_000
                if 0 <= mi < 390:
                    O[di, k, mi] = float(p[2])
                    C[di, k, mi] = float(p[3])
        if di % 100 == 0:
            print(day, di, len(files), round(time.time() - t0), "s", flush=True)
    np.savez_compressed(OUT, days=np.array(days), tickers=np.array(TICKERS), O=O, C=C)
    filled = float(np.isfinite(C).mean())
    print("saved", OUT, "days", len(days), "filled", round(filled, 3), round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
