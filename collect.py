import os
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

from fetch import OUT, universe

ARC = os.path.join(OUT, "archive")


def merge(path, new):
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0)
        old.index = pd.to_datetime(old.index, utc=True)
        new = pd.concat([old, new])
    new = new[~new.index.duplicated(keep="last")].sort_index()
    new.to_csv(path)
    return len(new)


def main():
    up = os.path.join(OUT, "universe_all.csv")
    try:
        u = universe(cap=None)
        if os.path.exists(up):
            u = pd.concat([pd.read_csv(up, dtype={"Code": str}), u]).drop_duplicates("yf", keep="last")
        u.to_csv(up, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(datetime.now().isoformat(timespec="seconds"), "universe failed, using cached:", e, flush=True)
        u = pd.read_csv(up, dtype={"Code": str})
    syms = list(u.yf)
    for iv, per in (("1m", "8d"), ("5m", "60d"), ("60m", "60d")):
        os.makedirs(os.path.join(ARC, iv), exist_ok=True)
        n = 0
        for i in range(0, len(syms), 40):
            part = syms[i:i + 40]
            df = yf.download(part, interval=iv, period=per, group_by="ticker", auto_adjust=False, progress=False, threads=False)
            for s in part:
                try:
                    x = df[s].dropna(how="all")
                except KeyError:
                    continue
                if len(x):
                    x.index = pd.to_datetime(x.index, utc=True)
                    merge(os.path.join(ARC, iv, s + ".csv"), x)
                    n += 1
            time.sleep(3)
        print(datetime.now().isoformat(timespec="seconds"), iv, "symbols", n, flush=True)


if __name__ == "__main__":
    main()
