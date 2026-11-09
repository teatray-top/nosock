import os
import sys
import time

import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def universe(cap=2000, low=100):
    df = fdr.StockListing("KRX")
    d = df[df.Market.isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
    d = d[~d.Name.str.contains("스팩|리츠")]
    d = d[d.Code.str.endswith("0")]
    d = d[~d.Dept.fillna("").str.contains("관리|투자주의|SPAC")]
    if cap:
        d = d[(d.Close <= cap) & (d.Close >= low)]
    d["yf"] = d.Code + d.Market.map({"KOSPI": ".KS", "KOSDAQ": ".KQ", "KOSDAQ GLOBAL": ".KQ"})
    return d[["Code", "Name", "Market", "Close", "Amount", "yf"]]


def fetch(iv, period, root=None, ufile="universe.csv"):
    root = root or OUT
    u = pd.read_csv(os.path.join(OUT, ufile), dtype={"Code": str})
    os.makedirs(os.path.join(root, iv), exist_ok=True)
    syms = list(u.yf)
    retry = []
    for i in range(0, len(syms), 40):
        part = syms[i:i + 40]
        df = yf.download(part, interval=iv, period=period, group_by="ticker", auto_adjust=False, progress=False, threads=False)
        for s in part:
            try:
                x = df[s].dropna(how="all")
            except KeyError:
                retry.append(s)
                continue
            if len(x):
                x.to_csv(os.path.join(root, iv, s + ".csv"))
            else:
                retry.append(s)
        print(iv, i + len(part), "/", len(syms), flush=True)
        time.sleep(3)
    if retry and period.endswith("d") and int(period[:-1]) > 365:
        print(iv, "retry", len(retry), "with 1y", flush=True)
        for i in range(0, len(retry), 40):
            part = retry[i:i + 40]
            df = yf.download(part, interval=iv, period="1y", group_by="ticker", auto_adjust=False, progress=False, threads=False)
            for s in part:
                try:
                    x = df[s].dropna(how="all")
                except KeyError:
                    continue
                if len(x):
                    x.to_csv(os.path.join(root, iv, s + ".csv"))
            time.sleep(3)


if __name__ == "__main__":
    if sys.argv[1:2] == ["all"]:
        u = universe(cap=None)
        u.to_csv(os.path.join(OUT, "universe_krx.csv"), index=False, encoding="utf-8-sig")
        print("universe", len(u), flush=True)
        for iv, per in (("60m", "720d"), ("5m", "60d")):
            fetch(iv, per, os.path.join(OUT, "all"), "universe_krx.csv")
        sys.exit()
    if sys.argv[1:2] == ["all1d"]:
        fetch("1d", "3y", os.path.join(OUT, "all"), "universe_krx.csv")
        sys.exit()
    if len(sys.argv) > 2:
        fetch(sys.argv[1], sys.argv[2])
        sys.exit()
    os.makedirs(OUT, exist_ok=True)
    u = universe()
    u.to_csv(os.path.join(OUT, "universe.csv"), index=False, encoding="utf-8-sig")
    print("universe", len(u), flush=True)
    fetch("5m", "60d")
    fetch("1d", "2y")
