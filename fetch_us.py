import io
import os
import sys
import time
import urllib.request

import pandas as pd
import yfinance as yf

from fetch import OUT

US = os.path.join(OUT, "us")
SRC = ["https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt", "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"]


def listing():
    rows = []
    for url in SRC:
        t = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
        df = pd.read_csv(io.StringIO(t), sep="|")
        df = df[~df.iloc[:, 0].astype(str).str.startswith("File Creation")]
        sym = df["Symbol"] if "Symbol" in df else df["ACT Symbol"]
        etf = df["ETF"] if "ETF" in df else "N"
        test = df["Test Issue"]
        name = df["Security Name"].astype(str)
        d = pd.DataFrame({"sym": sym.astype(str), "name": name, "etf": etf, "test": test})
        rows.append(d)
    d = pd.concat(rows)
    d = d[(d.test == "N") & (d.etf == "N")]
    d = d[~d.sym.str.contains(r"[\$\.\^=]")]
    d = d[~d.name.str.contains(r"\b(Warrants?|Rights?|Units?)\b|Preferred|Depositary|Notes|%|Acquisition Corp|Acquisition Co\b|SPAC|Blank Check", case=False, regex=True)]
    return d.drop_duplicates("sym")


def rank(d, top=1500):
    syms = list(d.sym)
    stats = []
    for i in range(0, len(syms), 200):
        part = syms[i:i + 200]
        x = yf.download(part, period="1mo", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        for s in part:
            try:
                c = x[s]["Close"].dropna()
                v = x[s]["Volume"].dropna()
            except KeyError:
                continue
            if len(c) >= 15:
                stats.append((s, float(c.iloc[-1]), float((c * v).median())))
        print("rank", i + len(part), "/", len(syms), flush=True)
        time.sleep(2)
    s = pd.DataFrame(stats, columns=["sym", "close", "dollar_vol"])
    s = s[(s.close >= 1.0) & (s.dollar_vol >= 2e6)]
    return s.sort_values("dollar_vol", ascending=False).head(top)


def fetch(u, iv, period):
    os.makedirs(os.path.join(US, iv), exist_ok=True)
    fresh = time.time() - 20 * 3600
    done = lambda s: os.path.exists(os.path.join(US, iv, s + ".csv")) and os.path.getmtime(os.path.join(US, iv, s + ".csv")) > fresh
    syms = [str(x) for x in u.sym if not done(str(x))]
    for i in range(0, len(syms), 40):
        part = syms[i:i + 40]
        df = yf.download(part, interval=iv, period=period, group_by="ticker", auto_adjust=False, progress=False, threads=False, prepost=False)
        for s in part:
            try:
                x = df[s].dropna(how="all")
            except KeyError:
                continue
            if len(x):
                x.to_csv(os.path.join(US, iv, s + ".csv"))
        print(iv, i + len(part), "/", len(syms), flush=True)
        time.sleep(3)


if __name__ == "__main__" and sys.argv[1:] == ["all"]:
    US = os.path.join(OUT, "usall")
    os.makedirs(US, exist_ok=True)
    up = os.path.join(US, "universe.csv")
    if not os.path.exists(up):
        listing().to_csv(up, index=False)
    u = pd.read_csv(up, keep_default_na=False)
    u = u[u.sym.astype(str).str.len() > 0]
    print("universe", len(u), flush=True)
    fetch(u, "5m", "60d")
    fetch(u, "60m", "720d")
elif __name__ == "__main__":
    os.makedirs(US, exist_ok=True)
    up = os.path.join(US, "universe.csv")
    if os.path.exists(up):
        u = pd.read_csv(up, keep_default_na=False)
    else:
        u = rank(listing())
        u.to_csv(up, index=False)
    print("universe", len(u), flush=True)
    fetch(u, "60m", "720d")
    fetch(u, "5m", "60d")
    fetch(u, "1d", "5y")
