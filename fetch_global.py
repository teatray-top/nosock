import hashlib
import json
import math
import os
import sys
from datetime import datetime

sys.dont_write_bytecode = True
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
TICKERS = {
    "crypto": ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "ADA-USD"],
    "metals": ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"],
    "energy": ["CL=F", "BZ=F", "NG=F", "HO=F", "RB=F"],
    "agri": ["ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F", "CC=F"],
    "equity_index": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^N225", "^HSI", "^STOXX50E", "^FTSE", "^GDAXI", "^KS11", "^KQ11", "000001.SS", "^TWII"],
    "macro": ["^VIX", "^TNX", "KRW=X", "DX-Y.NYB", "JPY=X", "EURUSD=X"],
}


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    out = os.path.join(HERE, "data", "global", "yf_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    allt = [t for g in TICKERS.values() for t in g]
    df = yf.download(allt, period="max", interval="1d", auto_adjust=True, group_by="ticker", progress=False, threads=True)
    manifest = dict(source="yfinance " + yf.__version__, auto_adjust=True, fetched_at=datetime.now().isoformat(timespec="seconds"), dropped_from=today, groups=TICKERS, files={})
    for t in allt:
        try:
            g = df[t].dropna(subset=["Open", "Close"])
        except KeyError:
            manifest["files"][t] = dict(error="missing")
            continue
        rows = [(i.strftime("%Y-%m-%d"), float(r.Open), float(r.High), float(r.Low), float(r.Close), float(r.Volume) if not math.isnan(r.Volume) else 0.0) for i, r in g.iterrows()]
        rows = [r for r in rows if r[0] < today and r[1] > 0 and r[4] > 0]
        rec = dict(ticker=t, days=[r[0] for r in rows], O=[r[1] for r in rows], H=[r[2] for r in rows], L=[r[3] for r in rows], C=[r[4] for r in rows], V=[r[5] for r in rows])
        name = t.replace("^", "IDX_").replace("=", "_").replace(".", "_") + ".json"
        data = json.dumps(rec).encode("utf-8")
        with open(os.path.join(out, name), "xb") as f:
            f.write(data)
        manifest["files"][t] = dict(file=name, sha256=hashlib.sha256(data).hexdigest(), rows=len(rows), first=rec["days"][0] if rows else None, last=rec["days"][-1] if rows else None)
    data = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
    with open(os.path.join(out, "manifest.json"), "xb") as f:
        f.write(data)
    for t, v in manifest["files"].items():
        print(t, v.get("rows"), v.get("first"), v.get("last"), v.get("error", ""))
    print(out, hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
