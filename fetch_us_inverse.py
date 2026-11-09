import hashlib
import json
import math
import os
import sys
from datetime import datetime

sys.dont_write_bytecode = True
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
ETFS = {
    "PSQ": ("us_inv", "Nasdaq 100 inverse 1x"), "DOG": ("us_inv", "Dow 30 inverse 1x"), "RWM": ("us_inv", "Russell 2000 inverse 1x"),
    "SEF": ("us_inv", "Financials inverse 1x"), "REK": ("us_inv", "Real estate inverse 1x"), "EFZ": ("us_inv", "Developed ex-US inverse 1x"),
    "EUM": ("us_inv", "Emerging inverse 1x"), "TBF": ("us_inv", "US Treasury 20y+ inverse 1x"), "SJB": ("us_inv", "High yield inverse 1x"),
}
FX = "KRW=X"


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    out = os.path.join(HERE, "data", "us_etf", "yf_us_inv1x_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    manifest = dict(source="yfinance " + yf.__version__, fetched_at=datetime.now().isoformat(timespec="seconds"), dropped_from=today, names={k: v[1] for k, v in ETFS.items()}, classes={k: v[0] for k, v in ETFS.items()}, files={})
    for adj in (True, False):
        tick = list(ETFS)
        df = yf.download(tick, period="max", interval="1d", auto_adjust=adj, group_by="ticker", progress=False, threads=True)
        for code in tick:
            key = f"{code.replace('=', '_')}_{'adj' if adj else 'raw'}"
            try:
                g = df[code].dropna(subset=["Open", "Close"])
            except KeyError:
                manifest["files"][key] = dict(error="missing")
                continue
            rows = [(i.strftime("%Y-%m-%d"), float(r.Open), float(r.High), float(r.Low), float(r.Close), float(r.Volume) if not math.isnan(r.Volume) else 0.0) for i, r in g.iterrows()]
            rows = [r for r in rows if r[0] < today and r[1] > 0 and r[4] > 0]
            rec = dict(code=code, adjusted=adj, days=[r[0] for r in rows], O=[r[1] for r in rows], H=[r[2] for r in rows], L=[r[3] for r in rows], C=[r[4] for r in rows], V=[r[5] for r in rows])
            data = json.dumps(rec).encode("utf-8")
            with open(os.path.join(out, key + ".json"), "xb") as f:
                f.write(data)
            manifest["files"][key] = dict(file=key + ".json", sha256=hashlib.sha256(data).hexdigest(), rows=len(rows), first=rec["days"][0] if rows else None, last=rec["days"][-1] if rows else None)
    data = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
    with open(os.path.join(out, "manifest.json"), "xb") as f:
        f.write(data)
    for k, v in manifest["files"].items():
        print(k, v.get("rows"), v.get("first"), v.get("last"), v.get("error", ""))
    print(out, hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
