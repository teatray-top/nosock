import hashlib
import json
import math
import os
import sys
from datetime import datetime

sys.dont_write_bytecode = True
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
ETFS_LEV = {"122630": "KODEX 레버리지", "233740": "KODEX 코스닥150레버리지", "225040": "TIGER 미국S&P500레버리지(합성 H)", "409820": "KODEX 미국나스닥100레버리지(합성 H)", "252670": "KODEX 200선물인버스2X"}
ETFS = {
    "069500": "KODEX 200", "229200": "KODEX 코스닥150", "114800": "KODEX 인버스", "091160": "KODEX 반도체", "091170": "KODEX 은행",
    "133690": "TIGER 미국나스닥100", "143850": "TIGER 미국S&P500선물(H)", "360750": "TIGER 미국S&P500", "192090": "TIGER 차이나CSI300",
    "132030": "KODEX 골드선물(H)", "130680": "TIGER 원유선물Enhanced(H)", "261240": "KODEX 미국달러선물",
    "148070": "KOSEF 국고채10년", "114260": "KODEX 국고채3년", "153130": "KODEX 단기채권",
}


def main(which="base"):
    global ETFS
    if which == "lev":
        ETFS = ETFS_LEV
    today = datetime.now().strftime("%Y-%m-%d")
    out = os.path.join(HERE, "data", "kr_etf", ("yf_lev_" if which == "lev" else "yf_etfs_") + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    manifest = dict(source="yfinance " + yf.__version__, fetched_at=datetime.now().isoformat(timespec="seconds"), dropped_from=today, names=ETFS, files={})
    for adj in (True, False):
        tick = [k + ".KS" for k in ETFS]
        df = yf.download(tick, period="max", interval="1d", auto_adjust=adj, group_by="ticker", progress=False, threads=True)
        for code in ETFS:
            t = code + ".KS"
            key = f"{code}_{'adj' if adj else 'raw'}"
            try:
                g = df[t].dropna(subset=["Open", "Close"])
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
    main(*sys.argv[1:2])
