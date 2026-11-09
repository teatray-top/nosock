import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime

sys.dont_write_bytecode = True
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
ETFS = {
    "245340": ("us_eq", "TIGER 미국다우존스30"),
    "280930": ("us_small", "KODEX 미국러셀2000(H)"),
    "287180": ("us_tech", "PLUS 미국나스닥테크"),
    "381180": ("us_semi", "TIGER 미국필라델피아반도체나스닥"),
    "200030": ("us_sector", "KODEX 미국S&P500산업재(합성)"),
    "218420": ("us_energy", "KODEX 미국S&P500에너지(합성)"),
    "185680": ("us_health", "KODEX 미국S&P바이오(합성)"),
    "182480": ("us_reit", "TIGER 미국MSCI리츠(합성 H)"),
    "213630": ("us_div", "PLUS 미국다우존스고배당주(합성 H)"),
    "219390": ("us_energy", "RISE 미국S&P원유생산기업(합성 H)"),
    "305080": ("us_bond", "TIGER 미국채10년선물"),
    "267440": ("us_bond", "RISE 미국장기국채선물(H)"),
    "225030": ("us_inv", "TIGER 미국S&P500선물인버스(H)"),
}


def fetch(adj, tries=3):
    tick = [k + ".KS" for k in ETFS]
    for k in range(tries):
        df = yf.download(tick, period="max", interval="1d", auto_adjust=adj, group_by="ticker", progress=False, threads=False)
        bad = [t for t in tick if t not in df.columns.get_level_values(0) or df[t].dropna(subset=["Open", "Close"]).empty]
        if not bad:
            return df, []
        time.sleep(3)
    return df, bad


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    out = os.path.join(HERE, "data", "kr_etf", "yf_kr_us_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    manifest = dict(source="yfinance " + yf.__version__, fetched_at=datetime.now().isoformat(timespec="seconds"), dropped_from=today, names={k: v[1] for k, v in ETFS.items()}, classes={k: v[0] for k, v in ETFS.items()}, files={})
    for adj in (True, False):
        df, bad = fetch(adj)
        for code in ETFS:
            t = code + ".KS"
            key = f"{code}_{'adj' if adj else 'raw'}"
            if t in bad:
                manifest["files"][key] = dict(error="empty after retries")
                continue
            g = df[t].dropna(subset=["Open", "Close"])
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
