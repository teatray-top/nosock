import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toss

SYM = "114800"
MIN_FROM = "2022-11-23"
LAST_DAY = "2026-11-06"
GAP = 0.25
CAP = 2600


class NoIssue(Exception):
    pass


def client():
    t = toss.Toss()
    if not t.tok or t.exp < time.time() + 900:
        raise SystemExit("cached token missing or under 15 minutes; not issuing")

    def refuse():
        raise NoIssue("token issue refused")

    t._issue = refuse
    return t


def page(t, interval, before, adjusted, log, n):
    if n[0] >= CAP:
        raise SystemExit("request cap reached")
    n[0] += 1
    params = dict(symbol=SYM, interval=interval, count=200, adjusted="true" if adjusted else "false")
    if before:
        params["before"] = before
    t0 = time.time()
    r = t.get("/api/v1/candles", **params)
    log.write((json.dumps(dict(n=n[0], interval=interval, adjusted=adjusted, before=before, at=datetime.now().isoformat(timespec="milliseconds"), elapsed=round(time.time() - t0, 3), result=r), ensure_ascii=False) + "\n").encode("utf-8"))
    time.sleep(GAP)
    return r


def daily(t, adjusted, log, n):
    rows, before = {}, None
    while True:
        r = page(t, "1d", before, adjusted, log, n)
        c = r.get("candles") or []
        for x in c:
            rows[x["timestamp"][:10]] = [float(x["openPrice"]), float(x["highPrice"]), float(x["lowPrice"]), float(x["closePrice"]), float(x["volume"])]
        before = r.get("nextBefore")
        if not c or not before or len(c) < 200:
            break
    days = sorted(d for d in rows if d <= LAST_DAY)
    return dict(days=days, O=[rows[d][0] for d in days], H=[rows[d][1] for d in days], L=[rows[d][2] for d in days], C=[rows[d][3] for d in days], V=[rows[d][4] for d in days])


def minutes(t, log, n):
    bars, before = {}, LAST_DAY + "T23:59:59+09:00"
    while True:
        r = page(t, "1m", before, False, log, n)
        c = r.get("candles") or []
        for x in c:
            ts = x["timestamp"][:19]
            if ts[:10] < MIN_FROM or ts[:10] > LAST_DAY:
                continue
            row = (x["openPrice"], x["highPrice"], x["lowPrice"], x["closePrice"], x["volume"])
            if ts in bars and bars[ts] != row:
                raise SystemExit(f"conflicting duplicate {ts}")
            bars[ts] = row
        before = r.get("nextBefore")
        if not c or not before or c[-1]["timestamp"][:10] < MIN_FROM:
            break
    return bars


def main():
    out = os.path.join(HERE, "data", "kr_etf", "inverse_114800_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    t = client()
    n = [0]
    with gzip.open(os.path.join(out, "responses.jsonl.gz"), "wb") as log:
        raw = daily(t, False, log, n)
        adj = daily(t, True, log, n)
        if raw["days"] != adj["days"]:
            raise SystemExit("raw and adjusted calendars differ")
        bars = minutes(t, log, n)
    with open(os.path.join(out, "daily_raw.json"), "x", encoding="utf-8") as f:
        json.dump({SYM: raw}, f)
    with open(os.path.join(out, "daily_adj.json"), "x", encoding="utf-8") as f:
        json.dump({SYM: adj}, f)
    with open(os.path.join(out, SYM + ".csv"), "x", encoding="utf-8", newline="") as f:
        f.write("ts,open,high,low,close,volume\n")
        for ts in sorted(bars):
            f.write(ts + "," + ",".join(str(v) for v in bars[ts]) + "\n")
    per_day = {}
    for ts in bars:
        per_day[ts[:10]] = per_day.get(ts[:10], 0) + 1
    summary = dict(symbol=SYM, requests=n[0], daily_days=len(raw["days"]), daily_first=raw["days"][0], daily_last=raw["days"][-1], minute_days=len(per_day), minute_bars=len(bars), minute_first=min(bars), minute_last=max(bars), days_not_390=sorted((d, k) for d, k in per_day.items() if k != 390), finished=datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(out, "summary.json"), "x", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    for name in ("daily_raw.json", "daily_adj.json", SYM + ".csv", "summary.json"):
        p = os.path.join(out, name)
        print(name, hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
    print(json.dumps({k: v for k, v in summary.items() if k != "days_not_390"}, ensure_ascii=False))
    print("days_not_390", len(summary["days_not_390"]), summary["days_not_390"][:12])
    print(out)


if __name__ == "__main__":
    main()
