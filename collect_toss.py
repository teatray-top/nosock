import csv
import json
import os
import sys
import time
from datetime import datetime

from toss import ApiError, Toss

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "toss")
RPS = float(os.environ.get("TOSS_RPS", 12))
MARKETS = {"kr": (("KOSPI", "KOSDAQ"), "005930", "T15:31:00+09:00", "09:01", "15:31"), "us": (("NYSE", "NASDAQ", "AMEX"), "AAPL", None, None, None)}


class Pacer:
    def __init__(self, rps):
        self.dt, self.t = 1.0 / rps, 0.0

    def wait(self):
        now = time.time()
        if now < self.t + self.dt:
            time.sleep(self.t + self.dt - now)
        self.t = time.time()


def log(*a):
    msg = datetime.now().isoformat(timespec="seconds") + " " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(ROOT, "collect.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def universe(t, mk):
    path = os.path.join(ROOT, f"universe_{mk}.json")
    syms = []
    for m in MARKETS[mk][0]:
        for s in t.stocks_all(m, status="ACTIVE", security_type="STOCK" if mk == "kr" else None, common=True):
            syms.append(dict(symbol=s["symbol"], name=s["name"], market=m, type=s["securityType"]))
    json.dump(syms, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return [s["symbol"] for s in syms]


def trading_days(t, ref, n):
    days, before = [], None
    while len(days) < n:
        r = t.candles(ref, "1d", count=200, before=before)
        days += [c["timestamp"][:10] for c in r["candles"]]
        before = r.get("nextBefore")
        if not before:
            break
    return sorted(set(days))[-n:]


def candles_retry(t, pacer, *a, **kw):
    for i in range(5):
        pacer.wait()
        try:
            return t.candles(*a, **kw)
        except (OSError, ApiError) as e:
            if i == 4:
                raise
            log("retry", i + 1, type(e).__name__, str(e)[:120])
            time.sleep(5 * 3 ** i)


def day_bars_kr(t, pacer, sym, day):
    out, before = [], day + MARKETS["kr"][2]
    for _ in range(2):
        r = candles_retry(t, pacer, sym, "1m", count=200, before=before)
        for c in r["candles"]:
            ts = c["timestamp"]
            if ts[:10] == day and "09:01" <= ts[11:16] <= "15:31":
                out.append((ts[:19], c["openPrice"], c["highPrice"], c["lowPrice"], c["closePrice"], c["volume"]))
        before = r.get("nextBefore")
        if not before or r["candles"][-1]["timestamp"][11:16] <= "09:01" or r["candles"][-1]["timestamp"][:10] < day:
            break
    return sorted(out)


def append(sym, rows):
    path = os.path.join(ROOT, "1m", sym + ".csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "open", "high", "low", "close", "volume"])
        w.writerows(rows)


def main(mk="kr", ndays=250):
    os.makedirs(os.path.join(ROOT, "1m"), exist_ok=True)
    t = Toss()
    pacer = Pacer(RPS)
    syms = universe(t, mk)
    days = trading_days(t, MARKETS[mk][1], ndays)
    today = datetime.now().strftime("%Y-%m-%d")
    if days and days[-1] == today and datetime.now().strftime("%H:%M") < "15:45":
        days = days[:-1]
    pp = os.path.join(ROOT, f"progress_{mk}.json")
    done = set(json.load(open(pp)).get("days", [])) if os.path.exists(pp) else set()
    log(f"{mk}: {len(syms)} symbols, {len(days)} days ({days[0]}..{days[-1]}), {len(done)} done, rps {RPS}")
    for day in reversed(days):
        if day in done:
            continue
        t0, n, miss = time.time(), 0, 0
        for sym in syms:
            try:
                rows = day_bars_kr(t, pacer, sym, day)
            except ApiError as e:
                if e.status in (404,):
                    miss += 1
                    continue
                log(f"{day} {sym} error {e}")
                time.sleep(5)
                try:
                    rows = day_bars_kr(t, pacer, sym, day)
                except ApiError as e2:
                    log(f"{day} {sym} skipped {e2}")
                    miss += 1
                    continue
            if rows:
                append(sym, rows)
                n += 1
        done.add(day)
        json.dump(dict(days=sorted(done)), open(pp, "w"))
        log(f"{day}: {n} symbols with bars, {miss} missing, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kr", int(sys.argv[2]) if len(sys.argv) > 2 else 250)
