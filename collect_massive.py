import csv
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from massive import ApiError, Client

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "massive")
ET = ZoneInfo("America/New_York")


def log(*a):
    msg = datetime.now().isoformat(timespec="seconds") + " " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(ROOT, "collect.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def last_closed_et():
    return (datetime.now(ET) - timedelta(days=1)).date()


def daily(start):
    d0 = date.fromisoformat(start)
    out = os.path.join(ROOT, "daily")
    os.makedirs(out, exist_ok=True)
    c = Client()
    d, end = d0, last_closed_et()
    n_req, t0 = 0, time.time()
    while d <= end:
        path = os.path.join(out, f"{d}.csv")
        if d.weekday() < 5 and not os.path.exists(path):
            try:
                rows = c.grouped_daily(str(d))
            except ApiError as e:
                log(f"{d} error {e}")
                if e.status == 403:
                    d += timedelta(days=1)
                    continue
                time.sleep(30)
                continue
            n_req += 1
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ticker", "open", "high", "low", "close", "volume", "vwap", "n", "t"])
                for r in rows:
                    w.writerow([r.get("T"), r.get("o"), r.get("h"), r.get("l"), r.get("c"), r.get("v"), r.get("vw"), r.get("n"), r.get("t")])
            if n_req % 20 == 0:
                log(f"daily {d}: {len(rows)} tickers, {n_req} req, {time.time() - t0:.0f}s")
        d += timedelta(days=1)
    log(f"daily done {d0}..{end}, {n_req} requests, {time.time() - t0:.0f}s")


def minute(symfile, start):
    syms = json.load(open(symfile, encoding="utf-8"))
    out = os.path.join(ROOT, "1m")
    os.makedirs(out, exist_ok=True)
    c = Client()
    end = str(last_closed_et())
    pp = os.path.join(ROOT, "progress_1m.json")
    done = json.load(open(pp)).get("done", {}) if os.path.exists(pp) else {}
    log(f"minute: {len(syms)} symbols {start}..{end}, {len(done)} done")
    t0 = time.time()
    for i, s in enumerate(syms):
        if done.get(s) == end:
            continue
        frm = done[s] if s in done else start
        try:
            rows = c.minute_bars(s, frm, end)
        except ApiError as e:
            log(f"{s} error {e}")
            if e.status in (403, 404):
                done[s] = end
                json.dump(dict(done=done), open(pp, "w"))
            continue
        path = os.path.join(out, s + ".csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["t", "open", "high", "low", "close", "volume", "vwap", "n"])
            for r in rows:
                if not new and str(datetime.fromtimestamp(r["t"] / 1000, ET).date()) <= frm:
                    continue
                w.writerow([r["t"], r["o"], r["h"], r["l"], r["c"], r["v"], r.get("vw"), r.get("n")])
        done[s] = end
        json.dump(dict(done=done), open(pp, "w"))
        if (i + 1) % 25 == 0:
            log(f"minute {i + 1}/{len(syms)} {s}: {len(rows)} bars, {time.time() - t0:.0f}s")
    log(f"minute done {len(syms)} symbols, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    if sys.argv[1] == "daily":
        daily(sys.argv[2] if len(sys.argv) > 2 else "2024-11-01")
    elif sys.argv[1] == "minute":
        minute(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "2024-11-05")
