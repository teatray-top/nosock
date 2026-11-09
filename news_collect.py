import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
KST = ZoneInfo("Asia/Seoul")
OUT = os.path.join(HERE, "data", "news", "naver_mainnews")
ITEMS = os.path.join(OUT, "items.jsonl")
STATE = os.path.join(OUT, "backfill_state.json")
URL = "https://m.stock.naver.com/api/news/list?category=mainnews&pageSize=20&page={}"
PAUSE = 2.0


def fetch(page):
    req = urllib.request.Request(URL.format(page), headers={"User-Agent": "Mozilla/5.0 research-reader"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def known_keys():
    keys = set()
    if os.path.exists(ITEMS):
        for line in open(ITEMS, encoding="utf-8"):
            x = json.loads(line)
            keys.add((x["oid"], x["aid"]))
    return keys


def append(items, keys, page):
    n = 0
    now = datetime.now(KST).isoformat(timespec="seconds")
    with open(ITEMS, "a", encoding="utf-8") as f:
        for it in items:
            k = (it.get("oid"), it.get("aid"))
            if k in keys:
                continue
            keys.add(k)
            f.write(json.dumps(dict(oid=it.get("oid"), aid=it.get("aid"), dt=it.get("dt"), press=it.get("ohnm"), title=it.get("tit"), summary=it.get("subcontent"), page=page, fetched_at=now), ensure_ascii=False) + "\n")
            n += 1
    return n


def backfill(until):
    os.makedirs(OUT, exist_ok=True)
    st = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else dict(next_page=1, oldest=None)
    keys = known_keys()
    errs = 0
    while True:
        p = st["next_page"]
        try:
            items = fetch(p)
            errs = 0
        except Exception as e:
            errs += 1
            print(datetime.now(KST).strftime("%H:%M:%S"), "page", p, "error", type(e).__name__, flush=True)
            if errs >= 5:
                time.sleep(300)
                errs = 0
            else:
                time.sleep(PAUSE * 5)
            continue
        if not items:
            print("empty page", p, "stop", flush=True)
            break
        n = append(items, keys, p)
        st.update(next_page=p + 1, oldest=items[-1].get("dt"))
        json.dump(st, open(STATE, "w", encoding="utf-8"))
        if p % 50 == 0:
            print(datetime.now(KST).strftime("%H:%M:%S"), "page", p, "oldest", st["oldest"], "new", n, "total", len(keys), flush=True)
        if st["oldest"] and st["oldest"][:8] < until:
            print("reached", st["oldest"], "stop", flush=True)
            break
        time.sleep(PAUSE)


def live(pages=3, every=300):
    os.makedirs(OUT, exist_ok=True)
    keys = known_keys()
    while True:
        now = datetime.now(KST)
        if now.weekday() < 5 and 7 <= now.hour < 17 or now.minute < 5:
            got = 0
            for p in range(1, pages + 1):
                try:
                    got += append(fetch(p), keys, p)
                except Exception as e:
                    print(now.strftime("%m-%d %H:%M"), "live error", type(e).__name__, flush=True)
                time.sleep(PAUSE)
            print(now.strftime("%m-%d %H:%M"), "live new", got, flush=True)
        time.sleep(every)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "backfill":
        backfill(sys.argv[2] if len(sys.argv) > 2 else "20221101")
    elif cmd == "live":
        live()
    else:
        print("usage: news_collect.py backfill YYYYMMDD | live")
