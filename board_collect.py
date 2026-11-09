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
URL = "https://m.stock.naver.com/front-api/discussion/list?discussionType=domesticStock&itemCode={code}&pageSize=50{off}"
PAUSE = 2.0


def paths(code):
    out = os.path.join(HERE, "data", "community", f"naver_board_{code}")
    return out, os.path.join(out, "posts.jsonl"), os.path.join(out, "backfill_state.json")


def fetch(code, offset):
    off = f"&offset={offset}" if offset is not None else ""
    req = urllib.request.Request(URL.format(code=code, off=off), headers={"User-Agent": "Mozilla/5.0 research-reader"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read().decode("utf-8"))
    res = j.get("result") or {}
    return res.get("posts") or [], res.get("lastOffset")


def known(path):
    ids = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            ids.add(json.loads(line)["id"])
    return ids


def append(path, posts, ids):
    n = 0
    now = datetime.now(KST).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        for p in posts:
            if p.get("id") in ids:
                continue
            ids.add(p.get("id"))
            f.write(json.dumps(dict(id=p.get("id"), order=p.get("orderNo"), at=p.get("writtenAt"), title=p.get("title"), body=(p.get("contentSwReplaced") or "")[:500], up=p.get("recommendCount"), down=p.get("notRecommendCount"), views=p.get("viewCount"), reply_depth=p.get("replyDepth"), holder=p.get("isHolderVerified"), fetched_at=now), ensure_ascii=False) + "\n")
            n += 1
    return n


def backfill(code, until):
    out, posts_path, state_path = paths(code)
    os.makedirs(out, exist_ok=True)
    st = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else dict(offset=None, oldest=None, pages=0)
    ids = known(posts_path)
    errs = 0
    while True:
        try:
            posts, last = fetch(code, st["offset"])
            errs = 0
        except Exception as e:
            errs += 1
            print(datetime.now(KST).strftime("%H:%M:%S"), "error", type(e).__name__, flush=True)
            time.sleep(300 if errs >= 5 else PAUSE * 5)
            continue
        if not posts:
            print("no more posts", flush=True)
            break
        n = append(posts_path, posts, ids)
        st.update(offset=last, oldest=posts[-1].get("writtenAt"), pages=st["pages"] + 1)
        json.dump(st, open(state_path, "w", encoding="utf-8"))
        if st["pages"] % 50 == 0:
            print(datetime.now(KST).strftime("%H:%M:%S"), "pages", st["pages"], "oldest", st["oldest"], "new", n, "total", len(ids), flush=True)
        if st["oldest"] and st["oldest"][:10].replace("-", "") < until:
            print("reached", st["oldest"], flush=True)
            break
        time.sleep(PAUSE)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "backfill":
        backfill(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "20221101")
    else:
        print("usage: board_collect.py backfill CODE [YYYYMMDD]")
