import json
import os
import sys
import time
import urllib.parse
import urllib.request

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reddit")
BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"
FIELDS = "id,created_utc,subreddit,title,selftext,score,num_comments,author,link_flair_text"
UA = os.environ.get("REDDIT_UA", "penny-research/0.1")
PAUSE = float(os.environ.get("REDDIT_PAUSE", 2.0))


def get(params):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for i in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            wait = 60
            if e.code == 429:
                try:
                    wait = max(5, int(e.headers.get("X-RateLimit-Reset", "60")))
                except ValueError:
                    pass
                print(f"429, sleeping {wait}s", flush=True)
            else:
                print(f"HTTP {e.code}, retry {i + 1}", flush=True)
            time.sleep(wait if e.code == 429 else 10 * 2 ** i)
        except OSError as e:
            print(f"{type(e).__name__} {str(e)[:80]}, retry {i + 1}", flush=True)
            time.sleep(10 * 2 ** i)
    raise RuntimeError("giving up")


def main(subs, start, end):
    os.makedirs(D, exist_ok=True)
    end_ts = int(time.mktime(time.strptime(end, "%Y-%m-%d"))) + 86400
    start_ts = int(time.mktime(time.strptime(start, "%Y-%m-%d")))
    desc = os.environ.get("REDDIT_ORDER", "asc") == "desc"
    for sub in subs:
        out = os.path.join(D, f"{sub}{'_desc' if desc else ''}.jsonl")
        state = os.path.join(D, f"{sub}{'_desc' if desc else ''}.state")
        cur = int(open(state).read()) if os.path.exists(state) else (end_ts if desc else start_ts)
        n, t0 = 0, time.time()
        with open(out, "a", encoding="utf-8") as f:
            while (cur > start_ts) if desc else (cur < end_ts):
                q = {"subreddit": sub, "sort": "desc" if desc else "asc", "limit": "auto", "fields": FIELDS}
                q.update({"before": cur, "after": start_ts} if desc else {"after": cur, "before": end_ts})
                rows = get(q).get("data", [])
                if not rows:
                    break
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += len(rows)
                last = int(rows[-1]["created_utc"])
                cur = (last - 1 if last <= cur else cur - 1) if desc else (last + 1 if last >= cur else cur + 1)
                open(state, "w").write(str(cur))
                if n % 5000 < len(rows):
                    print(f"{sub}: {n} posts, at {time.strftime('%Y-%m-%d', time.gmtime(cur))}, {time.time() - t0:.0f}s", flush=True)
                time.sleep(PAUSE)
        print(f"{sub}: done {n} new posts, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2], sys.argv[3])
