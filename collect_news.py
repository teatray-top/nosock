import json
import os
import sys
import time

import numpy as np

import massive

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "massive", "news")
URL = "https://api.massive.com/v2/reference/news"
KEEP = ("id", "published_utc", "tickers", "title", "publisher", "keywords")


def fetch(c, ticker, start, end, out):
    tmp = out + ".part"
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for page in c.pages(URL, {"ticker": ticker, "published_utc.gte": start + "T00:00:00Z", "published_utc.lte": end + "T23:59:59Z", "limit": 1000, "order": "asc"}):
            for a in page.get("results", []):
                row = {k: a.get(k) for k in KEEP}
                row["publisher"] = (a.get("publisher") or {}).get("name")
                ins = [i for i in a.get("insights") or [] if i.get("ticker") == ticker]
                row["sentiment"] = ins[0].get("sentiment") if ins else None
                row["reason"] = ins[0].get("sentiment_reasoning") if ins else None
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    os.replace(tmp, out)
    return n


def main(panel, start, end, tag):
    z = np.load(panel, allow_pickle=True)
    syms = [str(s) for s in z["syms"]]
    out_dir = os.path.join(D, tag)
    os.makedirs(out_dir, exist_ok=True)
    c = massive.Client(rpm=float(os.environ.get("MASSIVE_RPM", 240)))
    t0 = time.time()
    total = 0
    for i, s in enumerate(syms):
        out = os.path.join(out_dir, s + ".jsonl")
        if os.path.exists(out):
            continue
        for attempt in range(3):
            try:
                total += fetch(c, s, start, end, out)
                break
            except Exception as e:
                print(f"{s}: {type(e).__name__} {str(e)[:100]} (attempt {attempt + 1})", flush=True)
                time.sleep(10 * 3 ** attempt)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(syms)} articles {total} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(syms)} tickers, {total} new articles, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
