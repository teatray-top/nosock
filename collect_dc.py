import html
import json
import os
import re
import sys
import time
import urllib.request

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dc")
UA = os.environ.get("DC_UA", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) penny-research/0.1")
PAUSE = float(os.environ.get("DC_PAUSE", 0.7))
ROW = re.compile(r'<tr class="ub-content us-post" data-no="(\d+)" data-type="([^"]*)"[^>]*>(.*?)</tr>', re.S)
TIT = re.compile(r'<td class="gall_tit ub-word">\s*<a href="[^"]*"[^>]*>(.*?)</a>', re.S)
DATE = re.compile(r'<td class="gall_date"(?:\s+title="([^"]*)")?[^>]*>([^<]*)</td>')
NUM = re.compile(r'<td class="gall_count">([^<]*)</td>\s*<td class="gall_recommend">([^<]*)</td>')
REPLY = re.compile(r'<span class="reply_num">\[(\d+)\]</span>')
SUBJ = re.compile(r'<td class="gall_subject">(?:<b>)?([^<]*)')


def page(gid, p):
    url = f"https://gall.dcinside.com/mgallery/board/lists/?id={gid}&page={p}"
    for i in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ko", "Referer": f"https://gall.dcinside.com/mgallery/board/lists/?id={gid}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.HTTPError, OSError) as e:
            print(f"page {p}: {type(e).__name__} {str(e)[:60]}, retry {i + 1}", flush=True)
            time.sleep(15 * 2 ** i)
    raise RuntimeError("giving up")


def parse(s):
    out = []
    for no, typ, body in ROW.findall(s):
        if typ == "icon_notice" or "icon_notice" in body[:400]:
            continue
        t = TIT.search(body)
        d = DATE.search(body)
        n = NUM.search(body)
        rp = REPLY.search(body)
        sj = SUBJ.search(body)
        title = html.unescape(re.sub(r"<[^>]+>", "", t.group(1))).strip() if t else ""
        out.append(dict(no=int(no), subject=sj.group(1).strip() if sj else "", title=title, dt=(d.group(1) or d.group(2)) if d else "", views=n.group(1) if n else "", rec=n.group(2) if n else "", replies=int(rp.group(1)) if rp else 0))
    return out


def probe(gid, pages):
    for p in pages:
        rows = parse(page(gid, int(p)))
        print(p, len(rows), rows[0]["dt"] if rows else None, rows[-1]["dt"] if rows else None, rows[0]["no"] if rows else None, flush=True)
        time.sleep(PAUSE)


def find_page(gid, date, lo=1, hi=485000):
    while hi - lo > 1:
        mid = (lo + hi) // 2
        rows = [r for r in parse(page(gid, mid)) if len(r["dt"]) >= 10]
        time.sleep(PAUSE)
        if not rows:
            lo = mid
            continue
        d = rows[-1]["dt"][:10]
        print(f"page {mid}: {d}", flush=True)
        if d >= date:
            lo = mid
        else:
            hi = mid
    return lo


def scrape(gid, start_date, p0=1):
    os.makedirs(D, exist_ok=True)
    out = os.path.join(D, f"{gid}.jsonl")
    state = os.path.join(D, f"{gid}.state")
    p = int(open(state).read()) if os.path.exists(state) else p0
    seen = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                seen.add(json.loads(line)["no"])
    n, t0 = 0, time.time()
    with open(out, "a", encoding="utf-8") as f:
        while True:
            rows = parse(page(gid, p))
            dated = [r for r in rows if len(r["dt"]) >= 10]
            for r in rows:
                if r["no"] not in seen:
                    seen.add(r["no"]); f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
            f.flush()
            open(state, "w").write(str(p + 1))
            if dated and dated[-1]["dt"][:10] < start_date:
                break
            if p % 200 == 0:
                print(f"page {p}: {n} new posts, oldest {dated[-1]['dt'] if dated else '?'}, {time.time() - t0:.0f}s", flush=True)
            p += 1
            time.sleep(PAUSE)
    print(f"done at page {p}, {n} new posts, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "probe":
        probe(sys.argv[2], sys.argv[3:])
    elif cmd == "find":
        print("page for", sys.argv[3], "=", find_page(sys.argv[2], sys.argv[3]))
    else:
        scrape(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
