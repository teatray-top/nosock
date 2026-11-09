import csv
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

import toss

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = ("069500", "229200")
START, END = "2022-11-23", "2026-11-05"
CAL_FILE = os.path.join(HERE, "data", "kr_etf", "kodex_daily_raw.json")
BASE_DIR = os.path.join(HERE, "data", "kr_etf")
HARD_MAX_REQ, HARD_MIN_GAP = 4000, 1.0
MAX_REQ = min(int(os.environ.get("KR_ETF_MAX_REQ", HARD_MAX_REQ)), HARD_MAX_REQ)
MIN_GAP = max(float(os.environ.get("KR_ETF_MIN_GAP", HARD_MIN_GAP)), HARD_MIN_GAP)
TOKEN_MARGIN = 60
PAGES = 2
FIRST, LAST = "09:01", "15:31"
AUCTION_FILL = {f"15:{m:02d}" for m in range(21, 30)}
FIRST_SESSION_DAYS = {"2023-01-02", "2024-01-02", "2025-01-02", "2026-01-02"}
SUNEUNG_DAYS = {"2022-11-17", "2023-11-16", "2024-11-14", "2025-11-13", "2026-11-19"}


def session_window(day):
    if day in SUNEUNG_DAYS:
        return "10:01", "16:30", {f"16:{m:02d}" for m in range(21, 30)}, "shifted_session_special", 390
    if day in FIRST_SESSION_DAYS:
        return "10:01", "15:30", AUCTION_FILL, "late_open_special", 330
    return "09:01", "15:30", AUCTION_FILL, "normal", 390
HDR_KEEP = ("x-request-id", "date", "retry-after", "content-type")
HDR_SUBSTR = ("ratelimit", "rate-limit")


class Stop(Exception):
    def __init__(self, msg, headers=None):
        super().__init__(msg)
        self.headers = headers or {}


def sha_self():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_new(path, data):
    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def cached_token():
    try:
        t = json.load(open(toss.TOKEN_FILE, encoding="utf-8"))
    except (OSError, ValueError):
        return None, "token cache unreadable"
    exp = float(t.get("exp", 0))
    if exp - time.time() < TOKEN_MARGIN:
        return None, f"token expires in {exp - time.time():.0f}s (< {TOKEN_MARGIN}s), not issuing"
    tok = t.get("access_token")
    if not tok:
        return None, "token cache has no access_token"
    return tok, None


def pick_headers(h):
    out = {}
    for k, v in h.items():
        kl = k.lower()
        if kl in HDR_KEEP or any(s in kl for s in HDR_SUBSTR):
            out[kl] = v
    return out


def fetch(params):
    tok, why = cached_token()
    if not tok:
        raise Stop(why)
    url = toss.BASE + "/api/v1/candles?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read(), pick_headers(r.headers)
    except urllib.error.HTTPError as e:
        body = e.read()
        raise Stop(f"http {e.code}: {body[:300].decode('utf-8', 'ignore')}", pick_headers(e.headers)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise Stop(f"transport: {e}") from None


def calendar(sym):
    k = json.load(open(CAL_FILE, encoding="utf-8"))[sym]
    return [d for d in k["days"] if START <= d <= END]


def minutes_expected(first, last):
    out, (h, m) = [], (int(first[:2]), int(first[3:]))
    while f"{h:02d}:{m:02d}" <= last:
        out.append(f"{h:02d}:{m:02d}")
        m += 1
        if m == 60:
            h, m = h + 1, 0
    return out


def summarize_day(day, bars, incomplete):
    s_first, s_last, fill_set, expect_cls, expect_n = session_window(day)
    keys = sorted(k for k in bars if k[:10] == day and s_first <= k[11:16] <= s_last)
    mins = [k[11:16] for k in keys]
    zero = [m for m, k in zip(mins, keys) if str(bars[k][0]["volume"]) in ("0", "0.0")]
    n = len(keys)
    if incomplete:
        cls = "incomplete_pages"
    elif n == expect_n and mins[0] == s_first and mins[-1] == s_last:
        cls = expect_cls
    elif n == 0:
        cls = "no_bars"
    else:
        cls = "short_or_missing"
    exp = minutes_expected(s_first, s_last)
    have = set(mins)
    missing = [m for m in exp if m not in have]
    dup_ts = {k: len(v) for k, v in bars.items() if len(v) > 1}
    conflicts = [k for k, v in dup_ts.items() if any(json.dumps(x, sort_keys=True) != json.dumps(bars[k][0], sort_keys=True) for x in bars[k][1:])]
    day_keys = sorted(k for k in bars if k[:10] == day)
    return dict(n=n, n_day=len(day_keys), session=[s_first, s_last], day_first=day_keys[0][11:16] if day_keys else None, day_last=day_keys[-1][11:16] if day_keys else None, first=mins[0] if mins else None, last=mins[-1] if mins else None, cls=cls, incomplete=incomplete, zero_vol=len(zero), fill_outside_auction=[m for m in zero if m not in fill_set][:30], missing=missing[:30], n_missing=len(missing), duplicate_ts=len(dup_ts), duplicate_sample=sorted(dup_ts)[:5], content_conflicts=conflicts[:10], out_of_day=sum(1 for k in bars if k[:10] != day), out_of_day_latest=max((k for k in bars if k[:10] > day), default=None), out_of_session=sum(1 for k in bars if k[:10] == day and not (s_first <= k[11:16] <= s_last)))


def selftest():
    assert MAX_REQ <= HARD_MAX_REQ and MIN_GAP >= HARD_MIN_GAP, (MAX_REQ, MIN_GAP)
    probe = os.path.join(BASE_DIR, "1m_probe_20261106")
    for sym, day, fill in (("069500", "2022-11-24", []), ("229200", "2022-11-24", ["11:10", "12:54", "13:37"])):
        bars = {}
        for p in (0, 1):
            j = json.load(open(os.path.join(probe, f"{sym}_{day}_p{p}.json"), encoding="utf-8"))
            for x in j["response"]["result"]["candles"]:
                bars.setdefault(x["timestamp"], []).append(x)
        s = summarize_day(day, bars, False)
        assert s["cls"] == "normal" and s["n"] == 390 and s["n_day"] == 390 and s["fill_outside_auction"] == fill and s["duplicate_ts"] == 0 and s["out_of_day"] == 10, s
    def synth(day, first, last):
        return {f"{day}T{h:02d}:{m:02d}:00.000+09:00": [dict(volume="1", openPrice="1", highPrice="1", lowPrice="1", closePrice="1")] for h in range(9, 17) for m in range(60) if first <= f"{h:02d}:{m:02d}" <= last}
    su = synth("2024-11-14", "10:01", "16:30")
    s = summarize_day("2024-11-14", su, False)
    assert s["n"] == 390 and s["n_day"] == 390 and s["day_last"] == "16:30" and s["out_of_session"] == 0 and s["n_missing"] == 0 and s["cls"] == "shifted_session_special" and s["session"] == ["10:01", "16:30"], s
    su2 = {k: v for k, v in su.items() if k[11:16] != "16:00"}
    s2 = summarize_day("2024-11-14", su2, False)
    assert s2["cls"] == "short_or_missing" and s2["n_missing"] == 1 and s2["missing"] == ["16:00"], s2
    fd = synth("2025-01-02", "10:01", "15:30")
    s3 = summarize_day("2025-01-02", fd, False)
    assert s3["cls"] == "late_open_special" and s3["n"] == 330 and s3["n_missing"] == 0, s3
    odd = synth("2026-01-05", "10:01", "16:30")
    s4 = summarize_day("2026-01-05", odd, False)
    assert s4["cls"] == "short_or_missing" and s4["out_of_session"] == 60 and s4["n_missing"] == 60, s4
    s5 = summarize_day("2024-11-14", synth("2024-11-14", "10:01", "15:30"), False)
    assert s5["cls"] == "short_or_missing" and s5["n_missing"] == 60
    k = next(iter(bars))
    bars[k].append(dict(bars[k][0], volume="999"))
    s = summarize_day(day, bars, False)
    assert s["duplicate_ts"] == 1 and s["content_conflicts"] == [k], s
    assert summarize_day(day, bars, True)["cls"] == "incomplete_pages"
    saved = toss.TOKEN_FILE
    toss.TOKEN_FILE = os.path.join(HERE, "nonexistent_token.json")
    try:
        fetch(dict(symbol="x"))
        raise AssertionError("fetch did not stop without token")
    except Stop as e:
        assert "unreadable" in str(e)
    finally:
        toss.TOKEN_FILE = saved
    import email.message
    import io
    import tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(dict(client_id="selftest", access_token="selftest-token", exp=time.time() + 3600), tmp)
    tmp.close()
    hdr = email.message.Message()
    hdr["Retry-After"] = "1"
    hdr["X-RateLimit-Remaining"] = "0"
    hdr["X-Request-Id"] = "selftest"
    hdr["Authorization"] = "must-not-appear"

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", hdr, io.BytesIO(b'{"error":{"code":"rate-limit-exceeded"}}'))
    saved_open, saved_tok = urllib.request.urlopen, toss.TOKEN_FILE
    urllib.request.urlopen, toss.TOKEN_FILE = fake_urlopen, tmp.name
    try:
        fetch(dict(symbol="x"))
        raise AssertionError("fetch did not stop on 429")
    except Stop as e:
        assert "http 429" in str(e) and e.headers.get("retry-after") == "1" and e.headers.get("x-ratelimit-remaining") == "0" and "authorization" not in e.headers, (str(e), e.headers)
    finally:
        urllib.request.urlopen, toss.TOKEN_FILE = saved_open, saved_tok
        os.unlink(tmp.name)
    assert len(calendar("069500")) == 964 and len(calendar("229200")) == 964
    first = os.path.join(BASE_DIR, "1m_backfill_20261106_125425")
    if os.path.isdir(first):
        done, used, partial = prior_runs([first])
        assert used == 128 and len(done) == 63 and "069500_2022-11-23" in done and "069500_2023-02-23" not in done, (used, len(done))
        assert list(partial) == ["069500_2023-02-23"] and list(partial["069500_2023-02-23"]) == [0], partial
        bars, nb, reused = reuse_pages(partial["069500_2023-02-23"])
        assert len(reused) == 1 and reused[0]["n_candles"] == 200 and nb and min(bars)[11:16] > FIRST, (reused, nb)
    print("selftest ok", dict(max_req=MAX_REQ, min_gap=MIN_GAP, sha=sha_self()))


def prior_runs(dirs):
    done, used, partial = set(), 0, {}
    for d in dirs:
        d = os.path.join(HERE, d) if not os.path.isabs(d) else d
        for fn in os.listdir(os.path.join(d, "days")):
            done.add(fn[:-5])
        with open(os.path.join(d, "attempts.jsonl"), encoding="utf-8") as f:
            used += sum(1 for l in f if json.loads(l)["phase"] == "before")
        for fn in os.listdir(os.path.join(d, "raw")):
            key, page = fn[:-len("_p0.json.gz")], int(fn[-len("0.json.gz")])
            if key not in done:
                partial.setdefault(key, {})[page] = os.path.join(d, "raw", fn)
    return done, used, partial


def reuse_pages(partial_pages):
    bars, before, reused = {}, None, []
    for page in sorted(partial_pages):
        if page != len(reused):
            break
        with gzip.open(partial_pages[page], "rb") as f:
            res = json.loads(f.read()).get("result", {})
        c = res.get("candles", [])
        for x in c:
            bars.setdefault(x["timestamp"], []).append(x)
        reused.append(dict(page=page, raw=partial_pages[page], n_candles=len(c), next_before=res.get("nextBefore")))
        before = res.get("nextBefore")
        if not c or not before or c[-1]["timestamp"][11:16] <= FIRST:
            break
    return bars, before, reused


def main(skip_from=()):
    done, used, partial = prior_runs(skip_from)
    cap = MAX_REQ - used
    if cap <= 0:
        raise SystemExit(f"prior runs already used {used} attempts; cap {MAX_REQ}")
    run_dir = os.path.join(BASE_DIR, "1m_backfill_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=False)
    for sub in ("raw", "days", "csv"):
        os.makedirs(os.path.join(run_dir, sub))
    attempts_path = os.path.join(run_dir, "attempts.jsonl")
    meta = dict(script_sha256=sha_self(), started=now_iso(), syms=SYMS, start=START, end=END, max_req_total=MAX_REQ, prior_runs=list(skip_from), prior_attempts=used, prior_days_skipped=len(done), prior_partial_days=sorted(partial), cap_this_run=cap, min_gap_s=MIN_GAP, pages=PAGES, calendar_file=os.path.relpath(CAL_FILE, HERE), calendar_sha256=hashlib.sha256(open(CAL_FILE, "rb").read()).hexdigest()[:16], token_policy="shared cache read before each request; never issued here", run_dir=os.path.relpath(run_dir, HERE))
    write_new(os.path.join(run_dir, "meta.json"), json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    n_att, last_t, stopped, days_done = 0, 0.0, None, {}
    print("run dir", run_dir, "cap", cap, "skipping", len(done), flush=True)
    try:
        for sym in SYMS:
            for day in calendar(sym):
                if f"{sym}_{day}" in done:
                    continue
                bars, before, pages, incomplete = {}, day + "T23:59:59+09:00", [], False
                first_page = 0
                if f"{sym}_{day}" in partial:
                    bars, nb, reused = reuse_pages(partial[f"{sym}_{day}"])
                    if reused:
                        first_page = len(reused)
                        pages.extend(dict(r, reused=True) for r in reused)
                        with open(attempts_path, "ab") as f:
                            f.write((json.dumps(dict(sym=sym, day=day, phase="reused", pages=reused), ensure_ascii=False) + "\n").encode("utf-8"))
                        last_c = bars and min(bars)
                        if not nb or (last_c and last_c[11:16] <= FIRST) or first_page >= PAGES:
                            first_page = PAGES
                        else:
                            before = nb
                for page in range(first_page, PAGES):
                    if n_att >= cap:
                        raise Stop(f"attempt cap reached (this run {cap}, total {MAX_REQ})")
                    wait = last_t + MIN_GAP - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    params = dict(symbol=sym, interval="1m", count=200, before=before, adjusted="false")
                    n_att += 1
                    att = dict(n=n_att, sym=sym, day=day, page=page, params=params, planned_ts=now_iso())
                    with open(attempts_path, "ab") as f:
                        f.write((json.dumps(dict(att, phase="before")) + "\n").encode("utf-8"))
                        f.flush()
                        os.fsync(f.fileno())
                    t0 = time.time()
                    last_t = t0
                    try:
                        status, body, hdrs = fetch(params)
                    except Stop as e:
                        with open(attempts_path, "ab") as f:
                            f.write((json.dumps(dict(att, phase="failed", recv_ts=now_iso(), error=str(e), headers=e.headers), ensure_ascii=False) + "\n").encode("utf-8"))
                        raise
                    t1 = time.time()
                    rawp = os.path.join(run_dir, "raw", f"{sym}_{day}_p{page}.json.gz")
                    write_new(rawp, gzip.compress(body))
                    try:
                        res = json.loads(body).get("result", {})
                        c = res.get("candles", [])
                    except (ValueError, AttributeError):
                        raise Stop(f"invalid json {sym} {day} p{page}")
                    rec = dict(att, phase="received", status=status, recv_ts=datetime.fromtimestamp(t1).isoformat(timespec="milliseconds"), elapsed_s=round(t1 - t0, 3), n_candles=len(c), oldest=c[-1]["timestamp"] if c else None, newest=c[0]["timestamp"] if c else None, next_before=res.get("nextBefore"), headers=hdrs, raw=os.path.relpath(rawp, run_dir))
                    with open(attempts_path, "ab") as f:
                        f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
                    pages.append({k: rec[k] for k in ("n", "page", "status", "elapsed_s", "n_candles", "oldest", "newest", "next_before", "headers")})
                    for x in c:
                        bars.setdefault(x["timestamp"], []).append(x)
                    nb = res.get("nextBefore")
                    oldest = c[-1]["timestamp"] if c else None
                    if not c or not nb or oldest[:10] < day or oldest[11:16] <= FIRST:
                        break
                    before = nb
                    if page == PAGES - 1:
                        incomplete = True
                d = summarize_day(day, bars, incomplete)
                d["pages"] = pages
                d["rows"] = [[k[:19], bars[k][0]["openPrice"], bars[k][0]["highPrice"], bars[k][0]["lowPrice"], bars[k][0]["closePrice"], bars[k][0]["volume"]] for k in sorted(bars) if k[:10] == day]
                write_atomic(os.path.join(run_dir, "days", f"{sym}_{day}.json"), json.dumps(d, ensure_ascii=False).encode("utf-8"))
                days_done[f"{sym}_{day}"] = d["cls"]
                if len(days_done) % 50 == 0:
                    print(datetime.now().strftime("%H:%M:%S"), sym, day, d["cls"], d["n"], "attempts", n_att, flush=True)
    except Stop as e:
        stopped = str(e)
        print("STOP", e, flush=True)
    except KeyboardInterrupt:
        stopped = "interrupted"
    for sym in SYMS:
        rows = []
        for fn in sorted(os.listdir(os.path.join(run_dir, "days"))):
            if fn.startswith(sym + "_"):
                rows.extend(json.load(open(os.path.join(run_dir, "days", fn), encoding="utf-8"))["rows"])
        buf = ["ts,open,high,low,close,volume"] + [",".join(map(str, r)) for r in rows]
        write_atomic(os.path.join(run_dir, "csv", f"{sym}.csv"), ("\n".join(buf) + "\n").encode("utf-8"))
    cls = {}
    for v in days_done.values():
        cls[v] = cls.get(v, 0) + 1
    summary = dict(meta, finished=now_iso(), attempts=n_att, attempts_total_incl_prior=used + n_att, days_done=len(days_done), days_done_incl_prior=len(days_done) + len(done), classes=cls, stopped=stopped, expected_days={s: len(calendar(s)) for s in SYMS})
    write_atomic(os.path.join(run_dir, "summary.json"), json.dumps(summary, ensure_ascii=False, indent=1).encode("utf-8"))
    print(json.dumps(dict(attempts=n_att, days_done=len(days_done), classes=cls, stopped=stopped), ensure_ascii=False), flush=True)
    sys.exit(2 if stopped else 0)


if __name__ == "__main__":
    if sys.argv[1:] == ["sha"]:
        print(sha_self())
    elif sys.argv[1:] == ["selftest"]:
        selftest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "--skip-from":
        main(tuple(sys.argv[2].split(",")))
    elif len(sys.argv) == 1:
        main()
    else:
        print("usage: backfill_kr_etf_1m.py [--skip-from RUN_DIR[,RUN_DIR]] | selftest | sha")
