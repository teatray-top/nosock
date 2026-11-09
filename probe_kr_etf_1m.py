import json
import os
import sys
import time
from datetime import datetime

from toss import ApiError, Toss

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "kr_etf", "1m_probe_20261106")
SYMS = ("069500", "229200")
DAYS = ("2022-11-24", "2024-05-02", "2026-11-05")
MAX_REQ = 12
MIN_GAP = 1.0
EXPECT_FIRST, EXPECT_LAST = "09:01", "15:31"


def expected_minutes(day):
    out = []
    h, m = 9, 1
    while (h, m) <= (15, 31):
        out.append(f"{day}T{h:02d}:{m:02d}")
        m += 1
        if m == 60:
            h, m = h + 1, 0
    return out


def main():
    if os.path.exists(OUT) and os.listdir(OUT):
        raise SystemExit(f"{OUT} already has files; refusing to overwrite")
    os.makedirs(OUT, exist_ok=True)
    t = Toss()
    n_req, last = 0, 0.0
    report = dict(started=datetime.now().isoformat(timespec="seconds"), max_req=MAX_REQ, min_gap_s=MIN_GAP, syms=SYMS, days=DAYS, requests=[], per_day={}, stopped=None)
    try:
        for sym in SYMS:
            for day in DAYS:
                before = day + "T15:31:00+09:00"
                bars, pages = {}, []
                for page in range(2):
                    if n_req >= MAX_REQ:
                        report["stopped"] = f"request cap {MAX_REQ} reached before {sym} {day} p{page}"
                        raise StopIteration
                    wait = last + MIN_GAP - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    params = dict(symbol=sym, interval="1m", count=200, before=before, adjusted="false")
                    t0 = time.time()
                    last = t0
                    n_req += 1
                    try:
                        raw = t._req("GET", "/api/v1/candles", params, retry=0)
                        err = None
                    except ApiError as e:
                        raw, err = None, dict(status=e.status, code=e.code, message=e.message[:300], request_id=e.request_id)
                    t1 = time.time()
                    rec = dict(n=n_req, sym=sym, day=day, page=page, params=params, req_ts=datetime.fromtimestamp(t0).isoformat(timespec="milliseconds"), recv_ts=datetime.fromtimestamp(t1).isoformat(timespec="milliseconds"), elapsed_s=round(t1 - t0, 3), error=err)
                    path = os.path.join(OUT, f"{sym}_{day}_p{page}.json")
                    with open(path, "xb") as f:
                        f.write(json.dumps(dict(request=rec, response=raw), ensure_ascii=False, indent=1).encode("utf-8"))
                    rec["file"] = os.path.relpath(path, HERE)
                    if err is None:
                        c = raw.get("result", {}).get("candles", [])
                        rec["n_candles"] = len(c)
                        rec["first_ts"] = c[-1]["timestamp"] if c else None
                        rec["last_ts"] = c[0]["timestamp"] if c else None
                        rec["next_before"] = raw.get("result", {}).get("nextBefore")
                        rec["result_keys"] = sorted(raw.get("result", {}).keys())
                        for x in c:
                            bars.setdefault(x["timestamp"], []).append(x)
                    report["requests"].append(rec)
                    pages.append(rec)
                    if err is not None:
                        report["stopped"] = f"error on request {n_req}: {err}"
                        raise StopIteration
                    nb = rec["next_before"]
                    oldest = rec["first_ts"][11:16] if rec["first_ts"] else None
                    if not nb or (rec["first_ts"] and (rec["first_ts"][:10] < day or oldest <= EXPECT_FIRST)):
                        break
                    before = nb
                keys = sorted(bars)
                in_day = [k for k in keys if k[:10] == day]
                in_sess = [k for k in in_day if EXPECT_FIRST <= k[11:16] <= EXPECT_LAST]
                exp = expected_minutes(day)
                have = {k[:16] for k in in_sess}
                missing = [e for e in exp if e not in have]
                dup = {k: len(v) for k, v in bars.items() if len(v) > 1}
                zero_vol = sum(1 for k in in_sess for x in bars[k] if str(x.get("volume")) in ("0", "0.0"))
                report["per_day"][f"{sym}_{day}"] = dict(pages=len(pages), total_candles=sum(p.get("n_candles", 0) for p in pages), unique_ts=len(keys), in_day=len(in_day), in_session=len(in_sess), expected=len(exp), missing=len(missing), missing_sample=missing[:12], out_of_day=[k for k in keys if k[:10] != day][:6], out_of_session=[k for k in in_day if not (EXPECT_FIRST <= k[11:16] <= EXPECT_LAST)][:6], duplicates=dup, zero_volume_bars=zero_vol, first=in_day[0] if in_day else None, last=in_day[-1] if in_day else None, sample_first=bars[in_day[0]][0] if in_day else None, sample_last=bars[in_day[-1]][0] if in_day else None)
    except StopIteration:
        pass
    report["n_requests"] = n_req
    report["finished"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(OUT, "report.json"), "xb") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=1).encode("utf-8"))
    print(json.dumps(dict(n_requests=n_req, stopped=report["stopped"], per_day={k: {kk: v[kk] for kk in ("pages", "total_candles", "in_session", "expected", "missing", "duplicates", "zero_volume_bars", "first", "last", "out_of_day", "out_of_session")} for k, v in report["per_day"].items()}, elapsed=[r["elapsed_s"] for r in report["requests"]]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
