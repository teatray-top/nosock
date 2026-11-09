import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

sys.dont_write_bytecode = True
import backfill_kr_etf_1m as B

HERE = os.path.dirname(os.path.abspath(__file__))
KST = timezone(timedelta(hours=9))
SYMS = ("069500", "229200")
EXTRA_FIRST_SESSION_DAYS = {"2027-01-04"}
CONFIRM_DELAY = timedelta(minutes=40)
MIN_GAP = 1.0
MAX_REQ_PER_RUN = 20
TRANSPORT_RETRIES = 2
RETRY_WAIT = 5.0
ORIGIN_MAX_PAGES = 48
RECOVERY_PAGES = {"069500": 32, "229200": 16}
MIN_OVERLAP = 150
OUT_MIN = os.path.join(HERE, "data", "kr_etf", "forward_1m")
OUT_ADJ = os.path.join(HERE, "data", "kr_etf", "forward_daily_adj")


def session(day):
    if day in EXTRA_FIRST_SESSION_DAYS:
        return "10:01", "15:30", B.AUCTION_FILL, "late_open_special", 330
    return B.session_window(day)


def summarize_local(day, bars):
    s_first, s_last, fill_set, expect_cls, expect_n = session(day)
    day_keys = sorted(k for k in bars if k[:10] == day)
    in_sess = [k for k in day_keys if s_first <= k[11:16] <= s_last]
    out_sess = [k for k in day_keys if not (s_first <= k[11:16] <= s_last)]
    future = sorted(k for k in bars if k[:10] > day)
    conflicts = [k for k, v in bars.items() if len({json.dumps(x, sort_keys=True) for x in v}) > 1]
    expected = B.minutes_expected(s_first, s_last)
    have = {k[11:16] for k in in_sess}
    missing = [m for m in expected if m not in have]
    complete = len(in_sess) == expect_n and not missing
    cls = expect_cls if complete else ("no_bars" if not in_sess else "short_or_missing")
    reasons = []
    if cls != expect_cls:
        reasons.append("session class or bar count differs from the declared session")
    if conflicts:
        reasons.append("conflicting duplicate bars")
    if out_sess:
        reasons.append("bars outside the declared session")
    if future:
        reasons.append("bars dated after the requested day")
    return dict(n=len(in_sess), expected_n=expect_n, session=[s_first, s_last], cls=cls, expected_cls=expect_cls, missing=missing[:30], n_missing=len(missing), conflicts=conflicts[:10], out_of_session=out_sess[:10], future_rows=future[:10], zero_vol=sum(1 for k in in_sess if str(bars[k][0]["volume"]) in ("0", "0.0")), held=bool(reasons), hold_reasons=reasons), in_sess


def earliest_fetch(day):
    last = session(day)[1]
    return datetime.fromisoformat(f"{day}T{last}:00+09:00") + CONFIRM_DELAY


class Collector:
    def __init__(self, fetch=None, clock=None, sleep=None, out_min=OUT_MIN, out_adj=OUT_ADJ, max_req=MAX_REQ_PER_RUN, max_recovery_pages=ORIGIN_MAX_PAGES):
        self.max_req = max_req
        self.max_recovery_pages = max_recovery_pages
        self.fetch = fetch or B.fetch
        self.clock = clock or (lambda: datetime.now(KST))
        self.sleep = sleep or time.sleep
        self.out_min, self.out_adj = out_min, out_adj
        self.last_t = 0.0
        self.requests = []
        self.n_routine = 0
        self.n_recovery = 0
        self.vdir = None

    def persist(self, rec):
        rdir = os.path.join(self.vdir, "requests")
        os.makedirs(rdir, exist_ok=True)
        B.write_new(os.path.join(rdir, f"{len(self.requests):04d}.json"), json.dumps(rec, ensure_ascii=False).encode("utf-8"))
        self.requests.append({k: v for k, v in rec.items() if k != "body"})

    def call(self, params, budget="routine"):
        if self.vdir is None:
            raise B.Stop("no vintage directory before the first request")
        attempt = 0
        while True:
            if budget == "routine" and self.n_routine >= self.max_req:
                raise B.Stop(f"routine request budget reached ({self.max_req} per run)")
            if budget == "recovery" and self.n_recovery >= self.max_recovery_pages:
                raise B.Stop(f"recovery budget reached ({self.max_recovery_pages} pages)")
            wait = self.last_t + MIN_GAP - time.time()
            if wait > 0:
                self.sleep(wait)
            req_at = self.clock()
            if budget == "routine":
                self.n_routine += 1
            else:
                self.n_recovery += 1
            try:
                status, body, headers = self.fetch(params)
            except B.Stop as e:
                self.last_t = time.time()
                self.persist(dict(budget=budget, params=params, requested_at=req_at.isoformat(timespec="milliseconds"), failed_at=self.clock().isoformat(timespec="milliseconds"), error=str(e)[:300], headers=getattr(e, "headers", {})))
                if str(e).startswith("transport") and attempt < TRANSPORT_RETRIES:
                    attempt += 1
                    self.sleep(RETRY_WAIT)
                    continue
                raise
            rec_at = self.clock()
            self.last_t = time.time()
            self.persist(dict(budget=budget, params=params, requested_at=req_at.isoformat(timespec="milliseconds"), received_at=rec_at.isoformat(timespec="milliseconds"), status=status, sha256=hashlib.sha256(body).hexdigest(), headers=headers, body=body.decode("utf-8", "replace")))
            return json.loads(body), rec_at

    def open_vintage(self, root):
        vintage = self.clock().strftime("%Y%m%dT%H%M%S")
        self.vdir = os.path.join(root, vintage)
        os.makedirs(self.vdir, exist_ok=False)
        return vintage

    def minute_day(self, day, syms=SYMS):
        now = self.clock()
        if now < earliest_fetch(day):
            raise B.Stop(f"too early for {day}: now {now.isoformat(timespec='seconds')}, earliest {earliest_fetch(day).isoformat(timespec='seconds')}")
        vintage = self.open_vintage(self.out_min)
        vdir = self.vdir
        out = {}
        for sym in syms:
            bars, pages, before = {}, [], f"{day}T23:59:59+09:00"
            for page in range(B.PAGES):
                params = dict(symbol=sym, interval="1m", count=200, before=before, adjusted="false")
                data, rec_at = self.call(params)
                candles = (data.get("result") or data).get("candles") if isinstance(data, dict) else None
                if candles is None:
                    raise B.Stop(f"unexpected response shape for {sym} {day}")
                pages.append(dict(page=page, before=before, n=len(candles), received_at=rec_at.isoformat(timespec="milliseconds")))
                for c in candles:
                    ts = c["timestamp"][:19]
                    bars.setdefault(ts, []).append(dict(open=c["openPrice"], high=c["highPrice"], low=c["lowPrice"], close=c["closePrice"], volume=c["volume"]))
                if not candles:
                    break
                oldest = min(c["timestamp"] for c in candles)
                if datetime.fromisoformat(oldest) >= datetime.fromisoformat(before):
                    raise B.Stop(f"pagination did not progress for {sym} {day}: oldest {oldest} not before {before}")
                if oldest[:10] < day:
                    break
                before = oldest
            summary, in_sess = summarize_local(day, bars)
            rows = [[k, bars[k][0]["open"], bars[k][0]["high"], bars[k][0]["low"], bars[k][0]["close"], bars[k][0]["volume"]] for k in in_sess]
            rec = dict(symbol=sym, day=day, vintage=vintage, expected_session=list(session(day)[:2]), expected_class=session(day)[3], summary=summary, pages=pages, rows=rows, collector_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
            path = os.path.join(vdir, f"{day}_{sym}.json")
            B.write_new(path, json.dumps(rec, ensure_ascii=False).encode("utf-8"))
            out[sym] = dict(path=path, cls=summary["cls"], n=summary["n"], expected=session(day)[3], held=summary["held"], hold_reasons=summary["hold_reasons"])
        B.write_new(os.path.join(vdir, "meta.json"), json.dumps(dict(day=day, vintage=vintage, requests=self.requests, result=out), ensure_ascii=False, indent=1).encode("utf-8"))
        return vdir, out

    def adjusted_checked(self, stored, cutoff_day, origins, syms=SYMS, min_overlap=MIN_OVERLAP):
        vintage = self.open_vintage(self.out_adj)
        vdir = self.vdir
        out = {}
        for sym in syms:
            data, rec_at = self.call(dict(symbol=sym, interval="1d", count=200, adjusted="true"))
            candles = (data.get("result") or data).get("candles") or []
            page, page_conflicts, bad_prices = rows_from_candles(candles)
            days, closes = stored[sym]
            stored_map = dict(zip(days, closes))
            compared = {d: v for d, v in page.items() if d <= cutoff_day}
            overlap = sorted(set(compared) & set(stored_map))
            stored_last = days[-1] if days else ""
            rec = dict(symbol=sym, vintage=vintage, cutoff_day=cutoff_day, stored_last_day=stored_last, page_rows=len(page), rows_after_cutoff_excluded=sorted(d for d in page if d > cutoff_day), compared_rows=len(compared), overlap_rows=len(overlap), min_overlap=min_overlap, new_rows={d: compared[d] for d in sorted(compared) if d > stored_last}, unknown_days=sorted(d for d in compared if d <= stored_last and d not in stored_map), page_conflicts=page_conflicts[:10], non_finite_or_non_positive=bad_prices[:10])
            reasons = []
            if not candles:
                reasons.append("empty response")
            if page_conflicts:
                reasons.append("conflicting closes for the same date in the page")
            if bad_prices:
                reasons.append("non-finite or non-positive close in the page")
            if cutoff_day not in page:
                reasons.append("the latest completed session is not in the page")
            if len(overlap) < min_overlap:
                reasons.append(f"only {len(overlap)} overlapping dates with the stored history (minimum {min_overlap})")
            if reasons:
                rec.update(status="hold_invalid_page", hold_reasons=reasons)
            else:
                diffs = compare_adjusted([d for d in overlap], [stored_map[d] for d in overlap], [dict(timestamp=d + "T00:00:00+09:00", closePrice=str(compared[d])) for d in overlap])
                rec.update(diffs=diffs[:20], n_diffs=len(diffs))
                if not diffs:
                    rec["status"] = "unchanged"
                else:
                    try:
                        rows, conflicts, bad = full_adjusted_history(self, sym, origins[sym], RECOVERY_PAGES.get(sym))
                        kept = {d: v for d, v in rows.items() if d <= cutoff_day}
                        missing = sorted(d for d in stored_map if d <= cutoff_day and d not in kept)
                        problems = []
                        if conflicts:
                            problems.append("conflicting closes for the same date across pages")
                        if bad:
                            problems.append("non-finite or non-positive close in the rebuilt series")
                        if missing:
                            problems.append(f"{len(missing)} stored dates missing from the rebuilt series")
                        if cutoff_day not in kept:
                            problems.append("the rebuilt series does not reach the latest completed session")
                        if not kept or min(kept) > origins[sym]:
                            problems.append("the rebuilt series does not reach the origin")
                        rec.update(rebuilt_rows=len(kept), stored_rows_to_cutoff=sum(1 for d in stored_map if d <= cutoff_day), missing_stored_dates=missing[:30], n_missing_stored_dates=len(missing), rebuild_conflicts=conflicts[:10], rebuild_bad_prices=bad[:10])
                        if problems:
                            rec.update(status="hold_ribbon_signal", hold_reasons=problems)
                        else:
                            B.write_new(os.path.join(vdir, f"rebuilt_{sym}.json"), json.dumps(dict(symbol=sym, origin=origins[sym], cutoff_day=cutoff_day, days=sorted(kept), close=[kept[d] for d in sorted(kept)]), ensure_ascii=False).encode("utf-8"))
                            rec["status"] = "rebuilt_from_origin"
                    except B.Stop as e:
                        rec["status"] = "hold_ribbon_signal"
                        rec["rebuild_error"] = str(e)[:300]
            B.write_new(os.path.join(vdir, f"check_{sym}.json"), json.dumps(rec, ensure_ascii=False).encode("utf-8"))
            out[sym] = rec["status"]
        B.write_new(os.path.join(vdir, "meta.json"), json.dumps(dict(vintage=vintage, requests=self.requests, routine=self.n_routine, recovery=self.n_recovery, result=out), ensure_ascii=False, indent=1).encode("utf-8"))
        return vdir, out


def rows_from_candles(candles):
    import math
    rows, conflicts, bad = {}, [], []
    for c in candles:
        d = c["timestamp"][:10]
        try:
            v = float(c["closePrice"])
        except (TypeError, ValueError):
            v = float("nan")
        if not math.isfinite(v) or v <= 0:
            bad.append(dict(day=d, close=c.get("closePrice")))
            continue
        if d in rows and rows[d] != v:
            conflicts.append(dict(day=d, first=rows[d], second=v))
            continue
        rows[d] = v
    return rows, conflicts, bad


def full_adjusted_history(collector, sym, origin_day, max_pages=None):
    max_pages = max_pages or collector.max_recovery_pages
    rows, conflicts, bad, before = {}, [], [], None
    for page in range(max_pages):
        params = dict(symbol=sym, interval="1d", count=200, adjusted="true")
        if before:
            params["before"] = before
        data, _ = collector.call(params, budget="recovery")
        candles = (data.get("result") or data).get("candles")
        if not candles:
            break
        prow, pconf, pbad = rows_from_candles(candles)
        conflicts += pconf
        bad += pbad
        for d, v in prow.items():
            if d in rows and rows[d] != v:
                conflicts.append(dict(day=d, first=rows[d], second=v))
            else:
                rows[d] = v
        oldest = min(c["timestamp"] for c in candles)
        if before and datetime.fromisoformat(oldest) >= datetime.fromisoformat(before):
            raise B.Stop(f"adjusted pagination did not progress for {sym}")
        if oldest[:10] <= origin_day:
            return rows, conflicts, bad
        before = oldest
    raise B.Stop(f"origin {origin_day} not reached within {max_pages} pages for {sym}; hold the ribbon signal")


def compare_adjusted(stored_days, stored_close, new_candles, tol=1e-6):
    idx = {d: c for d, c in zip(stored_days, stored_close)}
    diffs = []
    for c in new_candles:
        d = c["timestamp"][:10]
        if d in idx:
            a, b = float(idx[d]), float(c["closePrice"])
            if abs(a - b) > tol * max(1.0, abs(a)):
                diffs.append(dict(day=d, stored=a, fetched=b))
    return diffs


def selftest():
    def fake_candles(day, start_min, n):
        out = []
        t = datetime.fromisoformat(f"{day}T09:01:00+09:00") + timedelta(minutes=start_min)
        for i in range(n):
            ts = (t + timedelta(minutes=i)).isoformat()
            out.append(dict(timestamp=ts, openPrice="100", highPrice="101", lowPrice="99", closePrice="100", volume="0" if "15:21" <= ts[11:16] <= "15:29" else "10"))
        return out

    calls = []

    def fake_fetch(params):
        calls.append(params)
        day = params["before"][:10]
        if params["before"].endswith("T23:59:59+09:00"):
            body = dict(candles=fake_candles(day, 190, 200))
        else:
            body = dict(candles=fake_candles(day, 0, 190) + fake_candles("2026-11-06", 389, 1))
        return 200, json.dumps(body).encode("utf-8"), {}

    def at(h, m, day=9):
        return lambda: datetime(2026, 11, day, h, m, tzinfo=KST)

    def mk(day, start_hm, n, extra=()):
        t0 = datetime.fromisoformat(f"{day}T{start_hm}:00")
        bars = {}
        for i in range(n):
            ts = (t0 + timedelta(minutes=i)).isoformat()
            bars.setdefault(ts, []).append(dict(open="100", high="101", low="99", close="100", volume="10"))
        for ts, close in extra:
            bars.setdefault(ts, []).append(dict(open="100", high="101", low="99", close=close, volume="10"))
        return bars

    with tempfile.TemporaryDirectory() as tmp:
        try:
            Collector(fetch=fake_fetch, clock=at(16, 5), sleep=lambda s: None, out_min=os.path.join(tmp, "m0")).minute_day("2026-11-09")
            raise AssertionError("must refuse before 16:10")
        except B.Stop as e:
            assert "too early" in str(e)
        assert earliest_fetch("2026-11-19").strftime("%H:%M") == "17:10" and earliest_fetch("2026-11-09").strftime("%H:%M") == "16:10" and session("2027-01-04")[3] == "late_open_special"
        c = Collector(fetch=fake_fetch, clock=at(16, 11), sleep=lambda s: None, out_min=os.path.join(tmp, "m"))
        vdir, out = c.minute_day("2026-11-09", syms=("069500",))
        assert out["069500"]["n"] == 390 and out["069500"]["cls"] == "normal" and not out["069500"]["held"], out
        assert len(c.requests) == 2 and len(os.listdir(os.path.join(vdir, "requests"))) == 2
        saved = json.load(open(os.path.join(vdir, "requests", "0000.json"), encoding="utf-8"))
        assert "body" in saved and saved["sha256"] == hashlib.sha256(saved["body"].encode("utf-8")).hexdigest()
        try:
            Collector(fetch=fake_fetch, clock=at(16, 11), sleep=lambda s: None, out_min=os.path.join(tmp, "m")).minute_day("2026-11-09", syms=("069500",))
            raise AssertionError("same vintage must not overwrite")
        except FileExistsError:
            pass

        def fetch429(params):
            raise B.Stop("http 429: rate-limit-exceeded", {"retry-after": "1"})
        c429 = Collector(fetch=fetch429, clock=at(16, 11), sleep=lambda s: None, out_min=os.path.join(tmp, "m429"))
        try:
            c429.minute_day("2026-11-09", syms=("069500",))
            raise AssertionError("429 must stop")
        except B.Stop as e:
            assert "429" in str(e)
        rq = os.listdir(os.path.join(c429.vdir, "requests"))
        rec429 = json.load(open(os.path.join(c429.vdir, "requests", rq[0]), encoding="utf-8"))
        assert len(rq) == 1 and "429" in rec429["error"] and rec429["headers"].get("retry-after") == "1"
        nT = []

        def fetchT(params):
            nT.append(1)
            raise B.Stop("transport: timed out")
        cT = Collector(fetch=fetchT, clock=at(16, 11), sleep=lambda s: None, out_min=os.path.join(tmp, "mT"))
        try:
            cT.minute_day("2026-11-09", syms=("069500",))
            raise AssertionError("transport must stop after retries")
        except B.Stop:
            assert len(nT) == 1 + TRANSPORT_RETRIES and len(os.listdir(os.path.join(cT.vdir, "requests"))) == 1 + TRANSPORT_RETRIES

        def fetch_stuck(params):
            return 200, json.dumps(dict(candles=fake_candles("2026-11-09", 190, 200) if params["before"].endswith("23:59:59+09:00") else fake_candles("2026-11-09", 190, 200))).encode("utf-8"), {}
        cS = Collector(fetch=fetch_stuck, clock=at(16, 11), sleep=lambda s: None, out_min=os.path.join(tmp, "mS"))
        try:
            cS.minute_day("2026-11-09", syms=("069500",))
            raise AssertionError("non-progressing pagination must stop")
        except B.Stop as e:
            assert "did not progress" in str(e) and len(os.listdir(os.path.join(cS.vdir, "requests"))) == 2
        cC = Collector(fetch=fake_fetch, clock=at(16, 11), sleep=lambda s: None, out_min=os.path.join(tmp, "mC"), max_req=1)
        try:
            cC.minute_day("2026-11-09", syms=("069500",))
            raise AssertionError("routine budget must stop")
        except B.Stop as e:
            assert "routine request budget" in str(e)

        def daily(day0, n, overrides=None, step=1):
            overrides = overrides or {}
            out = []
            d = datetime.fromisoformat(day0)
            for k in range(n):
                ds = (d - timedelta(days=k * step)).strftime("%Y-%m-%d")
                out.append(dict(timestamp=ds + "T00:00:00+09:00", closePrice=overrides.get(ds, "100")))
            return out

        def stored_from(candles):
            ds = sorted(c["timestamp"][:10] for c in candles)
            return (ds, [100.0] * len(ds))

        def make_fetch(first_page, history_pages=None):
            pages = list(history_pages or [])

            def f(params):
                if "before" not in params and f.first:
                    f.first = False
                    return 200, json.dumps(dict(candles=first_page)).encode("utf-8"), {}
                page = pages.pop(0) if pages else []
                return 200, json.dumps(dict(candles=page)).encode("utf-8"), {}
            f.first = True
            return f

        full = daily("2026-11-06", 200)
        stored = {"069500": stored_from(full)}

        def run_case(name, first_page, history_pages=None, stored_override=None, origin="2026-01-01", max_rec=5):
            c = Collector(fetch=make_fetch(first_page, history_pages), clock=at(16, 11), sleep=lambda s: None, out_adj=os.path.join(tmp, "adj_" + name), max_req=1, max_recovery_pages=max_rec)
            v, o = c.adjusted_checked(stored_override or stored, "2026-11-06", {"069500": origin}, syms=("069500",))
            return o["069500"], json.load(open(os.path.join(v, "check_069500.json"), encoding="utf-8")), v

        st_, rec, _ = run_case("same", full)
        assert st_ == "unchanged" and rec["overlap_rows"] == 200, (st_, rec)
        st_, rec, _ = run_case("empty", [])
        assert st_ == "hold_invalid_page" and "empty response" in rec["hold_reasons"]
        st_, rec, _ = run_case("nooverlap", daily("2026-11-06", 200), stored_override={"069500": (["2020-01-01"], [1.0])})
        assert st_ == "hold_invalid_page" and any("overlapping" in r for r in rec["hold_reasons"])
        st_, rec, _ = run_case("aftercutoff", daily("2026-11-10", 3))
        assert st_ == "hold_invalid_page" and "the latest completed session is not in the page" in rec["hold_reasons"]
        st_, rec, _ = run_case("nan", daily("2026-11-06", 200, overrides={"2026-11-03": "nan"}))
        assert st_ == "hold_invalid_page" and "non-finite or non-positive close in the page" in rec["hold_reasons"]
        dup = daily("2026-11-06", 200) + [dict(timestamp="2026-11-05T00:00:00+09:00", closePrice="80")]
        dup[1]["closePrice"] = "90"
        st_, rec, _ = run_case("dupconflict", dup)
        assert st_ == "hold_invalid_page" and "conflicting closes for the same date in the page" in rec["hold_reasons"]
        changed = daily("2026-11-06", 200, overrides={"2026-11-05": "101"})
        hist_full = [daily("2026-11-06", 200, overrides={"2026-11-05": "101"}), daily(full[-1]["timestamp"][:10], 200)[1:] + daily("2025-01-01", 5)]
        st_, rec, v = run_case("rebuilt", changed, history_pages=hist_full, origin="2025-01-01")
        assert st_ == "rebuilt_from_origin" and rec["n_missing_stored_dates"] == 0 and os.path.exists(os.path.join(v, "rebuilt_069500.json")), (st_, rec)
        sparse = [[dict(timestamp="2026-11-06T00:00:00+09:00", closePrice="101"), dict(timestamp="2025-01-01T00:00:00+09:00", closePrice="50")]]
        st_, rec, _ = run_case("sparse", daily("2026-11-06", 200, overrides={"2026-11-05": "101"}), history_pages=sparse, origin="2025-01-01")
        assert st_ == "hold_ribbon_signal" and rec["n_missing_stored_dates"] > 0 and any("missing from the rebuilt series" in r for r in rec["hold_reasons"]), (st_, rec)
        conf_hist = [daily("2026-11-06", 200, overrides={"2026-11-05": "101"}), [dict(timestamp="2026-11-05T00:00:00+09:00", closePrice="102")] + daily("2025-01-01", 1)]
        st_, rec, _ = run_case("rebuildconflict", daily("2026-11-06", 200, overrides={"2026-11-05": "101"}), history_pages=conf_hist, origin="2025-01-01")
        assert st_ == "hold_ribbon_signal" and any("conflicting closes" in r for r in rec["hold_reasons"]), (st_, rec)
        st_, rec, _ = run_case("budget", daily("2026-11-06", 200, overrides={"2026-11-05": "101"}), history_pages=[daily("2026-11-06", 200)], origin="2002-10-14", max_rec=1)
        assert st_ == "hold_ribbon_signal" and ("recovery budget" in rec.get("rebuild_error", "") or "not reached" in rec.get("rebuild_error", "")), (st_, rec)
        sm, _ = summarize_local("2026-11-09", mk("2026-11-09", "09:01", 390))
        assert sm["cls"] == "normal" and not sm["held"], sm
        sm, _ = summarize_local("2026-11-09", mk("2026-11-09", "09:01", 390, extra=[("2026-11-09T10:00:00", "999")]))
        assert sm["held"] and "conflicting duplicate bars" in sm["hold_reasons"]
        sm, rows_in = summarize_local("2026-11-09", mk("2026-11-09", "09:01", 390, extra=[("2026-11-09T08:59:00", "100")]))
        assert sm["held"] and "bars outside the declared session" in sm["hold_reasons"] and len(rows_in) == 390
        sm, _ = summarize_local("2026-11-09", mk("2026-11-09", "09:01", 390, extra=[("2026-11-10T09:01:00", "100")]))
        assert sm["held"] and "bars dated after the requested day" in sm["hold_reasons"]
        sm, _ = summarize_local("2027-01-04", mk("2027-01-04", "10:01", 330))
        assert sm["cls"] == "late_open_special" and not sm["held"] and sm["n_missing"] == 0, sm
        sm, _ = summarize_local("2026-11-19", mk("2026-11-19", "10:01", 390))
        assert sm["cls"] == "shifted_session_special" and not sm["held"], sm
        sm, _ = summarize_local("2026-11-09", mk("2026-11-09", "09:01", 389))
        assert sm["held"] and sm["cls"] == "short_or_missing"
    print("selftest ok (fake fetch only, temp directories, no network, no token use)")


def main():
    ap = argparse.ArgumentParser(description="DRAFT forward collector for KODEX 1m bars and adjusted daily vintages; not to be run before review")
    ap.add_argument("cmd", choices=["selftest", "minute", "adjusted"])
    ap.add_argument("--day")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "minute":
        print(json.dumps(Collector().minute_day(a.day), ensure_ascii=False, indent=1))
    else:
        raise SystemExit("adjusted collection needs the stored series, cutoff and origins; not wired to the CLI in this draft")


if __name__ == "__main__":
    main()
