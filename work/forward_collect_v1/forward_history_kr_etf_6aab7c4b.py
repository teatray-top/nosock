import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ft"))
import backfill_kr_etf_1m as B
import forward_collect_kr_etf as FC

SYMS = FC.SYMS
KST = FC.KST
ORIGINS = {"069500": "2002-10-14", "229200": "2015-10-01"}
BASE_ADJ = os.path.join(HERE, "data", "kr_etf", "kodex_daily_adj.json")
BASE_ADJ_SHA = "844b27891727b1faa227ea3259cb47eee258f8bf85ec30a3fc6a41b5a299a609"
BASE_LAST_DAY = "2026-11-05"
BASE_MIN = {
    "069500": (os.path.join(HERE, "ft", "user_data", "data_kodex", "source_snapshot", "069500.csv"), "ee10cfe3239bac46a065bb646d0fee3313a8ddb53888866968c7641d9e3f8f93"),
    "229200": (os.path.join(HERE, "ft", "user_data", "data_kosdaq150_v1", "source_snapshot", "229200.csv"), "5213ea897086b2eb2e57172de4f7fe7885d47acb6c40e2f90069225813f4edb5"),
}
DECLARED_FIRST = "2026-11-06"
DECLARED_LAST = "2027-01-29"
DECLARED_CLOSED = {"2026-12-25", "2026-12-31", "2027-01-01"}
REFETCH_AT = "08:30"
SUPPLEMENT_AT = "2026-11-08T12:00:00+09:00"
HIST_ROOT = os.path.join(HERE, "data", "kr_etf", "forward_daily_adj_history")
OBS_ROOT = os.path.join(HERE, "data", "kr_etf", "forward_obs_inputs")
BASE_VINTAGE = "0000_base"


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")


def declared_sessions(first=DECLARED_FIRST, last=DECLARED_LAST, closed=DECLARED_CLOSED):
    d, out = date.fromisoformat(first), []
    while d <= date.fromisoformat(last):
        if d.weekday() < 5 and d.isoformat() not in closed:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def latest_completed_session(now, sessions):
    if now.date().isoformat() > sessions[-1]:
        raise B.Stop(f"{now.date()} is past the declared schedule (last {sessions[-1]}); declare the next schedule first")
    done = [d for d in sessions if FC.earliest_fetch(d) <= now]
    if not done:
        raise B.Stop("no declared session is complete yet")
    return done[-1]


def schedule_for(day, sessions):
    i = sessions.index(day)
    at = FC.earliest_fetch(day).isoformat()
    jobs = [dict(at=at, job="minute", day=day, budget="routine", requests=2 * len(SYMS)), dict(at=at, job="adjusted", cutoff_day=day, budget="routine", requests=len(SYMS), recovery_pages_if_changed=dict(FC.RECOVERY_PAGES), recovery_run_total=FC.ORIGIN_MAX_PAGES)]
    if i + 1 < len(sessions):
        jobs.append(dict(at=f"{sessions[i + 1]}T{REFETCH_AT}:00+09:00", job="minute_refetch", day=day, budget="routine", requests=2 * len(SYMS)))
    else:
        jobs.append(dict(at=None, job="minute_refetch", day=day, note="the next session is outside the declared schedule"))
    return jobs


def base_history(path=BASE_ADJ, want_sha=BASE_ADJ_SHA, last_day=BASE_LAST_DAY, origins=ORIGINS, syms=SYMS):
    got = sha_file(path)
    if got != want_sha:
        raise B.Stop(f"base adjusted history hash {got[:8]} differs from {want_sha[:8]}")
    src = json.load(open(path, encoding="utf-8"))
    out, dropped = {}, {}
    for sym in syms:
        days, closes = src[sym]["days"], src[sym]["C"]
        if days[0] != origins[sym]:
            raise B.Stop(f"{sym} base history starts {days[0]}, origin {origins[sym]}")
        if days != sorted(set(days)):
            raise B.Stop(f"{sym} base history has unsorted or duplicate dates")
        keep = [i for i, d in enumerate(days) if d <= last_day]
        if not keep or days[keep[-1]] != last_day:
            raise B.Stop(f"{sym} base history does not contain {last_day}")
        out[sym] = dict(days=[days[i] for i in keep], close=[float(closes[i]) for i in keep])
        dropped[sym] = dict(rows=[dict(day=days[i], close=closes[i]) for i in range(len(days)) if days[i] > last_day], fetched=src[sym].get("fetched"))
    return out, dict(source=os.path.basename(path), source_sha256=got, cut_after=last_day, dropped_rows=dropped, reason="rows after the cut were fetched before the session completed (15:31 KST) and are not closes")


def write_base_vintage(hist_root, base, meta, origins=ORIGINS):
    vdir = os.path.join(hist_root, BASE_VINTAGE)
    os.makedirs(vdir, exist_ok=False)
    for sym, h in base.items():
        B.write_new(os.path.join(vdir, f"{sym}.json"), dump(dict(symbol=sym, vintage=BASE_VINTAGE, kind="base", cutoff_day=h["days"][-1], origin=origins[sym], days=h["days"], close=h["close"], parent=None, source=meta)))
    B.write_new(os.path.join(vdir, "meta.json"), dump(dict(vintage=BASE_VINTAGE, kind="base", source=meta, files={sym: sha_file(os.path.join(vdir, f"{sym}.json")) for sym in base})))
    return vdir


def history_files(hist_root, sym):
    out = []
    for v in sorted(os.listdir(hist_root)):
        p = os.path.join(hist_root, v, f"{sym}.json")
        if os.path.exists(p):
            out.append(p)
    return out


def latest_history(hist_root, sym):
    files = history_files(hist_root, sym)
    if not files:
        raise B.Stop(f"no history vintage for {sym}")
    return files[-1], json.load(open(files[-1], encoding="utf-8"))


def stored_for_check(hist_root, syms=SYMS):
    out = {}
    for sym in syms:
        _, h = latest_history(hist_root, sym)
        out[sym] = (h["days"], h["close"])
    return out


def new_day_problems(prev_last, new_days, sessions, cutoff_day):
    want = [d for d in sessions if prev_last < d <= cutoff_day]
    problems = []
    if cutoff_day not in sessions:
        problems.append(f"cutoff {cutoff_day} is not a declared session")
    if prev_last >= cutoff_day:
        problems.append(f"history already reaches {prev_last}, cutoff {cutoff_day}")
    extra = sorted(set(new_days) - set(want))
    missing = [d for d in want if d not in set(new_days)]
    if extra:
        problems.append(f"rows on dates that are not declared sessions: {extra}")
    if missing:
        problems.append(f"declared sessions missing from the rows: {missing}")
    return want, problems


def extend_history(hist_root, adj_vdir, cutoff_day, sessions):
    vintage = os.path.basename(adj_vdir.rstrip("/\\"))
    meta_path = os.path.join(adj_vdir, "meta.json")
    adj_meta = json.load(open(meta_path, encoding="utf-8"))
    files, result = {}, {}
    for sym in adj_meta["result"]:
        check_path = os.path.join(adj_vdir, f"check_{sym}.json")
        check = json.load(open(check_path, encoding="utf-8"))
        ppath, prev = latest_history(hist_root, sym)
        prev_last = prev["days"][-1]
        reasons = []
        if check["cutoff_day"] != cutoff_day:
            reasons.append(f"check cutoff {check['cutoff_day']} differs from {cutoff_day}")
        if check.get("stored_last_day") != prev_last:
            reasons.append(f"the check compared against history ending {check.get('stored_last_day')}, latest history ends {prev_last}")
        source = dict(adjusted_vintage=vintage, check_sha256=sha_file(check_path), adjusted_meta_sha256=sha_file(meta_path), check_status=check["status"])
        if check["status"] == "unchanged":
            if check.get("unknown_days"):
                reasons.append(f"page has dates the history lacks: {check['unknown_days'][:10]}")
            want, problems = new_day_problems(prev_last, list(check.get("new_rows", {})), sessions, cutoff_day)
            reasons += problems
            if not reasons:
                days = prev["days"] + want
                close = prev["close"] + [float(check["new_rows"][d]) for d in want]
                kind = "appended"
        elif check["status"] == "rebuilt_from_origin":
            rb_path = os.path.join(adj_vdir, f"rebuilt_{sym}.json")
            rb = json.load(open(rb_path, encoding="utf-8"))
            source["rebuilt_sha256"] = sha_file(rb_path)
            rb_set, prev_set = set(rb["days"]), set(prev["days"])
            lacking = [d for d in prev["days"] if d not in rb_set]
            unknown = [d for d in rb["days"] if d <= prev_last and d not in prev_set]
            if lacking:
                reasons.append(f"rebuilt series lacks {len(lacking)} stored dates")
            if unknown:
                reasons.append(f"rebuilt series has {len(unknown)} dates the history lacks: {unknown[:10]}")
            if rb["days"][0] != prev["origin"]:
                reasons.append(f"rebuilt series starts {rb['days'][0]}, origin {prev['origin']}")
            want, problems = new_day_problems(prev_last, [d for d in rb["days"] if d > prev_last], sessions, cutoff_day)
            reasons += problems
            if not reasons:
                days, close, kind = list(rb["days"]), [float(v) for v in rb["close"]], "rebuilt"
        else:
            reasons.append(f"adjusted check status {check['status']}")
        if reasons:
            result[sym] = dict(status="held", reasons=reasons, latest_history=os.path.relpath(ppath, hist_root).replace("\\", "/"))
            continue
        files[sym] = dict(symbol=sym, vintage=vintage, kind=kind, cutoff_day=cutoff_day, origin=prev["origin"], days=days, close=close, parent=dict(path=os.path.relpath(ppath, hist_root).replace("\\", "/"), sha256=sha_file(ppath)), source=source)
        result[sym] = dict(status=kind, n_days=len(days), new_days=want)
    vdir = os.path.join(hist_root, vintage)
    os.makedirs(vdir, exist_ok=False)
    for sym, payload in files.items():
        B.write_new(os.path.join(vdir, f"{sym}.json"), dump(payload))
    B.write_new(os.path.join(vdir, "meta.json"), dump(dict(vintage=vintage, cutoff_day=cutoff_day, result=result, files={sym: sha_file(os.path.join(vdir, f"{sym}.json")) for sym in files})))
    return vdir, result


def verify_chain(hist_root, sym):
    path, h = latest_history(hist_root, sym)
    chain = []
    while True:
        chain.append(os.path.relpath(path, hist_root).replace("\\", "/"))
        if h["kind"] == "base":
            if h["parent"] is not None:
                raise B.Stop(f"{sym} base vintage has a parent")
            return chain
        ppath = os.path.join(hist_root, h["parent"]["path"])
        if sha_file(ppath) != h["parent"]["sha256"]:
            raise B.Stop(f"{sym} parent hash differs at {chain[-1]}")
        p = json.load(open(ppath, encoding="utf-8"))
        if not p["cutoff_day"] < h["cutoff_day"]:
            raise B.Stop(f"{sym} cutoff does not advance at {chain[-1]}")
        if h["kind"] == "appended" and (h["days"][:len(p["days"])] != p["days"] or h["close"][:len(p["close"])] != p["close"]):
            raise B.Stop(f"{sym} appended vintage does not keep its parent as a prefix at {chain[-1]}")
        if h["kind"] == "rebuilt" and [d for d in h["days"] if d <= p["cutoff_day"]] != p["days"]:
            raise B.Stop(f"{sym} rebuilt vintage changes the parent's calendar at {chain[-1]}")
        path, h = ppath, p


def minute_vintages(min_root, day, sym):
    if not os.path.isdir(min_root):
        return []
    return [os.path.join(min_root, v, f"{day}_{sym}.json") for v in sorted(os.listdir(min_root)) if os.path.exists(os.path.join(min_root, v, f"{day}_{sym}.json"))]


def adjusted_for(kind, hist_root, sym, through_day):
    files = history_files(hist_root, sym)
    loaded = [(p, json.load(open(p, encoding="utf-8"))) for p in files]
    if kind == "as_received":
        hits = [(p, h) for p, h in loaded if h["cutoff_day"] == through_day]
        if not hits:
            before = [h["cutoff_day"] for _, h in loaded if h["cutoff_day"] < through_day]
            return None, f"no history vintage was made with cutoff {through_day} (latest before: {before[-1] if before else None})"
        return hits[0], None
    hits = [(p, h) for p, h in loaded if h["cutoff_day"] >= through_day]
    if not hits:
        return None, f"no history vintage reaches {through_day}"
    return hits[-1], None


def observation_input(kind, through_day, sessions, min_root, hist_root, out_root, built_at, base_min=BASE_MIN, base_last=BASE_LAST_DAY, syms=SYMS):
    if kind not in ("as_received", "revised"):
        raise ValueError(kind)
    forward, held, complete_through = [], [], base_last
    for d in [s for s in sessions if base_last < s <= through_day]:
        recs, reasons = [], []
        for sym in syms:
            vs = minute_vintages(min_root, d, sym)
            if not vs:
                reasons.append(f"{sym}: no minute vintage")
                continue
            p = vs[0] if kind == "as_received" else vs[-1]
            rec = json.load(open(p, encoding="utf-8"))
            if rec["day"] != d or rec["symbol"] != sym:
                reasons.append(f"{sym}: file content is for {rec['symbol']} {rec['day']}")
            if rec["summary"]["held"]:
                reasons.append(f"{sym}: chosen vintage {rec['vintage']} held: {rec['summary']['hold_reasons']}")
            recs.append(dict(day=d, symbol=sym, vintage=rec["vintage"], path=os.path.relpath(p, min_root).replace("\\", "/"), sha256=sha_file(p), rows_sha256=sha_bytes(dump(rec["rows"])), n=len(rec["rows"]), cls=rec["summary"]["cls"], n_vintages=len(vs)))
        if reasons:
            held.append(dict(day=d, reasons=reasons))
            break
        forward += recs
        complete_through = d
    adjusted = {}
    for sym in syms:
        hit, why = adjusted_for(kind, hist_root, sym, complete_through)
        if hit is None:
            adjusted[sym] = dict(held=True, reason=why)
        else:
            p, h = hit
            adjusted[sym] = dict(held=False, path=os.path.relpath(p, hist_root).replace("\\", "/"), sha256=sha_file(p), vintage=h["vintage"], kind=h["kind"], cutoff_day=h["cutoff_day"], truncated_to=complete_through, n_days=sum(1 for x in h["days"] if x <= complete_through))
    man = dict(version="forward-observation-input-v1", kind=kind, requested_through=through_day, complete_through=complete_through, built_at=built_at, rule=("first minute vintage of each day and the history vintage made with that day's cutoff" if kind == "as_received" else "newest minute vintage of each day and the newest history vintage, truncated to the last complete day"), base_minute={sym: dict(path=os.path.basename(base_min[sym][0]), sha256=base_min[sym][1], last_day=base_last) for sym in syms}, forward_minute=forward, held=held, adjusted=adjusted, collector_sha256=sha_file(FC.__file__), history_code_sha256=sha_file(os.path.abspath(__file__)))
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, f"{kind}_{complete_through}_{built_at}.json")
    B.write_new(path, dump(man))
    return path, man


def materialize(manifest_path, min_root, hist_root, base_min=BASE_MIN):
    import pandas as pd
    m = json.load(open(manifest_path, encoding="utf-8"))
    minute = {}
    for sym, b in m["base_minute"].items():
        bpath = base_min[sym][0]
        if sha_file(bpath) != b["sha256"]:
            raise B.Stop(f"base minute hash differs for {sym}")
        df = pd.read_csv(bpath, dtype={"ts": str})
        if df["ts"].iloc[-1][:10] != b["last_day"]:
            raise B.Stop(f"base minute for {sym} ends {df['ts'].iloc[-1][:10]}, manifest says {b['last_day']}")
        rows = [df[["ts", "open", "high", "low", "close", "volume"]]]
        for f in [x for x in m["forward_minute"] if x["symbol"] == sym]:
            p = os.path.join(min_root, f["path"])
            if sha_file(p) != f["sha256"]:
                raise B.Stop(f"minute vintage hash differs: {f['path']}")
            rec = json.load(open(p, encoding="utf-8"))
            if sha_bytes(dump(rec["rows"])) != f["rows_sha256"]:
                raise B.Stop(f"minute rows hash differs: {f['path']}")
            rows.append(pd.DataFrame(rec["rows"], columns=["ts", "open", "high", "low", "close", "volume"]))
        out = pd.concat(rows, ignore_index=True)
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = out[c].astype(float)
        if not out["ts"].is_monotonic_increasing or out["ts"].duplicated().any():
            raise B.Stop(f"{sym} minute rows are not strictly increasing")
        minute[sym] = out
    adjusted = {}
    for sym, a in m["adjusted"].items():
        if a["held"]:
            adjusted[sym] = None
            continue
        p = os.path.join(hist_root, a["path"])
        if sha_file(p) != a["sha256"]:
            raise B.Stop(f"history vintage hash differs: {a['path']}")
        h = json.load(open(p, encoding="utf-8"))
        k = [i for i, d in enumerate(h["days"]) if d <= a["truncated_to"]]
        adjusted[sym] = (h["days"][:len(k)], h["close"][:len(k)])
    return dict(manifest=m, minute=minute, adjusted=adjusted)


def to_5m(df_1m, tmp_dir, name):
    import kr_to_ft as K
    p = os.path.join(tmp_dir, f"{name}.csv")
    df_1m.to_csv(p, index=False)
    return K.resample(K.load_1m(p), "5m")


def selftest():
    import pandas as pd
    tmp = tempfile.mkdtemp(prefix="fwd_hist_")
    try:
        sessions = declared_sessions()
        assert sessions[0] == "2026-11-06" and "2026-11-07" not in sessions and "2026-12-25" not in sessions and "2026-12-31" not in sessions and "2027-01-01" not in sessions and "2027-01-04" in sessions
        t = lambda s: datetime.fromisoformat(s)
        assert latest_completed_session(t("2026-11-08T12:00:00+09:00"), sessions) == "2026-11-06"
        assert latest_completed_session(t("2026-11-09T16:09:59+09:00"), sessions) == "2026-11-06"
        assert latest_completed_session(t("2026-11-09T16:10:00+09:00"), sessions) == "2026-11-09"
        assert latest_completed_session(t("2026-11-19T17:09:00+09:00"), sessions) == "2026-11-18"
        assert latest_completed_session(t("2026-11-19T17:10:00+09:00"), sessions) == "2026-11-19"
        assert latest_completed_session(t("2026-12-25T20:00:00+09:00"), sessions) == "2026-12-24"
        assert latest_completed_session(t("2027-01-04T08:30:00+09:00"), sessions) == "2026-12-30"
        assert latest_completed_session(t("2027-01-04T16:10:00+09:00"), sessions) == "2027-01-04"
        try:
            latest_completed_session(t("2027-02-01T09:00:00+09:00"), sessions)
            raise AssertionError("past the declared schedule must stop")
        except B.Stop:
            pass
        sch = schedule_for("2026-11-19", sessions)
        assert sch[0]["at"] == "2026-11-19T17:10:00+09:00" and sch[2]["at"] == "2026-11-20T08:30:00+09:00"
        assert schedule_for("2026-12-24", sessions)[2]["at"] == "2026-12-28T08:30:00+09:00"
        assert schedule_for("2026-12-30", sessions)[2]["at"] == "2027-01-04T08:30:00+09:00"
        assert schedule_for("2027-01-04", sessions)[0]["at"] == "2027-01-04T16:10:00+09:00"
        assert schedule_for("2027-01-29", sessions)[2]["at"] is None
        want, pr = new_day_problems("2026-12-24", ["2026-12-25", "2026-12-28"], sessions, "2026-12-28")
        assert want == ["2026-12-28"] and any("not declared sessions" in x for x in pr)
        want, pr = new_day_problems("2026-11-05", ["2026-11-09"], sessions, "2026-11-09")
        assert any("missing" in x for x in pr)

        origins = {"069500": "2026-01-01", "229200": "2026-01-01"}
        base_days = [d for d in (date(2026, 1, 1) + timedelta(days=i) for i in range(400)) if d.weekday() < 5 and d.isoformat() <= "2026-11-05"]
        base_days = [d.isoformat() for d in base_days]
        px0 = {"069500": 80000.0, "229200": 11000.0}
        world = dict(adj={sym: {d: px0[sym] + 5 * (i % 13) for i, d in enumerate(base_days)} for sym in SYMS}, adj_last="2026-11-05", min_override={}, drop=set())
        fake_src = {sym: dict(days=base_days + ["2026-11-06"], C=[world["adj"][sym][d] for d in base_days] + [1.0], fetched="2026-11-06T11:54:53") for sym in SYMS}
        base_adj_path = os.path.join(tmp, "base_adj.json")
        open(base_adj_path, "w", encoding="utf-8").write(json.dumps(fake_src))
        try:
            base_history(base_adj_path, "0" * 64, origins=origins)
            raise AssertionError("wrong base hash must stop")
        except B.Stop:
            pass
        base, bmeta = base_history(base_adj_path, sha_file(base_adj_path), origins=origins)
        assert base["069500"]["days"][-1] == "2026-11-05" and bmeta["dropped_rows"]["069500"]["rows"] == [dict(day="2026-11-06", close=1.0)]

        def day_minutes(day):
            first, last = FC.session(day)[:2]
            return B.minutes_expected(first, last)

        def bar(sym, day, hm, i):
            p = px0[sym] + 5 * (i % 7)
            ov = world["min_override"].get((sym, day, hm))
            close = ov if ov is not None else p
            vol = "0" if hm in FC.session(day)[2] else str(1000 + i)
            return dict(timestamp=f"{day}T{hm}:00+09:00", openPrice=str(p), highPrice=str(max(p, close) + 5), lowPrice=str(min(p, close) - 5), closePrice=str(close), volume=vol)

        min_days = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"]

        def all_bars(sym):
            return [bar(sym, d, hm, i) for d in min_days for i, hm in enumerate(day_minutes(d)) if (sym, d, hm) not in world["drop"]]

        base_min = {}
        for sym in SYMS:
            rows = [b for b in all_bars(sym) if b["timestamp"][:10] <= "2026-11-05"]
            p = os.path.join(tmp, f"base_{sym}.csv")
            pd.DataFrame(dict(ts=[b["timestamp"][:19] for b in rows], open=[int(float(b["openPrice"])) for b in rows], high=[int(float(b["highPrice"])) for b in rows], low=[int(float(b["lowPrice"])) for b in rows], close=[int(float(b["closePrice"])) for b in rows], volume=[int(b["volume"]) for b in rows])).to_csv(p, index=False)
            base_min[sym] = (p, sha_file(p))

        calls = []

        def fetch(params):
            calls.append(dict(params))
            sym = params["symbol"]
            if params["interval"] == "1m":
                bs = [b for b in all_bars(sym) if b["timestamp"] < params["before"]]
                return 200, json.dumps(dict(candles=bs[-200:])).encode("utf-8"), {}
            if sym in world.get("adj_empty", ()):
                return 200, json.dumps(dict(candles=[])).encode("utf-8"), {}
            days = sorted(d for d in world["adj"][sym] if d <= world["adj_last"])
            if "before" in params:
                days = [d for d in days if d < params["before"][:10]]
            days = days[-200:]
            return 200, json.dumps(dict(candles=[dict(timestamp=f"{d}T00:00:00+09:00", closePrice=str(world["adj"][sym][d])) for d in days])).encode("utf-8"), {}

        min_root, adj_root, hist_root, obs_root = (os.path.join(tmp, x) for x in ("min", "adj", "hist", "obs"))

        def collector(at_iso):
            return FC.Collector(fetch=fetch, clock=lambda: t(at_iso), sleep=lambda s: None, out_min=min_root, out_adj=adj_root)

        write_base_vintage(hist_root, base, bmeta, origins=origins)
        try:
            write_base_vintage(hist_root, base, bmeta, origins=origins)
            raise AssertionError("base vintage must not be rewritten")
        except FileExistsError:
            pass

        def routine(at_iso, day):
            assert latest_completed_session(t(at_iso), sessions) == day
            v1, o1 = collector(at_iso).minute_day(day)
            v2, o2 = collector(at_iso[:17] + "30" + at_iso[19:]).adjusted_checked(stored_for_check(hist_root), day, origins)
            v3, o3 = extend_history(hist_root, v2, day, sessions)
            return o1, o2, o3, v2

        world["adj"] = {sym: dict(world["adj"][sym], **{"2026-11-06": px0[sym] + 7}) for sym in SYMS}
        world["adj_last"] = "2026-11-06"
        o1, o2, o3, _ = routine(SUPPLEMENT_AT, "2026-11-06")
        assert all(not v["held"] for v in o1.values()) and set(o2.values()) == {"unchanged"} and all(v["status"] == "appended" and v["new_days"] == ["2026-11-06"] for v in o3.values()), (o1, o2, o3)
        _, o = collector("2026-11-09T08:30:00+09:00").minute_day("2026-11-06")
        try:
            collector("2026-11-09T16:09:00+09:00").minute_day("2026-11-09")
            raise AssertionError("too early must stop")
        except B.Stop:
            pass
        world["adj"] = {sym: dict(world["adj"][sym], **{"2026-11-09": px0[sym] + 9}) for sym in SYMS}
        world["adj_last"] = "2026-11-09"
        o1, o2, o3, adj_v_1109 = routine("2026-11-09T16:10:00+09:00", "2026-11-09")
        assert set(o2.values()) == {"unchanged"} and all(v["status"] == "appended" and v["new_days"] == ["2026-11-09"] for v in o3.values()), (o2, o3)
        try:
            extend_history(hist_root, adj_v_1109, "2026-11-09", sessions)
            raise AssertionError("a history vintage must not be rewritten")
        except FileExistsError:
            pass
        world["min_override"][("069500", "2026-11-09", "15:30")] = px0["069500"] + 100
        collector("2026-11-10T08:30:00+09:00").minute_day("2026-11-09")
        a_path, a_man = observation_input("as_received", "2026-11-09", sessions, min_root, hist_root, obs_root, "20261110T083100", base_min=base_min)
        r_path, r_man = observation_input("revised", "2026-11-09", sessions, min_root, hist_root, obs_root, "20261110T083101", base_min=base_min)
        assert a_man["complete_through"] == r_man["complete_through"] == "2026-11-09" and not a_man["held"] and not r_man["held"]
        av = {(f["day"], f["symbol"]): f for f in a_man["forward_minute"]}
        rv = {(f["day"], f["symbol"]): f for f in r_man["forward_minute"]}
        assert av[("2026-11-09", "069500")]["vintage"] == "20261109T161000" and rv[("2026-11-09", "069500")]["vintage"] == "20261110T083000"
        assert av[("2026-11-06", "069500")]["vintage"] == "20261108T120000" and rv[("2026-11-06", "069500")]["vintage"] == "20261109T083000" and av[("2026-11-06", "069500")]["rows_sha256"] == rv[("2026-11-06", "069500")]["rows_sha256"]
        assert av[("2026-11-09", "229200")]["rows_sha256"] == rv[("2026-11-09", "229200")]["rows_sha256"] and av[("2026-11-09", "069500")]["rows_sha256"] != rv[("2026-11-09", "069500")]["rows_sha256"]
        A = materialize(a_path, min_root, hist_root, base_min=base_min)
        R = materialize(r_path, min_root, hist_root, base_min=base_min)
        assert len(A["minute"]["069500"]) == 4 * 390 and A["minute"]["069500"]["ts"].iloc[-1] == "2026-11-09T15:30:00"
        diff = (A["minute"]["069500"][["open", "high", "low", "close", "volume"]] != R["minute"]["069500"][["open", "high", "low", "close", "volume"]]).any(axis=1)
        assert list(A["minute"]["069500"]["ts"][diff]) == ["2026-11-09T15:30:00"], list(A["minute"]["069500"]["ts"][diff])
        assert A["minute"]["229200"].equals(R["minute"]["229200"])
        assert A["adjusted"]["069500"] == R["adjusted"]["069500"] and A["adjusted"]["069500"][0][-2:] == ["2026-11-06", "2026-11-09"]
        a5, r5 = to_5m(A["minute"]["069500"], tmp, "a069500"), to_5m(R["minute"]["069500"], tmp, "r069500")
        d5 = (a5[["open", "high", "low", "close", "volume"]] != r5[["open", "high", "low", "close", "volume"]]).any(axis=1)
        assert len(a5) == 4 * 78 and [str(x) for x in a5["date"][d5]] == ["2026-11-09 06:25:00+00:00"], [str(x) for x in a5["date"][d5]]

        world["adj"] = {sym: dict(world["adj"][sym], **{"2026-11-10": px0[sym] + 11}) for sym in SYMS}
        world["adj"]["229200"] = {d: (round(v * 0.99, 6) if d < "2026-11-10" else v) for d, v in world["adj"]["229200"].items()}
        world["adj_last"] = "2026-11-10"
        o1, o2, o3, _ = routine("2026-11-10T16:10:00+09:00", "2026-11-10")
        assert o2 == {"069500": "unchanged", "229200": "rebuilt_from_origin"} and o3["069500"]["status"] == "appended" and o3["229200"]["status"] == "rebuilt", (o2, o3)
        for sym in SYMS:
            ch = verify_chain(hist_root, sym)
            assert ch[-1] == f"{BASE_VINTAGE}/{sym}.json" and len(ch) == 4, ch
        a2_path, _ = observation_input("as_received", "2026-11-09", sessions, min_root, hist_root, obs_root, "20261110T161100", base_min=base_min)
        r2_path, r2_man = observation_input("revised", "2026-11-09", sessions, min_root, hist_root, obs_root, "20261110T161101", base_min=base_min)
        A2 = materialize(a2_path, min_root, hist_root, base_min=base_min)
        R2 = materialize(r2_path, min_root, hist_root, base_min=base_min)
        assert r2_man["adjusted"]["229200"]["kind"] == "rebuilt" and r2_man["adjusted"]["229200"]["truncated_to"] == "2026-11-09"
        assert A2["adjusted"]["229200"][0] == R2["adjusted"]["229200"][0] and all(abs(r - round(a * 0.99, 6)) < 1e-9 for a, r in zip(A2["adjusted"]["229200"][1], R2["adjusted"]["229200"][1]))
        assert A2["adjusted"]["069500"] == R2["adjusted"]["069500"] == A["adjusted"]["069500"]

        min_days.append("2026-11-11")
        world["drop"].add(("229200", "2026-11-11", "15:30"))
        world["adj"] = {sym: dict(world["adj"][sym], **{"2026-11-11": px0[sym] + 13}) for sym in SYMS}
        world["adj_last"] = "2026-11-11"
        o1, o2, o3, _ = routine("2026-11-11T16:10:00+09:00", "2026-11-11")
        assert o1["229200"]["held"] and not o1["069500"]["held"]
        a3_path, a3_man = observation_input("as_received", "2026-11-11", sessions, min_root, hist_root, obs_root, "20261111T161100", base_min=base_min)
        assert a3_man["complete_through"] == "2026-11-10" and a3_man["held"][0]["day"] == "2026-11-11" and "229200" in a3_man["held"][0]["reasons"][0]
        assert a3_man["adjusted"]["229200"]["cutoff_day"] == "2026-11-10"

        min_days += ["2026-11-12", "2026-11-13"]
        world["adj"] = {sym: dict(world["adj"][sym], **{"2026-11-12": px0[sym] + 15}) for sym in SYMS}
        world["adj_last"] = "2026-11-12"
        world["adj_empty"] = {"069500"}
        o1, o2, o3, _ = routine("2026-11-12T16:10:00+09:00", "2026-11-12")
        assert o2["069500"] == "hold_invalid_page" and o3["069500"]["status"] == "held" and o3["229200"]["status"] == "appended", (o2, o3)
        world["adj_empty"] = set()
        world["adj"] = {sym: dict(world["adj"][sym], **{"2026-11-13": px0[sym] + 17}) for sym in SYMS}
        world["adj_last"] = "2026-11-13"
        o1, o2, o3, _ = routine("2026-11-13T16:10:00+09:00", "2026-11-13")
        assert o3["069500"]["new_days"] == ["2026-11-12", "2026-11-13"] and o3["229200"]["new_days"] == ["2026-11-13"], o3
        _, a4 = observation_input("as_received", "2026-11-12", sessions, min_root, hist_root, obs_root, "20261113T161100", base_min=base_min, syms=("069500",))
        assert a4["complete_through"] == "2026-11-12" and a4["adjusted"]["069500"]["held"] and "latest before: 2026-11-11" in a4["adjusted"]["069500"]["reason"], a4["adjusted"]
        r4_path, r4 = observation_input("revised", "2026-11-12", sessions, min_root, hist_root, obs_root, "20261113T161101", base_min=base_min, syms=("069500",))
        R4 = materialize(r4_path, min_root, hist_root, base_min=base_min)
        assert r4["adjusted"]["069500"]["cutoff_day"] == "2026-11-13" and R4["adjusted"]["069500"][0][-1] == "2026-11-12" and R4["adjusted"]["069500"][1][-1] == px0["069500"] + 15

        tampered = os.path.join(min_root, rv[("2026-11-09", "069500")]["path"])
        with open(tampered, "ab") as f:
            f.write(b" ")
        try:
            materialize(r_path, min_root, hist_root, base_min=base_min)
            raise AssertionError("tampered minute vintage must stop")
        except B.Stop as e:
            assert "hash differs" in str(e)
        hp = os.path.join(hist_root, BASE_VINTAGE, "229200.json")
        os.chmod(hp, 0o666)
        with open(hp, "ab") as f:
            f.write(b" ")
        try:
            verify_chain(hist_root, "229200")
            raise AssertionError("tampered parent must stop")
        except B.Stop as e:
            assert "parent hash differs" in str(e)

        sym_calls = []

        def slow_fetch(params):
            sym_calls.append(params)
            return 200, json.dumps(dict(candles=[dict(timestamp=f"2026-0{9 - len(sym_calls)}-01T00:00:00+09:00", closePrice="1")])).encode("utf-8"), {}
        c = FC.Collector(fetch=slow_fetch, clock=lambda: t("2026-11-12T16:10:00+09:00"), sleep=lambda s: None, out_adj=os.path.join(tmp, "cap"))
        c.open_vintage(os.path.join(tmp, "cap"))
        try:
            FC.full_adjusted_history(c, "229200", "2015-10-01", 3)
            raise AssertionError("per-symbol cap must stop")
        except B.Stop as e:
            assert "not reached within 3 pages" in str(e) and len(sym_calls) == 3
        n_min = sum(1 for x in calls if x["interval"] == "1m")
        n_adj = sum(1 for x in calls if x["interval"] == "1d")
        print(f"fixture ok: fake fetch only, temp dir, no network; calls 1m {n_min}, 1d {n_adj}; history chains verified; as_received/revised differ only where the fixture changed data")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def base_check(out_path):
    import pandas as pd
    base, meta = base_history()
    sessions = declared_sessions()
    rec = dict(version="forward-base-check-v1", base_adjusted=dict(source=meta["source"], source_sha256=meta["source_sha256"], cut_after=meta["cut_after"], dropped_rows=meta["dropped_rows"], days={s: [base[s]["days"][0], base[s]["days"][-1], len(base[s]["days"])] for s in SYMS}), base_minute={}, collector_sha256=sha_file(FC.__file__), history_code_sha256=sha_file(os.path.abspath(__file__)), declared_schedule=dict(first=DECLARED_FIRST, last=DECLARED_LAST, closed=sorted(DECLARED_CLOSED), sessions=len(sessions), specials={d: FC.session(d)[3] for d in sessions if FC.session(d)[3] != "normal"}), supplement_at=SUPPLEMENT_AT)
    feathers = {"069500": os.path.join(HERE, "ft", "user_data", "data_kodex_nfi_v1_2_algo", "binance", "ALGO_USDT-5m.feather"), "229200": os.path.join(HERE, "ft", "user_data", "data_kosdaq150_v1", "binance", "ALGO_USDT-5m.feather")}
    tmp = tempfile.mkdtemp(prefix="fwd_base_")
    try:
        for sym in SYMS:
            p, want = BASE_MIN[sym]
            got = sha_file(p)
            df = pd.read_csv(p, dtype={"ts": str})
            f5 = to_5m(df[["ts", "open", "high", "low", "close", "volume"]], tmp, sym)
            fe = pd.read_feather(feathers[sym])
            fe["date"] = pd.to_datetime(fe["date"], utc=True)
            same_len = len(f5) == len(fe)
            dates_equal = same_len and bool((f5["date"].values == fe["date"].values).all())
            px_equal = same_len and all(bool(((f5[c] / 10000 - fe[c]).abs() < 1e-9).all()) for c in ("open", "high", "low", "close")) and bool((f5["volume"] == fe["volume"]).all())
            rec["base_minute"][sym] = dict(path=os.path.relpath(p, HERE).replace("\\", "/"), sha256=got, sha_matches=got == want, rows=len(df), first=df["ts"].iloc[0], last=df["ts"].iloc[-1], feather=os.path.relpath(feathers[sym], HERE).replace("\\", "/"), feather_sha256=sha_file(feathers[sym]), resampled_5m_rows=len(f5), feather_rows=len(fe), dates_equal=dates_equal, prices_div_10000_and_volume_equal=px_equal)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    B.write_new(out_path, json.dumps(rec, ensure_ascii=False, indent=1).encode("utf-8"))
    print(json.dumps({k: v for k, v in rec.items() if k != "base_adjusted"}, ensure_ascii=False)[:1500])
    print("base adjusted", json.dumps(rec["base_adjusted"], ensure_ascii=False)[:600])


def main():
    ap = argparse.ArgumentParser(description="DRAFT forward history and observation inputs for KODEX; local files only, no network")
    ap.add_argument("cmd", choices=["selftest", "base_check"])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        base_check(a.out)


if __name__ == "__main__":
    main()
