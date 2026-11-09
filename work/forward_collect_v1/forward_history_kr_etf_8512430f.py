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
AS_RECEIVED_USE = "per_day lists, for every day, the minute vintage and the history vintage (or the hold) made by that day's routine jobs; a cumulative replay of past decisions must take each day's inputs from per_day; the top-level adjusted is only the snapshot made with complete_through's cutoff"
REVISED_USE = "latest snapshot of every file; it does not record what was available at past decision times and must not drive a cumulative replay of past decisions"
STAGING = "_staging"
DECISION_RULE = "an input may be used for a decision at time T only if its available_at is not after T (minute: seal time of the vintage meta, written last; history: time of the commit record appended after the vintage directory is renamed into place); unless decision times are given, each day's decision time is the open of the next declared session (next-session contract); materialize(cumulative=True) applies the rule and lists late inputs"


class SimulatedCrash(Exception):
    pass


def stamp(clock=None):
    return ((clock or (lambda: datetime.now(KST)))()).isoformat(timespec="milliseconds")


def next_session_open(day, sessions):
    i = sessions.index(day)
    if i + 1 >= len(sessions):
        return None
    nd = sessions[i + 1]
    return (datetime.fromisoformat(f"{nd}T{FC.session(nd)[0]}:00+09:00") - timedelta(minutes=1)).isoformat()


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")


def now_iso():
    return datetime.now(KST).isoformat(timespec="milliseconds")


def rel(p, root):
    return os.path.relpath(p, root).replace("\\", "/")


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


def base_files(base, meta, origins=ORIGINS):
    files, shas = {}, {}
    for sym, h in base.items():
        data = dump(dict(symbol=sym, vintage=BASE_VINTAGE, kind="base", cutoff_day=h["days"][-1], origin=origins[sym], days=h["days"], close=h["close"], parent=None, source=meta))
        files[f"{sym}.json"] = data
        shas[sym] = sha_bytes(data)
    files["meta.json"] = dump(dict(vintage=BASE_VINTAGE, kind="base", cutoff_day=meta["cut_after"], source=meta, files=shas, result={s: dict(status="base") for s in shas}))
    return files


def write_base_vintage(hist_root, base, meta, origins=ORIGINS, clock=None, crash=None):
    return commit_vintage(hist_root, BASE_VINTAGE, base_files(base, meta, origins), clock=clock, crash=crash)


def staged_vintages(hist_root):
    p = os.path.join(hist_root, STAGING)
    return sorted(os.listdir(p)) if os.path.isdir(p) else []


def commit_vintage(hist_root, vintage, files, clock=None, crash=None):
    final = os.path.join(hist_root, vintage)
    if os.path.exists(final):
        raise FileExistsError(final)
    stage = os.path.join(hist_root, STAGING, vintage)
    os.makedirs(stage, exist_ok=True)
    names = sorted(n for n in files if n != "meta.json") + ["meta.json"]
    for i, n in enumerate(names):
        sp = os.path.join(stage, n)
        if os.path.exists(sp):
            if sha_file(sp) != sha_bytes(files[n]):
                raise B.Stop(f"staged {vintage}/{n} differs from the recomputed content; manual action needed, the staged files are kept")
        else:
            B.write_new(sp, files[n])
        if crash == "after_first_file" and i == 0:
            raise SimulatedCrash(f"{vintage}: after staging {n}")
    recs = FC.load_seal_records(hist_root)
    for i, n in enumerate(names):
        rp = f"{vintage}/{n}"
        want = sha_bytes(files[n])
        if rp in recs:
            if recs[rp]["sha256"] != want:
                raise B.Stop(f"seal for {rp} differs from the recomputed content; manual action needed")
        else:
            FC.append_seal(hist_root, rp, want, len(files[n]), stamp(clock))
        if crash == "after_first_seal" and i == 0:
            raise SimulatedCrash(f"{vintage}: after sealing {n}")
    if crash == "before_rename":
        raise SimulatedCrash(f"{vintage}: before the rename")
    os.rename(stage, final)
    if crash == "after_rename":
        raise SimulatedCrash(f"{vintage}: after the rename, before the commit record")
    FC.append_seal(hist_root, f"{vintage}/.commit", sha_bytes(files["meta.json"]), 0, stamp(clock))
    return final


def uncommitted_vintages(hist_root, recs=None):
    recs = FC.load_seal_records(hist_root) if recs is None else recs
    if not os.path.isdir(hist_root):
        return []
    return sorted(v for v in os.listdir(hist_root) if not v.startswith("_") and os.path.isdir(os.path.join(hist_root, v)) and f"{v}/.commit" not in recs)


def pending_vintages(hist_root):
    return staged_vintages(hist_root) + uncommitted_vintages(hist_root)


def vintage_meta(hist_root, v, seals):
    p = os.path.join(hist_root, v, "meta.json")
    FC.check_sealed(hist_root, p, seals)
    return json.load(open(p, encoding="utf-8"))


def load_history(hist_root, path, seals):
    got = FC.check_sealed(hist_root, path, seals)
    h = json.load(open(path, encoding="utf-8"))
    m = vintage_meta(hist_root, os.path.basename(os.path.dirname(path)), seals)
    if m["files"].get(h["symbol"]) != got:
        raise B.Stop(f"{rel(path, hist_root)} is not the file its vintage meta lists")
    return h


def history_vintages(hist_root, recs=None):
    recs = FC.load_seal_records(hist_root) if recs is None else recs
    return sorted(v for v in os.listdir(hist_root) if not v.startswith("_") and os.path.isdir(os.path.join(hist_root, v)) and f"{v}/.commit" in recs) if os.path.isdir(hist_root) else []


def history_files(hist_root, sym, recs=None):
    return [os.path.join(hist_root, v, f"{sym}.json") for v in history_vintages(hist_root, recs) if os.path.exists(os.path.join(hist_root, v, f"{sym}.json"))]


def verify_chain(hist_root, sym, seals=None):
    seals = FC.load_seals(hist_root) if seals is None else seals
    for v in uncommitted_vintages(hist_root, seals):
        for n in sorted(os.listdir(os.path.join(hist_root, v))):
            FC.check_sealed(hist_root, os.path.join(hist_root, v, n), seals)
    files = history_files(hist_root, sym, seals)
    if not files:
        raise B.Stop(f"no history vintage for {sym}")
    path = files[-1]
    h = load_history(hist_root, path, seals)
    chain, seen = [], set()
    while True:
        chain.append(rel(path, hist_root))
        seen.add(rel(path, hist_root))
        if h["kind"] == "base":
            if h["parent"] is not None:
                raise B.Stop(f"{sym} base vintage has a parent")
            break
        ppath = os.path.join(hist_root, h["parent"]["path"])
        p = load_history(hist_root, ppath, seals)
        if sha_file(ppath) != h["parent"]["sha256"]:
            raise B.Stop(f"{sym} parent hash differs at {chain[-1]}")
        if not p["cutoff_day"] < h["cutoff_day"]:
            raise B.Stop(f"{sym} cutoff does not advance at {chain[-1]}")
        if h["kind"] == "appended" and (h["days"][:len(p["days"])] != p["days"] or h["close"][:len(p["close"])] != p["close"]):
            raise B.Stop(f"{sym} appended vintage does not keep its parent as a prefix at {chain[-1]}")
        if h["kind"] == "rebuilt" and [d for d in h["days"] if d <= p["cutoff_day"]] != p["days"]:
            raise B.Stop(f"{sym} rebuilt vintage changes the parent's calendar at {chain[-1]}")
        path, h = ppath, p
    stray = [rel(f, hist_root) for f in files if rel(f, hist_root) not in seen]
    if stray:
        raise B.Stop(f"{sym} history files outside the chain: {stray[:5]}")
    return chain


def latest_history(hist_root, sym, seals=None):
    seals = FC.load_seals(hist_root) if seals is None else seals
    chain = verify_chain(hist_root, sym, seals)
    path = os.path.join(hist_root, chain[0])
    return path, load_history(hist_root, path, seals)


def stored_for_check(hist_root, syms=SYMS):
    seals = FC.load_seals(hist_root)
    out = {}
    for sym in syms:
        _, h = latest_history(hist_root, sym, seals)
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


def build_extension(hist_root, adj_vdir, cutoff_day, sessions):
    adj_vdir = os.path.normpath(adj_vdir)
    adj_root = os.path.dirname(adj_vdir)
    adj_seals = FC.load_seals(adj_root)
    hist_seals = FC.load_seals(hist_root)
    vintage = os.path.basename(adj_vdir)
    meta_path = os.path.join(adj_vdir, "meta.json")
    FC.check_sealed(adj_root, meta_path, adj_seals)
    adj_meta = json.load(open(meta_path, encoding="utf-8"))
    files, result = {}, {}
    for sym in adj_meta["result"]:
        check_path = os.path.join(adj_vdir, f"check_{sym}.json")
        check_sha = FC.check_sealed(adj_root, check_path, adj_seals)
        check = json.load(open(check_path, encoding="utf-8"))
        ppath, prev = latest_history(hist_root, sym, hist_seals)
        prev_last = prev["days"][-1]
        reasons = []
        if check["cutoff_day"] != cutoff_day:
            reasons.append(f"check cutoff {check['cutoff_day']} differs from {cutoff_day}")
        if check.get("stored_last_day") != prev_last:
            reasons.append(f"the check compared against history ending {check.get('stored_last_day')}, latest history ends {prev_last}")
        source = dict(adjusted_vintage=vintage, check_sha256=check_sha, adjusted_meta_sha256=sha_file(meta_path), check_status=check["status"])
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
            source["rebuilt_sha256"] = FC.check_sealed(adj_root, rb_path, adj_seals)
            rb = json.load(open(rb_path, encoding="utf-8"))
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
            result[sym] = dict(status="held", reasons=reasons, latest_history=rel(ppath, hist_root))
            continue
        files[sym] = dict(symbol=sym, vintage=vintage, kind=kind, cutoff_day=cutoff_day, origin=prev["origin"], days=days, close=close, parent=dict(path=rel(ppath, hist_root), sha256=sha_file(ppath)), source=source)
        result[sym] = dict(status=kind, n_days=len(days), new_days=want)
    times = [r.get("received_at") or r.get("failed_at") for r in adj_meta.get("requests", [])]
    out = {f"{sym}.json": dump(payload) for sym, payload in files.items()}
    out["meta.json"] = dump(dict(vintage=vintage, cutoff_day=cutoff_day, result=result, files={sym: sha_bytes(out[f"{sym}.json"]) for sym in files}, adjusted_received_at=max((t for t in times if t), default=None)))
    return vintage, out, result


def extend_history(hist_root, adj_vdir, cutoff_day, sessions, clock=None, crash=None):
    pend = pending_vintages(hist_root)
    if pend:
        raise B.Stop(f"history vintages {pend} are pending (staged or without a commit record); run recover_history first")
    vintage, files, result = build_extension(hist_root, adj_vdir, cutoff_day, sessions)
    return commit_vintage(hist_root, vintage, files, clock=clock, crash=crash), result


def recover_history(hist_root, adj_root, sessions, clock=None, base_loader=None, origins=ORIGINS):
    out = []
    recs = FC.load_seal_records(hist_root)
    seals = {k: r["sha256"] for k, r in recs.items()}
    for v in uncommitted_vintages(hist_root, recs):
        vdir = os.path.join(hist_root, v)
        names = sorted(os.listdir(vdir))
        if "meta.json" not in names:
            raise B.Stop(f"{v} has no meta.json; manual action needed")
        for n in names:
            FC.check_sealed(hist_root, os.path.join(vdir, n), seals)
        m = json.load(open(os.path.join(vdir, "meta.json"), encoding="utf-8"))
        if set(names) != {f"{sym}.json" for sym in m["files"]} | {"meta.json"} or any(sha_file(os.path.join(vdir, f"{sym}.json")) != h for sym, h in m["files"].items()):
            raise B.Stop(f"{v} files do not match its meta; manual action needed")
        FC.append_seal(hist_root, f"{v}/.commit", sha_file(os.path.join(vdir, "meta.json")), 0, stamp(clock))
        out.append(dict(vintage=v, status="committed", how="final directory found without its commit record"))
    for v in staged_vintages(hist_root):
        if v == BASE_VINTAGE:
            base, meta = (base_loader or base_history)()
            commit_vintage(hist_root, v, base_files(base, meta, origins), clock=clock)
            out.append(dict(vintage=v, status="committed", kind="base"))
            continue
        adj_vdir = os.path.join(adj_root, v)
        seals = FC.load_seals(adj_root)
        mp = os.path.join(adj_vdir, "meta.json")
        FC.check_sealed(adj_root, mp, seals)
        m = json.load(open(mp, encoding="utf-8"))
        cutoffs = set()
        for sym in m["result"]:
            cp = os.path.join(adj_vdir, f"check_{sym}.json")
            FC.check_sealed(adj_root, cp, seals)
            cutoffs.add(json.load(open(cp, encoding="utf-8"))["cutoff_day"])
        if len(cutoffs) != 1:
            raise B.Stop(f"adjusted vintage {v} has cutoffs {sorted(cutoffs)}; manual action needed")
        cutoff = cutoffs.pop()
        _, files, result = build_extension(hist_root, adj_vdir, cutoff, sessions)
        commit_vintage(hist_root, v, files, clock=clock)
        out.append(dict(vintage=v, status="committed", cutoff_day=cutoff, result=result))
    return out


def minute_vintages(min_root, day, sym):
    if not os.path.isdir(min_root):
        return []
    return [os.path.join(min_root, v, f"{day}_{sym}.json") for v in sorted(os.listdir(min_root)) if os.path.exists(os.path.join(min_root, v, f"{day}_{sym}.json"))]


def split_minute_vintages(min_root, day, sym, recs):
    done, cut = [], []
    for p in minute_vintages(min_root, day, sym):
        v = os.path.basename(os.path.dirname(p))
        if not os.path.exists(os.path.join(min_root, v, "meta.json")):
            cut.append(p)
        elif f"{v}/meta.json" not in recs:
            raise B.Stop(f"no seal for {v}/meta.json although the file exists; manual action needed")
        else:
            done.append(p)
    return done, cut


def minute_entry(min_root, p, d, sym, seals, recs):
    got = FC.check_sealed(min_root, p, seals)
    rec = json.load(open(p, encoding="utf-8"))
    v = os.path.basename(os.path.dirname(p))
    mpath = os.path.join(min_root, v, "meta.json")
    FC.check_sealed(min_root, mpath, seals)
    meta = json.load(open(mpath, encoding="utf-8"))
    times = [r.get("received_at") or r.get("failed_at") for r in meta["requests"]]
    reasons = []
    if rec["day"] != d or rec["symbol"] != sym:
        reasons.append(f"{sym}: file content is for {rec['symbol']} {rec['day']}")
    if rec["summary"]["held"]:
        reasons.append(f"{sym}: vintage {rec['vintage']} held: {rec['summary']['hold_reasons']}")
    return dict(day=d, symbol=sym, vintage=rec["vintage"], path=rel(p, min_root), sha256=got, rows_sha256=sha_bytes(dump(rec["rows"])), n=len(rec["rows"]), cls=rec["summary"]["cls"], received_at=max(t for t in times if t), available_at=recs[f"{v}/meta.json"]["sealed_at"]), reasons


def history_entry(hist_root, v, sym, seals, metas, recs):
    m = metas[v]
    st = m["result"].get(sym, {})
    times = dict(received_at=m.get("adjusted_received_at"), available_at=recs[f"{v}/.commit"]["sealed_at"])
    if st.get("status") in ("appended", "rebuilt", "base"):
        p = os.path.join(hist_root, v, f"{sym}.json")
        h = load_history(hist_root, p, seals)
        return dict(held=False, vintage=v, path=rel(p, hist_root), sha256=sha_file(p), kind=h["kind"], cutoff_day=h["cutoff_day"], **times)
    return dict(held=True, vintage=v, cutoff_day=m.get("cutoff_day"), reason=st.get("reasons") or "no result for this symbol", **times)


def observation_input(kind, through_day, sessions, min_root, hist_root, out_root, built_at, base_min=BASE_MIN, base_last=BASE_LAST_DAY, syms=SYMS, clock=None):
    if kind not in ("as_received", "revised"):
        raise ValueError(kind)
    min_recs, hist_recs = FC.load_seal_records(min_root), FC.load_seal_records(hist_root)
    min_seals, hist_seals = {k: r["sha256"] for k, r in min_recs.items()}, {k: r["sha256"] for k, r in hist_recs.items()}
    for sym in syms:
        verify_chain(hist_root, sym, hist_seals)
    metas = {v: vintage_meta(hist_root, v, hist_seals) for v in history_vintages(hist_root, hist_recs)}
    forward, held, per_day, ignored, complete_through = [], [], [], [], base_last
    for d in [x for x in sessions if base_last < x <= through_day]:
        recs, reasons = [], []
        for sym in syms:
            done, cut = split_minute_vintages(min_root, d, sym, min_recs)
            ignored += [rel(c, min_root) for c in cut]
            if not done:
                reasons.append(f"{sym}: no complete minute vintage")
                continue
            e, why = minute_entry(min_root, done[0] if kind == "as_received" else done[-1], d, sym, min_seals, min_recs)
            e["n_vintages"] = len(done)
            recs.append(e)
            reasons += why
        if reasons:
            held.append(dict(day=d, reasons=reasons))
            break
        forward += recs
        complete_through = d
        if kind == "as_received":
            made = [v for v in sorted(metas) if metas[v].get("cutoff_day") == d]
            adj = {sym: (history_entry(hist_root, made[0], sym, hist_seals, metas, hist_recs) if made else dict(held=True, reason=f"no history job ran with cutoff {d}")) for sym in syms}
            per_day.append(dict(day=d, decision_at_next_session=next_session_open(d, sessions), minute={e["symbol"]: dict(vintage=e["vintage"], received_at=e["received_at"], available_at=e["available_at"]) for e in recs}, adjusted=adj))
    adjusted = {}
    for sym in syms:
        if kind == "as_received":
            made = [v for v in sorted(metas) if metas[v].get("cutoff_day") == complete_through]
            e = history_entry(hist_root, made[0], sym, hist_seals, metas, hist_recs) if made else dict(held=True, reason=f"no history job ran with cutoff {complete_through}")
        else:
            cands = [f for f in history_files(hist_root, sym) if load_history(hist_root, f, hist_seals)["cutoff_day"] >= complete_through]
            e = history_entry(hist_root, os.path.basename(os.path.dirname(cands[-1])), sym, hist_seals, metas, hist_recs) if cands else dict(held=True, reason=f"no history vintage reaches {complete_through}")
        if e["held"]:
            before = [h["cutoff_day"] for h in (load_history(hist_root, f, hist_seals) for f in history_files(hist_root, sym)) if h["cutoff_day"] < complete_through]
            e["latest_cutoff_before"] = before[-1] if before else None
        else:
            e["truncated_to"] = complete_through
        adjusted[sym] = e
    man = dict(version="forward-observation-input-v3", kind=kind, allowed_use=AS_RECEIVED_USE if kind == "as_received" else REVISED_USE, decision_rule=DECISION_RULE if kind == "as_received" else None, requested_through=through_day, complete_through=complete_through, built_at=built_at, base_minute={sym: dict(path=os.path.basename(base_min[sym][0]), sha256=base_min[sym][1], last_day=base_last) for sym in syms}, forward_minute=forward, held=held, incomplete_minute_vintages_ignored=sorted(set(ignored)), pending_history_vintages=pending_vintages(hist_root), adjusted=adjusted, per_day=per_day if kind == "as_received" else None, collector_sha256=sha_file(FC.__file__), history_code_sha256=sha_file(os.path.abspath(__file__)))
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, f"{kind}_{complete_through}_{built_at}.json")
    FC.write_sealed(out_root, path, dump(man), stamp(clock))
    return path, man


def value_problems(df):
    import numpy as np
    o, h, l, c, v = (df[k].astype(float).to_numpy() for k in ("open", "high", "low", "close", "volume"))
    fin = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(v)
    with np.errstate(invalid="ignore"):
        bad = ~fin | (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (v < 0) | (l > o) | (l > c) | (o > h) | (c > h)
    return df["ts"][bad].tolist()


def load_adjusted(hist_root, a, seals, upto):
    p = os.path.join(hist_root, a["path"])
    if FC.check_sealed(hist_root, p, seals) != a["sha256"]:
        raise B.Stop(f"history vintage hash differs from the manifest: {a['path']}")
    h = load_history(hist_root, p, seals)
    k = sum(1 for d in h["days"] if d <= upto)
    return h["days"][:k], h["close"][:k]


def materialize(manifest_path, min_root, hist_root, base_min=BASE_MIN, cumulative=False, decision_at=None):
    import pandas as pd
    FC.check_sealed(os.path.dirname(os.path.abspath(manifest_path)), manifest_path)
    m = json.load(open(manifest_path, encoding="utf-8"))
    if cumulative and m["kind"] != "as_received":
        raise B.Stop(f"{m['kind']} input: {REVISED_USE}")
    min_seals, hist_seals = FC.load_seals(min_root), FC.load_seals(hist_root)
    for sym in m["adjusted"]:
        verify_chain(hist_root, sym, hist_seals)
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
            if FC.check_sealed(min_root, p, min_seals) != f["sha256"]:
                raise B.Stop(f"minute vintage hash differs from the manifest: {f['path']}")
            rec = json.load(open(p, encoding="utf-8"))
            if sha_bytes(dump(rec["rows"])) != f["rows_sha256"]:
                raise B.Stop(f"minute rows hash differs: {f['path']}")
            rows.append(pd.DataFrame(rec["rows"], columns=["ts", "open", "high", "low", "close", "volume"]))
        out = pd.concat(rows, ignore_index=True)
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        if not out["ts"].is_monotonic_increasing or out["ts"].duplicated().any():
            raise B.Stop(f"{sym} minute rows are not strictly increasing")
        bad = value_problems(out)
        if bad:
            raise B.Stop(f"{sym} has {len(bad)} minute rows with invalid values, first {bad[:3]}")
        minute[sym] = out
    adjusted = {sym: (None if a["held"] else load_adjusted(hist_root, a, hist_seals, a["truncated_to"])) for sym, a in m["adjusted"].items()}
    res = dict(manifest=m, minute=minute, adjusted=adjusted)
    if cumulative:
        late, per_adj, minute_ok, used = [], {}, {}, {}
        for e in m["per_day"]:
            d = e["day"]
            dec = (decision_at or {}).get(d, e["decision_at_next_session"])
            used[d] = dec
            on_time = lambda t: dec is not None and t is not None and datetime.fromisoformat(t) <= datetime.fromisoformat(dec)
            minute_ok[d] = all(on_time(x["available_at"]) for x in e["minute"].values())
            late += [dict(day=d, input="minute", symbol=sym, available_at=x["available_at"], decision_at=dec) for sym, x in e["minute"].items() if not on_time(x["available_at"])]
            per_adj[d] = {}
            for sym, a in e["adjusted"].items():
                if a["held"]:
                    per_adj[d][sym] = None
                elif not on_time(a["available_at"]):
                    per_adj[d][sym] = None
                    late.append(dict(day=d, input="adjusted", symbol=sym, available_at=a["available_at"], decision_at=dec))
                else:
                    per_adj[d][sym] = load_adjusted(hist_root, a, hist_seals, d)
        res.update(per_day_adjusted=per_adj, per_day_minute_on_time=minute_ok, late_inputs=late, decision_at_used=used)
    return res


def to_5m(df_1m, tmp_dir, name):
    import kr_to_ft as K
    p = os.path.join(tmp_dir, f"{name}.csv")
    df_1m.to_csv(p, index=False)
    return K.resample(K.load_1m(p), "5m")


class FakeWorld:
    def __init__(self, tmp):
        import pandas as pd
        self.tmp = tmp
        self.sessions = declared_sessions()
        self.origins = {"069500": "2026-01-01", "229200": "2026-01-01"}
        self.px0 = {"069500": 80000.0, "229200": 11000.0}
        base_days = [x.isoformat() for x in (date(2026, 1, 1) + timedelta(days=i) for i in range(400)) if x.weekday() < 5 and x.isoformat() <= "2026-11-05"]
        self.adj = {sym: {d: self.px0[sym] + 5 * (i % 13) for i, d in enumerate(base_days)} for sym in SYMS}
        self.adj_last, self.adj_empty = "2026-11-05", set()
        self.min_override, self.min_field, self.drop = {}, {}, set()
        self.min_days = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]
        self.calls = []
        src = {sym: dict(days=base_days + ["2026-11-06"], C=[self.adj[sym][d] for d in base_days] + [1.0], fetched="2026-11-06T11:54:53") for sym in SYMS}
        self.base_adj_path = os.path.join(tmp, "base_adj.json")
        open(self.base_adj_path, "w", encoding="utf-8").write(json.dumps(src))
        self.base, self.bmeta = base_history(self.base_adj_path, sha_file(self.base_adj_path), origins=self.origins)
        self.base_min = {}
        for sym in SYMS:
            rows = [b for b in self.all_bars(sym) if b["timestamp"][:10] <= "2026-11-05"]
            p = os.path.join(tmp, f"base_{sym}.csv")
            pd.DataFrame(dict(ts=[b["timestamp"][:19] for b in rows], open=[int(float(b["openPrice"])) for b in rows], high=[int(float(b["highPrice"])) for b in rows], low=[int(float(b["lowPrice"])) for b in rows], close=[int(float(b["closePrice"])) for b in rows], volume=[int(b["volume"]) for b in rows])).to_csv(p, index=False)
            self.base_min[sym] = (p, sha_file(p))
        self.roots = {k: os.path.join(tmp, k) for k in ("min", "adj", "hist", "obs")}
        write_base_vintage(self.roots["hist"], self.base, self.bmeta, origins=self.origins, clock=lambda: datetime.fromisoformat("2026-11-08T11:59:00+09:00"))

    def bar(self, sym, day, hm, i):
        p = self.px0[sym] + 5 * (i % 7)
        ov = self.min_override.get((sym, day, hm))
        close = ov if ov is not None else p
        b = dict(timestamp=f"{day}T{hm}:00+09:00", openPrice=str(p), highPrice=str(max(p, close) + 5), lowPrice=str(min(p, close) - 5), closePrice=str(close), volume="0" if hm in FC.session(day)[2] else str(1000 + i))
        b.update(self.min_field.get((sym, day, hm), {}))
        return b

    def all_bars(self, sym):
        out = []
        for d in self.min_days:
            first, last = FC.session(d)[:2]
            out += [self.bar(sym, d, hm, i) for i, hm in enumerate(B.minutes_expected(first, last)) if (sym, d, hm) not in self.drop]
        return out

    def fetch(self, params):
        self.calls.append(dict(params))
        sym = params["symbol"]
        if params["interval"] == "1m":
            bs = [b for b in self.all_bars(sym) if b["timestamp"] < params["before"]]
            return 200, json.dumps(dict(candles=bs[-200:])).encode("utf-8"), {}
        if sym in self.adj_empty:
            return 200, json.dumps(dict(candles=[])).encode("utf-8"), {}
        days = sorted(d for d in self.adj[sym] if d <= self.adj_last)
        if "before" in params:
            days = [d for d in days if d < params["before"][:10]]
        days = days[-200:]
        return 200, json.dumps(dict(candles=[dict(timestamp=f"{d}T00:00:00+09:00", closePrice=str(self.adj[sym][d])) for d in days])).encode("utf-8"), {}

    def collector(self, at_iso):
        return FC.Collector(fetch=self.fetch, clock=lambda: datetime.fromisoformat(at_iso), sleep=lambda s: None, out_min=self.roots["min"], out_adj=self.roots["adj"])

    def add_adj(self, day, step):
        for sym in SYMS:
            self.adj[sym][day] = self.px0[sym] + step
        self.adj_last = day

    def routine(self, at_iso, day, adj_at=None, crash=None):
        assert latest_completed_session(datetime.fromisoformat(at_iso), self.sessions) == day
        v1, o1 = self.collector(at_iso).minute_day(day)
        adj_at = adj_at or (at_iso[:17] + "30" + at_iso[19:])
        v2, o2 = self.collector(adj_at).adjusted_checked(stored_for_check(self.roots["hist"]), day, self.origins)
        self.last_adj_vdir = v2
        v3, o3 = extend_history(self.roots["hist"], v2, day, self.sessions, clock=lambda: datetime.fromisoformat(adj_at) + timedelta(seconds=1), crash=crash)
        return o1, o2, o3, v2

    def refetch(self, at_iso, day):
        return self.collector(at_iso).minute_day(day)

    def obs(self, kind, through, built_at, syms=SYMS):
        return observation_input(kind, through, self.sessions, self.roots["min"], self.roots["hist"], self.roots["obs"], built_at, base_min=self.base_min, syms=syms)

    def mat(self, path, cumulative=False, decision_at=None):
        return materialize(path, self.roots["min"], self.roots["hist"], base_min=self.base_min, cumulative=cumulative, decision_at=decision_at)

    def recover(self, at_iso):
        return recover_history(self.roots["hist"], self.roots["adj"], self.sessions, clock=lambda: datetime.fromisoformat(at_iso), base_loader=lambda: (self.base, self.bmeta), origins=self.origins)

    def standard(self):
        self.add_adj("2026-11-06", 7)
        self.routine(SUPPLEMENT_AT, "2026-11-06")
        self.refetch("2026-11-09T08:30:00+09:00", "2026-11-06")
        self.add_adj("2026-11-09", 9)
        self.routine("2026-11-09T16:10:00+09:00", "2026-11-09")
        self.min_override[("069500", "2026-11-09", "15:30")] = self.px0["069500"] + 100
        self.refetch("2026-11-10T08:30:00+09:00", "2026-11-09")


def rewrite_json(path, fn):
    d = json.load(open(path, encoding="utf-8"))
    fn(d)
    os.chmod(path, 0o666)
    open(path, "wb").write(dump(d))


def expect_stop(fn, text):
    try:
        fn()
    except B.Stop as e:
        assert text in str(e), (text, str(e))
        return str(e)
    raise AssertionError(f"expected a stop containing: {text}")


def counterexamples():
    cases = []

    def case(name):
        def deco(f):
            cases.append((name, f))
            return f
        return deco

    @case("tip close changed after sealing")
    def _(w):
        tip = history_files(w.roots["hist"], "069500")[-1]
        rewrite_json(tip, lambda d: d["close"].__setitem__(-1, d["close"][-1] + 1))
        n = len(w.calls)
        expect_stop(lambda: latest_history(w.roots["hist"], "069500"), "sealed hash differs")
        expect_stop(lambda: verify_chain(w.roots["hist"], "069500"), "sealed hash differs")
        expect_stop(lambda: w.obs("as_received", "2026-11-09", "b1"), "sealed hash differs")
        w.add_adj("2026-11-10", 11)
        expect_stop(lambda: w.collector("2026-11-10T16:10:30+09:00").adjusted_checked(stored_for_check(w.roots["hist"]), "2026-11-10", w.origins), "sealed hash differs")
        assert len(w.calls) == n

    @case("middle vintage changed after sealing")
    def _(w):
        mid = history_files(w.roots["hist"], "229200")[1]
        rewrite_json(mid, lambda d: d["close"].__setitem__(0, 1.0))
        expect_stop(lambda: verify_chain(w.roots["hist"], "229200"), "sealed hash differs")
        expect_stop(lambda: w.obs("revised", "2026-11-09", "b2"), "sealed hash differs")

    @case("base vintage changed after sealing")
    def _(w):
        rewrite_json(os.path.join(w.roots["hist"], BASE_VINTAGE, "069500.json"), lambda d: d["close"].__setitem__(5, 2.0))
        expect_stop(lambda: stored_for_check(w.roots["hist"]), "sealed hash differs")
        expect_stop(lambda: w.obs("as_received", "2026-11-06", "b3"), "sealed hash differs")

    @case("vintage meta and file changed together, seal log unchanged")
    def _(w):
        tip = history_files(w.roots["hist"], "229200")[-1]
        rewrite_json(tip, lambda d: d["close"].__setitem__(-1, 5.0))
        new_sha = sha_file(tip)
        rewrite_json(os.path.join(os.path.dirname(tip), "meta.json"), lambda d: d["files"].__setitem__("229200", new_sha))
        expect_stop(lambda: verify_chain(w.roots["hist"], "229200"), "sealed hash differs")

    @case("minute file changed before the manifest is written")
    def _(w):
        p = minute_vintages(w.roots["min"], "2026-11-09", "069500")[0]
        rewrite_json(p, lambda d: d["rows"][-1].__setitem__(4, "1"))
        expect_stop(lambda: w.obs("as_received", "2026-11-09", "b4"), "sealed hash differs")
        w.obs("revised", "2026-11-09", "b5")

    @case("manifest changed after it is written")
    def _(w):
        p, _ = w.obs("revised", "2026-11-09", "b6")
        rewrite_json(p, lambda d: d.__setitem__("complete_through", "2026-11-10"))
        expect_stop(lambda: w.mat(p), "sealed hash differs")

    @case("unsealed vintage copied in")
    def _(w):
        src = os.path.dirname(minute_vintages(w.roots["min"], "2026-11-09", "069500")[-1])
        shutil.copytree(src, os.path.join(w.roots["min"], "20261111T000000"))
        expect_stop(lambda: w.obs("revised", "2026-11-09", "b7"), "no seal")
        hsrc = os.path.dirname(history_files(w.roots["hist"], "069500")[-1])
        shutil.copytree(hsrc, os.path.join(w.roots["hist"], "20261111T000000"))
        expect_stop(lambda: verify_chain(w.roots["hist"], "069500"), "no seal")

    @case("invalid bar values with every minute present")
    def _(w):
        w.add_adj("2026-11-10", 11)
        w.min_field[("229200", "2026-11-10", "10:00")] = dict(highPrice="1", lowPrice="2")
        w.min_field[("069500", "2026-11-10", "11:00")] = dict(closePrice="NaN")
        o1, _, _, _ = w.routine("2026-11-10T16:10:00+09:00", "2026-11-10")
        assert o1["229200"]["held"] and "invalid bar values" in o1["229200"]["hold_reasons"] and o1["069500"]["held"], o1
        _, man = w.obs("as_received", "2026-11-10", "b8")
        assert man["complete_through"] == "2026-11-09" and "invalid bar values" in json.dumps(man["held"])
        w.min_field.clear()
        w.min_field[("069500", "2026-11-10", "11:00")] = dict(volume="-3")
        w.refetch("2026-11-11T08:30:00+09:00", "2026-11-10")
        _, man = w.obs("revised", "2026-11-10", "b9")
        assert man["complete_through"] == "2026-11-09" and "invalid bar values" in json.dumps(man["held"])

    @case("adjusted held then recovered; per-day record and cumulative use")
    def _(w):
        w.add_adj("2026-11-10", 11)
        w.adj_empty = {"069500"}
        _, o2, o3, _ = w.routine("2026-11-10T16:10:00+09:00", "2026-11-10")
        assert o2["069500"] == "hold_invalid_page" and o3["069500"]["status"] == "held" and o3["229200"]["status"] == "appended"
        w.adj_empty = set()
        w.refetch("2026-11-11T08:30:00+09:00", "2026-11-10")
        w.add_adj("2026-11-11", 13)
        _, _, o3, _ = w.routine("2026-11-11T16:10:00+09:00", "2026-11-11")
        assert o3["069500"]["new_days"] == ["2026-11-10", "2026-11-11"] and o3["229200"]["new_days"] == ["2026-11-11"], o3
        ap, am = w.obs("as_received", "2026-11-11", "b10")
        pdd = {e["day"]: e for e in am["per_day"]}
        assert [e["day"] for e in am["per_day"]] == ["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"]
        assert pdd["2026-11-10"]["adjusted"]["069500"]["held"] and not pdd["2026-11-10"]["adjusted"]["229200"]["held"] and not pdd["2026-11-11"]["adjusted"]["069500"]["held"]
        A = w.mat(ap, cumulative=True)
        assert A["per_day_adjusted"]["2026-11-10"]["069500"] is None and A["per_day_adjusted"]["2026-11-10"]["229200"][0][-1] == "2026-11-10"
        assert A["per_day_adjusted"]["2026-11-11"]["069500"][0][-2:] == ["2026-11-10", "2026-11-11"] and A["adjusted"]["069500"][0][-1] == "2026-11-11"
        assert A["per_day_adjusted"]["2026-11-09"]["069500"][0][-1] == "2026-11-09"
        rp, rm = w.obs("revised", "2026-11-11", "b11")
        assert rm["per_day"] is None and "must not drive a cumulative replay" in rm["allowed_use"]
        expect_stop(lambda: w.mat(rp, cumulative=True), "must not drive a cumulative replay")
        w.mat(rp)
        _, a10 = w.obs("as_received", "2026-11-10", "b12")
        assert a10["adjusted"]["069500"]["held"] and a10["adjusted"]["069500"]["latest_cutoff_before"] == "2026-11-09", a10["adjusted"]

    @case("invalid value already in the frozen base minute file")
    def _(w):
        import pandas as pd
        path = w.base_min["229200"][0]
        df = pd.read_csv(path, dtype={"ts": str})
        df.loc[100, "volume"] = -1
        df.to_csv(path, index=False)
        w.base_min["229200"] = (path, sha_file(path))
        p, _ = w.obs("as_received", "2026-11-09", "b13")
        expect_stop(lambda: w.mat(p), "invalid values")

    @case("base parent changed after a normal manifest; materialize")
    def _(w):
        ap, _ = w.obs("as_received", "2026-11-09", "c1")
        rp, _ = w.obs("revised", "2026-11-09", "c2")
        w.mat(ap, cumulative=True)
        rewrite_json(os.path.join(w.roots["hist"], BASE_VINTAGE, "069500.json"), lambda d: d["close"].__setitem__(3, 9.0))
        expect_stop(lambda: w.mat(ap, cumulative=True), "sealed hash differs")
        expect_stop(lambda: w.mat(rp), "sealed hash differs")

    def commit_cut(point):
        def f(w):
            w.add_adj("2026-11-10", 11)
            try:
                w.routine("2026-11-10T16:10:00+09:00", "2026-11-10", crash=point)
                raise AssertionError("simulated crash expected")
            except SimulatedCrash:
                pass
            v = os.path.basename(w.last_adj_vdir)
            assert pending_vintages(w.roots["hist"]) == [v] and (point == "after_rename") == os.path.exists(os.path.join(w.roots["hist"], v))
            st = stored_for_check(w.roots["hist"])
            assert st["069500"][0][-1] == "2026-11-09" and verify_chain(w.roots["hist"], "229200")[0].startswith("20261109T161030")
            expect_stop(lambda: extend_history(w.roots["hist"], w.last_adj_vdir, "2026-11-10", w.sessions), "pending")
            _, man = w.obs("as_received", "2026-11-10", f"cut_{point}")
            assert man["pending_history_vintages"] == [v] and all(man["per_day"][-1]["adjusted"][s]["held"] for s in SYMS)
            n = len(w.calls)
            rec = w.recover("2026-11-10T16:20:00+09:00")
            assert rec[0]["vintage"] == v and (point == "after_rename" or all(rec[0]["result"][s]["status"] == "appended" for s in SYMS)) and pending_vintages(w.roots["hist"]) == [] and len(w.calls) == n, rec
            for sym in SYMS:
                assert verify_chain(w.roots["hist"], sym)[0] == f"{v}/{sym}.json"
            w.add_adj("2026-11-11", 13)
            _, _, o3, _ = w.routine("2026-11-11T16:10:00+09:00", "2026-11-11")
            assert all(o3[s]["status"] == "appended" and o3[s]["new_days"] == ["2026-11-11"] for s in SYMS), o3
            _, man = w.obs("as_received", "2026-11-11", f"after_{point}")
            pdd = {e["day"]: e for e in man["per_day"]}
            assert not pdd["2026-11-10"]["adjusted"]["069500"]["held"] and pdd["2026-11-10"]["adjusted"]["069500"]["available_at"].startswith("2026-11-10T16:20"), pdd["2026-11-10"]
        return f

    for point in ("after_first_file", "after_first_seal", "before_rename", "after_rename"):
        case(f"history commit cut {point.replace('_', ' ')}; explicit recovery, then the next day")(commit_cut(point))

    @case("staged file changed before recovery")
    def _(w):
        w.add_adj("2026-11-10", 11)
        try:
            w.routine("2026-11-10T16:10:00+09:00", "2026-11-10", crash="before_rename")
            raise AssertionError("simulated crash expected")
        except SimulatedCrash:
            pass
        v = os.path.basename(w.last_adj_vdir)
        rewrite_json(os.path.join(w.roots["hist"], STAGING, v, "069500.json"), lambda d: d["close"].__setitem__(-1, 1.0))
        expect_stop(lambda: w.recover("2026-11-10T16:20:00+09:00"), "manual action")
        assert staged_vintages(w.roots["hist"]) == [v]

    @case("base commit cut; explicit recovery")
    def _(w):
        tmp2 = os.path.join(w.tmp, "fresh_hist")
        try:
            write_base_vintage(tmp2, w.base, w.bmeta, origins=w.origins, crash="after_first_seal")
            raise AssertionError("simulated crash expected")
        except SimulatedCrash:
            pass
        assert staged_vintages(tmp2) == [BASE_VINTAGE] and history_files(tmp2, "069500") == []
        out = recover_history(tmp2, w.roots["adj"], w.sessions, base_loader=lambda: (w.base, w.bmeta), origins=w.origins)
        assert out == [dict(vintage=BASE_VINTAGE, status="committed", kind="base")] and verify_chain(tmp2, "069500") == [f"{BASE_VINTAGE}/069500.json"]

    @case("minute vintage cut before its meta")
    def _(w):
        def broken(params):
            if params["symbol"] == "229200":
                raise B.Stop("http 500: fake")
            return w.fetch(params)
        c = FC.Collector(fetch=broken, clock=lambda: datetime.fromisoformat("2026-11-10T09:00:00+09:00"), sleep=lambda s: None, out_min=w.roots["min"], out_adj=w.roots["adj"])
        expect_stop(lambda: c.minute_day("2026-11-09"), "http 500")
        assert os.path.exists(os.path.join(w.roots["min"], "20261110T090000", "2026-11-09_069500.json")) and not os.path.exists(os.path.join(w.roots["min"], "20261110T090000", "meta.json"))
        _, man = w.obs("revised", "2026-11-09", "c3")
        rv = {(f["day"], f["symbol"]): f for f in man["forward_minute"]}
        assert rv[("2026-11-09", "069500")]["vintage"] == "20261110T083000" and man["incomplete_minute_vintages_ignored"] == ["20261110T090000/2026-11-09_069500.json"], man["incomplete_minute_vintages_ignored"]

    @case("late adjusted receipt and the decision-time rule")
    def _(w):
        w.add_adj("2026-11-10", 11)
        w.routine("2026-11-10T16:10:00+09:00", "2026-11-10", adj_at="2026-11-11T10:00:00+09:00")
        w.refetch("2026-11-11T10:05:00+09:00", "2026-11-10")
        w.add_adj("2026-11-11", 13)
        w.routine("2026-11-11T16:10:00+09:00", "2026-11-11", adj_at="2026-11-11T21:00:00+09:00")
        ap, am = w.obs("as_received", "2026-11-11", "c4")
        pdd = {e["day"]: e for e in am["per_day"]}
        assert pdd["2026-11-10"]["adjusted"]["069500"]["received_at"].startswith("2026-11-11T10:00") and pdd["2026-11-10"]["adjusted"]["069500"]["available_at"].startswith("2026-11-11T10:00:01")
        assert pdd["2026-11-10"]["decision_at_next_session"] == "2026-11-11T09:00:00+09:00" and pdd["2026-11-11"]["decision_at_next_session"] == "2026-11-12T09:00:00+09:00"
        A = w.mat(ap, cumulative=True)
        assert A["per_day_adjusted"]["2026-11-10"]["069500"] is None and A["per_day_adjusted"]["2026-11-11"]["069500"][0][-1] == "2026-11-11", A["late_inputs"]
        assert sorted((x["day"], x["input"], x["symbol"]) for x in A["late_inputs"]) == [("2026-11-10", "adjusted", "069500"), ("2026-11-10", "adjusted", "229200")]
        B2 = w.mat(ap, cumulative=True, decision_at={"2026-11-10": "2026-11-11T12:00:00+09:00", "2026-11-11": "2026-11-11T16:40:00+09:00"})
        assert B2["per_day_adjusted"]["2026-11-10"]["069500"][0][-1] == "2026-11-10" and B2["per_day_adjusted"]["2026-11-11"]["229200"] is None
        assert sorted((x["day"], x["input"]) for x in B2["late_inputs"]) == [("2026-11-11", "adjusted"), ("2026-11-11", "adjusted")]

    results = []
    for name, f in cases:
        tmp = tempfile.mkdtemp(prefix="fwd_cx_")
        try:
            w = FakeWorld(tmp)
            w.standard()
            f(w)
            results.append(name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return results


def selftest():
    sessions = declared_sessions()
    assert sessions[0] == "2026-11-06" and "2026-11-07" not in sessions and "2026-12-25" not in sessions and "2026-12-31" not in sessions and "2027-01-01" not in sessions and "2027-01-04" in sessions and len(sessions) == 58
    t = lambda s: datetime.fromisoformat(s)
    for at, want in (("2026-11-08T12:00:00+09:00", "2026-11-06"), ("2026-11-09T16:09:59+09:00", "2026-11-06"), ("2026-11-09T16:10:00+09:00", "2026-11-09"), ("2026-11-19T17:09:00+09:00", "2026-11-18"), ("2026-11-19T17:10:00+09:00", "2026-11-19"), ("2026-12-25T20:00:00+09:00", "2026-12-24"), ("2027-01-04T08:30:00+09:00", "2026-12-30"), ("2027-01-04T16:10:00+09:00", "2027-01-04")):
        assert latest_completed_session(t(at), sessions) == want, (at, want)
    expect_stop(lambda: latest_completed_session(t("2027-02-01T09:00:00+09:00"), sessions), "past the declared schedule")
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

    tmp = tempfile.mkdtemp(prefix="fwd_hist_")
    try:
        w = FakeWorld(tmp)
        expect_stop(lambda: base_history(w.base_adj_path, "0" * 64, origins=w.origins), "differs")
        assert w.base["069500"]["days"][-1] == "2026-11-05" and w.bmeta["dropped_rows"]["069500"]["rows"] == [dict(day="2026-11-06", close=1.0)]
        try:
            write_base_vintage(w.roots["hist"], w.base, w.bmeta, origins=w.origins)
            raise AssertionError("base vintage must not be rewritten")
        except FileExistsError:
            pass
        w.add_adj("2026-11-06", 7)
        o1, o2, o3, _ = w.routine(SUPPLEMENT_AT, "2026-11-06")
        assert all(not v["held"] for v in o1.values()) and set(o2.values()) == {"unchanged"} and all(v["status"] == "appended" and v["new_days"] == ["2026-11-06"] for v in o3.values()), (o1, o2, o3)
        w.refetch("2026-11-09T08:30:00+09:00", "2026-11-06")
        expect_stop(lambda: w.collector("2026-11-09T16:09:00+09:00").minute_day("2026-11-09"), "too early")
        w.add_adj("2026-11-09", 9)
        o1, o2, o3, adj_v = w.routine("2026-11-09T16:10:00+09:00", "2026-11-09")
        assert set(o2.values()) == {"unchanged"} and all(v["status"] == "appended" and v["new_days"] == ["2026-11-09"] for v in o3.values())
        try:
            extend_history(w.roots["hist"], adj_v, "2026-11-09", sessions)
            raise AssertionError("a history vintage must not be rewritten")
        except FileExistsError:
            pass
        w.min_override[("069500", "2026-11-09", "15:30")] = w.px0["069500"] + 100
        w.refetch("2026-11-10T08:30:00+09:00", "2026-11-09")
        a_path, a_man = w.obs("as_received", "2026-11-09", "20261110T083100")
        r_path, r_man = w.obs("revised", "2026-11-09", "20261110T083101")
        assert a_man["complete_through"] == r_man["complete_through"] == "2026-11-09" and not a_man["held"] and not r_man["held"]
        av = {(f["day"], f["symbol"]): f for f in a_man["forward_minute"]}
        rv = {(f["day"], f["symbol"]): f for f in r_man["forward_minute"]}
        assert av[("2026-11-09", "069500")]["vintage"] == "20261109T161000" and rv[("2026-11-09", "069500")]["vintage"] == "20261110T083000"
        assert av[("2026-11-06", "069500")]["vintage"] == "20261108T120000" and rv[("2026-11-06", "069500")]["vintage"] == "20261109T083000" and av[("2026-11-06", "069500")]["rows_sha256"] == rv[("2026-11-06", "069500")]["rows_sha256"]
        assert av[("2026-11-09", "229200")]["rows_sha256"] == rv[("2026-11-09", "229200")]["rows_sha256"] and av[("2026-11-09", "069500")]["rows_sha256"] != rv[("2026-11-09", "069500")]["rows_sha256"]
        assert a_man["per_day"][-1]["adjusted"]["069500"]["vintage"] == a_man["adjusted"]["069500"]["vintage"] and a_man["per_day"][0]["minute"]["069500"]["available_at"].startswith("2026-11-08T12:00")
        A, R = w.mat(a_path), w.mat(r_path)
        assert len(A["minute"]["069500"]) == 4 * 390 and A["minute"]["069500"]["ts"].iloc[-1] == "2026-11-09T15:30:00"
        cols = ["open", "high", "low", "close", "volume"]
        diff = (A["minute"]["069500"][cols] != R["minute"]["069500"][cols]).any(axis=1)
        assert list(A["minute"]["069500"]["ts"][diff]) == ["2026-11-09T15:30:00"]
        assert A["minute"]["229200"].equals(R["minute"]["229200"]) and A["adjusted"] == R["adjusted"] and A["adjusted"]["069500"][0][-2:] == ["2026-11-06", "2026-11-09"]
        a5, r5 = to_5m(A["minute"]["069500"], tmp, "a"), to_5m(R["minute"]["069500"], tmp, "r")
        d5 = (a5[cols] != r5[cols]).any(axis=1)
        assert len(a5) == 4 * 78 and [str(x) for x in a5["date"][d5]] == ["2026-11-09 06:25:00+00:00"]
        w.add_adj("2026-11-10", 11)
        w.adj["229200"] = {d: (round(v * 0.99, 6) if d < "2026-11-10" else v) for d, v in w.adj["229200"].items()}
        _, o2, o3, _ = w.routine("2026-11-10T16:10:00+09:00", "2026-11-10")
        assert o2 == {"069500": "unchanged", "229200": "rebuilt_from_origin"} and o3["069500"]["status"] == "appended" and o3["229200"]["status"] == "rebuilt", (o2, o3)
        for sym in SYMS:
            assert verify_chain(w.roots["hist"], sym)[-1] == f"{BASE_VINTAGE}/{sym}.json"
        a2, _ = w.obs("as_received", "2026-11-09", "20261110T161100")
        r2, r2m = w.obs("revised", "2026-11-09", "20261110T161101")
        A2, R2 = w.mat(a2), w.mat(r2)
        assert r2m["adjusted"]["229200"]["kind"] == "rebuilt" and r2m["adjusted"]["229200"]["truncated_to"] == "2026-11-09"
        assert A2["adjusted"]["229200"][0] == R2["adjusted"]["229200"][0] and all(abs(r - round(a * 0.99, 6)) < 1e-9 for a, r in zip(A2["adjusted"]["229200"][1], R2["adjusted"]["229200"][1]))
        assert A2["adjusted"]["069500"] == R2["adjusted"]["069500"] == A["adjusted"]["069500"]
        w.drop.add(("229200", "2026-11-11", "15:30"))
        w.add_adj("2026-11-11", 13)
        o1, _, _, _ = w.routine("2026-11-11T16:10:00+09:00", "2026-11-11")
        assert o1["229200"]["held"] and not o1["069500"]["held"]
        _, a3 = w.obs("as_received", "2026-11-11", "20261111T161100")
        assert a3["complete_through"] == "2026-11-10" and a3["held"][0]["day"] == "2026-11-11" and "229200" in a3["held"][0]["reasons"][0] and a3["adjusted"]["229200"]["cutoff_day"] == "2026-11-10"
        sym_calls = []

        def slow_fetch(params):
            sym_calls.append(params)
            return 200, json.dumps(dict(candles=[dict(timestamp=f"2026-0{9 - len(sym_calls)}-01T00:00:00+09:00", closePrice="1")])).encode("utf-8"), {}
        c = FC.Collector(fetch=slow_fetch, clock=lambda: t("2026-11-12T16:10:00+09:00"), sleep=lambda s: None, out_adj=os.path.join(tmp, "cap"))
        c.open_vintage(os.path.join(tmp, "cap"))
        expect_stop(lambda: FC.full_adjusted_history(c, "229200", "2015-10-01", 3), "not reached within 3 pages")
        assert len(sym_calls) == 3
        n1, nd = sum(1 for x in w.calls if x["interval"] == "1m"), sum(1 for x in w.calls if x["interval"] == "1d")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    done = counterexamples()
    print(f"fixture ok: fake fetch only, temp dirs, no network; main flow calls 1m {n1}, 1d {nd}; {len(done)} independent counterexamples pass: " + "; ".join(done))


def base_check(out_path):
    import pandas as pd
    base, meta = base_history()
    sessions = declared_sessions()
    rec = dict(version="forward-base-check-v2", base_adjusted=dict(source=meta["source"], source_sha256=meta["source_sha256"], cut_after=meta["cut_after"], dropped_rows=meta["dropped_rows"], days={s: [base[s]["days"][0], base[s]["days"][-1], len(base[s]["days"])] for s in SYMS}, non_finite_or_non_positive_closes={s: sum(1 for v in base[s]["close"] if not (v == v and 0 < v < float("inf"))) for s in SYMS}), base_minute={}, collector_sha256=sha_file(FC.__file__), history_code_sha256=sha_file(os.path.abspath(__file__)), declared_schedule=dict(first=DECLARED_FIRST, last=DECLARED_LAST, closed=sorted(DECLARED_CLOSED), sessions=len(sessions), specials={d: FC.session(d)[3] for d in sessions if FC.session(d)[3] != "normal"}), supplement_at=SUPPLEMENT_AT)
    feathers = {"069500": os.path.join(HERE, "ft", "user_data", "data_kodex_nfi_v1_2_algo", "binance", "ALGO_USDT-5m.feather"), "229200": os.path.join(HERE, "ft", "user_data", "data_kosdaq150_v1", "binance", "ALGO_USDT-5m.feather")}
    tmp = tempfile.mkdtemp(prefix="fwd_base_")
    try:
        for sym in SYMS:
            p, want = BASE_MIN[sym]
            got = sha_file(p)
            df = pd.read_csv(p, dtype={"ts": str})
            bad = value_problems(df)
            f5 = to_5m(df[["ts", "open", "high", "low", "close", "volume"]], tmp, sym)
            fe = pd.read_feather(feathers[sym])
            fe["date"] = pd.to_datetime(fe["date"], utc=True)
            same_len = len(f5) == len(fe)
            dates_equal = same_len and bool((f5["date"].values == fe["date"].values).all())
            px_equal = same_len and all(bool(((f5[c] / 10000 - fe[c]).abs() < 1e-9).all()) for c in ("open", "high", "low", "close")) and bool((f5["volume"] == fe["volume"]).all())
            rec["base_minute"][sym] = dict(path=rel(p, HERE), sha256=got, sha_matches=got == want, rows=len(df), first=df["ts"].iloc[0], last=df["ts"].iloc[-1], invalid_value_rows=len(bad), invalid_value_sample=bad[:5], feather=rel(feathers[sym], HERE), feather_sha256=sha_file(feathers[sym]), resampled_5m_rows=len(f5), feather_rows=len(fe), dates_equal=dates_equal, prices_div_10000_and_volume_equal=px_equal)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    B.write_new(out_path, json.dumps(rec, ensure_ascii=False, indent=1).encode("utf-8"))
    print(json.dumps({s: {k: v for k, v in r.items() if k in ("sha_matches", "rows", "invalid_value_rows", "invalid_value_sample", "dates_equal", "prices_div_10000_and_volume_equal")} for s, r in rec["base_minute"].items()}, ensure_ascii=False))
    print("base adjusted", json.dumps({k: rec["base_adjusted"][k] for k in ("cut_after", "days", "non_finite_or_non_positive_closes")}, ensure_ascii=False), "| history", rec["history_code_sha256"][:8], "collector", rec["collector_sha256"][:8])


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
