import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backfill_kr_etf_1m as B
import forward_collect_kr_etf as FC
import forward_history_kr_etf as FH

LOG = os.path.join(HERE, "runs", "forward_runner_log.jsonl")
LOCK = os.path.join(HERE, "runs", "forward_runner.lock")
LATE_LIMIT = timedelta(hours=6)
TRIES = 1 + FC.TRANSPORT_RETRIES
JOB_CAP = {
    "base": dict(routine=0, recovery=0),
    "minute": dict(routine=2 * len(FC.SYMS) * TRIES, recovery=0),
    "minute_refetch": dict(routine=2 * len(FC.SYMS) * TRIES, recovery=0),
    "adjusted": dict(routine=len(FC.SYMS) * TRIES, recovery=FC.ORIGIN_MAX_PAGES),
}
TOTAL_CAP = dict(routine=1728, recovery=192)
TERMINAL = {"ok", "error", "skipped_late", "recovered_ok", "interrupted_incomplete"}


class SimulatedCrash(Exception):
    pass


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def compact(at_iso):
    return datetime.fromisoformat(at_iso).strftime("%Y%m%dT%H%M%S")


def plan(sessions, supplement_at=FH.SUPPLEMENT_AT, first=FH.DECLARED_FIRST):
    jobs = [dict(id="base", at=supplement_at, job="base", vintage=FH.BASE_VINTAGE)]
    for d in sessions:
        for j in FH.schedule_for(d, sessions):
            if j["at"] is None:
                continue
            at = supplement_at if (d == first and j["job"] in ("minute", "adjusted")) else j["at"]
            jid = f"{j['job']}_{d}"
            jobs.append(dict(id=jid, at=at, job=j["job"], day=d, vintage=f"{compact(at)}_{jid}"))
    order = {"base": 0, "minute": 1, "adjusted": 2, "minute_refetch": 3}
    return sorted(jobs, key=lambda j: (j["at"], order[j["job"]], j.get("day", "")))


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_log(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def acquire_lock(path, now):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise B.Stop(f"another runner holds the lock {path}: {open(path, encoding='utf-8').read()[:200]}; check that process before removing the lock by hand")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(pid=os.getpid(), started_at=now.isoformat(timespec="seconds"))))
        f.flush()
        os.fsync(f.fileno())


def job_root(j, roots):
    return roots["hist"] if j["job"] == "base" else roots["adj"] if j["job"] == "adjusted" else roots["min"]


def request_files(vdir):
    p = os.path.join(vdir, "requests")
    return len(os.listdir(p)) if os.path.isdir(p) else 0


def check_log(rows, by_id):
    started, terminal = {}, {}
    for r in rows:
        if r["id"] == "runner":
            continue
        if r["id"] not in by_id:
            raise B.Stop(f"log row for an unknown job {r['id']}")
        if r["status"] == "started":
            if r["id"] in started:
                raise B.Stop(f"two started rows for {r['id']}")
            if r["vintage"] != by_id[r["id"]]["vintage"]:
                raise B.Stop(f"{r['id']} started with vintage {r['vintage']}, plan says {by_id[r['id']]['vintage']}")
            started[r["id"]] = r
        elif r["status"] in TERMINAL:
            if r["id"] in terminal:
                raise B.Stop(f"two terminal rows for {r['id']}")
            if r["status"] != "skipped_late" and r["id"] not in started:
                raise B.Stop(f"terminal row without a started row for {r['id']}")
            terminal[r["id"]] = r
        else:
            raise B.Stop(f"unknown status {r['status']} for {r['id']}")
    return started, terminal


def used_budget(started, terminal):
    used = dict(routine=0, recovery=0)
    for jid, s in started.items():
        t = terminal.get(jid)
        c = s["reserved"] if t is None else (t.get("charged") or t.get("attempts") or {})
        for k in used:
            used[k] += c.get(k, 0)
    return used


def reconcile(j, srow, roots, sessions):
    vdir = os.path.join(job_root(j, roots), j["vintage"])
    row = dict(id=j["id"], job=j["job"], day=j.get("day"), vintage=j["vintage"], request_files=request_files(vdir), charged=srow["reserved"])
    try:
        FC.check_sealed(job_root(j, roots), os.path.join(vdir, "meta.json"))
        m = json.load(open(os.path.join(vdir, "meta.json"), encoding="utf-8"))
        if j["job"] == "adjusted":
            row["charged"] = dict(routine=m["routine"], recovery=m["recovery"])
        elif j["job"] != "base":
            row["charged"] = dict(routine=sum(1 for q in m["requests"] if q.get("budget", "routine") == "routine"), recovery=sum(1 for q in m["requests"] if q.get("budget") == "recovery"))
        else:
            row["charged"] = dict(routine=0, recovery=0)
        if j["job"] == "adjusted":
            hdir = os.path.join(roots["hist"], j["vintage"])
            if os.path.isdir(hdir):
                FC.check_sealed(roots["hist"], os.path.join(hdir, "meta.json"))
                hres, how = json.load(open(os.path.join(hdir, "meta.json"), encoding="utf-8"))["result"], "history vintage already written"
            else:
                _, hres = FH.extend_history(roots["hist"], vdir, j["day"], sessions)
                how = "history extended from the sealed adjusted vintage, no request"
            row.update(status="recovered_ok", result=m["result"], history=hres, how=how)
        else:
            row.update(status="recovered_ok", result=m.get("result"), how="sealed vintage found, no request")
    except (B.Stop, OSError) as e:
        row.update(status="interrupted_incomplete", reason=f"{type(e).__name__}: {e}"[:300], how="not rerun; the next refetch or adjusted job covers the day")
    return row


def verify_results(terminal, by_id, roots):
    bad = []
    for jid, r in terminal.items():
        if r["status"] not in ("ok", "recovered_ok"):
            continue
        j = by_id[jid]
        try:
            FC.check_sealed(job_root(j, roots), os.path.join(job_root(j, roots), j["vintage"], "meta.json"))
            if j["job"] == "adjusted":
                FC.check_sealed(roots["hist"], os.path.join(roots["hist"], j["vintage"], "meta.json"))
        except (B.Stop, OSError) as e:
            bad.append(f"{jid}: {e}")
    if bad:
        raise B.Stop(f"logged results do not match the sealed files: {bad[:5]}")


def run_job(j, now, roots, sessions, fetch=None, clock=None, sleep=None, origins=FH.ORIGINS, base_loader=None, crash_point=None):
    cap = JOB_CAP[j["job"]]
    c = None
    out = dict(vintage=j["vintage"])

    def mk():
        return FC.Collector(fetch=fetch, clock=clock, sleep=sleep, out_min=roots["min"], out_adj=roots["adj"], max_req=cap["routine"], max_recovery_pages=cap["recovery"], vintage=j["vintage"])
    try:
        if j["job"] == "base":
            base, meta = (base_loader or FH.base_history)()
            FH.write_base_vintage(roots["hist"], base, meta, origins=origins)
            out.update(status="ok", result="base vintage written")
        elif j["job"] in ("minute", "minute_refetch"):
            c = mk()
            _, res = c.minute_day(j["day"])
            out.update(status="ok", result=res)
        else:
            if FH.latest_completed_session(now, sessions) < j["day"]:
                raise B.Stop(f"{j['day']} is not complete at {now.isoformat(timespec='seconds')}")
            stored = FH.stored_for_check(roots["hist"])
            c = mk()
            v, res = c.adjusted_checked(stored, j["day"], origins)
            if crash_point == "after_collect":
                raise SimulatedCrash(j["id"])
            _, hres = FH.extend_history(roots["hist"], v, j["day"], sessions)
            out.update(status="ok", result=res, history=hres)
    except (B.Stop, FileExistsError) as e:
        out.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
    out["attempts"] = dict(routine=c.n_routine if c else 0, recovery=c.n_recovery if c else 0)
    out["request_files"] = request_files(c.vdir) if c is not None and c.vdir else 0
    return out


def loop(roots, log_path, lock_path, until, sessions=None, fetch=None, clock=None, sleep=None, origins=FH.ORIGINS, base_loader=None, say=print, crash=None, total_cap=TOTAL_CAP):
    sessions = sessions or FH.declared_sessions()
    clock = clock or (lambda: datetime.now(FC.KST))
    sleep = sleep or time.sleep
    jobs = plan(sessions)
    by_id = {j["id"]: j for j in jobs}
    acquire_lock(lock_path, clock())
    crashed = False
    try:
        started, terminal = check_log(read_log(log_path), by_id)
        for jid, s in started.items():
            if jid not in terminal:
                row = reconcile(by_id[jid], s, roots, sessions)
                row["reconciled_at"] = clock().isoformat(timespec="seconds")
                append_log(log_path, row)
                say(f"{jid} {row['status']}: {row['how']}")
        started, terminal = check_log(read_log(log_path), by_id)
        verify_results(terminal, by_id, roots)
        used = used_budget(started, terminal)
        say(f"attempts used {used}, remaining { {k: total_cap[k] - used[k] for k in used} }")
        while True:
            started, terminal = check_log(read_log(log_path), by_id)
            pending = [j for j in jobs if j["id"] not in terminal]
            if not pending:
                say("no pending jobs in the declared schedule")
                return
            j = pending[0]
            at = datetime.fromisoformat(j["at"])
            if at > until:
                say(f"next job {j['id']} at {j['at']} is after the run limit {until.isoformat(timespec='seconds')}")
                return
            now = clock()
            if now < at:
                sleep(min((at - now).total_seconds(), 600))
                continue
            stamp = dict(id=j["id"], job=j["job"], day=j.get("day"), vintage=j["vintage"], planned_at=j["at"], late_s=round((now - at).total_seconds()))
            if now - at > LATE_LIMIT and j["job"] != "base":
                append_log(log_path, dict(stamp, status="skipped_late", attempts=dict(routine=0, recovery=0), at=now.isoformat(timespec="seconds"), reason=f"{stamp['late_s']} s after the planned time (limit {int(LATE_LIMIT.total_seconds())} s)"))
                say(f"{j['id']} skipped_late")
                continue
            used = used_budget(started, terminal)
            cap = JOB_CAP[j["job"]]
            if any(used[k] + cap[k] > total_cap[k] for k in cap):
                append_log(log_path, dict(id="runner", status="stopped_budget", next_job=j["id"], used=used, job_reserve=cap, total_cap=total_cap, at=now.isoformat(timespec="seconds")))
                say(f"stopped before {j['id']}: attempts used {used} + reserve {cap} exceed {total_cap}")
                return
            append_log(log_path, dict(stamp, status="started", started_at=now.isoformat(timespec="seconds"), reserved=cap, runner_sha256=sha_file(os.path.abspath(__file__)), collector_sha256=sha_file(FC.__file__), history_sha256=sha_file(FH.__file__)))
            if crash == (j["id"], "before_run"):
                crashed = True
                raise SimulatedCrash(j["id"])
            try:
                res = run_job(j, now, roots, sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader, crash_point=crash[1] if crash and crash[0] == j["id"] else None)
            except SimulatedCrash:
                crashed = True
                raise
            if crash == (j["id"], "before_terminal"):
                crashed = True
                raise SimulatedCrash(j["id"])
            append_log(log_path, dict(stamp, **res, finished_at=clock().isoformat(timespec="seconds")))
            say(f"{j['id']} {res['status']}" + (f": {res.get('error')}" if res["status"] != "ok" else ""))
            if j["job"] == "base" and res["status"] != "ok":
                say("base vintage failed; stopping")
                return
    finally:
        if not crashed:
            os.remove(lock_path)


def selftest():
    tmp = tempfile.mkdtemp(prefix="fwd_run_")
    try:
        sessions = FH.declared_sessions()
        p = plan(sessions)
        assert p[0]["id"] == "base" and [j["id"] for j in p[1:3]] == ["minute_2026-11-06", "adjusted_2026-11-06"] and p[1]["at"] == FH.SUPPLEMENT_AT
        assert p[1]["vintage"] == "20261108T120000_minute_2026-11-06" and p[2]["vintage"] == "20261108T120000_adjusted_2026-11-06"
        assert next(j for j in p if j["id"] == "minute_refetch_2026-11-06")["vintage"] == "20261109T083000_minute_refetch_2026-11-06"
        ids, vints = [j["id"] for j in p], [j["vintage"] for j in p]
        assert len(ids) == len(set(ids)) == 1 + 58 * 3 - 1 and len(set(vints)) == len(vints) and plan(sessions) == p
        assert [j["at"] for j in p] == sorted(j["at"] for j in p)
        planned_logical = sum(FH.schedule_for(d, sessions)[k]["requests"] for d in sessions for k in range(3) if FH.schedule_for(d, sessions)[k]["at"] is not None)
        cap_sum = sum(JOB_CAP[j["job"]]["routine"] for j in p)
        assert planned_logical == 576 and cap_sum == 576 * TRIES == TOTAL_CAP["routine"], (planned_logical, cap_sum)

        origins = {"069500": "2026-01-01", "229200": "2026-01-01"}
        days, d = [], datetime(2026, 1, 1)
        while d.strftime("%Y-%m-%d") <= "2026-11-13":
            if d.weekday() < 5:
                days.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        adj = {s: {x: 100.0 + i % 9 for i, x in enumerate(days)} for s in FC.SYMS}
        state = dict(now=datetime.fromisoformat("2026-11-08T11:00:00+09:00"), fail=set())
        clock = lambda: state["now"]

        def sleep(s):
            state["now"] += timedelta(seconds=max(s, 0.001))

        def visible_last():
            return [x for x in days if FC.earliest_fetch(x) <= state["now"] or x <= "2026-11-05"][-1]

        def bars(day):
            first, last = FC.session(day)[:2]
            return [dict(timestamp=f"{day}T{hm}:00+09:00", openPrice="100", highPrice="105", lowPrice="95", closePrice="100", volume="0" if hm in FC.session(day)[2] else "10") for hm in B.minutes_expected(first, last)]

        calls = []

        def fetch(params):
            calls.append(dict(params, at=state["now"].isoformat(timespec="seconds")))
            state["now"] += timedelta(seconds=1)
            if params["interval"] in state["fail"]:
                raise B.Stop("transport: fake timeout")
            if params["interval"] == "1m":
                bs = [b for x in days if "2026-11-04" <= x <= visible_last() for b in bars(x) if b["timestamp"] < params["before"]]
                return 200, json.dumps(dict(candles=bs[-200:])).encode("utf-8"), {}
            ds = [x for x in days if x <= visible_last()]
            if "before" in params:
                ds = [x for x in ds if x < params["before"][:10]]
            ds = ds[-200:]
            return 200, json.dumps(dict(candles=[dict(timestamp=f"{x}T00:00:00+09:00", closePrice=str(adj[params["symbol"]][x])) for x in ds])).encode("utf-8"), {}

        def base_loader():
            keep = [x for x in days if x <= "2026-11-05"]
            return {s: dict(days=keep, close=[adj[s][x] for x in keep]) for s in FC.SYMS}, dict(source="fixture", cut_after="2026-11-05")

        roots = {k: os.path.join(tmp, k) for k in ("min", "adj", "hist")}
        log_path, lock_path = os.path.join(tmp, "log.jsonl"), os.path.join(tmp, "runner.lock")
        msgs = []

        def go(until, **kw):
            loop(roots, log_path, lock_path, datetime.fromisoformat(until), sessions=sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader, say=msgs.append, **kw)

        def term(jid):
            rs = [r for r in read_log(log_path) if r["id"] == jid and r["status"] in TERMINAL]
            return rs[0] if rs else None

        go("2026-11-10T08:31:00+09:00")
        rows = [r for r in read_log(log_path) if r["status"] in TERMINAL]
        assert [r["id"] for r in rows] == ["base", "minute_2026-11-06", "adjusted_2026-11-06", "minute_refetch_2026-11-06", "minute_2026-11-09", "adjusted_2026-11-09", "minute_refetch_2026-11-09"]
        assert all(r["status"] == "ok" for r in rows) and not os.path.exists(lock_path)
        assert sum(r["attempts"]["routine"] for r in rows) == len(calls) == sum(r["request_files"] for r in rows) == 20
        assert os.path.isdir(os.path.join(roots["min"], "20261109T161000_minute_2026-11-09")) and os.path.isdir(os.path.join(roots["hist"], "20261109T161000_adjusted_2026-11-09"))
        n = len(calls)
        go("2026-11-10T08:31:00+09:00")
        assert len(calls) == n

        acquire_lock(lock_path, clock())
        try:
            go("2026-11-10T08:31:00+09:00")
            raise AssertionError("a second runner must not start while the lock exists")
        except B.Stop as e:
            assert "holds the lock" in str(e)
        os.remove(lock_path)

        try:
            go("2026-11-10T16:11:00+09:00", crash=("minute_2026-11-10", "before_terminal"))
            raise AssertionError("simulated crash expected")
        except SimulatedCrash:
            pass
        assert os.path.exists(lock_path) and term("minute_2026-11-10") is None
        n_1m = sum(1 for c in calls if c["interval"] == "1m")
        try:
            go("2026-11-10T16:11:00+09:00")
            raise AssertionError("the stale lock must block a restart")
        except B.Stop as e:
            assert "holds the lock" in str(e)
        os.remove(lock_path)
        go("2026-11-10T16:11:00+09:00")
        r = term("minute_2026-11-10")
        assert r["status"] == "recovered_ok" and r["charged"] == dict(routine=4, recovery=0) and r["request_files"] == 4 and sum(1 for c in calls if c["interval"] == "1m") == n_1m, r
        assert term("adjusted_2026-11-10")["status"] == "ok"

        state["now"] = datetime.fromisoformat("2026-11-11T08:30:00+09:00")
        n_1d = sum(1 for c in calls if c["interval"] == "1d")
        try:
            go("2026-11-11T16:11:00+09:00", crash=("adjusted_2026-11-11", "after_collect"))
            raise AssertionError("simulated crash expected")
        except SimulatedCrash:
            pass
        assert not os.path.isdir(os.path.join(roots["hist"], "20261111T161000_adjusted_2026-11-11")) and os.path.isdir(os.path.join(roots["adj"], "20261111T161000_adjusted_2026-11-11"))
        n_1d_after = sum(1 for c in calls if c["interval"] == "1d")
        os.remove(lock_path)
        go("2026-11-11T16:11:00+09:00")
        r = term("adjusted_2026-11-11")
        assert r["status"] == "recovered_ok" and "extended" in r["how"] and r["history"]["069500"]["new_days"] == ["2026-11-11"] and sum(1 for c in calls if c["interval"] == "1d") == n_1d_after > n_1d, r
        for s in FC.SYMS:
            assert FH.verify_chain(roots["hist"], s)[0].startswith("20261111T161000_adjusted_2026-11-11")

        state["now"] = datetime.fromisoformat("2026-11-12T08:30:00+09:00")
        try:
            go("2026-11-12T08:31:00+09:00", crash=("minute_refetch_2026-11-11", "before_run"))
            raise AssertionError("simulated crash expected")
        except SimulatedCrash:
            pass
        os.remove(lock_path)
        n = len(calls)
        go("2026-11-12T08:31:00+09:00")
        r = term("minute_refetch_2026-11-11")
        assert r["status"] == "interrupted_incomplete" and r["charged"] == JOB_CAP["minute_refetch"] and len(calls) == n, r

        state["now"] = datetime.fromisoformat("2026-11-12T16:10:00+09:00")
        state["fail"] = {"1m"}
        go("2026-11-12T16:10:00+09:00")
        r = term("minute_2026-11-12")
        assert r["status"] == "error" and "transport" in r["error"] and r["attempts"] == dict(routine=TRIES, recovery=0) and r["request_files"] == TRIES and r["vintage"] == "20261112T161000_minute_2026-11-12", r
        state["fail"] = set()
        go("2026-11-12T16:10:00+09:00")
        assert term("adjusted_2026-11-12")["status"] == "ok"
        st, tm = check_log(read_log(log_path), {j["id"]: j for j in p})
        used = used_budget(st, tm)
        assert used == dict(routine=len(calls) + JOB_CAP["minute_refetch"]["routine"], recovery=0), (used, len(calls))

        state["now"] = datetime.fromisoformat("2026-11-13T08:30:00+09:00")
        small = dict(routine=used["routine"] + JOB_CAP["minute_refetch"]["routine"] - 1, recovery=TOTAL_CAP["recovery"])
        go("2026-11-13T08:31:00+09:00", total_cap=small)
        last = read_log(log_path)[-1]
        assert last["id"] == "runner" and last["status"] == "stopped_budget" and last["next_job"] == "minute_refetch_2026-11-12" and term("minute_refetch_2026-11-12") is None, last

        state["now"] = datetime.fromisoformat("2026-11-14T09:00:00+09:00")
        go("2026-11-14T09:00:00+09:00")
        assert term("minute_refetch_2026-11-12")["status"] == "skipped_late" and term("minute_2026-11-13")["status"] == "skipped_late" and term("adjusted_2026-11-13")["status"] == "skipped_late"

        r = run_job(dict(id="x", job="adjusted", day="2026-11-16", vintage="x_adjusted"), datetime.fromisoformat("2026-11-14T09:00:00+09:00"), roots, sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins)
        assert r["status"] == "error" and "not complete" in r["error"] and r["attempts"] == dict(routine=0, recovery=0)

        bad_log = os.path.join(tmp, "bad.jsonl")
        shutil.copyfile(log_path, bad_log)
        append_log(bad_log, dict(term("adjusted_2026-11-12")))
        try:
            check_log(read_log(bad_log), {j["id"]: j for j in p})
            raise AssertionError("duplicate terminal rows must stop")
        except B.Stop as e:
            assert "two terminal rows" in str(e)
        print(f"runner selftest ok: fake clock and fetch, temp dir, no network; {len(calls)} fake attempts; lock, crash before run / after collect / before terminal, transport failures, attempt caps, late skips, duplicate log rows checked")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="DRAFT forward collection runner; runs only the declared jobs; not to be started before review")
    ap.add_argument("cmd", choices=["selftest", "plan"])
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        p = plan(FH.declared_sessions())
        for j in p[:8]:
            print(j)
        print(f"jobs {len(p)}, routine attempt cap per job {dict((k, v['routine']) for k, v in JOB_CAP.items())}, total {TOTAL_CAP}")


if __name__ == "__main__":
    main()
