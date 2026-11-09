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
LATE_LIMIT = timedelta(hours=6)


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def plan(sessions, supplement_at=FH.SUPPLEMENT_AT, first=FH.DECLARED_FIRST):
    jobs = [dict(id="base", at=supplement_at, job="base")]
    for d in sessions:
        for j in FH.schedule_for(d, sessions):
            if j["at"] is None:
                continue
            at = supplement_at if (d == first and j["job"] in ("minute", "adjusted")) else j["at"]
            jobs.append(dict(id=f"{j['job']}_{d}", at=at, job=j["job"], day=d))
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


def run_job(j, now, roots, sessions, fetch=None, clock=None, sleep=None, origins=FH.ORIGINS, base_loader=None):
    mk = lambda: FC.Collector(fetch=fetch, clock=clock, sleep=sleep, out_min=roots["min"], out_adj=roots["adj"])
    if j["job"] == "base":
        if os.path.isdir(os.path.join(roots["hist"], FH.BASE_VINTAGE)):
            return dict(result="base vintage already present", base_files={s: sha_file(os.path.join(roots["hist"], FH.BASE_VINTAGE, f"{s}.json")) for s in FC.SYMS})
        base, meta = (base_loader or FH.base_history)()
        v = FH.write_base_vintage(roots["hist"], base, meta, origins=origins)
        return dict(result="base vintage written", path=v)
    if j["job"] in ("minute", "minute_refetch"):
        c = mk()
        try:
            v, out = c.minute_day(j["day"])
        finally:
            used = dict(routine=c.n_routine, recovery=c.n_recovery, vintage=c.vdir)
        return dict(result=out, requests=used)
    if j["job"] == "adjusted":
        if FH.latest_completed_session(now, sessions) < j["day"]:
            raise B.Stop(f"{j['day']} is not complete at {now.isoformat(timespec='seconds')}")
        c = mk()
        try:
            v, out = c.adjusted_checked(FH.stored_for_check(roots["hist"]), j["day"], origins)
        finally:
            used = dict(routine=c.n_routine, recovery=c.n_recovery, vintage=c.vdir)
        hv, hres = FH.extend_history(roots["hist"], v, j["day"], sessions)
        return dict(result=out, history=hres, history_vintage=os.path.basename(hv), requests=used)
    raise ValueError(j["job"])


def loop(roots, log_path, until, sessions=None, fetch=None, clock=None, sleep=None, origins=FH.ORIGINS, base_loader=None, say=print):
    sessions = sessions or FH.declared_sessions()
    clock = clock or (lambda: datetime.now(FC.KST))
    sleep = sleep or time.sleep
    jobs = plan(sessions)
    while True:
        done = {r["id"] for r in read_log(log_path) if r.get("status") in ("ok", "error", "skipped_late")}
        pending = [j for j in jobs if j["id"] not in done]
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
        row = dict(id=j["id"], job=j["job"], day=j.get("day"), planned_at=j["at"], started_at=now.isoformat(timespec="seconds"), late_s=round((now - at).total_seconds()), runner_sha256=sha_file(os.path.abspath(__file__)), collector_sha256=sha_file(FC.__file__), history_sha256=sha_file(FH.__file__))
        if now - at > LATE_LIMIT and j["job"] != "base":
            row.update(status="skipped_late", reason=f"started {row['late_s']} s after the planned time (limit {int(LATE_LIMIT.total_seconds())} s)")
        else:
            try:
                row.update(status="ok", **run_job(j, now, roots, sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader))
            except B.Stop as e:
                row.update(status="error", error=str(e)[:500])
        row["finished_at"] = clock().isoformat(timespec="seconds")
        append_log(log_path, row)
        say(f"{row['id']} {row['status']}" + (f": {row.get('error') or row.get('reason')}" if row["status"] != "ok" else ""))
        if j["job"] == "base" and row["status"] != "ok":
            say("base vintage failed; stopping")
            return


def selftest():
    tmp = tempfile.mkdtemp(prefix="fwd_run_")
    try:
        sessions = FH.declared_sessions()
        p = plan(sessions)
        assert p[0]["id"] == "base" and [j["id"] for j in p[1:3]] == ["minute_2026-11-06", "adjusted_2026-11-06"] and p[1]["at"] == FH.SUPPLEMENT_AT
        assert next(j for j in p if j["id"] == "minute_refetch_2026-11-06")["at"] == "2026-11-09T08:30:00+09:00"
        ids = [j["id"] for j in p]
        assert len(ids) == len(set(ids)) == 1 + 58 * 3 - 1
        assert [j["at"] for j in p] == sorted(j["at"] for j in p)
        i19 = [j for j in p if j.get("day") == "2026-11-19"]
        assert [j["at"] for j in i19] == ["2026-11-19T17:10:00+09:00", "2026-11-19T17:10:00+09:00", "2026-11-20T08:30:00+09:00"]
        t1120 = [j["id"] for j in p if j["at"] == "2026-11-20T08:30:00+09:00"]
        assert t1120 == ["minute_refetch_2026-11-19"]

        origins = {"069500": "2026-01-01", "229200": "2026-01-01"}
        days = []
        d = datetime(2026, 1, 1)
        while d.strftime("%Y-%m-%d") <= "2026-11-10":
            if d.weekday() < 5:
                days.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        adj = {s: {x: 100.0 + i % 9 for i, x in enumerate(days)} for s in FC.SYMS}
        state = dict(now=datetime.fromisoformat("2026-11-08T11:00:00+09:00"))
        clock = lambda: state["now"]

        def sleep(s):
            state["now"] += timedelta(seconds=max(s, 0.001))

        def visible_last():
            done = [x for x in days if FC.earliest_fetch(x) <= state["now"] or x <= "2026-11-05"]
            return done[-1]

        def bars(sym, day):
            first, last = FC.session(day)[:2]
            return [dict(timestamp=f"{day}T{hm}:00+09:00", openPrice="100", highPrice="105", lowPrice="95", closePrice="100", volume="0" if hm in FC.session(day)[2] else "10") for hm in B.minutes_expected(first, last)]

        calls = []

        def fetch(params):
            calls.append(dict(params, at=state["now"].isoformat(timespec="seconds")))
            state["now"] += timedelta(seconds=1)
            if params["interval"] == "1m":
                lim = visible_last()
                bs = [b for x in days if "2026-11-04" <= x <= lim for b in bars(params["symbol"], x) if b["timestamp"] < params["before"]]
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
        log_path = os.path.join(tmp, "log.jsonl")
        msgs = []
        loop(roots, log_path, datetime.fromisoformat("2026-11-10T08:31:00+09:00"), sessions=sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader, say=msgs.append)
        rows = read_log(log_path)
        assert [r["id"] for r in rows] == ["base", "minute_2026-11-06", "adjusted_2026-11-06", "minute_refetch_2026-11-06", "minute_2026-11-09", "adjusted_2026-11-09", "minute_refetch_2026-11-09"], [r["id"] for r in rows]
        assert all(r["status"] == "ok" for r in rows), rows
        assert rows[2]["history"] == {s: dict(status="appended", n_days=len([x for x in days if x <= "2026-11-06"]), new_days=["2026-11-06"]) for s in FC.SYMS}, rows[2]["history"]
        assert rows[5]["history"]["069500"]["new_days"] == ["2026-11-09"]
        assert rows[4]["started_at"] == "2026-11-09T16:10:00+09:00" and rows[6]["started_at"] == "2026-11-10T08:30:00+09:00"
        assert all(c["at"] >= "2026-11-08T12:00:00+09:00" for c in calls) and not [c for c in calls if "2026-11-09T08:30" <= c["at"] < "2026-11-09T16:10" and c["interval"] == "1d"]
        n_routine = sum(r["requests"]["routine"] for r in rows if "requests" in r)
        assert n_routine == len(calls) == 4 + 2 + 4 + 4 + 2 + 4, (n_routine, len(calls))
        assert msgs[-1].startswith("next job adjusted_2026-11-10") or msgs[-1].startswith("next job minute_2026-11-10"), msgs[-1]
        for s in FC.SYMS:
            assert len(FH.verify_chain(roots["hist"], s)) == 3
        n_before = len(calls)
        loop(roots, log_path, datetime.fromisoformat("2026-11-10T08:31:00+09:00"), sessions=sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader, say=msgs.append)
        assert len(calls) == n_before and len(read_log(log_path)) == 7

        state["now"] = datetime.fromisoformat("2026-11-11T09:00:00+09:00")
        loop(roots, log_path, datetime.fromisoformat("2026-11-11T09:00:00+09:00"), sessions=sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader, say=msgs.append)
        late = read_log(log_path)[7:]
        assert [r["id"] for r in late] == ["minute_2026-11-10", "adjusted_2026-11-10", "minute_refetch_2026-11-10"] and [r["status"] for r in late] == ["skipped_late", "skipped_late", "ok"], [(r["id"], r["status"]) for r in late]

        tmp2 = os.path.join(tmp, "early")
        roots2 = {k: os.path.join(tmp2, k) for k in ("min", "adj", "hist")}
        state["now"] = datetime.fromisoformat("2026-11-08T12:00:00+09:00")
        try:
            run_job(dict(id="x", job="adjusted", day="2026-11-09"), state["now"], roots2, sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins)
            raise AssertionError("an adjusted job before completion must stop")
        except B.Stop as e:
            assert "not complete" in str(e)
        print(f"runner selftest ok: fake clock and fetch, temp dir, no network; {len(calls)} fake requests; jobs ran at their planned times, reruns skip done jobs, late jobs over {int(LATE_LIMIT.total_seconds() / 3600)} h are skipped and logged")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="DRAFT forward collection runner; runs only the declared jobs; not to be started before review")
    ap.add_argument("cmd", choices=["selftest", "plan"])
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        for j in plan(FH.declared_sessions())[:12]:
            print(j)


if __name__ == "__main__":
    main()
