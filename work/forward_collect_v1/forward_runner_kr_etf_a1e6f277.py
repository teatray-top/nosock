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
ROOTS = dict(min=FC.OUT_MIN, adj=FC.OUT_ADJ, hist=FH.HIST_ROOT)
OUT, ERR = os.path.join(HERE, "runs", "forward_runner.out"), os.path.join(HERE, "runs", "forward_runner.err")
CODE_FILES = ("forward_runner_kr_etf.py", "forward_collect_kr_etf.py", "forward_history_kr_etf.py", "backfill_kr_etf_1m.py", "toss.py", "cred.py")
TOKEN_MIN_LEFT = 900


SimulatedCrash = FH.SimulatedCrash


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


def reconcile(j, srow, roots, sessions, clock=None, base_loader=None, origins=FH.ORIGINS):
    vdir = os.path.join(job_root(j, roots), j["vintage"])
    row = dict(id=j["id"], job=j["job"], day=j.get("day"), vintage=j["vintage"], request_files=request_files(vdir), charged=srow["reserved"])
    try:
        recovered = []
        if j["job"] in ("base", "adjusted") and j["vintage"] in FH.pending_vintages(roots["hist"]):
            if j["job"] == "adjusted":
                FC.check_sealed(roots["adj"], os.path.join(vdir, "meta.json"))
            recovered = FH.recover_history(roots["hist"], roots["adj"], sessions, clock=clock, base_loader=base_loader, origins=origins)
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
                if f"{j['vintage']}/.commit" not in FC.load_seals(roots["hist"]):
                    raise B.Stop(f"history vintage {j['vintage']} has no commit record")
                hres = json.load(open(os.path.join(hdir, "meta.json"), encoding="utf-8"))["result"]
                how = "pending history vintage committed by explicit recovery, no request" if recovered else "history vintage already committed"
            else:
                _, hres = FH.extend_history(roots["hist"], vdir, j["day"], sessions)
                how = "history extended from the sealed adjusted vintage, no request"
            row.update(status="recovered_ok", result=m["result"], history=hres, how=how)
        elif j["job"] == "base":
            if f"{FH.BASE_VINTAGE}/.commit" not in FC.load_seals(roots["hist"]):
                raise B.Stop("base vintage has no commit record")
            row.update(status="recovered_ok", result="base vintage committed", how="pending base vintage committed by explicit recovery" if recovered else "base vintage already committed")
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
            if j["job"] in ("base", "adjusted"):
                FC.check_sealed(roots["hist"], os.path.join(roots["hist"], j["vintage"], "meta.json"))
        except (B.Stop, OSError) as e:
            bad.append(f"{jid}: {e}")
    if bad:
        raise B.Stop(f"logged results do not match the sealed files: {bad[:5]}")


def ensure_token(min_left=TOKEN_MIN_LEFT):
    import toss
    t = None
    for i in range(2):
        try:
            t = json.load(open(toss.TOKEN_FILE, encoding="utf-8"))
            break
        except (OSError, ValueError):
            time.sleep(2)
    left = float(t.get("exp", 0)) - time.time() if t else -1.0
    if t and t.get("access_token") and left >= min_left:
        return dict(issued=False, seconds_left=round(left))
    c = toss.Toss()
    c._issue()
    return dict(issued=True, previous_seconds_left=round(left), seconds_left=round(c.exp - time.time()))


def run_job(j, now, roots, sessions, fetch=None, clock=None, sleep=None, origins=FH.ORIGINS, base_loader=None, crash_point=None, token=None):
    cap = JOB_CAP[j["job"]]
    c = None
    out = dict(vintage=j["vintage"])

    def mk():
        return FC.Collector(fetch=fetch, clock=clock, sleep=sleep, out_min=roots["min"], out_adj=roots["adj"], max_req=cap["routine"], max_recovery_pages=cap["recovery"], vintage=j["vintage"])
    try:
        if j["job"] == "base":
            base, meta = (base_loader or FH.base_history)()
            FH.write_base_vintage(roots["hist"], base, meta, origins=origins, clock=clock, crash=crash_point[8:] if crash_point and crash_point.startswith("history_") else None)
            out.update(status="ok", result="base vintage written")
        elif j["job"] in ("minute", "minute_refetch"):
            if now < FC.earliest_fetch(j["day"]):
                raise B.Stop(f"too early for {j['day']} at {now.isoformat(timespec='seconds')}")
            take_token(token, out)
            c = mk()
            _, res = c.minute_day(j["day"])
            out.update(status="ok", result=res)
        else:
            if FH.latest_completed_session(now, sessions) < j["day"]:
                raise B.Stop(f"{j['day']} is not complete at {now.isoformat(timespec='seconds')}")
            stored = FH.stored_for_check(roots["hist"])
            take_token(token, out)
            c = mk()
            v, res = c.adjusted_checked(stored, j["day"], origins)
            if crash_point == "after_collect":
                raise SimulatedCrash(j["id"])
            _, hres = FH.extend_history(roots["hist"], v, j["day"], sessions, clock=clock, crash=crash_point[8:] if crash_point and crash_point.startswith("history_") else None)
            out.update(status="ok", result=res, history=hres)
    except (B.Stop, FileExistsError) as e:
        out.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
    out["attempts"] = dict(routine=c.n_routine if c else 0, recovery=c.n_recovery if c else 0)
    out["request_files"] = request_files(c.vdir) if c is not None and c.vdir else 0
    return out


def take_token(token, out):
    if token is None:
        return
    try:
        out["token"] = token()
    except Exception as e:
        out["token"] = dict(issued=False, error=f"{type(e).__name__}: {str(e)[:200]}")
        raise B.Stop(f"token step failed: {type(e).__name__}")


def loop(roots, log_path, lock_path, until, sessions=None, fetch=None, clock=None, sleep=None, origins=FH.ORIGINS, base_loader=None, say=print, crash=None, total_cap=TOTAL_CAP, token=None):
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
                row = reconcile(by_id[jid], s, roots, sessions, clock=clock, base_loader=base_loader, origins=origins)
                row["reconciled_at"] = clock().isoformat(timespec="seconds")
                append_log(log_path, row)
                say(f"{jid} {row['status']}: {row['how']}")
        pend = FH.pending_vintages(roots["hist"])
        if pend:
            raise B.Stop(f"history vintages {pend} are still pending after reconciliation; manual action needed")
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
                res = run_job(j, now, roots, sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader, crash_point=crash[1] if crash and crash[0] == j["id"] else None, token=token)
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


def window_jobs(until_iso, sessions=None):
    until = datetime.fromisoformat(until_iso)
    return [j["id"] for j in plan(sessions or FH.declared_sessions()) if datetime.fromisoformat(j["at"]) <= until]


def launch_spec(window, until_iso, here=HERE):
    relp = lambda x: os.path.relpath(x, here).replace("\\", "/")
    jobs = window_jobs(until_iso)
    by_id = {j["id"]: j for j in plan(FH.declared_sessions())}
    reserve = {k: sum(JOB_CAP[by_id[i]["job"]][k] for i in jobs) for k in ("routine", "recovery")}
    return dict(version="forward-runner-launch-v1", window=window, until=until_iso, cwd=here, python=dict(executable=sys.executable, version=sys.version.split()[0]), command=["python", "-X", "utf8", "forward_runner_kr_etf.py", "run", "--launch", "<this file, relative to cwd>", "--launch-sha", "<sha256 of this file>"], launcher="PowerShell: Set-Location <cwd>; Start-Process python -ArgumentList <command[1:]> -RedirectStandardOutput runs\\forward_runner.out -RedirectStandardError runs\\forward_runner.err -PassThru -NoNewWindow, then PriorityClass Idle and WaitForExit, as a background job of this session", code_sha256={f: sha_file(os.path.join(here, f)) for f in CODE_FILES}, roots={k: relp(v) for k, v in ROOTS.items()}, log=relp(LOG), lock=relp(LOCK), stdout=relp(OUT), stderr=relp(ERR), base_adjusted=dict(file=relp(FH.BASE_ADJ), sha256=FH.BASE_ADJ_SHA, cut_after=FH.BASE_LAST_DAY), jobs=jobs, logical_routine_requests=sum({"minute": 4, "minute_refetch": 4, "adjusted": 2, "base": 0}[by_id[i]["job"]] for i in jobs), attempt_reserve=reserve, total_cap=TOTAL_CAP, token_policy=f"before each minute or adjusted job the runner reads the shared token cache; only if it has less than {TOKEN_MIN_LEFT} s left does it issue one token through toss.Toss (credentials from Windows Credential Manager), at most once per job; the token is never printed or logged", priority="Idle", no_orders=True)


def make_launch(window, until_iso, out_path):
    spec = launch_spec(window, until_iso)
    B.write_new(out_path, json.dumps(spec, ensure_ascii=False, indent=1).encode("utf-8"))
    return sha_file(out_path), spec


def load_launch(path, want_sha, here=HERE, cwd=None):
    got = sha_file(path)
    if got != want_sha:
        raise B.Stop(f"launch file hash {got[:8]} differs from the given {want_sha[:8]}")
    spec = json.load(open(path, encoding="utf-8"))
    problems = []
    for f, h in spec["code_sha256"].items():
        if sha_file(os.path.join(here, f)) != h:
            problems.append(f"code hash differs: {f}")
    if set(spec["code_sha256"]) != set(CODE_FILES):
        problems.append("code file list differs")
    if os.path.normcase(os.path.abspath(cwd or os.getcwd())) != os.path.normcase(os.path.abspath(spec["cwd"])):
        problems.append(f"cwd {cwd or os.getcwd()} differs from {spec['cwd']}")
    if sys.version.split()[0] != spec["python"]["version"]:
        problems.append(f"python {sys.version.split()[0]} differs from {spec['python']['version']}")
    want = launch_spec(spec["window"], spec["until"], here)
    for k in ("roots", "log", "lock", "stdout", "stderr", "base_adjusted", "jobs", "attempt_reserve", "total_cap"):
        if spec[k] != want[k]:
            problems.append(f"{k} differs from the code")
    if problems:
        raise B.Stop("launch check failed: " + "; ".join(problems))
    return spec


def say_out(msg):
    print(f"{datetime.now(FC.KST).isoformat(timespec='seconds')} {msg}", flush=True)


def run(launch_path, launch_sha):
    spec = load_launch(os.path.abspath(launch_path), launch_sha)
    say_out(f"launch {spec['window']} verified: until {spec['until']}, jobs {len(spec['jobs'])}, code {', '.join(f'{k} {v[:8]}' for k, v in spec['code_sha256'].items())}")
    loop(dict(ROOTS), LOG, LOCK, datetime.fromisoformat(spec["until"]), say=say_out, token=ensure_token)
    say_out("runner finished its window")


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
        try:
            go("2026-11-12T16:10:00+09:00", crash=("adjusted_2026-11-12", "history_after_first_seal"))
            raise AssertionError("simulated crash expected")
        except SimulatedCrash:
            pass
        r = term("minute_2026-11-12")
        assert r["status"] == "error" and "transport" in r["error"] and r["attempts"] == dict(routine=TRIES, recovery=0) and r["request_files"] == TRIES and r["vintage"] == "20261112T161000_minute_2026-11-12", r
        assert FH.pending_vintages(roots["hist"]) == ["20261112T161000_adjusted_2026-11-12"] and FH.verify_chain(roots["hist"], "069500")[0].startswith("20261111T161000")
        state["fail"] = set()
        os.remove(lock_path)
        n = len(calls)
        go("2026-11-12T16:10:00+09:00")
        r = term("adjusted_2026-11-12")
        assert r["status"] == "recovered_ok" and "explicit recovery" in r["how"] and r["history"]["069500"]["new_days"] == ["2026-11-12"] and len(calls) == n and FH.pending_vintages(roots["hist"]) == [], r
        assert FH.verify_chain(roots["hist"], "229200")[0] == "20261112T161000_adjusted_2026-11-12/229200.json"
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

        roots_b = {k: os.path.join(tmp, "b_" + k) for k in ("min", "adj", "hist")}
        log_b, lock_b = os.path.join(tmp, "b_log.jsonl"), os.path.join(tmp, "b.lock")
        state["now"] = datetime.fromisoformat("2026-11-08T11:59:00+09:00")
        try:
            loop(roots_b, log_b, lock_b, datetime.fromisoformat("2026-11-08T12:01:00+09:00"), sessions=sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader, say=msgs.append, crash=("base", "history_after_first_file"))
            raise AssertionError("simulated crash expected")
        except SimulatedCrash:
            pass
        assert FH.pending_vintages(roots_b["hist"]) == [FH.BASE_VINTAGE]
        os.remove(lock_b)
        loop(roots_b, log_b, lock_b, datetime.fromisoformat("2026-11-08T12:01:00+09:00"), sessions=sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader, say=msgs.append)
        rb = [r for r in read_log(log_b) if r["status"] in TERMINAL]
        assert [(r["id"], r["status"]) for r in rb] == [("base", "recovered_ok"), ("minute_2026-11-06", "ok"), ("adjusted_2026-11-06", "ok")] and "explicit recovery" in rb[0]["how"], rb
        roots_c = {k: os.path.join(tmp, "c_" + k) for k in ("min", "adj", "hist")}
        log_c, lock_c = os.path.join(tmp, "c_log.jsonl"), os.path.join(tmp, "c.lock")
        state["now"] = datetime.fromisoformat("2026-11-08T11:59:00+09:00")
        goc = lambda **kw: loop(roots_c, log_c, lock_c, datetime.fromisoformat("2026-11-08T12:01:00+09:00"), sessions=sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, base_loader=base_loader, say=msgs.append, **kw)
        try:
            goc(crash=("adjusted_2026-11-06", "history_before_rename"))
            raise AssertionError("simulated crash expected")
        except SimulatedCrash:
            pass
        staged = os.path.join(roots_c["hist"], FH.STAGING, "20261108T120000_adjusted_2026-11-06", "069500.json")
        os.chmod(staged, 0o666)
        with open(staged, "ab") as f:
            f.write(b" ")
        os.remove(lock_c)
        n = len(calls)
        try:
            goc()
            raise AssertionError("a pending vintage that cannot be recovered must stop the runner")
        except B.Stop as e:
            assert "still pending" in str(e) and "manual action" in str(e)
        rc = [r for r in read_log(log_c) if r["id"] == "adjusted_2026-11-06" and r["status"] in TERMINAL]
        assert len(rc) == 1 and rc[0]["status"] == "interrupted_incomplete" and "manual action" in rc[0]["reason"] and len(calls) == n and not os.path.exists(lock_c), rc
        try:
            goc()
            raise AssertionError("the runner must keep refusing until the pending vintage is handled by hand")
        except B.Stop as e:
            assert "still pending" in str(e)
        state["now"] = datetime.fromisoformat("2026-11-14T09:00:00+09:00")
        tk = []
        r = run_job(dict(id="t0", job="minute", day="2026-11-16", vintage="t0_minute"), state["now"], roots, sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, token=lambda: tk.append(1))
        assert r["status"] == "error" and "too early" in r["error"] and tk == [] and "token" not in r

        def fake_token():
            tk.append(1)
            return dict(issued=len(tk) == 1, seconds_left=3600)
        r = run_job(dict(id="t", job="minute", day="2026-11-12", vintage="t_minute"), datetime.fromisoformat("2026-11-14T09:00:00+09:00"), roots, sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, token=fake_token)
        assert r["status"] == "ok" and r["token"] == dict(issued=True, seconds_left=3600) and len(tk) == 1

        def bad_token():
            raise OSError("credential store unavailable")
        n = len(calls)
        r = run_job(dict(id="t2", job="adjusted", day="2026-11-13", vintage="t2_adjusted"), datetime.fromisoformat("2026-11-14T09:00:00+09:00"), roots, sessions, fetch=fetch, clock=clock, sleep=sleep, origins=origins, token=bad_token)
        assert r["status"] == "error" and "token step failed" in r["error"] and r["token"]["error"].startswith("OSError") and r["attempts"] == dict(routine=0, recovery=0) and len(calls) == n, r

        lp = os.path.join(tmp, "launch_test.json")
        sha, spec = make_launch("wtest", "2026-11-10T08:40:00+09:00", lp)
        assert spec["jobs"] == ["base", "minute_2026-11-06", "adjusted_2026-11-06", "minute_refetch_2026-11-06", "minute_2026-11-09", "adjusted_2026-11-09", "minute_refetch_2026-11-09"] and spec["logical_routine_requests"] == 20 and spec["attempt_reserve"] == dict(routine=60, recovery=96), spec
        assert load_launch(lp, sha, cwd=HERE)["window"] == "wtest"
        expect = lambda f, text: (lambda e: text in str(e))(_catch(f))
        assert expect(lambda: load_launch(lp, "0" * 64, cwd=HERE), "launch file hash")
        assert expect(lambda: load_launch(lp, sha, cwd=tmp), "cwd")
        bad = dict(spec, code_sha256=dict(spec["code_sha256"], **{"toss.py": "0" * 64}))
        lp2 = os.path.join(tmp, "launch_bad.json")
        open(lp2, "w", encoding="utf-8").write(json.dumps(bad))
        assert expect(lambda: load_launch(lp2, sha_file(lp2), cwd=HERE), "code hash differs: toss.py")
        bad = dict(spec, jobs=spec["jobs"][:-1])
        open(lp2, "w", encoding="utf-8").write(json.dumps(bad))
        assert expect(lambda: load_launch(lp2, sha_file(lp2), cwd=HERE), "jobs differs")
        bad = dict(spec, lock="runs/other.lock")
        open(lp2, "w", encoding="utf-8").write(json.dumps(bad))
        assert expect(lambda: load_launch(lp2, sha_file(lp2), cwd=HERE), "lock differs")
        bad_log = os.path.join(tmp, "bad.jsonl")
        shutil.copyfile(log_path, bad_log)
        append_log(bad_log, dict(term("adjusted_2026-11-12")))
        try:
            check_log(read_log(bad_log), {j["id"]: j for j in p})
            raise AssertionError("duplicate terminal rows must stop")
        except B.Stop as e:
            assert "two terminal rows" in str(e)
        print(f"runner selftest ok: fake clock and fetch, temp dir, no network; {len(calls)} fake attempts; lock, crash before run / after collect / inside the history commit / inside the base commit / before terminal, unrecoverable staged file, transport failures, attempt caps, late skips, duplicate log rows checked")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _catch(f):
    try:
        f()
    except B.Stop as e:
        return e
    raise AssertionError("expected a stop")


def main():
    ap = argparse.ArgumentParser(description="forward collection runner; runs only the declared jobs of a verified launch file")
    ap.add_argument("cmd", choices=["selftest", "plan", "make_launch", "check_launch", "run"])
    ap.add_argument("--launch")
    ap.add_argument("--launch-sha")
    ap.add_argument("--window")
    ap.add_argument("--until")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "make_launch":
        sha, spec = make_launch(a.window, a.until, a.launch)
        print(json.dumps(dict(launch=a.launch, sha256=sha, jobs=spec["jobs"], logical_routine_requests=spec["logical_routine_requests"], attempt_reserve=spec["attempt_reserve"]), ensure_ascii=False))
    elif a.cmd == "check_launch":
        spec = load_launch(os.path.abspath(a.launch), a.launch_sha)
        print(json.dumps(dict(ok=True, window=spec["window"], until=spec["until"], jobs=len(spec["jobs"]), python=spec["python"], code={k: v[:8] for k, v in spec["code_sha256"].items()}), ensure_ascii=False))
    elif a.cmd == "run":
        run(a.launch, a.launch_sha)
    else:
        p = plan(FH.declared_sessions())
        for j in p[:8]:
            print(j)
        print(f"jobs {len(p)}, routine attempt cap per job {dict((k, v['routine']) for k, v in JOB_CAP.items())}, total {TOTAL_CAP}")


if __name__ == "__main__":
    main()
