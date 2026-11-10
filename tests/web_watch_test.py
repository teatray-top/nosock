import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
tmp = tempfile.mkdtemp(prefix="TEST_FIXTURE_nosock_web_")
runs, web, inbox = (os.path.join(tmp, x) for x in ("runs", "web", "inbox"))
for d in (runs, web, inbox):
    os.makedirs(d)
LOG, LEDGER = os.path.join(runs, "live_nosock.jsonl"), os.path.join(runs, "live_nosock_state.json")
os.environ.update(NOSOCK_RUNS=runs, NOSOCK_LOG=LOG, NOSOCK_STATE=LEDGER, NOSOCK_SCENARIO_LOG=os.path.join(runs, "scenario_desk.jsonl"))
for k in ("NOSOCK_BACKUP", "DOBBY_LIVE", "NOSOCK_SEED"):
    os.environ.pop(k, None)
import toss
from toss import ApiError


class NoToss:
    def __init__(self, *a, **k):
        raise AssertionError("TEST_FIXTURE: tests never build a real Toss client")


toss.Toss = NoToss
import live_nosock as L

os.environ.update(NOSOCK_INBOX=inbox, NOSOCK_STATE=web, NOSOCK_LIVE_LOG=LOG, NOSOCK_LIVE_STATE=LEDGER, NOSOCK_DESK_LOG=os.path.join(runs, "scenario_desk.jsonl"), NOSOCK_SCORECARD=os.path.join(runs, "scorecard.json"), NOSOCK_WEB_PUSH="0", NOSOCK_APP=os.path.join(tmp, "app"), NOSOCK_PY=os.path.join(tmp, "no_python"), CREDENTIALS_DIRECTORY=os.path.join(tmp, "cred"))
WEB_SERVER = next(p for p in (os.path.join(ROOT, "deploy", "seoha", "web", "server.py"), os.path.join(os.path.dirname(ROOT), "web", "server.py")) if os.path.exists(p))
spec = importlib.util.spec_from_file_location("nosock_web_server", WEB_SERVER)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

for p in (L.RUNS, L.LOG, L.STATE, L.STOP, L.MARK_FLAG, L.FX_FLAG, L.BEAT, L.BACKUP, L.SCEN_LOG, S.INBOX, S.STATE, S.LIVE_LOG, S.LIVE_STATE, S.DESK_LOG, S.SCORECARD, S.MARK_FLAG, S.FX_FLAG, S.BEAT, S.BT_DIR, S.APP_DIR):
    assert os.path.abspath(p).startswith(os.path.abspath(tmp)), p
assert not S.WEB_PUSH
REAL_RUNS = os.path.join(ROOT, "runs")
REAL_BEFORE = sorted(x for x in os.listdir(REAL_RUNS) if x.startswith(("live_nosock", "backup"))) if os.path.isdir(REAL_RUNS) else []
PASS, FAIL = [], []
KST = ZoneInfo("Asia/Seoul")
REAL_DT = S.datetime


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


def kst(d, h, m=0, s=0):
    return datetime(2026, 11, d, h, m, s, tzinfo=KST)


def row(kind, at, **kw):
    return dict(ts=at.isoformat(timespec="seconds"), kind=kind, live=True, **kw)


def led(**kw):
    s = dict(book=dict(krw=0.0, usd=0.0, qty={}, cost={}, cost_krw={}, realized=0.0), markets={"069500": "KR", "SPY": "US"}, checked_day="2026-11-10", kr_done_for="2026-11-10", us_done_for="2026-11-09")
    s.update(kw)
    return s


def run(n, lr, st, beat_age=30):
    try:
        info, out = S.watch(lr, st, None if beat_age is None else n.timestamp() - beat_age, n)
    except Exception as e:
        return dict(error=repr(e)), [f"CRASH {e!r}"], []
    return info, [a["id"] for a in out], out


def fresh(n):
    return [row("start", n - timedelta(hours=30)), row("mark", n - timedelta(minutes=5), equity=1.0)]


def put(p, text):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def put_rows(lr):
    put(LOG, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lr))


def wipe():
    for p in (LOG, LEDGER, LEDGER + ".prev", LEDGER + ".tmp", L.BEAT, os.path.join(web, "pushed.json"), os.path.join(web, "fx_done.json")):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(L.BACKUP, ignore_errors=True)
    L.BEAT_AT[0] = 0.0


def age_beat(sec, base=None):
    t = (time.time() if base is None else base) - sec
    put(L.BEAT, "TEST_FIXTURE")
    os.utime(L.BEAT, (t, t))
    return t


check("paths: the runner and the web server agree on the heartbeat file", os.path.abspath(S.BEAT) == os.path.abspath(L.BEAT), (S.BEAT, L.BEAT))
check("paths: the web server marks the same windows as the runner", all(S.mark_window(kst(d, h, m)) == (L.in_window(kst(d, h, m), (9, 0), (15, 35)) or L.us_mark_window(kst(d, h, m))) for d in range(9, 17) for h in range(24) for m in (0, 4, 5, 6, 19, 20, 21, 34, 35, 36, 59)))

n = kst(10, 10)
info, got, out = run(n, fresh(n), led())
check("normal: a live runner inside the session raises nothing", got == [] and info == dict(beat="2026-11-10T09:59:30+09:00", age=30, silent=False), (got, info))

info, got, out = run(n, fresh(n), led(), beat_age=600)
check("silent: ten minutes without a heartbeat is not yet an alert", got == [] and info["silent"] is False, (got, info))
info, got, out = run(n, fresh(n), led(), beat_age=601)
sil = f"sil-{int(n.timestamp() - 601)}"
check("silent: past ten minutes raises one alert that names the last signal", got == [sil] and info["silent"] is True and info["age"] == 601 and out and out[0]["title"] == "NoSock 실행기 멈춤" and "11-10 09:49 이후 신호가 없습니다" in out[0]["body"], (got, info, out))
later = S.watch(fresh(n), led(), n.timestamp() - 601, n + timedelta(hours=3))[1]
check("silent: the alert keeps one id for as long as the same stall lasts", [a["id"] for a in later][:1] == [sil], later)
stopped = fresh(n) + [row("stop", n - timedelta(minutes=20), reason="stop file")]
info, got, out = run(n, stopped, led(), beat_age=1200)
check("silent: a runner that was stopped on purpose is described as switched off", len(got) == 1 and out and "꺼져 있습니다" in out[0]["body"] and "죽었거나" not in out[0]["body"], out)
restarted = stopped + [row("start", n - timedelta(minutes=15))]
info, got, out = run(n, restarted, led(), beat_age=800)
check("silent: a runner that started and then went quiet is described as dead or stuck", len(got) == 1 and out and "죽었거나 멈춘" in out[0]["body"], out)
info, got, out = run(n, fresh(n), led(), beat_age=None)
check("silent: no heartbeat file at all (old runner) raises nothing", got == [] and info == dict(beat=None, age=None, silent=False), (got, info))

old = [row("start", n - timedelta(hours=30)), row("mark", kst(10, 9, 20), equity=1.0)]
info, got, out = run(n, old, led())
check("quiet: forty minutes without a record inside the session raises an alert", got == ["quiet-2026-11-10T09:20:00+09:00"] and out and out[0]["title"] == "NoSock 확인 기록 끊김" and "11-10 09:20 이후 30분 넘게" in out[0]["body"], (got, out))
got = run(n, [row("mark", kst(10, 9, 30), equity=1.0)], led())[1]
check("quiet: exactly thirty minutes is not an alert", got == [], got)
got = run(n, [row("mark", kst(10, 9, 29, 59), equity=1.0)], led())[1]
check("quiet: one second past thirty minutes is an alert", got == ["quiet-2026-11-10T09:29:59+09:00"], got)
got = run(n, old, led(markets={}))[1]
check("quiet: a ledger with no instruments yet raises nothing", got == [], got)
got = run(n, old, led(), beat_age=700)[1]
check("quiet: a silent runner raises the silent alert only", got == [f"sil-{int(n.timestamp() - 700)}"], got)
quiet_ok = {k: run(n, old + [row(k, n - timedelta(minutes=5), code="SPY")], led())[1] for k in ("start", "mark", "order", "dry_order", "order_error", "day_plan")}
check("quiet: a recent start, mark, order, dry order, order error or day plan counts as activity", all(v == [] for v in quiet_ok.values()), quiet_ok)
quiet_no = {k: run(n, old + [row(k, n - timedelta(minutes=5))], led())[1] for k in ("error", "token_renew", "backup", "skip", "stop")}
check("quiet: error, token, backup, skip and stop rows do not count as activity", all(v == ["quiet-2026-11-10T09:20:00+09:00"] for v in quiet_no.values()), quiet_no)
info, got, out = run(n, [row("token_renew", n)], led())
check("quiet: no activity record at all is reported in plain words", got == ["quiet-none"] and out and "기록이 하나도 없습니다" in out[0]["body"] and "None" not in out[0]["body"], (got, out))
info, got, out = run(n, [dict(kind="mark"), dict(kind="mark", ts="broken"), dict(kind="order", ts=None)], led())
check("quiet: rows with a missing or broken time never crash the check", got == ["quiet-none"], got)
eve = [row("mark", kst(9, 15, 30), equity=1.0)]
got = run(kst(10, 9, 20), eve, led(kr_done_for="2026-11-09", checked_day="2026-11-09"))[1]
check("quiet: the first half hour after the open is not judged", got == [], got)
got = run(kst(10, 9, 31), eve, led())[1]
check("quiet: half an hour into the session with no record is an alert", got == ["quiet-2026-11-09T15:30:00+09:00"], got)
got = run(kst(10, 7, 30), eve, led())[1]
check("quiet: outside both sessions nothing is expected", got == [], got)
us_old = [row("mark", kst(11, 2, 20), equity=1.0)]
got = run(kst(11, 3), us_old, led(us_done_for="2026-11-10"))[1]
check("quiet: the US session is watched too", got == ["quiet-2026-11-11T02:20:00+09:00"], got)
got = run(kst(15, 3), [row("mark", kst(14, 15, 30), equity=1.0)], led(checked_day="2026-11-13", kr_done_for="2026-11-13", us_done_for="2026-11-13"))[1]
check("quiet: New York Saturday is not a session", got == [], got)

behind = led(checked_day="2026-11-09", kr_done_for="2026-11-09", us_done_for="2026-11-09")
got = run(kst(10, 9, 29), fresh(kst(10, 9, 29)), behind)[1]
check("missed KR: before 09:30 nothing is raised", got == [], got)
info, got, out = run(kst(10, 9, 30), fresh(kst(10, 9, 30)), behind)
check("missed KR: at 09:30 with no KR step today an alert is raised", got == ["miss-kr-2026-11-10"] and out and out[0]["title"] == "NoSock 국내 주문 안 돎" and "휴장일" in out[0]["body"], (got, out))
got = run(kst(10, 23), fresh(kst(10, 23)), behind)[1]
check("missed KR: the alert keeps one id for the whole day", got == ["miss-kr-2026-11-10"], got)
got = run(kst(10, 9, 31), fresh(kst(10, 9, 31)), led())[1]
check("missed KR: nothing once today's KR step has run", got == [], got)
sat = led(checked_day="2026-11-13", kr_done_for="2026-11-13", us_done_for="2026-11-13")
got = run(kst(14, 10), fresh(kst(14, 10)), sat)[1]
check("missed KR: Saturday is not a trading day", got == [], got)
got = run(kst(10, 9, 31), fresh(kst(10, 9, 31)), dict(markets={"SPY": "US"}))[1]
check("missed KR: no ledger yet (first start) raises nothing", got == [], got)
got = run(kst(10, 9, 31), fresh(kst(10, 9, 31)), None, beat_age=900)[1]
check("missed KR: an unreadable ledger still lets the silent alert through", got == [f"sil-{int(kst(10, 9, 31).timestamp() - 900)}"], got)

got = run(kst(10, 12), fresh(kst(10, 12)), led())[1]
check("missed US: on the KR day itself, before New York reaches that date, nothing is raised", got == [], got)
got = run(kst(11, 0, 14), fresh(kst(11, 0, 14)), led())[1]
check("missed US: forty-four minutes after the US open nothing is raised", got == [], got)
info, got, out = run(kst(11, 0, 15), fresh(kst(11, 0, 15)), led())
check("missed US: forty-five minutes after the US open an alert is raised", got == ["miss-us-2026-11-10"] and out and out[0]["title"] == "NoSock 미국 주문 안 돎" and "2026-11-10 계획" in out[0]["body"], (got, out))
got = run(kst(11, 0, 16), fresh(kst(11, 0, 16)), led(us_done_for="2026-11-10"))[1]
check("missed US: nothing once the US step has run for that plan", got == [], got)
got = run(kst(11, 0, 16), fresh(kst(11, 0, 16)), led(kr_done_for="2026-11-09"))[1]
check("missed US: nothing while the KR step for that plan has not run", got == [], got)
fri = led(checked_day="2026-11-13", kr_done_for="2026-11-13", us_done_for="2026-11-12")
got = run(kst(14, 12), fresh(kst(14, 12)), fri)[1]
check("missed US: a Friday plan is still reported on Saturday morning in Korea", got == ["miss-us-2026-11-13"], got)
got = run(kst(15, 12), fresh(kst(15, 12)), fri)[1]
check("missed US: the weekend itself raises nothing new", got == [], got)
both = run(kst(11, 9, 31), [row("mark", kst(11, 9, 25), equity=1.0)], led())[1]
check("missed KR and US: both are reported together when both are behind", both == ["miss-kr-2026-11-11", "miss-us-2026-11-10"], both)

wipe()
check("files: a missing ledger reads as none", S.live_book() is None)
put(LEDGER, "")
check("files: an empty ledger file reads as none", S.live_book() is None)
put(LEDGER, "[1, 2]")
check("files: a ledger file that is not a ledger reads as none", S.live_book() is None)
check("files: a missing heartbeat reads as none", S.beat_time() is None)
wipe()
L.save_state(led())
L.log("start", note="TEST_FIXTURE")
L.log("mark", phase="tick", equity=10_000.0, note="TEST_FIXTURE")
L.beat()
bt = S.beat_time()
check("files: the web server sees the heartbeat the runner writes", bt is not None and abs(time.time() - bt) < 5 and isinstance(S.live_book(), dict), bt)
st = S.state()
check("state: a fresh heartbeat shows the runner as alive", st["live"]["runner"]["silent"] is False and 0 <= st["live"]["runner"]["age"] < 5 and not [a for a in st["alerts"] if a["id"].startswith("sil-")], st["live"]["runner"])
check("state: the whole answer still turns into JSON", bool(json.dumps(st, ensure_ascii=False)))
os.remove(L.BEAT)
st = S.state()
check("state: with no heartbeat file (old runner, new web server) nothing is raised", st["live"]["runner"] == dict(beat=None, age=None, silent=False) and not [a for a in st["alerts"] if a["id"].startswith("sil-")], st["live"]["runner"])
t_old = age_beat(900)
st = S.state()
sil_alerts = [a for a in st["alerts"] if a["id"].startswith("sil-")]
check("state: a fifteen-minute-old heartbeat shows the runner as silent and raises the alert", st["live"]["runner"]["silent"] is True and 895 <= st["live"]["runner"]["age"] <= 910 and [a["id"] for a in sil_alerts] == [f"sil-{int(t_old)}"] and sil_alerts[0]["title"] == "NoSock 실행기 멈춤", (st["live"]["runner"], sil_alerts))


class Dead:
    def place(self, *a, **k):
        raise ApiError(502, "http", "TEST_FIXTURE bad gateway")


keep_resend, L.RESEND_FOR = L.RESEND_FOR, 0
os.environ["DOBBY_LIVE"] = "1"
try:
    lost = L.place(Dead(), led(), "TESTCYCLE", dict(code="SPY", side="BUY"), "US", 1400.0, amount="5.00")
finally:
    os.environ.pop("DOBBY_LIVE", None)
    L.RESEND_FOR = keep_resend
al = S.alerts(S.live_rows())
hit = [a for a in al if "SPY" in a["body"] and a["title"] == "NoSock 주문 문제"]
check("alerts: an order whose answer was lost reaches the phone with the instruction to check the app", lost.get("kind") == "order_error" and lost.get("lost") is True and len(hit) == 1 and "토스 앱에서 SPY 주문이 들어갔는지 확인해 주세요" in hit[0]["body"], (lost, al))

os.remove(LEDGER)
refused = L.open_ledger()
lr = S.live_rows()
al = S.alerts(lr)
hit = [a for a in al if "새 장부를 만들지 않고 멈춥니다" in a["body"]]
st = S.state()
sil_alerts = [a for a in st["alerts"] if a["id"].startswith("sil-")]
check("alerts: a refused start reaches the phone and the runner shows as switched off", refused is None and len(hit) == 1 and hit[0]["title"] == "NoSock 주문 문제" and st["live"]["last_start"]["kind"] == "stop" and len(sil_alerts) == 1 and "꺼져 있습니다" in sil_alerts[0]["body"], (hit, st["live"]["last_start"], sil_alerts))


class FakeDT(REAL_DT):
    fixed = None

    @classmethod
    def now(cls, tz=None):
        return cls.fixed.astimezone(tz) if tz else cls.fixed


sent = []
keep_push = S.send_push
check("push: with web push switched off the real sender does nothing", keep_push(dict(id="TEST_FIXTURE", title="TEST_FIXTURE", body="TEST_FIXTURE")) == 0)
S.send_push = lambda p: (sent.append(p), 1)[1]
S.datetime = FakeDT
try:
    wipe()
    n = kst(10, 10)
    FakeDT.fixed = n
    put_rows([row("start", kst(10, 8)), row("mark", kst(10, 9, 55), equity=1.0), row("fx_request", kst(10, 9, 7), id="fx-TEST_FIXTURE", direction="KRW->USD", amount_won=10000, amount_usd=7.0, fx_mid=1400.0)])
    L.save_state(behind)
    t_old = age_beat(900, n.timestamp())
    S.watch_once()
    first = [p["id"] for p in sent]
    pushed = json.load(open(os.path.join(web, "pushed.json"), encoding="utf-8"))
    check("push: an exchange request, a silent runner and a missed KR step are each pushed once", first == ["fx-TEST_FIXTURE", f"sil-{int(t_old)}", "miss-kr-2026-11-10"] and sorted(pushed) == sorted(first) and sent[1]["title"] == "NoSock 실행기 멈춤" and sent[1]["url"] == "/nosock/", (first, pushed))
    S.watch_once()
    S.watch_once()
    check("push: the same alerts are never pushed twice", [p["id"] for p in sent] == first, [p["id"] for p in sent])
    FakeDT.fixed = kst(10, 10, 40)
    age_beat(10, FakeDT.fixed.timestamp())
    S.watch_once()
    S.watch_once()
    check("push: when the runner comes back but records nothing, that is pushed once too", [p["id"] for p in sent] == first + ["quiet-2026-11-10T09:55:00+09:00"], [p["id"] for p in sent])
    put(LEDGER, "")
    put(LOG, "not json\n" + json.dumps(row("mark", kst(10, 10, 35), equity=1.0)) + "\n{broken\n")
    try:
        S.watch_once()
        crash = None
    except Exception as e:
        crash = e
    check("push: a broken ledger or broken log lines never crash the watcher", crash is None and [p["id"] for p in sent] == first + ["quiet-2026-11-10T09:55:00+09:00"], (crash, [p["id"] for p in sent]))
finally:
    S.datetime = REAL_DT
    S.send_push = keep_push

real_after = sorted(x for x in os.listdir(REAL_RUNS) if x.startswith(("live_nosock", "backup"))) if os.path.isdir(REAL_RUNS) else []
check("fixture: nothing was written into the real runs folder", real_after == REAL_BEFORE and not os.path.exists(os.path.join(REAL_RUNS, "live_nosock.beat")), (REAL_BEFORE, real_after))
shutil.rmtree(tmp, ignore_errors=True)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
