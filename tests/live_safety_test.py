import http.client
import json
import os
import shutil
import sys
import tempfile
import time as real_time
import urllib.error
from datetime import date, datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
tmp = tempfile.mkdtemp(prefix="TEST_FIXTURE_nosock_safety_")
os.environ.update(NOSOCK_RUNS=tmp, NOSOCK_LOG=os.path.join(tmp, "live_nosock.jsonl"), NOSOCK_STATE=os.path.join(tmp, "live_nosock_state.json"), NOSOCK_SCENARIO_LOG=os.path.join(tmp, "scenario_desk.jsonl"))
for k in ("NOSOCK_BACKUP", "DOBBY_LIVE", "NOSOCK_SEED"):
    os.environ.pop(k, None)
import toss
from toss import ApiError


class NoToss:
    def __init__(self, *a, **k):
        raise AssertionError("TEST_FIXTURE: tests never build a real Toss client")


toss.Toss = NoToss
import live_nosock as L

for p in (L.RUNS, L.LOG, L.STATE, L.STOP, L.MARK_FLAG, L.FX_FLAG, L.BEAT, L.BACKUP, L.SCEN_LOG):
    assert os.path.abspath(p).startswith(os.path.abspath(tmp)), p
REAL_RUNS = os.path.join(ROOT, "runs")
REAL_BEFORE = sorted(x for x in os.listdir(REAL_RUNS) if x.startswith(("live_nosock", "backup"))) if os.path.isdir(REAL_RUNS) else []
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


def rows():
    if not os.path.exists(L.LOG):
        return []
    return [json.loads(x) for x in open(L.LOG, encoding="utf-8")]


def put(p, text):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def reset():
    for p in (L.STATE, L.STATE + ".prev", L.STATE + ".tmp", L.LOG, L.BEAT, L.STOP):
        if os.path.exists(p):
            os.remove(p)
    if os.path.isfile(L.BACKUP):
        os.remove(L.BACKUP)
    shutil.rmtree(L.BACKUP, ignore_errors=True)
    L.BEAT_AT[0] = 0.0


def book(usd=0.0, krw=0.0):
    return dict(krw=krw, usd=usd, qty={}, cost={}, cost_krw={}, vault_krw=0.0, vault_usd=0.0, tax={}, tax_usd=0.0, realized=0.0, hwm=0.0, deposits=0.0, contributed=0.0, pending={})


def history():
    L.log("mark", phase="view", equity=10_000.0, note="TEST_FIXTURE")


def opened():
    n0 = len(rows())
    st = L.open_ledger()
    return st, rows()[n0:]


def kst(h, m, d=10):
    return datetime(2026, 11, d, h, m, tzinfo=L.KST)


reset()
s1 = dict(book=book(krw=10_000.0), current="c1")
L.save_state(s1)
check("save: first save writes the ledger, no temp file left, no .prev yet", json.loads(read(L.STATE)) == s1 and not os.path.exists(L.STATE + ".tmp") and not os.path.exists(L.STATE + ".prev"))
first = read(L.STATE)
s2 = dict(book=book(krw=7_000.0), current="c1")
L.save_state(s2)
check("save: second save keeps the previous ledger as .prev", os.path.exists(L.STATE + ".prev") and read(L.STATE + ".prev") == first and json.loads(read(L.STATE)) == s2)
check("load: a good ledger loads back unchanged", L.load_state() == s2)
for label, bad in (("0-byte", ""), ("cut-off", first[: len(first) // 2]), ("not-a-ledger", "[1, 2]")):
    put(L.STATE, bad)
    try:
        L.load_state()
        got = None
    except L.LedgerError as e:
        got = str(e)
    check(f"load: {label} ledger file is refused instead of being read as empty", got is not None, got)
put(L.STATE, "")
L.save_state(dict(book=book(krw=5_000.0)))
check("save: a broken ledger file never replaces the good .prev copy", os.path.exists(L.STATE + ".prev") and read(L.STATE + ".prev") == first and L.load_state()["book"]["krw"] == 5_000.0)
os.remove(L.STATE)
check("load: no ledger file at all reads as empty (first start)", L.load_state() == {})

reset()
st, new = opened()
check("start: true first start (no ledger, no history) is allowed", st == {} and not new, (st, new))
L.log("start", note="TEST_FIXTURE")
st, new = opened()
check("start: no ledger and a log without any balance check is still a first start", st == {} and not new, (st, new))

reset()
put(L.STATE, "")
st, new = opened()
check("start: 0-byte ledger -> refuse + stop rows, reason says how to recover", st is None and [r["kind"] for r in new] == ["refuse", "stop"] and "읽지 못했습니다" in new[0]["reason"] and ".prev" in new[0]["reason"] and "backup" in new[0]["reason"], new)
check("start: the broken ledger file is left untouched", os.path.getsize(L.STATE) == 0 and not os.path.exists(L.STATE + ".tmp"))

reset()
history()
st, new = opened()
check("start: ledger file gone but trade history exists -> refuse, no new book", st is None and [r["kind"] for r in new] == ["refuse", "stop"] and not os.path.exists(L.STATE), new)
put(L.STATE, "{}")
st, new = opened()
check("start: empty ledger {} with trade history -> refuse", st is None and new and new[0]["kind"] == "refuse", new)

reset()
os.makedirs(L.BACKUP)
put(os.path.join(L.BACKUP, "live_nosock_state.2026-11-09.json"), json.dumps(dict(book=book())))
put(L.STATE, json.dumps(dict(current="c1")))
st, new = opened()
check("start: ledger without a book while a backup exists -> refuse", st is None and new and new[0]["kind"] == "refuse", new)

reset()
good = dict(book=book(krw=1.0), current="c1")
put(L.STATE, json.dumps(good))
history()
st, new = opened()
check("start: normal restart (ledger + history) loads the ledger", st == good and not new, (st, new))


class StubToss:
    built = 0

    def __init__(self):
        StubToss.built += 1
        self.tok, self.exp, self.account = "TEST_FIXTURE", real_time.time() + 36000, None

    def accounts(self):
        return [dict(accountSeq=1)]

    def _load_token(self):
        pass

    def _issue(self):
        raise AssertionError("TEST_FIXTURE: no token issue in tests")


toss.Toss = StubToss
steps = []


def fake_step(t, st, n, ms):
    steps.append(n)
    put(L.STOP, "TEST_FIXTURE")


def one_pass():
    keep = L.step, L.nap, L.now
    L.step, L.nap, L.now = fake_step, (lambda sec: None), (lambda: kst(22, 0))
    try:
        L.loop()
    finally:
        L.step, L.nap, L.now = keep
    return [r["kind"] for r in rows()]


reset()
put(L.STATE, "")
kinds = one_pass()
check("loop: a refused start returns before any Toss client is built", StubToss.built == 0 and not steps and kinds == ["refuse", "stop"], (StubToss.built, kinds))
reset()
history()
kinds = one_pass()
check("loop: a missing ledger with trade history also stops before Toss", StubToss.built == 0 and not steps and kinds == ["mark", "refuse", "stop"], (StubToss.built, kinds))

reset()
put(L.STATE, json.dumps(good))
kinds = one_pass()
check("loop: one pass writes the heartbeat, takes the daily backup, then stops on the stop file", StubToss.built == 1 and len(steps) == 1 and kinds == ["start", "backup", "stop"] and os.path.exists(L.BEAT) and os.path.exists(os.path.join(L.BACKUP, "live_nosock_state.2026-11-10.json")), (StubToss.built, kinds))
toss.Toss = NoToss

reset()
st = dict(book=book(krw=3.0), current="c1")
history()
before = read(L.LOG)
row = L.backup(st, "2026-11-10")
bp = os.path.join(L.BACKUP, "live_nosock_state.2026-11-10.json")
lb = os.path.join(L.BACKUP, os.path.basename(L.LOG))
check("backup: dated ledger copy and a copy of the trade log are written", json.loads(read(bp)) == st and read(lb) == before and row["kind"] == "backup" and not [x for x in os.listdir(L.BACKUP) if x.endswith(".tmp")], os.listdir(L.BACKUP))
st2 = dict(book=book(krw=99.0))
check("backup: a second call the same day changes nothing", L.backup(st2, "2026-11-10") is None and json.loads(read(bp)) == st)
for i in range(35):
    L.backup(st, (date(2026, 12, 1) + timedelta(days=i)).isoformat())
kept = sorted(x for x in os.listdir(L.BACKUP) if x.startswith("live_nosock_state."))
check("backup: only the newest 30 daily ledger copies are kept", len(kept) == L.BACKUP_KEEP == 30 and kept[0] == "live_nosock_state.2026-12-06.json" and kept[-1] == "live_nosock_state.2027-01-04.json", (len(kept), kept[:1], kept[-1:]))
check("backup: the log copy follows the growing log", read(L.LOG).startswith(read(lb)) and len(read(lb)) > len(before), (len(read(lb)), len(before)))
size = os.path.getsize(lb)
put(L.LOG, "")
L.backup(st, "2027-02-01")
r = rows()
check("backup: a shrunken trade log never overwrites the bigger copy, and an error row says so", os.path.getsize(lb) == size and any(x["kind"] == "error" and x.get("where") == "backup" and "작아졌습니다" in x["message"] for x in r), r)

reset()
ms = {}
L.daily_backup({}, ms, kst(22, 0))
check("daily backup: nothing before a book exists", not os.path.isdir(L.BACKUP) and not ms)
history()
L.daily_backup(st, ms, kst(9, 6))
check("daily backup: waits while orders are about to go out (09:06 KST)", not os.path.isdir(L.BACKUP) and not ms)
L.daily_backup(st, ms, kst(23, 30))
check("daily backup: waits at the US open (09:30 New York)", not os.path.isdir(L.BACKUP) and not ms)
L.daily_backup(st, ms, kst(9, 20))
n0 = len(rows())
L.daily_backup(st2, ms, kst(15, 0))
check("daily backup: taken once a day", json.loads(read(os.path.join(L.BACKUP, "live_nosock_state.2026-11-10.json"))) == st and ms["backup_try"] == "2026-11-10" and len(rows()) == n0, ms)
L.daily_backup(st2, ms, kst(0, 1, d=11))
check("daily backup: next day gets its own copy", json.loads(read(os.path.join(L.BACKUP, "live_nosock_state.2026-11-11.json"))) == st2)

reset()
put(L.BACKUP, "TEST_FIXTURE a file where the backup folder should be")
history()
ms = {}
try:
    L.daily_backup(st, ms, kst(22, 0))
    raised = None
except Exception as e:
    raised = e
r = rows()
check("daily backup: a failure is logged and never stops the runner", raised is None and r[-1]["kind"] == "error" and r[-1]["where"] == "backup" and "장부 백업 실패" in r[-1]["message"] and ms.get("backup_try") == "2026-11-10", (raised, r[-1]))


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t0 = self.t = t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def __getattr__(self, k):
        return getattr(real_time, k)


class Broker:
    def __init__(self, clock, script=(), default="ok", pending_polls=0):
        self.clock, self.script, self.default, self.pending_polls = clock, list(script), default, pending_polls
        self.made, self.calls, self.by_cid, self.polls = [], [], {}, 0

    def accept(self, cid, body):
        hit = self.by_cid.get(cid)
        if hit and self.clock.t - hit["at"] <= 600:
            if hit["body"] != body:
                raise ApiError(422, "idempotency-key-conflict", "TEST_FIXTURE")
            return hit["oid"]
        oid = f"TEST_FIXTURE_o{len(self.made) + 1}"
        self.made.append(dict(oid=oid, cid=cid, body=body))
        self.by_cid[cid] = dict(oid=oid, body=body, at=self.clock.t)
        return oid

    def place(self, symbol, side, order_type, client_order_id=None, quantity=None, amount=None):
        body = dict(symbol=symbol, side=side, order_type=order_type, quantity=quantity, amount=amount)
        self.calls.append(dict(cid=client_order_id, body=body, at=self.clock.t))
        act = self.script.pop(0) if self.script else self.default
        if act == "lost":
            self.accept(client_order_id, body)
            self.clock.t += 20
            raise TimeoutError("TEST_FIXTURE response lost")
        if act == "unreachable":
            self.clock.t += 20
            raise urllib.error.URLError("TEST_FIXTURE unreachable")
        if isinstance(act, BaseException):
            raise act
        return dict(orderId=self.accept(client_order_id, body))

    def order(self, oid):
        self.polls += 1
        m = next(x for x in self.made if x["oid"] == oid)
        amt = float(m["body"]["amount"])
        if self.polls <= self.pending_polls:
            return dict(status="PENDING", execution={})
        return dict(status="FILLED", execution=dict(filledQuantity=str(amt / 500.0), averageFilledPrice="500", filledAmount=str(amt), commission="0.01", tax="0"))


def order_case(script=(), default="ok", live=True, pending_polls=0):
    reset()
    ck = Clock()
    br = Broker(ck, script, default, pending_polls)
    stx = dict(book=book(usd=20.0), current="c1")
    L.time = ck
    if live:
        os.environ["DOBBY_LIVE"] = "1"
    try:
        row = L.place(br, stx, "c1", dict(code="SPY", side="BUY", px=500.0), "US", 1400.0, amount="5.00")
    finally:
        os.environ.pop("DOBBY_LIVE", None)
        L.time = real_time
    return row, br, stx, ck


def same_request(br):
    return len({c["cid"] for c in br.calls}) == 1 and all(c["body"] == br.calls[0]["body"] for c in br.calls)


def filled_once(stx):
    b = stx["book"]
    return abs(b["qty"].get("SPY", 0.0) - 0.01) < 1e-12 and abs(b["usd"] - 14.99) < 1e-9


def untouched(stx):
    return stx["book"]["qty"] == {} and stx["book"]["usd"] == 20.0 and not os.path.exists(L.STATE)


row, br, stx, ck = order_case()
check("order: normal case is one request, no resend", row["kind"] == "order" and row["resent"] == 0 and len(br.calls) == 1 and len(br.made) == 1 and filled_once(stx), row)
check("order: sending an order writes the heartbeat", os.path.exists(L.BEAT))

row, br, stx, ck = order_case(["lost"])
check("resend: response lost after Toss took the order -> same id and body sent again, one order, booked once", row["kind"] == "order" and row["resent"] == 1 and len(br.calls) == 2 and same_request(br) and len(br.made) == 1 and filled_once(stx) and json.loads(read(L.STATE)) == stx, (row, br.calls, br.made))

row, br, stx, ck = order_case(["lost", ApiError(409, "request-in-progress", "TEST_FIXTURE 처리 중")])
check("resend: 409 still-processing answer is waited out, one order", row["kind"] == "order" and row["resent"] == 2 and len(br.calls) == 3 and same_request(br) and len(br.made) == 1 and filled_once(stx), (row, br.calls))

row, br, stx, ck = order_case(["unreachable", ApiError(503, "http", "TEST_FIXTURE")])
check("resend: request that never arrived, then a 503 -> third try places exactly one order", row["kind"] == "order" and row["resent"] == 2 and len(br.made) == 1 and same_request(br) and filled_once(stx), (row, br.calls))

row, br, stx, ck = order_case(default=ApiError(502, "http", "TEST_FIXTURE bad gateway"))
span = ck.t - ck.t0
check("resend: endless 502 -> gives up after about two minutes with a clear alert text", row["kind"] == "order_error" and row["lost"] is True and row["resent"] >= 1 and row["status"] == 502 and "토스 앱에서 SPY" in (row.get("message") or "") and f"{row['resent']}번" in (row.get("message") or "") and L.RESEND_FOR <= span <= L.RESEND_FOR + 35, (row, span))
check("resend: nothing is booked when the result is unknown", untouched(stx) and len(br.made) == 0 and same_request(br), stx["book"])

row, br, stx, ck = order_case(default="lost")
last = max(c["at"] for c in br.calls) - ck.t0
check("resend: Toss took the order but no answer ever comes back -> still exactly one order at the broker", row["kind"] == "order_error" and row["lost"] is True and row["error"] == "TimeoutError" and len(br.made) == 1 and len(br.calls) >= 3 and same_request(br), (row, len(br.calls), br.made))
check("resend: every resend goes out well inside Toss's 10-minute same-id window", last < 300 and ck.t - ck.t0 < 300, (last, ck.t - ck.t0))
check("resend: that unknown fill is not booked and the alert asks to check the app", untouched(stx) and "확인해 주세요" in (row.get("message") or ""), stx["book"])
ck.t = ck.t0 + 601 + last
late = br.accept(br.calls[0]["cid"], br.calls[0]["body"])
check("fixture: the fake broker models the 10-minute rule (same id later = a second order)", len(br.made) == 2 and late != br.made[0]["oid"])

row, br, stx, ck = order_case(default=ApiError(422, "prerequisite-required", "TEST_FIXTURE 위험 고지 동의가 필요합니다", "rq1"))
check("resend: a 422 refusal is final, sent once, Toss's reason kept", row["kind"] == "order_error" and row["lost"] is False and row["resent"] == 0 and len(br.calls) == 1 and "위험 고지" in row["message"] and row["request_id"] == "rq1" and row["code_detail"] == "prerequisite-required", row)

row, br, stx, ck = order_case(default=ApiError(409, "already-filled", "TEST_FIXTURE"))
check("resend: other 409 answers are not resent", row["kind"] == "order_error" and row["lost"] is False and len(br.calls) == 1, row)

row, br, stx, ck = order_case(default=PermissionError("TEST_FIXTURE live orders disabled"))
check("resend: the live-order switch refusal is not resent", row["kind"] == "order_error" and row["lost"] is False and len(br.calls) == 1, row)

row, br, stx, ck = order_case(live=False)
check("dry run: no request leaves, a dry_order row is logged", row["kind"] == "dry_order" and len(br.calls) == 0 and untouched(stx), row)

yes = [TimeoutError("x"), urllib.error.URLError("x"), ConnectionResetError("x"), http.client.RemoteDisconnected("x"), http.client.IncompleteRead(b""), json.JSONDecodeError("x", "", 0)] + [ApiError(s, "http", "x") for s in (408, 429, 500, 502, 503, 504)] + [ApiError(409, "request-in-progress", "x")]
no = [PermissionError("x"), KeyError("result"), TypeError("x")] + [ApiError(s, "http", "x") for s in (400, 401, 403, 404, 422)] + [ApiError(409, c, "x") for c in ("already-filled", "already-canceled", "already-rejected", "opposite-pending-order-exists")]
check("resend: only no-answer cases count as unknown", all(L.unsure(e) for e in yes) and not any(L.unsure(e) for e in no), ([type(e).__name__ for e in yes if not L.unsure(e)], [str(e) for e in no if L.unsure(e)]))

row, br, stx, ck = order_case(pending_polls=3)
check("order: slow fill is waited for and booked once", row["kind"] == "order" and row["status"] == "FILLED" and br.polls == 4 and filled_once(stx), (row, br.polls))

reset()
ck = Clock()
br = Broker(ck, pending_polls=2)
oid = br.accept("TEST_FIXTURE_cid", dict(symbol="SPY", side="BUY", order_type="MARKET", quantity=None, amount="5.00"))
L.time = ck
try:
    L.wait_order(br, oid)
    waited = os.path.exists(L.BEAT)
    reset()
    L.beat()
    a = os.path.exists(L.BEAT)
    stamp = read(L.BEAT)
    os.remove(L.BEAT)
    ck.t += 3
    L.beat()
    b = os.path.exists(L.BEAT)
    ck.t += 3
    L.beat()
    c = os.path.exists(L.BEAT)
    keep_beat, L.BEAT = L.BEAT, os.path.join(tmp, "TEST_FIXTURE_missing_folder", "x.beat")
    ck.t += 10
    try:
        L.beat()
        beat_err = None
    except Exception as e:
        beat_err = e
    L.BEAT = keep_beat
finally:
    L.time = real_time
check("beat: waiting for a fill keeps the heartbeat alive", waited)
check("beat: heartbeat file holds a time and is written at most once per 5 s", a and not b and c and datetime.fromisoformat(stamp).tzinfo is not None, (a, b, c, stamp))
check("beat: an unwritable heartbeat path never raises", beat_err is None, beat_err)

real_after = sorted(x for x in os.listdir(REAL_RUNS) if x.startswith(("live_nosock", "backup"))) if os.path.isdir(REAL_RUNS) else []
check("fixture: nothing was written into the real runs folder", real_after == REAL_BEFORE and not os.path.exists(os.path.join(REAL_RUNS, "live_nosock.beat")), (REAL_BEFORE, real_after))
shutil.rmtree(tmp, ignore_errors=True)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
