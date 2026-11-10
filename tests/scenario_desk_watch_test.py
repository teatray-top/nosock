import json
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp(prefix="desk_watch_test_")
os.environ["NOSOCK_SCENARIO_LOG"] = os.path.join(tmp, "desk.jsonl")
os.environ["NOSOCK_SCENARIO_WORK"] = os.path.join(tmp, "work")
os.environ["NOSOCK_RUNS"] = tmp
os.environ.pop("NOSOCK_DROPBOX", None)
sys.path.insert(0, os.path.dirname(HERE))
import scenario_desk as D

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


START = 1_000_000.0
clock = [START]
limit = [0.0]
base = [datetime(2026, 11, 8, 21, 44, tzinfo=D.KST)]


class FakeTime:
    @staticmethod
    def time():
        return clock[0]

    @staticmethod
    def sleep(s):
        clock[0] += s
        if clock[0] > START + limit[0]:
            open(D.STOP, "w").close()


def rows():
    return [json.loads(x) for x in open(os.environ["NOSOCK_SCENARIO_LOG"], encoding="utf-8")]


def reset(hours, start_at):
    open(os.environ["NOSOCK_SCENARIO_LOG"], "w").close()
    if os.path.exists(D.STOP):
        os.remove(D.STOP)
    clock[0] = START
    limit[0] = hours * 3600
    base[0] = start_at


scored = []
fake = types.ModuleType("scorecard")
fake.main = lambda: scored.append(D.now().strftime("%H:%M"))
sys.modules["scorecard"] = fake
D.time = FakeTime
D.dropbox_cycle = lambda st: None
D.pull = lambda: 0
D.now = lambda: base[0] + timedelta(seconds=clock[0] - START)

runs = []


def fail_run(trigger="morning"):
    runs.append((clock[0] - START, trigger))
    raise RuntimeError("cached token missing or expiring; not issuing")


D.run = fail_run
reset(4, datetime(2026, 11, 8, 21, 44, tzinfo=D.KST))
open(D.URGENT, "w").write("dropbox-test")
D.watch()
errs = [r for r in rows() if r.get("kind") == "scenario_error"]
check("failed urgent run retried 3 times, 30+ min apart", len(runs) == 3 and all(t == "urgent:dropbox-test" for _, t in runs) and runs[1][0] - runs[0][0] >= 1800, runs)
check("error row keeps the reason", errs and errs[0].get("detail", "").startswith("cached token"), errs)
check("gives up after 3 attempts", errs[-1].get("retry_in_min") is None and not os.path.exists(D.URGENT), errs[-1])

D.run = lambda trigger="morning": None
reset(11, datetime(2026, 11, 8, 6, 50, tzinfo=D.KST))
D.watch()
check("scorecard runs once after US close and once after KR close", scored == ["07:10", "16:10"], scored)
check("scorecard runs logged", [r["slot"] for r in rows() if r.get("kind") == "scorecard"] == ["after_us_close", "after_kr_close"])


def boom():
    raise ValueError("price fetch failed")


fake.main = boom
reset(1, datetime(2026, 11, 8, 7, 5, tzinfo=D.KST))
D.watch()
e = [r for r in rows() if r.get("where") == "scorecard"]
check("scorecard failure logged, loop keeps going", len(e) == 1 and e[0]["detail"] == "price fetch failed", e)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
