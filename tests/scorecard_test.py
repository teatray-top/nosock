import json
import os
import sys
import tempfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scorecard as S

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


def near(a, b, eps=1e-9):
    return abs(a - b) < eps


def kst(s):
    return datetime.fromisoformat(s + "+09:00")


s = {
    "A": dict(market="KR", days=["2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"], po=[100, 100, 102, 104], pc=[100, 100, 104, 103],
              ko=[100, 100, 102, 104], kc=[100, 100, 104, 103], cost=[0.001] * 4),
    "U": dict(market="US", days=["2026-11-05", "2026-11-06", "2026-11-09"], po=[50, 50, 51], pc=[50, 50, 52],
              ko=[50000, 50000, 51000], kc=[50000, 50000, 52000], cost=[0.001] * 3),
}


def call(code, direction, at, horizon=1, author="dot"):
    return dict(batch="b", at=kst(at), author=author, title="t", urgency="high", event_id=None, signal=None, code=code, direction=direction, magnitude=None, horizon=horizon)


r = S.score_call(call("A", "up", "2026-11-08T13:00"), s)
check("weekend KR call enters Monday open, not Friday close", r["entry_day"] == "2026-11-09" and near(r["ret"], 104 / 102 - 1), r)
check("1-day call matured and hit", r["matured"] and r["hit"] and r["days"] == 1, r)
check("net = called return minus two one-way costs", near(r["net"], 104 / 102 - 1 - 0.002), r)
r = S.score_call(call("A", "down", "2026-11-09T10:00", horizon=2), s)
check("call after KR open enters next day open", r["entry_day"] == "2026-11-10", r)
check("2-day call with 1 day of data -> open, interim scored", not r["matured"] and r["days"] == 1 and r["hit"] and near(r["ret"], 103 / 104 - 1), r)
r = S.score_call(call("U", "up", "2026-11-09T21:44"), s)
check("US call at 21:44 KST enters that night's US open", r["entry_day"] == "2026-11-09" and r["matured"] and r["hit"] and near(r["ret"], 52 / 51 - 1), r)
check("mixed direction not scored", S.score_call(call("A", "mixed", "2026-11-08T13:00"), s) is None)
check("unknown code not scored", S.score_call(call("ZZZ", "up", "2026-11-08T13:00"), s) is None)
r = S.score_call(call("A", "up", "2026-11-10T10:00"), s)
check("no session after the call yet -> waiting", r["entry_day"] is None and not r["matured"] and r["days"] == 0, r)

rows = [dict(at=kst("2026-11-08T22:00"), tilt={"A": 0.1}, trigger="m"), dict(at=kst("2026-11-09T12:00"), tilt={}, trigger="u")]
daily, tot = S.desk_pnl(rows, s)
g1 = 0.1 * (104 / 102 - 1)
check("desk tilt earns only from the open after it was made (no gap credit)", near(daily[0]["gross"], g1) and daily[0]["day"] == "2026-11-09", daily)
check("tilt removed intraday is closed at next open with cost", len(daily) == 2 and near(daily[1]["gross"], 0.0) and near(daily[1]["cost"], 0.1 * 0.001), daily)
check("desk total net = gross - entry and exit costs", near(tot["net"], g1 - 0.0002), tot)

rows2 = [dict(at=kst("2026-11-05T20:00"), tilt={"A": 0.1}, trigger="m")]
d2 = {d["day"]: d for d in S.desk_pnl(rows2, s)[0]}
check("tilt already held over the weekend earns the gap", near(d2["2026-11-09"]["gross"], 0.1 * (102 / 100 - 1) + 0.1 * (104 / 102 - 1)) and near(d2["2026-11-09"]["cost"], 0.0), d2)
rows3 = [dict(at=kst("2026-11-06T20:00"), tilt={"A": 0.1}, trigger="m")]
d3 = {d["day"]: d for d in S.desk_pnl(rows3, s)[0]}
check("tilt made Friday evening enters Monday open, no weekend gap", near(d3["2026-11-09"]["gross"], 0.1 * (104 / 102 - 1)), d3)


class FakeCost:
    def __call__(self, k, p):
        return 0.002


inst = {"A": dict(market="KR", days=["2026-11-09", "2026-11-10"], O=[1.0, 2.0], C=[1.0, 2.0], fxv=[1.0, 1.0], cost=FakeCost()),
        "U": dict(market="US", days=["2026-11-06", "2026-11-09"], O=[1400.0, 1420.0], C=[1400.0, 1430.0], fxv=[1400.0, 1400.0], cost=FakeCost())}
ser = S.series_from_inst(inst, kst("2026-11-10T12:00"))
check("today's KR bar before 15:30 is not a close -> dropped", ser["A"]["days"] == ["2026-11-09"], ser["A"])
check("US bar closed at 06:00 KST kept, price back in dollars", ser["U"]["days"] == ["2026-11-06", "2026-11-09"] and near(ser["U"]["pc"][1], 1430 / 1400), ser["U"])

tmp = tempfile.mkdtemp(prefix="scorecard_test_")
idir = os.path.join(tmp, "issues")
os.makedirs(idir)
check("server timestamp +0000 parsed (python 3.9)", S.parse_ts("2026-11-08T13:04:58+0000") == kst("2026-11-08T22:04:58"))
check("Z and +09:00 timestamps parsed", S.parse_ts("2026-11-08T04:00:00Z") == kst("2026-11-08T13:00") and S.parse_ts("2026-11-08T13:00:00+09:00") == kst("2026-11-08T13:00"))
batch = dict(id="x1", kind="issues", received_at="2026-11-08T04:00:00+0000", source="dot", payload=dict(author="dot", issues=[
    dict(title="a", affected={"A": dict(direction="up", horizon_days=1), "U": dict(direction="mixed", horizon_days=5)}),
    dict(title="b", affected={"A": dict(direction="down", horizon_days="2")})]))
json.dump(batch, open(os.path.join(idir, "x1.json"), "w", encoding="utf-8"))
open(os.path.join(idir, "bad.json"), "w").write("{")
dpath = os.path.join(tmp, "desk.jsonl")
with open(dpath, "w", encoding="utf-8") as f:
    f.write(json.dumps(dict(ts="2026-11-08T22:00:00+09:00", kind="scenario", opt_ok=True, tilt={"A": 0.1})) + "\n")
    f.write(json.dumps(dict(ts="2026-11-09T21:44:10+09:00", kind="scenario_error", where="urgent_run")) + "\n")
sc = S.build(s, kst("2026-11-11T08:00"), idir, dpath)
by = sc["calls"]["by_author"]["dot"]
check("build: 3 calls loaded, broken file skipped, mixed not scored", sc["calls"]["total"] == 3 and sc["calls"]["scored"] == 2, sc["calls"])
check("build: 1 up hit matured, 1 down call matured as miss", by["matured"] == 2 and by["hits"] == 1, by)
check("build: per-author count includes unscored calls (3 made, 2 scored)", by["all"] == 3 and by["calls"] == 2, by)
check("build: desk errors ignored, 1 scenario row", sc["desk"]["rows"] == 1, sc["desk"])
drow = [dict(at=kst("2026-11-06T08:50"), trigger="morning", tilt={}, probs=[0.6, 0.4], scen=[dict(returns_pct={"A": 10, "U": -2}), dict(returns_pct={"A": -5, "U": 3})]),
        dict(at=kst("2026-11-06T12:00"), trigger="urgent:x", tilt={}, probs=[1.0], scen=[dict(returns_pct={"A": -50})])]
dc = S.desk_calls(drow)
check("desk forecast: E[r] = sum p*r -> A up 4%, U 0% skipped, urgent row ignored", len(dc) == 1 and dc[0]["code"] == "A" and dc[0]["direction"] == "up" and near(dc[0]["magnitude"], 4.0) and dc[0]["horizon"] == 20, dc)
dc[0]["horizon"] = 2
it, by2 = S.summarize(dc, s)
check("desk forecast scored with magnitude error", by2["desk"]["matured"] == 1 and near(by2["desk"]["mag_bias"], (104 / 100 - 1) - 0.04), by2)
txt = S.report(sc)
check("report prints after-cost desk line", "비용 후" in txt and "dot" in txt, txt)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
