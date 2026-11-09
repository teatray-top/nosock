import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timedelta

import pandas as pd

import f1_fill_schedule as F
import fill_sensitivity as FS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPEC = os.path.join(HERE, "results", "conservative_fill_v1", "f1_spec_draft_v3.json")
SPEC_SHA = "59f3db14fc1e39eaa75255700a89416f6469a616e2bfd46e4df07537a8a07fda"
CAL = os.path.join(HERE, "session_calendar_kr_etf_f1_v1.json")
CAL_SHA = "aa12305513f739387e82fb31a586f034a1f798a4c81a47ed529dadec1a0d62af"
MIN_ROOT = os.path.join(ROOT, "data", "kr_etf", "forward_1m")
NOV6 = "20261107T082200_minute_2026-11-06"
OUT_DIR = os.path.join(HERE, "results", "conservative_fill_v1", "f1_fixtures_v3")
OUT = os.path.join(OUT_DIR, "fixtures_result.json")
CAPITAL = 100_000_000
FORBIDDEN = ("toss", "cred", "freqtrade", "forward_collect_kr_etf", "forward_history_kr_etf", "forward_runner_kr_etf", "backfill_kr_etf_1m", "dobby_kr", "fills_source", "sensitivity_apply")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def entry(first, last, cls="synthetic"):
    hm = lambda s, d: (datetime.strptime(s, "%H:%M") + timedelta(minutes=d)).strftime("%H:%M")
    return dict(cls=cls, first=first, last=last, session_start=hm(first, -1), continuous_end=hm(last, -10))


D1, D2, PREV, LATE, SHIFT = "2030-01-07", "2030-01-08", "2029-12-31", "2030-01-02", "2030-11-14"
SYN_CAL = {D1: entry("09:01", "15:30"), D2: entry("09:01", "15:30")}
LATE_CAL = {PREV: entry("09:01", "15:30"), LATE: entry("10:01", "15:30", "late_open_synthetic")}
SHIFT_CAL = {SHIFT: entry("10:01", "16:30", "shifted_session_synthetic")}
BASE = {D1: 10000, D2: 20000, PREV: 50000, LATE: 30000, SHIFT: 40000}


def lab(day, hhmm):
    return F.at(day, hhmm)


def px(day, cal, hhmm):
    return BASE[day] + int((lab(day, hhmm) - lab(day, cal[day]["first"])).total_seconds() // 60)


def gen(day, cal, zero=(), drop=(), dup=None, conflict=None, invalid=None):
    rows, t = [], lab(day, cal[day]["first"])
    while t <= lab(day, cal[day]["last"]):
        hh = t.strftime("%H:%M")
        o = px(day, cal, hh)
        r = dict(label=t, open=o, high=o + 5, low=o - 5, close=o + 1, volume=0 if hh in zero else 1000)
        if hh == invalid:
            r = dict(r, high=o - 1)
        if hh not in drop:
            rows.append(r)
        if hh == dup:
            rows.append(dict(r))
        if hh == conflict:
            rows.append(dict(r, volume=r["volume"] + 1))
        t += F.MIN
    return rows


def order(i, side, shares, day, hhmm, sec=0, tag=None, force=False, price=None):
    t = lab(day, hhmm) + timedelta(seconds=sec)
    return dict(order_id=f"fx:{i}", source_index=i, side=side, shares=shares, activation=t, price=price, tag=tag, is_force_exit=force)


def summary(res):
    if res["status"] != "ok":
        return dict(status=res["status"], problems=res["problems"], fills=0, pnl=res["pnl"])
    return dict(status="ok", orders={r["order_id"]: {k: r[k] for k in ("status", "requested", "filled", "cancelled", "expired", "rejected", "carried_unfilled_at_end", "expired_at")} for r in res["orders"]}, fills=[dict(order_id=f["order_id"], label=f["interval_label"], model_fill_time=f["model_fill_time"], model_notification_time=f["model_notification_time"], shares=f["shares"], price=f["price"]) for f in res["fills"]], counters=res["counters"], final={k: res["final"][k] for k in ("cash", "economic", "notified", "unnotified_sold", "reserved", "net_liquidation_value")})


def check(actual, expected):
    bad = []

    def walk(a, e, path):
        if isinstance(e, dict):
            if not isinstance(a, dict):
                bad.append(path)
                return
            for k, v in e.items():
                walk(a.get(k) if isinstance(a, dict) else None, v, f"{path}.{k}")
        elif isinstance(e, list):
            if not isinstance(a, list) or len(a) != len(e):
                bad.append(path)
                return
            for i, (x, y) in enumerate(zip(a, e)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(e, float):
            if not (F.number(a) and F.number(e)) or abs(a - e) > 1e-6:
                bad.append(path)
        elif a != e:
            bad.append(path)
    walk(actual, expected, "")
    return bad


def fl(order_id, day, label, price, shares):
    t = lab(day, label)
    return dict(order_id=order_id, label=f"{day} {label}", model_fill_time=F.iso(t - F.MIN), model_notification_time=F.iso(t), shares=shares, price=float(price))


def load_real():
    cal = json.load(open(CAL, encoding="utf-8"))
    seals = {}
    for line in open(os.path.join(MIN_ROOT, "seals.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["file"] in seals:
            raise SystemExit(f"two seals for {r['file']}")
        seals[r["file"]] = r
    out, files = {}, {}
    for sym in ("069500", "229200"):
        relp = f"{NOV6}/2026-11-06_{sym}.json"
        p = os.path.join(MIN_ROOT, relp)
        h = sha_file(p)
        if seals.get(relp, {}).get("sha256") != h:
            raise SystemExit(f"seal mismatch {relp}")
        d = json.load(open(p, encoding="utf-8"))
        rows = [dict(label=datetime.fromisoformat(r[0]).replace(tzinfo=F.KST), open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]), volume=int(r[5])) for r in d["rows"]]
        out[sym] = rows
        files[sym] = dict(file=rel(p), sha256=h, sealed_at=seals[relp]["sealed_at"], received_at=max(pg["received_at"] for pg in d["pages"]))
    meta = f"{NOV6}/meta.json"
    if seals.get(meta, {}).get("sha256") != sha_file(os.path.join(MIN_ROOT, meta)):
        raise SystemExit("seal mismatch meta")
    return {"2026-11-06": cal["days"]["2026-11-06"]}, out, files, seals[meta]["sealed_at"]


def baseline(orders, minutes, day):
    rows = sorted(minutes[day], key=lambda r: r["label"])
    groups = {}
    for r in rows:
        s = r["label"] - F.MIN
        b = s.replace(minute=s.minute - s.minute % 5)
        groups.setdefault(b, []).append(r)
    bars = pd.DataFrame([dict(date=pd.Timestamp(b).tz_convert("UTC"), open=g[0]["open"], high=max(x["high"] for x in g), low=min(x["low"] for x in g), close=g[-1]["close"], volume=sum(x["volume"] for x in g)) for b, g in sorted(groups.items())])
    fo = [dict(time=pd.Timestamp(o["activation"]).tz_convert("UTC"), side=o["side"], shares=o["shares"], price=o["price"]) for o in orders]
    r = FS.run(fo, bars, {day: dict(last="15:30")}, CAPITAL, cap=False, tick=False)
    return dict(fills=[dict(side=f["side"], shares=f["shares"], price=f["price"]) for f in r["fills"]], statuses=[a["status"] for a in r["accounting"]], held=r["held"], blocked=r["notes"]["fills_blocked_zero_volume_or_auction"])


def cases():
    out = []

    def add(cid, desc, res, expected, extra=None):
        a = summary(res)
        if extra:
            a.update(extra)
        bad = check(a, expected)
        cons = F.conservation(res, CAPITAL)
        out.append(dict(id=cid, case=desc, expected=expected, actual=a, conservation=cons, mismatches=bad, passed=not bad and cons["ok"]))
        return res

    real_cal, real_min, real_files, meta_sealed = load_real()
    for cid, hhmm, label, prices in (("C1", "09:30", "09:34", {"069500": 77950.0, "229200": 10435.0}), ("C2", "09:00", "09:02", {"069500": 76935.0, "229200": 10315.0})):
        for sym in ("069500", "229200"):
            o = [order(0, "buy", 10, "2026-11-06", hhmm)]
            res = F.run(o, {"2026-11-06": real_min[sym]}, real_cal, "2026-11-06", "2026-11-06", CAPITAL)
            add(f"{cid}_{sym}", f"real 11-06 {sym}, market buy active at {hhmm}", res, dict(status="ok", orders={"fx:0": dict(status="filled", filled=10)}, fills=[fl("fx:0", "2026-11-06", label, prices[sym], 10)]))
            if cid == "C1":
                f = res["fills"][0]
                clocks = dict(input_received_at=real_files[sym]["received_at"], input_available_at=meta_sealed, model_fill_time=f["model_fill_time"], model_notification_time=f["model_notification_time"])
                out.append(dict(id=f"C15_{sym}", case="11-06 input clocks are separate fields from model times", expected=dict(separate=True, input_after_model=True), actual=dict(clocks, separate=len(set(clocks.values())) == 4, input_after_model=clocks["input_available_at"] > clocks["model_notification_time"]), conservation=dict(ok=True), mismatches=[], passed=len(set(clocks.values())) == 4 and datetime.fromisoformat(clocks["input_available_at"]) > datetime.fromisoformat(clocks["model_notification_time"])))

    z5 = ("09:31", "09:32", "09:33", "09:34", "09:35")
    z3 = ("09:31", "09:32", "09:33")
    m = {D1: gen(D1, SYN_CAL, zero=z5)}
    add("C3", "whole zero-volume 5-minute bar, buy waits across the boundary", F.run([order(0, "buy", 10, D1, "09:30")], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:0": dict(status="filled", filled=10)}, fills=[fl("fx:0", D1, "09:36", 10035, 10)]))
    m = {D1: gen(D1, SYN_CAL)}
    add("C4", "buy activated at 15:20 = continuous_end", F.run([order(0, "buy", 10, D1, "15:20")], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:0": dict(status="expired_at_activation", expired=10, filled=0, expired_at=F.iso(lab(D1, "15:20")))}, fills=[]))
    add("C16", "buy activated at 15:25 inside the closing auction", F.run([order(0, "buy", 10, D1, "15:25")], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:0": dict(status="expired_at_activation", expired=10, expired_at=F.iso(lab(D1, "15:25")))}, fills=[]))
    m2 = {D1: gen(D1, SYN_CAL), D2: gen(D2, SYN_CAL)}
    add("C5", "sell active at 15:20 carried; next session first interval excluded", F.run([order(0, "buy", 10, D1, "09:00"), order(1, "sell", 10, D1, "15:20")], m2, SYN_CAL, D1, D2, CAPITAL), dict(orders={"fx:0": dict(status="filled"), "fx:1": dict(status="filled", filled=10)}, fills=[fl("fx:0", D1, "09:02", 10001, 10), fl("fx:1", D2, "09:02", 20001, 10)]))
    m = {D1: gen(D1, SYN_CAL, zero=z3)}
    r6 = add("C6", "sell activated before the buy fill is notified", F.run([order(0, "buy", 10, D1, "09:30"), order(1, "sell", 10, D1, "09:33")], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:0": dict(status="filled", filled=10), "fx:1": dict(status="cancelled_no_sellable", cancelled=10, filled=0)}, fills=[fl("fx:0", D1, "09:34", 10033, 10)]))
    add("C7", "two same-side buys pending at once", F.run([order(0, "buy", 5, D1, "09:30"), order(1, "buy", 5, D1, "09:30")], m, SYN_CAL, D1, D1, CAPITAL), dict(counters=dict(pending_same_side_overlap=1), fills=[fl("fx:0", D1, "09:34", 10033, 5), fl("fx:1", D1, "09:34", 10033, 5)]))
    held = lambda day, why: dict(status="held_input_damaged", problems={day: [why]}, fills=0, pnl=None)
    add("C8", "one minute missing on one day of the range", F.run([order(0, "buy", 10, D1, "09:30")], {D1: gen(D1, SYN_CAL), D2: gen(D2, SYN_CAL, drop=("11:00",))}, SYN_CAL, D1, D2, CAPITAL), held(D2, "missing 1 from 11:00"))
    add("C9_duplicate", "duplicate minute row", F.run([order(0, "buy", 10, D1, "09:30")], {D1: gen(D1, SYN_CAL, dup="10:00")}, SYN_CAL, D1, D1, CAPITAL), held(D1, "duplicate 10:00"))
    add("C9_conflict", "conflicting minute row", F.run([order(0, "buy", 10, D1, "09:30")], {D1: gen(D1, SYN_CAL, conflict="10:00")}, SYN_CAL, D1, D1, CAPITAL), held(D1, "conflict 10:00"))
    add("C9_invalid", "invalid minute value (high below open)", F.run([order(0, "buy", 10, D1, "09:30")], {D1: gen(D1, SYN_CAL, invalid="10:00")}, SYN_CAL, D1, D1, CAPITAL), held(D1, "invalid 10:00"))
    m = {D1: gen(D1, SYN_CAL)}
    add("C10", "input ends with a pending sell", F.run([order(0, "buy", 10, D1, "09:00"), order(1, "sell", 10, D1, "15:25")], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:1": dict(status="carried_unfilled_at_end", carried_unfilled_at_end=10, filled=0)}, final=dict(reserved=10, economic=10)))
    r11 = F.run([order(0, "buy", 10, D1, "09:00"), order(1, "sell", 10, D1, "15:15", tag="force_exit", force=True)], m, SYN_CAL, D1, D1, CAPITAL)
    fees = 10 * 10001 * F.FEE + 10 * 10375 * F.FEE
    add("C11", "force_exit sell: one exit fee per share, no closing valuation of sold shares", r11, dict(fills=[fl("fx:0", D1, "09:02", 10001, 10), fl("fx:1", D1, "15:16", 10375, 10)], final=dict(economic=0, net_liquidation_value=float(CAPITAL - 10 * 10001 + 10 * 10375 - fees))), extra=dict(fees_total=sum(f["fee"] for f in r11["fills"])))
    out[-1]["passed"] = out[-1]["passed"] and abs(out[-1]["actual"]["fees_total"] - fees) < 1e-6 and abs(r11["final"]["net_liquidation_value"] - r11["final"]["cash"]) < 1e-9
    o18 = [order(0, "buy", 10, D1, "09:04"), order(1, "sell", 8, D1, "09:05")]
    r18 = add("C18", "notification and new sell at the same time point", F.run(o18, m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:1": dict(status="filled", filled=8, cancelled=0)}, fills=[fl("fx:0", D1, "09:05", 10004, 10), fl("fx:1", D1, "09:06", 10005, 8)]))
    at905 = [(e["kind"], e["order_id"]) for e in r18["events"] if e["t"] == F.iso(lab(D1, "09:05"))]
    again = F.run(o18, m, SYN_CAL, D1, D1, CAPITAL)
    h1, h2 = (hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest() for x in (r18, again))
    out.append(dict(id="C12", case="same time point order and deterministic rerun", expected=dict(order_at_0905=[["notify", "fx:0"], ["activate", "fx:1"], ["fill", "fx:1"]], same_hash=True), actual=dict(order_at_0905=[list(x) for x in at905], same_hash=h1 == h2, hash=h1), conservation=dict(ok=True), mismatches=[], passed=at905 == [("notify", "fx:0"), ("activate", "fx:1"), ("fill", "fx:1")] and h1 == h2))
    add("C13", "orders between sessions are rejected_non_trading_time, orders before or after the range rejected_outside_input_range; target not held", F.run([order(0, "buy", 10, D1, "15:40"), order(1, "buy", 10, D1, "15:30"), order(2, "buy", 10, "2030-01-06", "10:00"), order(3, "buy", 10, D2, "15:40")], m2, SYN_CAL, D1, D2, CAPITAL), dict(status="ok", orders={"fx:0": dict(status="rejected_non_trading_time", rejected=10), "fx:1": dict(status="rejected_non_trading_time", rejected=10), "fx:2": dict(status="rejected_outside_input_range", rejected=10), "fx:3": dict(status="rejected_outside_input_range", rejected=10)}, fills=[]))
    add("C17", "10 notified shares, two sells of 8 at the same time", F.run([order(0, "buy", 10, D1, "09:00"), order(1, "sell", 8, D1, "09:30"), order(2, "sell", 8, D1, "09:30")], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:1": dict(filled=8, cancelled=0), "fx:2": dict(filled=2, cancelled=6)}, fills=[fl("fx:0", D1, "09:02", 10001, 10), fl("fx:1", D1, "09:31", 10030, 8), fl("fx:2", D1, "09:31", 10030, 2)]))
    add("C19", "filled but unnotified sell shares are not reserved again", F.run([order(0, "buy", 10, D1, "09:00"), order(1, "sell", 8, D1, "09:04"), order(2, "sell", 8, D1, "09:04", sec=30)], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:1": dict(filled=8), "fx:2": dict(filled=2, cancelled=6)}, fills=[fl("fx:0", D1, "09:02", 10001, 10), fl("fx:1", D1, "09:05", 10004, 8), fl("fx:2", D1, "09:06", 10005, 2)]))
    add("C20", "activations before fills at the same time point", F.run([order(0, "buy", 10, D1, "09:00"), order(1, "sell", 8, D1, "09:04"), order(2, "sell", 8, D1, "09:04")], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:1": dict(filled=8), "fx:2": dict(filled=2, cancelled=6)}, fills=[fl("fx:0", D1, "09:02", 10001, 10), fl("fx:1", D1, "09:05", 10004, 8), fl("fx:2", D1, "09:05", 10004, 2)]))
    add("C21", "activation not on a minute boundary", F.run([order(0, "buy", 10, D1, "09:04", sec=30)], m, SYN_CAL, D1, D1, CAPITAL), dict(fills=[fl("fx:0", D1, "09:06", 10005, 10)]))
    add("C22", "buy filled in the interval starting 09:04 is not sellable for a sell activated at 09:04:30", F.run([order(0, "buy", 10, D1, "09:04"), order(1, "sell", 10, D1, "09:04", sec=30)], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:0": dict(status="filled", filled=10), "fx:1": dict(status="cancelled_no_sellable", cancelled=10)}, fills=[fl("fx:0", D1, "09:05", 10004, 10)]))
    add("C23", "FIFO uses the numeric source_index: source 2 before source 10", F.run([order(0, "buy", 10, D1, "09:00"), order(10, "sell", 8, D1, "09:30"), order(2, "sell", 8, D1, "09:30")], m, SYN_CAL, D1, D1, CAPITAL), dict(orders={"fx:2": dict(filled=8, cancelled=0), "fx:10": dict(filled=2, cancelled=6)}, fills=[fl("fx:0", D1, "09:02", 10001, 10), fl("fx:2", D1, "09:31", 10030, 8), fl("fx:10", D1, "09:31", 10030, 2)]))
    zl = tuple((datetime(2000, 1, 1, 15, 1) + timedelta(minutes=k)).strftime("%H:%M") for k in range(20))
    add("C24", "buy active at 15:00 with no positive volume before 15:20 expires at continuous_end", F.run([order(0, "buy", 10, D1, "15:00")], {D1: gen(D1, SYN_CAL, zero=zl), D2: gen(D2, SYN_CAL)}, SYN_CAL, D1, D2, CAPITAL), dict(orders={"fx:0": dict(status="expired_session_end", expired=10, filled=0, expired_at=F.iso(lab(D1, "15:20")))}, fills=[]))
    add("X1_late_open", "late-open calendar: buy at 10:00 skips the first interval; buy at 09:30 is outside the session", F.run([order(0, "buy", 10, LATE, "10:00"), order(1, "buy", 10, LATE, "09:30")], {PREV: gen(PREV, LATE_CAL), LATE: gen(LATE, LATE_CAL)}, LATE_CAL, PREV, LATE, CAPITAL), dict(orders={"fx:0": dict(status="filled"), "fx:1": dict(status="rejected_non_trading_time")}, fills=[fl("fx:0", LATE, "10:02", 30001, 10)]))
    add("X2_shifted", "shifted-session calendar: 15:25 is continuous trading, 16:20 is the continuous end", F.run([order(0, "buy", 10, SHIFT, "15:25"), order(1, "buy", 10, SHIFT, "16:20")], {SHIFT: gen(SHIFT, SHIFT_CAL)}, SHIFT_CAL, SHIFT, SHIFT, CAPITAL), dict(orders={"fx:0": dict(status="filled"), "fx:1": dict(status="expired_at_activation", expired_at=F.iso(lab(SHIFT, "16:20")))}, fills=[fl("fx:0", SHIFT, "15:26", 40325, 10)]))

    base = gen(D1, SYN_CAL, zero=z3)
    extra = dict(label=lab(D1, "09:31") + timedelta(seconds=30), open=9000, high=9005, low=8995, close=9001, volume=500)
    add("C25", "extra row off the minute grid (09:31:30) with 390 normal rows", F.run([order(0, "buy", 10, D1, "09:30")], {D1: base + [extra]}, SYN_CAL, D1, D1, CAPITAL), held(D1, "unexpected label " + F.iso(extra["label"])))
    add("C26", "two orders with the same order_id", F.run([order(0, "buy", 1, D1, "09:30"), dict(order(1, "buy", 1, D1, "09:35"), order_id="fx:0")], m, SYN_CAL, D1, D1, CAPITAL), dict(status="held_input_damaged", problems={"orders": ["duplicate order_id fx:0"]}, fills=0, pnl=None))
    add("C27", "side BUY after holding 2 shares", F.run([order(0, "buy", 2, D1, "09:00"), dict(order(1, "buy", 1, D1, "09:30"), side="BUY")], m, SYN_CAL, D1, D1, CAPITAL), dict(status="held_input_damaged", problems={"orders": ["bad side 'BUY' at 1"]}, fills=0, pnl=None))
    no_vol = [dict(r) for r in gen(D1, SYN_CAL)]
    del no_vol[10]["volume"]
    add("C28", "minute row without a volume key", F.run([order(0, "buy", 10, D1, "09:30")], {D1: no_vol}, SYN_CAL, D1, D1, CAPITAL), dict(status="held_input_damaged", problems={D1: ["malformed row 10", "missing 1 from 09:11"]}, fills=0, pnl=None))
    naive = [dict(r) for r in gen(D1, SYN_CAL)]
    naive[20] = dict(naive[20], label=naive[20]["label"].replace(tzinfo=None))
    add("C29", "naive minute label", F.run([order(0, "buy", 10, D1, "09:30")], {D1: naive}, SYN_CAL, D1, D1, CAPITAL), dict(status="held_input_damaged", problems={D1: ["naive or non-datetime label row 20", "missing 1 from 09:21"]}, fills=0, pnl=None))
    add("C30", "naive order activation", F.run([dict(order(0, "buy", 10, D1, "09:30"), activation=datetime(2030, 1, 7, 9, 30))], m, SYN_CAL, D1, D1, CAPITAL), dict(status="held_input_damaged", problems={"orders": ["naive or non-datetime activation at 0"]}, fills=0, pnl=None))
    r31 = F.run([dict(order(0, "buy", 1, D1, "09:30"), order_id=[])], m, SYN_CAL, D1, D1, CAPITAL)
    add("C31", "order_id is a list", r31, dict(status="held_input_damaged", problems={"orders": ["bad order_id at 0"]}, fills=0, pnl=None, events=0), extra=dict(events=len(r31["events"])))
    r32 = F.run([dict(order(0, "buy", 1, D1, "09:30"), source_index={})], m, SYN_CAL, D1, D1, CAPITAL)
    add("C32", "source_index is a dict", r32, dict(status="held_input_damaged", problems={"orders": ["bad source_index at 0"]}, fills=0, pnl=None, events=0), extra=dict(events=len(r32["events"])))
    nan_checks = dict(nan_actual=check(dict(cash=float("nan")), dict(cash=1.0)), nan_expected=check(dict(cash=1.0), dict(cash=float("nan"))), inf_actual=check(dict(cash=float("inf")), dict(cash=1.0)), equal=check(dict(cash=1.0), dict(cash=1.0)))
    out.append(dict(id="K1", case="fixture float comparison requires finite values", expected=dict(nan_actual=[".cash"], nan_expected=[".cash"], inf_actual=[".cash"], equal=[]), actual=nan_checks, conservation=dict(ok=True), mismatches=[], passed=nan_checks == dict(nan_actual=[".cash"], nan_expected=[".cash"], inf_actual=[".cash"], equal=[])))
    mb = {D1: gen(D1, SYN_CAL)}
    b1 = baseline([order(0, "buy", 10, D1, "09:30", price=9990)], mb, D1)
    f1b1 = F.run([order(0, "buy", 10, D1, "09:30", price=9990)], mb, SYN_CAL, D1, D1, CAPITAL)
    out.append(dict(id="B1", case="new buy whose original price differs from the 5-minute open", expected=dict(baseline_price=9990.0, f1_price=10030.0), actual=dict(baseline_price=b1["fills"][0]["price"], f1_price=f1b1["fills"][0]["price"]), conservation=F.conservation(f1b1, CAPITAL), mismatches=[], passed=b1["fills"][0]["price"] == 9990 and f1b1["fills"][0]["price"] == 10030))
    mz = {D1: gen(D1, SYN_CAL, zero=z5)}
    b2 = baseline([order(0, "buy", 10, D1, "09:30", price=10030)], mz, D1)
    out.append(dict(id="B2", case="buy in a zero-volume 5-minute bar", expected=dict(baseline_status=["cancelled"], baseline_held=0, baseline_blocked=1, f1="see C3: fills at label 09:36"), actual=dict(baseline_status=b2["statuses"], baseline_held=b2["held"], baseline_blocked=b2["blocked"]), conservation=dict(ok=True), mismatches=[], passed=b2["statuses"] == ["cancelled"] and b2["held"] == 0 and b2["blocked"] == 1))
    b3 = baseline([order(0, "buy", 5, D1, "09:30", price=10030), order(1, "buy", 5, D1, "09:30", price=10030)], mb, D1)
    out.append(dict(id="B3", case="two same-side orders in one bar", expected=dict(baseline_status=["filled", "filled"], baseline_held=10, f1="see C7: both kept, overlap counted"), actual=dict(baseline_status=b3["statuses"], baseline_held=b3["held"]), conservation=dict(ok=True), mismatches=[], passed=b3["statuses"] == ["filled", "filled"] and b3["held"] == 10))
    return out, real_files, meta_sealed


def main(write):
    if sha_file(SPEC) != SPEC_SHA or sha_file(CAL) != CAL_SHA:
        raise SystemExit("spec or calendar hash differs")
    before = set(sys.modules)
    res, real_files, meta_sealed = cases()
    project = sorted(n for n, mod in sys.modules.items() if getattr(mod, "__file__", None) and os.path.abspath(mod.__file__).startswith(ROOT))
    forbidden = sorted(n for n in sys.modules if n.split(".")[0] in FORBIDDEN)
    body = dict(version="f1-fixtures-v3", previous=[dict(file="ft/results/conservative_fill_v1/f1_fixtures_v1/fixtures_result.json", sha256="6786cbd130a783ab9a919babc47e6b49dc08cf8bb9a5a3c4f23af0188bde5b2f"), dict(file="ft/results/conservative_fill_v1/f1_fixtures_v2/fixtures_result.json", sha256="bf40b3287607a57524f60161f5f25ab461ee025795515f0005169aabbe45249f")], scope="pure event/accounting functions and small counterexamples only; no full replay, no engine or strategy import, no API or token access, no change to original data, collector or paper loop", inputs=dict(code={rel(p): sha_file(p) for p in (F.__file__, os.path.abspath(__file__), FS.__file__)}, spec=dict(file=rel(SPEC), sha256=SPEC_SHA), calendar=dict(file=rel(CAL), sha256=CAL_SHA), real_minutes=real_files, real_minutes_meta_sealed_at=meta_sealed), cases=res, all_passed=all(c["passed"] for c in res), n_cases=len(res), imports=dict(project_modules=project, forbidden_loaded=forbidden), write_paths=[rel(OUT)] if write else [])
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    doc = dict(body, deterministic_result_sha256=digest, run_meta=dict(generated_at=datetime.now(F.KST).isoformat(timespec="seconds"), python=platform.python_version(), excluded_from_hash=True))
    for c in res:
        print(("ok  " if c["passed"] else "FAIL"), c["id"], "" if c["passed"] else json.dumps(dict(mismatches=c["mismatches"], conservation=c["conservation"], actual=c["actual"]), ensure_ascii=False, default=str)[:1500])
    print("all passed", body["all_passed"], "cases", len(res), "forbidden", forbidden, "project", project, "result", digest)
    if write:
        os.makedirs(OUT_DIR, exist_ok=True)
        data = json.dumps(doc, ensure_ascii=False, indent=1, default=str).encode("utf-8")
        with open(OUT, "xb") as f:
            f.write(data)
        print(rel(OUT), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main("--write" in sys.argv)
