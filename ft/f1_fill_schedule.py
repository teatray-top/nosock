import math
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
FEE = 0.00015
MIN = timedelta(minutes=1)


def at(day, hhmm):
    return datetime.fromisoformat(f"{day}T{hhmm}:00").replace(tzinfo=KST)


def iso(t):
    return t.astimezone(KST).isoformat()


def bounds(day, cal):
    c = cal[day]
    return dict(start=at(day, c["session_start"]), cont_end=at(day, c["continuous_end"]), first=at(day, c["first"]), last=at(day, c["last"]))


FIELDS = ("label", "open", "high", "low", "close", "volume")


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def aware(t):
    return isinstance(t, datetime) and t.tzinfo is not None and t.utcoffset() is not None


def valid_row(r):
    px = (r["open"], r["high"], r["low"], r["close"])
    v = r["volume"]
    if not all(number(x) and x > 0 for x in px):
        return False
    if not (number(v) and v >= 0 and v == int(v)):
        return False
    return r["low"] <= min(r["open"], r["close"]) and max(r["open"], r["close"]) <= r["high"]


def precheck(days, minutes, cal):
    problems = {}
    if not days:
        return {"range": ["no calendar session in the range"]}
    for d in days:
        rows = minutes.get(d)
        if not isinstance(rows, list):
            problems[d] = ["no minute input"]
            continue
        b = bounds(d, cal)
        want, t = set(), b["first"]
        while t <= b["last"]:
            want.add(t)
            t += MIN
        seen, p = {}, []
        for k, r in enumerate(rows):
            if not isinstance(r, dict) or any(f not in r for f in FIELDS):
                p.append(f"malformed row {k}")
                continue
            if not aware(r["label"]):
                p.append(f"naive or non-datetime label row {k}")
                continue
            t = r["label"].astimezone(KST)
            if t not in want:
                p.append("unexpected label " + iso(r["label"]))
                continue
            if t in seen:
                p.append(("duplicate " if seen[t] == r else "conflict ") + t.strftime("%H:%M"))
                continue
            seen[t] = r
            if not valid_row(r):
                p.append("invalid " + t.strftime("%H:%M"))
        missing = sorted(t for t in want if t not in seen)
        if missing:
            p.append(f"missing {len(missing)} from {missing[0].strftime('%H:%M')}")
        if p:
            problems[d] = p
    return problems


def check_orders(orders):
    p, ids, idx = [], set(), set()
    if not isinstance(orders, list):
        return ["orders is not a list"]
    for k, o in enumerate(orders):
        if not isinstance(o, dict) or any(f not in o for f in ("order_id", "source_index", "side", "shares", "activation")):
            p.append(f"malformed order {k}")
            continue
        if not isinstance(o["order_id"], str) or not o["order_id"]:
            p.append(f"bad order_id at {k}")
        elif o["order_id"] in ids:
            p.append(f"duplicate order_id {o['order_id']}")
        else:
            ids.add(o["order_id"])
        if not isinstance(o["source_index"], int) or isinstance(o["source_index"], bool):
            p.append(f"bad source_index at {k}")
        elif o["source_index"] in idx:
            p.append(f"duplicate source_index {o['source_index']}")
        else:
            idx.add(o["source_index"])
        if o["side"] not in ("buy", "sell"):
            p.append(f"bad side {o['side']!r} at {k}")
        if not aware(o["activation"]):
            p.append(f"naive or non-datetime activation at {k}")
    return p


def fifo_key(o):
    return (0 if o["side"] == "sell" else 1, o["activation"], o["source_index"], o["order_id"])


def run(orders, minutes, cal, first_day, last_day, capital, fee=FEE):
    days = [d for d in sorted(cal) if first_day <= d <= last_day]
    problems = {}
    bad = check_orders(orders)
    if bad:
        problems["orders"] = bad
    problems.update(precheck(days, minutes, cal))
    if problems:
        return dict(status="held_input_damaged", problems=problems, events=[], orders=[], fills=[], final=None, pnl=None)
    bd = {d: bounds(d, cal) for d in days}
    ivs = []
    for d in days:
        rows = sorted(minutes[d], key=lambda r: r["label"].astimezone(KST))
        for k, r in enumerate(rows):
            e = r["label"].astimezone(KST)
            s = e - MIN
            ivs.append(dict(day=d, start=s, end=e, open=float(r["open"]), close=float(r["close"]), volume=r["volume"], first_of_session=k == 0, continuous=s < bd[d]["cont_end"]))
    by_start = {iv["start"]: iv for iv in ivs}
    range_start, range_end = bd[days[0]]["start"], bd[days[-1]]["last"]
    st = dict(cash=float(capital), economic=0, notified=0, unnotified_sold=0, reserved=0)
    min_cash = st["cash"]
    recs, active, unnotified, events, fills = {}, [], [], [], []
    counters = dict(pending_same_side_overlap=0)
    acts = {}
    for o in orders:
        recs[o["order_id"]] = dict(order_id=o["order_id"], source_index=o["source_index"], side=o["side"], activation=iso(o["activation"]), original_price=o.get("price"), tag=o.get("tag"), is_force_exit=bool(o.get("is_force_exit")), requested=o["shares"], filled=0, cancelled=0, expired=0, rejected=0, carried_unfilled_at_end=0, status=None, expired_at=None)
        acts.setdefault(o["activation"], []).append(o)
    points = sorted(set(iv["start"] for iv in ivs) | set(iv["end"] for iv in ivs) | set(acts) | set(b["cont_end"] for b in bd.values()))

    def ev(t, kind, oid, **kw):
        events.append(dict(t=iso(t), kind=kind, order_id=oid, **kw, state=dict(st)))

    for T in points:
        for f in sorted([f for f in unnotified if f["notify"] <= T], key=lambda f: (f["notify"], f["seq"])):
            unnotified.remove(f)
            if f["side"] == "buy":
                st["notified"] += f["shares"]
            else:
                st["notified"] -= f["shares"]
                st["unnotified_sold"] -= f["shares"]
            ev(T, "notify", f["order_id"], fill_id=f["fill_id"], shares=f["shares"])
        for a in sorted([a for a in active if a["side"] == "buy" and bd[a["session"]]["cont_end"] <= T], key=fifo_key):
            active.remove(a)
            r = recs[a["order_id"]]
            r["expired"] += a["remaining"]
            r["expired_at"] = iso(bd[a["session"]]["cont_end"])
            r["status"] = "expired_session_end"
            ev(T, "expire", a["order_id"], shares=a["remaining"])
        for o in sorted(acts.get(T, []), key=fifo_key):
            r = recs[o["order_id"]]
            n = o["shares"]
            if not (isinstance(n, int) and not isinstance(n, bool) and n > 0):
                r.update(rejected=n, status="rejected_non_integer_or_non_positive")
                ev(T, "reject", o["order_id"], reason=r["status"])
                continue
            if T < range_start or T >= range_end:
                r.update(rejected=n, status="rejected_outside_input_range")
                ev(T, "reject", o["order_id"], reason=r["status"])
                continue
            d = T.astimezone(KST).date().isoformat()
            if d not in bd or not (bd[d]["start"] <= T < bd[d]["last"]):
                r.update(rejected=n, status="rejected_non_trading_time")
                ev(T, "reject", o["order_id"], reason=r["status"])
                continue
            if any(a["side"] == o["side"] for a in active):
                counters["pending_same_side_overlap"] += 1
            a = dict(o, remaining=n, session=d)
            if o["side"] == "buy":
                if T >= bd[d]["cont_end"]:
                    r.update(expired=n, expired_at=iso(T), status="expired_at_activation")
                    ev(T, "expire", o["order_id"], shares=n)
                    continue
                active.append(a)
                ev(T, "activate", o["order_id"], shares=n)
            else:
                sellable = max(st["notified"] - st["unnotified_sold"] - st["reserved"], 0)
                take = min(n, sellable)
                if take < n:
                    r["cancelled"] += n - take
                if take == 0:
                    r["status"] = "cancelled_no_sellable"
                    ev(T, "cancel", o["order_id"], shares=n)
                    continue
                st["reserved"] += take
                a["remaining"] = take
                active.append(a)
                ev(T, "activate", o["order_id"], shares=take, cancelled=n - take)
        iv = by_start.get(T)
        if iv is None or iv["volume"] <= 0 or not iv["continuous"]:
            continue
        for a in sorted(active, key=fifo_key):
            if a["side"] == "buy" and a["session"] != iv["day"]:
                continue
            if iv["first_of_session"] and a["activation"] <= bd[iv["day"]]["start"]:
                continue
            n, px = a["remaining"], iv["open"]
            r = recs[a["order_id"]]
            if a["side"] == "buy":
                st["cash"] -= n * px * (1 + fee)
                st["economic"] += n
            else:
                st["cash"] += n * px * (1 - fee)
                st["economic"] -= n
                st["reserved"] -= n
                st["unnotified_sold"] += n
            min_cash = min(min_cash, st["cash"])
            active.remove(a)
            r["filled"] += n
            r["status"] = "filled"
            f = dict(fill_id=a["order_id"] + ":f", order_id=a["order_id"], side=a["side"], shares=n, price=px, fee=n * px * fee, model_fill_time=iso(iv["start"]), model_notification_time=iso(iv["end"]), interval_label=iv["end"].strftime("%Y-%m-%d %H:%M"), cash_after=st["cash"])
            fills.append(f)
            unnotified.append(dict(f, notify=iv["end"], seq=len(fills)))
            ev(T, "fill", a["order_id"], fill_id=f["fill_id"], shares=n, price=px)
    for a in sorted(active, key=fifo_key):
        r = recs[a["order_id"]]
        if a["side"] == "sell":
            r["carried_unfilled_at_end"] += a["remaining"]
            r["status"] = "carried_unfilled_at_end"
        else:
            raise AssertionError("a buy outlived its session")
    assert not unnotified
    last_close = ivs[-1]["close"]
    nlv = st["cash"] + st["economic"] * last_close * (1 - fee)
    out_orders = [recs[o["order_id"]] for o in sorted(orders, key=lambda o: o["source_index"])]
    return dict(status="ok", problems={}, events=events, orders=out_orders, fills=fills, counters=counters, final=dict(st, min_cash=min_cash, executable=min_cash >= -1e-6, last_close=last_close, net_liquidation_value=nlv), pnl=nlv - capital)


def conservation(res, capital, fee=FEE):
    if res["status"] != "ok":
        return dict(ok=res["fills"] == [] and res["pnl"] is None, held=True)
    per = all(r["requested"] == r["filled"] + r["cancelled"] + r["expired"] + r["rejected"] + r["carried_unfilled_at_end"] for r in res["orders"] if isinstance(r["requested"], int) and not isinstance(r["requested"], bool) and r["requested"] > 0)
    cash = capital - sum(f["shares"] * f["price"] * (1 + fee) for f in res["fills"] if f["side"] == "buy") + sum(f["shares"] * f["price"] * (1 - fee) for f in res["fills"] if f["side"] == "sell")
    held = sum(f["shares"] for f in res["fills"] if f["side"] == "buy") - sum(f["shares"] for f in res["fills"] if f["side"] == "sell")
    st = res["final"]
    ok = per and abs(cash - st["cash"]) < 1e-6 and held == st["economic"] == st["notified"] and st["reserved"] == sum(r["carried_unfilled_at_end"] for r in res["orders"]) and st["unnotified_sold"] == 0
    return dict(ok=ok, per_order=per, cash_identity=abs(cash - st["cash"]) < 1e-6, held=held)
