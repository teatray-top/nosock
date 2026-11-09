import json
import math
import sys

import pandas as pd

TICK = 5
FEE = 0.00015
CAP_SHARE = 0.05


def auction_buckets(calendar, kst_day):
    last = pd.Timestamp(f"{kst_day} {calendar[kst_day]['last']}")
    return {(last - pd.Timedelta(minutes=10)).strftime("%H:%M"), (last - pd.Timedelta(minutes=5)).strftime("%H:%M")}


def tick_price(px, side, enabled):
    if not enabled:
        return px
    if side == "buy":
        return math.ceil(px / TICK - 1e-9) * TICK + TICK
    return math.floor(px / TICK + 1e-9) * TICK - TICK


def run(orders, bars, calendar, capital, cap=True, tick=False):
    cash_limit = cap
    bars = bars.sort_values("date").reset_index(drop=True)
    first_bar, last_end = bars["date"].iloc[0], bars["date"].iloc[-1] + pd.Timedelta(minutes=5)
    starts = list(bars["date"])
    cash, held, reserved = float(capital), 0, 0
    min_cash = cash
    pending_sells = []
    fills = []
    acct = []
    notes = dict(buys_cut_shares=0, buy_orders_cut=0, sell_shares_cut_to_holdings=0, sells_carried=0, max_bars_to_complete_sell=0, fill_price_outside_ohlc=0, unfilled_attempt_price_outside_ohlc=0, fills_blocked_zero_volume_or_auction=0, invariant_violations=0)
    by_bar = {}
    for k, o in enumerate(sorted(orders, key=lambda o: (o["time"], 0 if o["side"] == "sell" else 1))):
        rec = dict(id=k, time=str(o["time"]), side=o["side"], requested=o["shares"], filled=0, cancelled=0, rejected=0, carried_unfilled_at_end=0, status=None)
        acct.append(rec)
        if abs(o["shares"] - round(o["shares"])) > 1e-9 or o["shares"] <= 0:
            rec.update(rejected=o["shares"], status="rejected_non_integer_or_non_positive")
            continue
        if not (first_bar <= o["time"] < last_end):
            rec.update(rejected=o["shares"], status="rejected_outside_input_range")
            continue
        i = None
        for j in range(len(starts)):
            if starts[j] <= o["time"] < starts[j] + pd.Timedelta(minutes=5):
                i = j
                break
        if i is None:
            rec.update(rejected=o["shares"], status="rejected_no_bar")
            continue
        by_bar.setdefault(i, []).append((dict(o, shares=int(round(o["shares"]))), rec))
    for i in range(len(bars)):
        b = bars.iloc[i]
        start = b["date"]
        kst = start.tz_convert("Asia/Seoul")
        blocked = b["volume"] <= 0 or kst.strftime("%H:%M") in auction_buckets(calendar, kst.strftime("%Y-%m-%d"))
        capacity = math.floor(CAP_SHARE * b["volume"]) if cap else float("inf")
        if blocked:
            capacity = 0
        new = by_bar.get(i, [])
        queue = [("carried", s) for s in pending_sells] + [("new", x) for x in new if x[0]["side"] == "sell"] + [("new", x) for x in new if x[0]["side"] == "buy"]
        pending_sells = []
        for kind, item in queue:
            o, rec = item
            side = o["side"]
            px = b["open"] if kind == "carried" else o["price"]
            px = tick_price(px, side, tick)
            outside = not (b["low"] <= px <= b["high"])
            if side == "sell":
                if kind == "new":
                    allowed = max(held - reserved, 0)
                    want = min(o["shares"], allowed)
                    if want < o["shares"]:
                        notes["sell_shares_cut_to_holdings"] += o["shares"] - want
                        rec["cancelled"] += o["shares"] - want
                    o["remaining"], o["first_bar"] = want, i
                    reserved += want
                want = o["remaining"]
                n = 0 if blocked else int(min(want, capacity))
                if blocked and want > 0:
                    notes["fills_blocked_zero_volume_or_auction"] += 1
                if outside:
                    notes["fill_price_outside_ohlc" if n > 0 else "unfilled_attempt_price_outside_ohlc"] += 1
                if n > 0:
                    cash += n * px - n * px * FEE
                    held -= n
                    reserved -= n
                    capacity -= n
                    rec["filled"] += n
                    fills.append(dict(time=str(start), side="sell", shares=n, price=px, cash_after=cash, held_after=held, reserved_after=reserved))
                o["remaining"] = want - n
                if o["remaining"] > 0:
                    if kind == "new":
                        notes["sells_carried"] += 1
                    pending_sells.append(item)
                else:
                    notes["max_bars_to_complete_sell"] = max(notes["max_bars_to_complete_sell"], i - o["first_bar"])
                    rec["status"] = rec["status"] or "filled"
            else:
                limit = o["shares"]
                if cap:
                    limit = min(limit, capacity)
                if cash_limit:
                    limit = min(limit, int(max(cash, 0) // (px * (1 + FEE))))
                n = 0 if blocked else int(limit)
                if blocked:
                    notes["fills_blocked_zero_volume_or_auction"] += 1
                if outside:
                    notes["fill_price_outside_ohlc" if n > 0 else "unfilled_attempt_price_outside_ohlc"] += 1
                if n < o["shares"]:
                    notes["buys_cut_shares"] += o["shares"] - n
                    notes["buy_orders_cut"] += 1
                    rec["cancelled"] += o["shares"] - n
                if n > 0:
                    cash -= n * px + n * px * FEE
                    held += n
                    capacity -= n
                    rec["filled"] += n
                    fills.append(dict(time=str(start), side="buy", shares=n, price=px, cash_after=cash, held_after=held, reserved_after=reserved))
                rec["status"] = "filled" if n == o["shares"] else ("partially_filled_rest_cancelled" if n > 0 else "cancelled")
            min_cash = min(min_cash, cash)
            if not (held >= reserved >= 0):
                notes["invariant_violations"] += 1
    for o, rec in pending_sells:
        rec["carried_unfilled_at_end"] = o["remaining"]
        rec["status"] = "carried_unfilled_at_end"
    last_close = float(bars.iloc[-1]["close"])
    nlv = cash + held * last_close - held * last_close * FEE
    for rec in acct:
        rec["balanced"] = rec["requested"] == rec["filled"] + rec["cancelled"] + rec["rejected"] + rec["carried_unfilled_at_end"]
    return dict(variant=dict(cap=cap, tick=tick, cash_limit=cash_limit), fills=fills, accounting=acct, cash=cash, min_cash=min_cash, executable=min_cash >= -1e-6, held=held, reserved=reserved, net_liquidation_value=nlv, notes=notes, caveat="bar volume is known only after the bar; fixed-order fill sensitivity, not a strategy re-execution and not a lower bound on profit")


def check_conservation(res, capital):
    buys = sum(f["shares"] for f in res["fills"] if f["side"] == "buy")
    sells = sum(f["shares"] for f in res["fills"] if f["side"] == "sell")
    cash = capital - sum(f["shares"] * f["price"] * (1 + FEE) for f in res["fills"] if f["side"] == "buy") + sum(f["shares"] * f["price"] * (1 - FEE) for f in res["fills"] if f["side"] == "sell")
    return dict(shares_conserved=buys - sells == res["held"], cash_conserved=abs(cash - res["cash"]) < 1e-6, held_ge_reserved_ge_0=all(f["held_after"] >= f["reserved_after"] >= 0 for f in res["fills"]) and res["notes"]["invariant_violations"] == 0, every_order_balanced=all(r["balanced"] for r in res["accounting"]))


def fixtures():
    cal = {"2026-01-05": dict(cls="normal", first="09:01", last="15:30"), "2026-11-19": dict(cls="shifted_session_special", first="10:01", last="16:30")}

    def bar(t, o, h, l, c, v):
        return dict(date=pd.Timestamp(t, tz="Asia/Seoul").tz_convert("UTC"), open=o, high=h, low=l, close=c, volume=v)
    bars = pd.DataFrame([bar("2026-01-05 09:00", 10000, 10050, 9950, 10000, 2000), bar("2026-01-05 09:05", 10000, 10100, 9990, 10050, 1000), bar("2026-01-05 09:10", 10050, 10060, 10000, 10010, 0), bar("2026-01-05 09:15", 10010, 10020, 9990, 10000, 1000), bar("2026-01-05 09:25", 10000, 10010, 9990, 10000, 1000), bar("2026-01-05 15:20", 10000, 10000, 10000, 10000, 0), bar("2026-01-05 15:25", 10000, 10010, 9990, 10005, 5000)])
    t = lambda s: pd.Timestamp(s, tz="Asia/Seoul").tz_convert("UTC")
    out = {}
    r = run([dict(time=t("2026-01-05 09:00"), side="buy", shares=100, price=10000)], bars, cal, 10_000_000)
    out["cap_exactly_floor_5pct"] = r["held"] == 100 and r["notes"]["buys_cut_shares"] == 0
    r = run([dict(time=t("2026-01-05 09:05"), side="buy", shares=100, price=10000), dict(time=t("2026-01-05 09:15"), side="sell", shares=100, price=10000)], bars, cal, 10_000_000)
    out["partial_buy_50_then_sell_capped_to_held"] = r["notes"]["buys_cut_shares"] == 50 and r["notes"]["sell_shares_cut_to_holdings"] == 50 and r["held"] == 0
    r = run([dict(time=t("2026-01-05 09:00"), side="buy", shares=100, price=10000), dict(time=t("2026-01-05 09:05"), side="sell", shares=100, price=10050)], bars, cal, 10_000_000)
    out["sell_capped_by_bar_then_carried_past_zero_volume"] = r["notes"]["sells_carried"] == 1 and r["held"] == 0 and r["notes"]["max_bars_to_complete_sell"] == 2
    r = run([dict(time=t("2026-01-05 09:10"), side="buy", shares=10, price=10050)], bars, cal, 10_000_000)
    out["zero_volume_bar_takes_no_fill"] = r["held"] == 0 and r["notes"]["fills_blocked_zero_volume_or_auction"] == 1 and r["accounting"][0]["status"] == "cancelled"
    r = run([dict(time=t("2026-01-05 15:25"), side="buy", shares=10, price=10000)], bars, cal, 10_000_000)
    out["auction_bucket_takes_no_fill"] = r["held"] == 0
    r = run([dict(time=t("2026-01-05 09:00"), side="buy", shares=10, price=10000)], bars, cal, 10_000_000, cap=True, tick=True)
    out["tick_raises_buy_by_5"] = r["fills"][0]["price"] == 10005
    r = run([dict(time=t("2026-01-05 09:00"), side="buy", shares=60, price=10000), dict(time=t("2026-01-05 09:00"), side="buy", shares=60, price=10000)], bars, cal, 10_000_000)
    out["cap_is_shared_by_orders_in_a_bar"] = r["held"] == 100 and r["notes"]["buys_cut_shares"] == 20
    r = run([dict(time=t("2026-01-05 09:00"), side="buy", shares=100, price=10000)], bars, cal, 500_000)
    out["cash_limits_buy_in_cap_variant"] = r["held"] == int(500_000 // (10000 * (1 + FEE)))
    r = run([dict(time=t("2026-01-05 09:15"), side="buy", shares=1, price=10020)], bars, cal, 10_000_000, cap=False, tick=True)
    out["tick_fill_outside_ohlc_flagged_as_fill_violation"] = r["notes"]["fill_price_outside_ohlc"] == 1 and r["notes"]["unfilled_attempt_price_outside_ohlc"] == 0
    r = run([dict(time=t("2026-01-05 09:10"), side="buy", shares=1, price=10060)], bars, cal, 10_000_000, cap=False, tick=True)
    out["blocked_attempt_outside_ohlc_counted_as_unfilled_attempt"] = r["notes"]["fill_price_outside_ohlc"] == 0 and r["notes"]["unfilled_attempt_price_outside_ohlc"] == 1 and r["held"] == 0
    combos = []
    for c in (False, True):
        for tk in (False, True):
            for when in ("2026-01-05 09:10", "2026-01-05 15:20", "2026-01-05 15:25"):
                rb = run([dict(time=t(when), side="buy", shares=10, price=10000)], bars, cal, 10_000_000, cap=c, tick=tk)
                rs = run([dict(time=t("2026-01-05 09:00"), side="buy", shares=10, price=10000), dict(time=t(when), side="sell", shares=10, price=10000)], bars, cal, 10_000_000, cap=c, tick=tk)
                no_sell_on_blocked_bar = not any(f["side"] == "sell" and f["time"] == str(t(when)) for f in rs["fills"])
                sell_after = [f for f in rs["fills"] if f["side"] == "sell"]
                sell_ok = all(pd.Timestamp(f["time"]) > t(when) for f in sell_after) and (rs["held"] + sum(f["shares"] for f in sell_after) == 10)
                combos.append(rb["held"] == 0 and rb["fills"] == [] and rb["accounting"][0]["status"] == "cancelled" and no_sell_on_blocked_bar and sell_ok and rs["accounting"][1]["balanced"])
    out["blocked_bars_take_no_buy_or_sell_in_all_4_variants"] = all(combos) and len(combos) == 12
    r = run([dict(time=t("2026-01-05 09:00"), side="buy", shares=100, price=10000)], bars, cal, 1_000_150, cap=False, tick=True)
    out["tick_only_keeps_quantity_and_marks_negative_cash"] = r["held"] == 100 and abs(r["cash"] - (-500.075)) < 1e-6 and r["executable"] is False
    r = run([dict(time=t("2026-01-05 15:40"), side="buy", shares=10, price=10000)], bars, cal, 10_000_000)
    out["order_after_last_bar_is_recorded_as_rejected"] = r["accounting"][0]["status"] == "rejected_outside_input_range" and r["accounting"][0]["rejected"] == 10
    r = run([dict(time=t("2026-01-05 09:20"), side="buy", shares=10, price=10000)], bars, cal, 10_000_000)
    out["order_on_missing_bar_is_rejected_not_moved"] = r["accounting"][0]["status"] == "rejected_no_bar" and r["held"] == 0
    r = run([dict(time=t("2026-01-05 09:00"), side="buy", shares=0.5, price=10000)], bars, cal, 10_000_000)
    out["non_integer_order_is_rejected_not_truncated"] = r["accounting"][0]["status"] == "rejected_non_integer_or_non_positive" and r["accounting"][0]["rejected"] == 0.5
    r = run([dict(time=t("2026-01-05 09:00"), side="buy", shares=100, price=10000), dict(time=t("2026-01-05 15:25"), side="sell", shares=100, price=10000)], bars, cal, 10_000_000)
    out["sell_in_last_auction_bucket_is_carried_unfilled_at_end"] = r["accounting"][1]["status"] == "carried_unfilled_at_end" and r["accounting"][1]["carried_unfilled_at_end"] == 100 and r["held"] == 100
    cons = [check_conservation(run([dict(time=t("2026-01-05 09:00"), side="buy", shares=100, price=10000), dict(time=t("2026-01-05 09:05"), side="sell", shares=100, price=10050), dict(time=t("2026-01-05 15:40"), side="buy", shares=5, price=10000)], bars, cal, 10_000_000, cap=c, tick=tk), 10_000_000) for c in (False, True) for tk in (False, True)]
    out["conservation_cash_shares_invariants_and_accounting"] = all(all(c.values()) for c in cons)
    return out


if __name__ == "__main__":
    res = fixtures()
    print(json.dumps(res, indent=1))
    if not all(res.values()):
        sys.exit(1)
    print("fixtures ok")
