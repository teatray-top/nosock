import os
from datetime import datetime, timezone

import pandas as pd

import sim

MAX_SPREAD_TICKS_KR = float(os.environ.get("BILL_KR_TICKS", 2))
MAX_SPREAD_US = float(os.environ.get("BILL_US_SPREAD", 0.003))
MAX_ASK_BID_RATIO = float(os.environ.get("BILL_IMB", 3.0))
MAX_STALE_SEC = float(os.environ.get("BILL_STALE", 90))
MAX_DRIFT = float(os.environ.get("BILL_DRIFT", 0.01))
MIN_DEPTH_X = float(os.environ.get("BILL_DEPTH_X", 3.0))
MAX_SELLER_FRAC = float(os.environ.get("BILL_SELLER", 0.8))


def kr_tick(p):
    for b, t in zip(sim._KR_B, sim._KR_T):
        if p < b:
            return t
    return sim._KR_T[-1]


def judge(mk, ob, trades, last_close, qty=None, notional=None):
    reasons = []
    asks, bids = ob.get("asks") or [], ob.get("bids") or []
    m = dict(levels=(len(asks), len(bids)))
    if not asks or not bids:
        return False, ["empty_book"], m
    a1, b1 = float(asks[0]["price"]), float(bids[0]["price"])
    mid = (a1 + b1) / 2
    spread = a1 - b1
    m.update(ask1=a1, bid1=b1, spread_pct=round(spread / mid * 100, 4))
    if mk == "kr":
        ticks = spread / kr_tick(mid)
        m["spread_ticks"] = round(ticks, 2)
        if ticks > MAX_SPREAD_TICKS_KR + 1e-9:
            reasons.append(f"spread {ticks:.1f} ticks")
    else:
        if spread / mid > MAX_SPREAD_US:
            reasons.append(f"spread {spread / mid * 100:.2f}%")
    av = sum(float(x["volume"]) for x in asks[:5])
    bv = sum(float(x["volume"]) for x in bids[:5])
    m.update(ask5=av, bid5=bv, ask_bid_ratio=round(av / max(bv, 1e-9), 2))
    if bv <= 0 or av / bv > MAX_ASK_BID_RATIO:
        reasons.append(f"ask/bid depth {av / max(bv, 1e-9):.1f}x")
    ts = ob.get("timestamp")
    if ts:
        age = (datetime.now(timezone.utc) - pd.Timestamp(ts).tz_convert("UTC").to_pydatetime()).total_seconds()
        m["book_age_s"] = round(age, 1)
        if age > MAX_STALE_SEC:
            reasons.append(f"stale book {age:.0f}s")
    else:
        reasons.append("no book timestamp")
    if last_close and last_close > 0:
        drift = a1 / last_close - 1
        m["ask_vs_close_pct"] = round(drift * 100, 3)
        if abs(drift) > MAX_DRIFT:
            reasons.append(f"ask moved {drift * 100:+.2f}% from bar close")
    if qty:
        within = sum(float(x["volume"]) for x in asks if float(x["price"]) <= a1 + 2 * kr_tick(a1)) if mk == "kr" else sum(float(x["volume"]) for x in asks[:3])
        m["ask_depth_for_us"] = within
        if within < MIN_DEPTH_X * qty:
            reasons.append(f"thin ask {within:.0f} vs need {MIN_DEPTH_X * qty:.0f}")
    if trades:
        seller = sum(1 for x in trades if float(x["price"]) <= b1)
        buyer = sum(1 for x in trades if float(x["price"]) >= a1)
        n = len(trades)
        m.update(trades_n=n, seller_frac=round(seller / n, 2), buyer_frac=round(buyer / n, 2))
        if n >= 20 and seller / n > MAX_SELLER_FRAC:
            reasons.append(f"sellers {seller / n * 100:.0f}% of last {n} trades")
    return not reasons, reasons, m
