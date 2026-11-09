import json
import os
import sys
import time
import uuid
from datetime import datetime

from toss import ApiError, Toss

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "live_trades.jsonl")


def wait_fill(t, oid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        o = t.order(oid)
        if o["status"] in ("FILLED", "CANCELED", "REJECTED"):
            return o
        time.sleep(1.5)
    return t.order(oid)


def rec(kind, **kw):
    row = dict(ts=datetime.now().isoformat(timespec="seconds"), kind=kind, **kw)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def main(symbol="AAPL", amount="1.00"):
    t = Toss()
    t.account = t.accounts()[0]["accountSeq"]
    bp = t.buying_power("USD")["cashBuyingPower"]
    ob = t.orderbook(symbol)
    ask, bid = float(ob["asks"][0]["price"]), float(ob["bids"][0]["price"])
    rec("pre", symbol=symbol, usd_buying_power=bp, ask=ask, bid=bid, spread_pct=(ask - bid) / ask * 100)
    buy = t.place(symbol, "BUY", "MARKET", amount=amount, client_order_id="test-" + uuid.uuid4().hex[:12])
    o = wait_fill(t, buy["orderId"])
    ex = o["execution"]
    qty = float(ex["filledQuantity"])
    rec("buy", order_id=o["orderId"], status=o["status"], qty=qty, avg_price=ex["averageFilledPrice"], amount=ex["filledAmount"], commission=ex["commission"], ask_at_order=ask, slip_pct=(float(ex["averageFilledPrice"]) / ask - 1) * 100 if ex["averageFilledPrice"] else None)
    if o["status"] != "FILLED" or qty <= 0:
        print("buy not filled, stop", flush=True)
        return
    time.sleep(3)
    sq = t.get("/api/v1/sellable-quantity", account=True, symbol=symbol)["sellableQuantity"]
    ob = t.orderbook(symbol)
    bid = float(ob["bids"][0]["price"])
    sell = t.place(symbol, "SELL", "MARKET", quantity=sq, client_order_id="test-" + uuid.uuid4().hex[:12])
    o2 = wait_fill(t, sell["orderId"])
    ex2 = o2["execution"]
    rec("sell", order_id=o2["orderId"], status=o2["status"], qty=ex2["filledQuantity"], avg_price=ex2["averageFilledPrice"], amount=ex2["filledAmount"], commission=ex2["commission"], tax=ex2["tax"], bid_at_order=bid, slip_pct=(float(ex2["averageFilledPrice"]) / bid - 1) * 100 if ex2["averageFilledPrice"] else None)
    if ex["filledAmount"] and ex2["filledAmount"]:
        cost = float(ex["filledAmount"]) - float(ex2["filledAmount"]) + float(ex["commission"] or 0) + float(ex2["commission"] or 0) + float(ex2["tax"] or 0)
        rec("roundtrip", symbol=symbol, net_cost_usd=round(cost, 4), net_cost_pct=round(cost / float(ex["filledAmount"]) * 100, 3), usd_buying_power_after=t.buying_power("USD")["cashBuyingPower"])


if __name__ == "__main__":
    if os.environ.get("DOBBY_LIVE") != "1":
        sys.exit("set DOBBY_LIVE=1 to run a live test order")
    main(*(sys.argv[1:3]))
