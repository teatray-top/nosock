import json
import sys
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from toss import Toss

sym, entry, log = sys.argv[1], float(sys.argv[2]), sys.argv[3]
t = Toss()
t.account = t.accounts()[0]["accountSeq"]
sq = t.get("/api/v1/sellable-quantity", account=True, symbol=sym)["sellableQuantity"]
print("sellable", sq, flush=True)
r = t.place(sym, "SELL", "MARKET", quantity=sq, client_order_id="manual-" + uuid.uuid4().hex[:10])
for _ in range(30):
    o = t.order(r["orderId"])
    if o["status"] in ("FILLED", "CANCELED", "REJECTED"):
        break
    time.sleep(2)
ex = o["execution"]
pnl = float(ex["filledAmount"] or 0) - float(ex["commission"] or 0) - float(ex["tax"] or 0) - entry * float(ex["filledQuantity"] or 0)
row = dict(ts=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"), kind="sell", symbol=sym, reason="manual_user", status=o["status"], qty=ex["filledQuantity"], price=ex["averageFilledPrice"], amount=ex["filledAmount"], commission=ex["commission"], tax=ex["tax"], entry=entry, pnl=round(pnl, 4))
open(log, "a", encoding="utf-8").write(json.dumps(row) + "\n")
print(json.dumps(row), flush=True)
print("USD buying power", t.buying_power("USD")["cashBuyingPower"], flush=True)
