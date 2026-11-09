import hashlib
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
KST = ZoneInfo("Asia/Seoul")
LEDGER = os.environ.get("DCA_LEDGER", os.path.join(HERE, "runs", "dca_kr_ledger.jsonl"))
SYMBOL = "069500"
FEE = 0.00015
TICK = 5.0
ORDER_WINDOW = ("09:00", "09:10")


def sha_self():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def load_ledger(path=LEDGER):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    return rows


def append(row, path=LEDGER):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = dict(ts=datetime.now(KST).isoformat(timespec="seconds"), **row)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def state(rows):
    qty = sum(r["qty"] for r in rows if r.get("kind") == "buy")
    cost = sum(r["cost"] for r in rows if r.get("kind") == "buy")
    deposits = sum(r["amount"] for r in rows if r.get("kind") == "deposit")
    cfg = next((r for r in reversed(rows) if r.get("kind") == "config"), None)
    return dict(qty=qty, cost=cost, deposits=deposits, config=cfg, buys=sum(1 for r in rows if r.get("kind") == "buy"))


def decide(now, cash, ask, st, cfg):
    reasons = []
    hhmm = now.strftime("%H:%M")
    if not (ORDER_WINDOW[0] <= hhmm <= ORDER_WINDOW[1]):
        reasons.append(f"outside order window {ORDER_WINDOW}")
    if cfg is None:
        reasons.append("no config")
    else:
        if now.date().isoformat() > cfg["end_date"]:
            reasons.append("past end_date")
        if st["cost"] + ask * (1 + FEE) > cfg["cap_krw"]:
            reasons.append("cap reached")
    if ask is None or ask <= 0:
        reasons.append("no ask")
    elif cash < ask + max(1.0, round(ask * FEE)):
        reasons.append(f"cash {cash:.0f} < ask {ask:.0f} + fee")
    today = now.date().isoformat()
    return (not reasons, reasons)


def report(rows, last_price):
    st = state(rows)
    value = st["qty"] * last_price if last_price else None
    pnl = (value - st["cost"]) if value is not None else None
    marks = [r for r in rows if r.get("kind") == "mark"]
    peak, dd = None, 0.0
    for m in marks:
        v = m["pnl"]
        peak = v if peak is None else max(peak, v)
        dd = min(dd, v - peak)
    return dict(symbol=SYMBOL, qty=st["qty"], cost=st["cost"], avg=(st["cost"] / st["qty"]) if st["qty"] else None, deposits=st["deposits"], last_price=last_price, value=value, pnl_vs_cost=pnl, pnl_pct=(pnl / st["cost"]) if (pnl is not None and st["cost"]) else None, max_dd_won_of_marks=dd, buys=st["buys"], config=st["config"])


def run_once(t, live=False, now=None, path=LEDGER):
    now = now or datetime.now(KST)
    rows = load_ledger(path)
    st = state(rows)
    cfg = st["config"]
    cash = float(t.buying_power("KRW")["cashBuyingPower"])
    ob = t.orderbook(SYMBOL)
    ask = float(ob["asks"][0]["price"]) if ob.get("asks") else None
    ok, reasons = decide(now, cash, ask, st, cfg)
    if not ok:
        return append(dict(kind="skip", cash=cash, ask=ask, reasons=reasons, live=live), path)
    if not live:
        px = ask + 0.0
        fee = round(px * FEE)
        return append(dict(kind="buy", mode="paper", qty=1, price=px, fee=fee, cost=px + fee, cash_before=cash), path)
    res = t.place(SYMBOL, "BUY", "MARKET", quantity=1, client_order_id=f"dca-{now.strftime('%Y%m%d')}")
    return append(dict(kind="order", mode="live", qty=1, ask=ask, cash_before=cash, order=res), path)


def selftest():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "l.jsonl")
    append(dict(kind="config", cap_krw=3_000_000, end_date="2027-11-05", note="test"), path)
    append(dict(kind="deposit", amount=10_000), path)
    rows = load_ledger(path)
    st = state(rows)
    now = datetime(2026, 11, 9, 9, 3, tzinfo=KST)
    assert decide(now, 9_999.0, 82_000.0, st, st["config"]) == (False, ["cash 9999 < ask 82000 + fee"])
    assert decide(now, 90_000.0, 82_000.0, st, st["config"])[0]
    assert "outside order window" in decide(datetime(2026, 11, 9, 10, 0, tzinfo=KST), 90_000.0, 82_000.0, st, st["config"])[1][0]
    assert "past end_date" in decide(datetime(2027, 11, 6, 9, 3, tzinfo=KST), 90_000.0, 82_000.0, st, st["config"])[1]
    big = dict(st, cost=2_950_000.0)
    assert "cap reached" in decide(now, 90_000.0, 82_000.0, big, st["config"])[1]

    class FakeToss:
        def buying_power(self, c):
            return {"cashBuyingPower": "90000"}

        def orderbook(self, s):
            return {"asks": [{"price": "82000", "qty": "100"}], "bids": [{"price": "81995", "qty": "100"}]}

        def place(self, *a, **k):
            raise AssertionError("live order must not be called in paper mode")
    r = run_once(FakeToss(), live=False, now=now, path=path)
    assert r["kind"] == "buy" and r["mode"] == "paper" and r["qty"] == 1 and r["cost"] == 82_000 + round(82_000 * FEE), r
    rows = load_ledger(path)
    rep = report(rows, 83_000.0)
    assert rep["qty"] == 1 and rep["avg"] == r["cost"] and rep["pnl_vs_cost"] == 83_000.0 - r["cost"] and rep["deposits"] == 10_000, rep
    r2 = run_once(FakeToss(), live=False, now=datetime(2026, 11, 9, 11, 0, tzinfo=KST), path=path)
    assert r2["kind"] == "skip"
    print("selftest ok", dict(sha=sha_self()))


if __name__ == "__main__":
    import toss
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif sys.argv[1:2] == ["config"] and len(sys.argv) == 4:
        print(append(dict(kind="config", cap_krw=float(sys.argv[2]), end_date=sys.argv[3])))
    elif sys.argv[1:2] == ["deposit"] and len(sys.argv) == 3:
        print(append(dict(kind="deposit", amount=float(sys.argv[2]))))
    elif sys.argv[1:] == ["run"]:
        print(json.dumps(run_once(toss.Toss(), live=False), ensure_ascii=False))
    elif sys.argv[1:] == ["run", "--live"]:
        if os.environ.get("DOBBY_LIVE") != "1":
            raise SystemExit("live run requires DOBBY_LIVE=1 and the owner's explicit instruction")
        print(json.dumps(run_once(toss.Toss(), live=True), ensure_ascii=False))
    elif sys.argv[1:] == ["report"]:
        t = toss.Toss()
        px = float(t.prices([SYMBOL])[0]["lastPrice"])
        rows = load_ledger()
        rep = report(rows, px)
        append(dict(kind="mark", price=px, pnl=rep["pnl_vs_cost"] or 0.0, value=rep["value"]))
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    else:
        print("usage: dca_kr.py selftest | config <cap_krw> <end_date> | deposit <krw> | run [--live] | report")
