import glob
import json
import os
import zipfile

import pandas as pd


def from_engine_zip(result_dir, strategy):
    z = sorted(glob.glob(os.path.join(result_dir, "*.zip")))[-1]
    with zipfile.ZipFile(z) as f:
        n = [x for x in f.namelist() if x.endswith(".json") and not x.endswith("_config.json")][0]
        st = json.loads(f.read(n))["strategy"][strategy]
    orders = []
    for t in st["trades"]:
        for o in t["orders"]:
            entry = bool(o["ft_is_entry"])
            fee_rate = t["fee_open"] if entry else t["fee_close"]
            orders.append(dict(time=pd.Timestamp(o["order_filled_timestamp"], unit="ms", tz="UTC"), side="buy" if entry else "sell", amount=o["amount"], price=o["safe_price"], fee_rate=fee_rate, tag=o.get("ft_order_tag"), is_force_exit=o.get("ft_order_tag") == "force_exit" or (not entry and t.get("exit_reason") == "force_exit" and o is t["orders"][-1])))
    meta = dict(source="engine_zip", path=z, trades=len(st["trades"]), engine_final_balance=st["final_balance"], starting_balance=st["starting_balance"], backtest_start=st["backtest_start"], wins=sum(1 for t in st["trades"] if t["profit_abs"] > 0), losses=sum(1 for t in st["trades"] if t["profit_abs"] <= 0), gross_win=sum(t["profit_abs"] for t in st["trades"] if t["profit_abs"] > 0), gross_loss=-sum(t["profit_abs"] for t in st["trades"] if t["profit_abs"] < 0))
    return orders, meta


def from_fillgate_log(log_path, result_dir=None, strategy=None):
    log = json.load(open(log_path, encoding="utf-8"))
    orders = []
    for f in log["fills"]:
        orders.append(dict(time=pd.Timestamp(f["bar_utc"]), side="buy" if f["is_entry"] else "sell", amount=f["amount"], price=f["price"], fee_rate=f["fee_rate"], tag=f.get("tag"), is_force_exit=bool(f.get("is_force_exit"))))
    meta = dict(source="fillgate_log", path=log_path, fills=len(orders), blocked=len(log["blocked"]), open_trades_at_exit=log["open_trades_at_exit"], blocked_force_exits=sum(1 for b in log["blocked"] if b.get("is_force_exit")))
    if result_dir and strategy:
        _, em = from_engine_zip(result_dir, strategy)
        meta.update(engine_final_balance=em["engine_final_balance"], starting_balance=em["starting_balance"], backtest_start=em["backtest_start"], engine_trades=em["trades"])
    return orders, meta


def replay(orders, bars, capital, drop_force_exits=False):
    orders = sorted(orders, key=lambda o: (o["time"], o["side"] != "buy"))
    cash, held = float(capital), 0.0
    used = []
    for o in orders:
        if drop_force_exits and o["is_force_exit"]:
            continue
        gross = o["amount"] * o["price"]
        fee = gross * o["fee_rate"]
        if o["side"] == "buy":
            cash -= gross + fee
            held += o["amount"]
        else:
            cash += gross - fee
            held -= o["amount"]
        used.append(o)
    last_close = float(bars.sort_values("date")["close"].iloc[-1])
    return dict(cash=cash, held=held, last_close=last_close, equity_at_last_close=cash + held * last_close, orders_used=len(used), force_exits_dropped=sum(1 for o in orders if o["is_force_exit"]) if drop_force_exits else 0)


def selftest():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    v2 = os.path.join(here, "fixtures", "fillgate_v2")
    bars2 = pd.read_feather(os.path.join(v2, "data", "binance", "ALGO_USDT-5m.feather"))
    o2, m2 = from_fillgate_log(os.path.join(v2, "out", "end", "gate_log.json"), os.path.join(v2, "out", "end"), "FillGateEnd")
    r2 = replay(o2, bars2, m2["starting_balance"])
    out["v2_end_self_ledger_equity_1007_448"] = abs(r2["equity_at_last_close"] - 1007.448) < 1e-6 and r2["held"] == 98.0 and abs(r2["cash"] - 19.902) < 1e-6
    out["v2_engine_export_differs_and_is_flagged"] = abs(m2["engine_final_balance"] - 19.902) < 1e-6 and m2["engine_trades"] == 0 and len(m2["open_trades_at_exit"]) == 1 and m2["blocked_force_exits"] == 1
    v1 = os.path.join(here, "fixtures", "fillgate_v1")
    bars1 = pd.read_feather(os.path.join(v1, "data", "binance", "ALGO_USDT-5m.feather"))
    o1, m1 = from_fillgate_log(os.path.join(v1, "out", "gated", "gate_log.json"), os.path.join(v1, "out", "gated"), "FillGateFixture")
    r1 = replay(o1, bars1, m1["starting_balance"])
    e1, em1 = from_engine_zip(os.path.join(v1, "out", "gated"), "FillGateFixture")
    re1 = replay(e1, bars1, em1["starting_balance"])
    out["v1_self_ledger_equals_engine_final_balance"] = r1["held"] == 0 and abs(r1["cash"] - m1["engine_final_balance"]) < 1e-6
    out["v1_self_ledger_equals_engine_zip_replay"] = abs(r1["cash"] - re1["cash"]) < 1e-9 and len(o1) == len(e1)
    out["v1_no_force_exits"] = not any(o["is_force_exit"] for o in o1)
    print(json.dumps(out, indent=1))
    if not all(out.values()):
        sys.exit(1)
    print("fills_source selftest ok")


if __name__ == "__main__":
    selftest()
