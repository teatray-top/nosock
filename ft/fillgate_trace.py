import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import fills_source as FS


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(params_path):
    P = json.load(open(os.path.join(HERE, params_path), encoding="utf-8"))
    bars = pd.read_feather(os.path.join(HERE, P["bars"]))
    bars["date"] = pd.to_datetime(bars["date"], utc=True)
    out = dict(version="fillgate-trace-v1", params=P, code_sha256=sha(os.path.abspath(__file__)), runs={})
    for name, src in P["runs"].items():
        log_path = os.path.join(HERE, src["log"])
        log = json.load(open(log_path, encoding="utf-8"))
        orders, meta = FS.from_fillgate_log(log_path, os.path.join(HERE, src["result_dir"]), "NostalgiaForInfinityX8")
        orders.sort(key=lambda o: (o["time"], o["side"] != "buy"))
        cash, held, trace, invariant_ok = float(meta["starting_balance"]), 0.0, [], True
        for o in orders:
            gross = o["amount"] * o["price"]
            fee = gross * o["fee_rate"]
            if o["side"] == "buy":
                cash -= gross + fee
                held += o["amount"]
            else:
                cash += gross - fee
                held -= o["amount"]
            if held < -1e-9 or cash < -1e-6:
                invariant_ok = False
            trace.append(dict(time_kst=str(o["time"].tz_convert("Asia/Seoul")), side=o["side"], amount=o["amount"], price_units=o["price"], fee_units=fee, cash_after=cash, held_after=held, tag=o["tag"], synthetic_terminal_exit=o["is_force_exit"]))
        open_amount = sum(t["amount"] for t in log["open_trades_at_exit"])
        last_close = float(bars["close"].iloc[-1])
        rec = dict(log=src["log"], log_sha256=sha(log_path), fills=len(orders), trace=trace, final_cash_units=cash, final_held=held, open_trades_at_exit=log["open_trades_at_exit"], open_amount_at_exit=open_amount, held_matches_open_trades=abs(held - open_amount) < 1e-9, cash_and_held_never_negative=invariant_ok, synthetic_terminal_exit_fills=[t for t in trace if t["synthetic_terminal_exit"]], blocked_attempts_unfilled=[dict(time_kst=str(pd.Timestamp(b["bar_utc"]).tz_convert("Asia/Seoul")), side=b["side"], amount=b["amount"], price_units=b["price"], reason=b["reason"], tag=b.get("tag"), is_force_exit=b.get("is_force_exit")) for b in log["blocked"]], engine_final_balance_units=meta["engine_final_balance"], engine_trades=meta["engine_trades"], self_equity_at_last_close_units=cash + held * last_close, self_cash_equals_engine_final=abs(cash - meta["engine_final_balance"]) < 1e-4 if open_amount == 0 else "not comparable: open position at exit", last_bar_utc=str(bars["date"].iloc[-1]))
        out["runs"][name] = rec
        print(name, json.dumps({k: rec[k] for k in ("fills", "final_cash_units", "final_held", "open_amount_at_exit", "held_matches_open_trades", "cash_and_held_never_negative", "engine_final_balance_units", "engine_trades", "self_cash_equals_engine_final", "last_bar_utc")}, default=str))
        print("   blocked:", [(b["time_kst"][:16], b["side"], b["reason"], b["is_force_exit"]) for b in rec["blocked_attempts_unfilled"]], "| synthetic terminal fills:", len(rec["synthetic_terminal_exit_fills"]))
    p = os.path.join(HERE, P["out"])
    json.dump(out, open(p, "x", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("saved", p, sha(p)[:8])


if __name__ == "__main__":
    main(sys.argv[1])
