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


def check(orders, bars, cal, scale):
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], utc=True)
    idx = bars.set_index("date")
    v = dict(non_integer_shares=0, off_5won_grid=0, outside_bar_range=0, zero_volume_bar=0, auction_bucket=0, no_bar=0)
    max_share, examples = 0.0, []
    for o in orders:
        ts = o["time"].tz_convert("UTC") if o["time"].tzinfo else o["time"].tz_localize("UTC")
        bad = []
        if abs(o["amount"] - round(o["amount"])) > 1e-9:
            bad.append("non_integer_shares")
        won = round(o["price"] * scale, 6)
        if abs(won - round(won)) > 1e-6 or int(round(won)) % 5 != 0:
            bad.append("off_5won_grid")
        if ts not in idx.index:
            bad.append("no_bar")
        else:
            b = idx.loc[ts]
            if not (b["low"] - 1e-12 <= o["price"] <= b["high"] + 1e-12):
                bad.append("outside_bar_range")
            if b["volume"] <= 0:
                bad.append("zero_volume_bar")
            else:
                max_share = max(max_share, o["amount"] / b["volume"])
            kst = ts.tz_convert("Asia/Seoul")
            last = pd.Timestamp(f"{kst.strftime('%Y-%m-%d')} {cal[kst.strftime('%Y-%m-%d')]['last']}")
            if kst.strftime("%H:%M") in {(last - pd.Timedelta(minutes=10)).strftime("%H:%M"), (last - pd.Timedelta(minutes=5)).strftime("%H:%M")}:
                bad.append("auction_bucket")
        for k in bad:
            v[k] += 1
        if bad and len(examples) < 8:
            examples.append(dict(time_kst=str(ts.tz_convert("Asia/Seoul")), side=o["side"], amount=o["amount"], price=o["price"], issues=bad))
    return dict(orders=len(orders), violations=v, violations_total=sum(v.values()), max_order_share_of_bar_volume=max_share, examples=examples)


def main(params_path):
    P = json.load(open(params_path, encoding="utf-8"))
    bars = pd.read_feather(os.path.join(HERE, P["bars"]))
    cal = json.load(open(os.path.join(HERE, P["session_calendar"]), encoding="utf-8"))["days"]
    out = dict(version="fill-log-order-checks-v1", params=P, code_sha256=sha(os.path.abspath(__file__)), fills_source_sha256=sha(os.path.join(HERE, "fills_source.py")), runs={})
    for name, src in P["runs"].items():
        orders, meta = FS.from_fillgate_log(os.path.join(HERE, src["log"]), os.path.join(HERE, src["result_dir"]), src["strategy"])
        res = check(orders, bars, cal, P.get("scale", 10000))
        rep = FS.replay(orders, bars, meta["starting_balance"])
        rep_nf = FS.replay(orders, bars, meta["starting_balance"], drop_force_exits=True)
        res.update(source=meta["path"], source_sha256=sha(meta["path"]), blocked_attempts=meta["blocked"], blocked_force_exits=meta["blocked_force_exits"], open_trades_at_exit=len(meta["open_trades_at_exit"]), engine_final_balance=meta.get("engine_final_balance"), self_ledger=rep, self_ledger_without_force_exits=rep_nf, engine_matches_self_ledger=(len(meta["open_trades_at_exit"]) == 0 and abs(rep["cash"] - meta.get("engine_final_balance", float("nan"))) < 1e-4), status="clean" if res["violations_total"] == 0 else "FILL_IMPOSSIBLE_PRESENT")
        out["runs"][name] = res
        print(name, json.dumps({k: res[k] for k in ("orders", "violations_total", "blocked_attempts", "blocked_force_exits", "open_trades_at_exit", "engine_final_balance", "engine_matches_self_ledger", "status")}, default=str), "equity", round(rep["equity_at_last_close"], 6))
    p = os.path.join(HERE, P["out"])
    json.dump(out, open(p, "x", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("saved", p, sha(p)[:8])


if __name__ == "__main__":
    main(sys.argv[1])
