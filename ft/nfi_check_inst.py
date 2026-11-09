import glob
import hashlib
import json
import math
import os
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(sys.argv[1], encoding="utf-8"))
ROOT = os.path.join(HERE, P["root"])
DATA = os.path.join(HERE, P["bars"])
SCALE = 10000
FEE = 0.00015
FIXED_START = P["fixed_start"]
CAL = json.load(open(os.path.join(HERE, P["session_calendar"]), encoding="utf-8"))["days"]


def auction_buckets(kst_day):
    last = pd.Timestamp(f"{kst_day} {CAL[kst_day]['last']}")
    return {(last - pd.Timedelta(minutes=10)).strftime("%H:%M"), (last - pd.Timedelta(minutes=5)).strftime("%H:%M")}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def on_grid(price_units):
    won = round(price_units * SCALE, 6)
    return abs(won - round(won)) < 1e-6 and int(round(won)) % 5 == 0


def tick_adjust(price_units, is_buy):
    won = price_units * SCALE
    adj = math.ceil(won / 5 - 1e-9) * 5 if is_buy else math.floor(won / 5 + 1e-9) * 5
    return adj / SCALE


def main():
    bars = pd.read_feather(DATA)
    bars["date"] = pd.to_datetime(bars["date"], utc=True)
    bars = bars.set_index("date")
    report = dict(version="nfi-instrument-order-checks-v1", params=P, code_sha256=sha(os.path.abspath(__file__)), data_sha256=sha(DATA), rule="per order: integer shares, 5-won grid, inside the 5m bar range, bar volume > 0, not in the closing-auction buckets of that date (the two 5m buckets in the 10 minutes before the date's last bar label: 15:20/15:25 on normal and late-open days, 16:20/16:25 on shifted Suneung days, from session_calendar); violations are reported as violations", runs={})
    for d in sorted(glob.glob(os.path.join(ROOT, "*", "*"))):
        if not os.path.isdir(d):
            continue
        w, v = d.split(os.sep)[-2:]
        z = sorted(glob.glob(os.path.join(d, "*.zip")))
        if not z:
            continue
        with zipfile.ZipFile(z[-1]) as f:
            n = [x for x in f.namelist() if x.endswith(".json") and not x.endswith("_config.json")][0]
            st = json.loads(f.read(n))["strategy"]["NostalgiaForInfinityX8"]
        run = dict(result_zip=os.path.relpath(z[-1], HERE), result_sha256=sha(z[-1]), backtest_start=st["backtest_start"], start_matches_fixed=st["backtest_start"] == FIXED_START[w], trades=st["total_trades"], profit_pct=round(100 * st["profit_total"], 4), final_balance=st["final_balance"], profit_factor=st.get("profit_factor"), max_drawdown_account=st.get("max_drawdown_account"), market_change_pct=round(100 * (st.get("market_change") or 0), 2), exit_reasons={}, orders=0, violations=dict(non_integer_shares=0, off_5won_grid=0, outside_bar_range=0, zero_volume_bar=0, excluded_auction_bucket=0, no_bar=0), examples=[])
        events = []
        for t in st["trades"]:
            run["exit_reasons"][t["exit_reason"]] = run["exit_reasons"].get(t["exit_reason"], 0) + 1
            for o in t["orders"]:
                run["orders"] += 1
                ts = pd.Timestamp(o["order_filled_timestamp"], unit="ms", tz="UTC")
                bar_ts = ts.floor("5min")
                px, amt, buy = o["safe_price"], o["amount"], bool(o["ft_is_entry"])
                events.append((ts, buy, px, amt, bar_ts))
                bad = []
                if abs(amt - round(amt)) > 1e-9:
                    bad.append("non_integer_shares")
                if not on_grid(px):
                    bad.append("off_5won_grid")
                if bar_ts not in bars.index:
                    bad.append("no_bar")
                else:
                    b = bars.loc[bar_ts]
                    if not (b["low"] - 1e-12 <= px <= b["high"] + 1e-12):
                        bad.append("outside_bar_range")
                    if b["volume"] == 0:
                        bad.append("zero_volume_bar")
                kst = bar_ts.tz_convert("Asia/Seoul")
                if kst.strftime("%H:%M") in auction_buckets(kst.strftime("%Y-%m-%d")):
                    bad.append("excluded_auction_bucket")
                for k in bad:
                    run["violations"][k] += 1
                if bad and len(run["examples"]) < 8:
                    run["examples"].append(dict(pair=t["pair"], side="buy" if buy else "sell", kst=str(ts.tz_convert("Asia/Seoul")), won=round(px * SCALE, 2), shares=amt, issues=bad))
                if bar_ts in bars.index and bars.loc[bar_ts]["volume"] > 0:
                    share = amt / bars.loc[bar_ts]["volume"]
                    run["max_order_share_of_bar_volume"] = max(run.get("max_order_share_of_bar_volume", 0.0), share)
                    if share > 0.05:
                        run["orders_above_5pct_of_bar_volume"] = run.get("orders_above_5pct_of_bar_volume", 0) + 1
        events.sort(key=lambda e: (e[0], not e[1]))
        for label, adjust in (("as_reported", False), ("tick5_adverse", True)):
            cash = float(st["starting_balance"])
            min_cash = cash
            out_range = 0
            for ts, buy, px, amt, bar_ts in events:
                p = tick_adjust(px, buy) if adjust else px
                if adjust and bar_ts in bars.index:
                    b = bars.loc[bar_ts]
                    if not (b["low"] - 1e-12 <= p <= b["high"] + 1e-12):
                        out_range += 1
                cash += (-amt * p * (1 + FEE)) if buy else (amt * p * (1 - FEE))
                min_cash = min(min_cash, cash)
            run[label] = dict(final_cash_units=round(cash, 6), final_won=round(cash * SCALE), return_pct=round(100 * (cash / st["starting_balance"] - 1), 4), min_cash_units=round(min_cash, 6), changed_prices_outside_bar=out_range if adjust else None)
        run["as_reported"]["parity_with_engine"] = abs(run["as_reported"]["final_cash_units"] - st["final_balance"]) < 1e-4
        run["violations_total"] = sum(run["violations"].values())
        report["runs"][f"{w}/{v}"] = run
        print(w, v, json.dumps({k: run[k] for k in ("backtest_start", "start_matches_fixed", "trades", "orders", "profit_pct", "profit_factor", "market_change_pct", "exit_reasons", "violations", "as_reported", "tick5_adverse")}, ensure_ascii=False, default=str))
        for ex in run["examples"]:
            print("   ", ex)
    out = os.path.join(ROOT, "order_checks.json")
    json.dump(report, open(out, "x", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("saved", out, sha(out)[:8])


if __name__ == "__main__":
    main()
