import glob
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUY_FEE, SELL_FEE = 0.00015, 0.00215
ADV = dict(profit_gt=0.0, pf_gt=1.1, trades_ge=30)
N_TESTS = 52
KR_DATADIR = os.path.join(HERE, "user_data", "data_kr_rt", "binance")
BIN_DATADIR = os.path.join(HERE, "user_data", "data", "binance")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_result(d):
    zips = sorted(glob.glob(os.path.join(d, "*.zip")))
    if not zips:
        return None, None
    with zipfile.ZipFile(zips[-1]) as f:
        name = [n for n in f.namelist() if n.endswith(".json") and not n.endswith("_config.json")][0]
        return json.loads(f.read(name)), zips[-1]


def last_error(log):
    if not os.path.exists(log):
        return None
    lines = [l.strip() for l in open(log, encoding="utf-8", errors="replace") if "ERROR" in l or "Error" in l or "Exception" in l]
    return lines[-1][:300] if lines else None


def fill_physics(st, datadir):
    import pandas as pd
    tf = st["timeframe"]
    cache = {}
    out = dict(fills=0, outside_input=0, price_out_of_bar_range=0, on_zero_volume_bar=0, zero_volume_examples=[])
    for t in st["trades"]:
        pair = t["pair"]
        if pair not in cache:
            df = pd.read_feather(os.path.join(datadir, pair.replace("/", "_") + f"-{tf}.feather"))
            df["date"] = pd.to_datetime(df["date"], utc=True)
            cache[pair] = df.set_index("date")
        b = cache[pair]
        for k, px in (("open_date", t["open_rate"]), ("close_date", t["close_rate"])):
            out["fills"] += 1
            ts = pd.Timestamp(t[k])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            if ts not in b.index:
                out["outside_input"] += 1
                continue
            bar = b.loc[ts]
            if not (bar["low"] - 1e-9 <= px <= bar["high"] + 1e-9):
                out["price_out_of_bar_range"] += 1
            if bar["volume"] == 0:
                out["on_zero_volume_bar"] += 1
                if len(out["zero_volume_examples"]) < 5:
                    out["zero_volume_examples"].append(dict(pair=pair, side=k, utc=str(ts), kst=str(ts.tz_convert("Asia/Seoul"))))
    out["violations"] = out["outside_input"] + out["price_out_of_bar_range"] + out["on_zero_volume_bar"]
    return out


def recompute_stock_cost(trades):
    delta = 0.0
    for t in trades:
        a, o, c = t["amount"], t["open_rate"], t["close_rate"]
        approx = a * c * (1 - t["fee_close"]) - a * o * (1 + t["fee_open"])
        exact = a * c * (1 - SELL_FEE) - a * o * (1 + BUY_FEE)
        delta += exact - approx
    return delta


def main(root, out_path):
    rows = []
    for market in sorted(os.listdir(root)):
        mdir = os.path.join(root, market)
        if not os.path.isdir(mdir) or market == "logs":
            continue
        for strat in sorted(os.listdir(mdir)):
            d = os.path.join(mdir, strat)
            exitf = os.path.join(d, ".exit")
            code = int(open(exitf).read().strip()) if os.path.exists(exitf) else None
            res, z = load_result(d)
            row = dict(market=market, strategy=strat, exit_code=code)
            if res is None or strat not in res.get("strategy", {}):
                row.update(status="error", error=last_error(os.path.join(root, "logs", f"{market}__{strat}.log")))
                rows.append(row)
                continue
            st = res["strategy"][strat]
            row.update(status="ok", result_zip=os.path.relpath(z, HERE), result_sha256=sha(z), timeframe=st.get("timeframe"), trades=st["total_trades"], profit_total_pct=round(100 * st["profit_total"], 2), profit_factor=st.get("profit_factor"), max_drawdown_account_pct=round(100 * (st.get("max_drawdown_account") or 0), 2), p_value=st.get("p_value"), winrate=st.get("winrate"), market_change_pct=round(100 * (st.get("market_change") or 0), 2))
            row["fill_physics"] = fill_physics(st, KR_DATADIR if market.startswith("kr") else BIN_DATADIR)
            row["fills_outside_input"] = row["fill_physics"]["outside_input"]
            row["fill_status"] = "clean" if row["fill_physics"]["violations"] == 0 else "FILL_IMPOSSIBLE_PRESENT (result not executable as reported; needs a fill-impossible re-execution rung)"
            if market.startswith("kr"):
                delta = recompute_stock_cost(st["trades"])
                row["stock_cost_recomputed_profit_pct"] = round(100 * (st["profit_total_abs"] + delta) / st["starting_balance"], 2)
                if strat == "HourBasedStrategy":
                    row["note"] = "reads the UTC clock hour; KR session is 00:00-06:30 UTC, so its hour rules see only those hours"
                row["advance"] = bool(row["stock_cost_recomputed_profit_pct"] > ADV["profit_gt"] and (row["profit_factor"] or 0) > ADV["pf_gt"] and row["trades"] >= ADV["trades_ge"] and row["fill_physics"]["violations"] == 0)
            else:
                row["cagr_pct"] = round(100 * (st.get("cagr") or 0), 2)
            row["p_below_bonferroni"] = bool(row["p_value"] is not None and row["p_value"] < 0.05 / N_TESTS)
            rows.append(row)
    out = dict(version="sweep-summary-v2", fill_physics_rule="a fill is impossible if its time is not an input bar, its price is outside that bar's low..high, or that bar has zero volume (KR: 15:20-15:30 closing call auction minutes and the 2026-06-08 09:05-09:33 stretch on all 10 stocks, consistent with a market-wide halt but cause unverified, are zero-volume provider bars); this is a post-run executability check, not a signal filter: signals still see every bar", code_sha256=sha(os.path.abspath(__file__)), root=root, stock_costs=dict(buy=BUY_FEE, sell=SELL_FEE), advance_rule=ADV, bonferroni_threshold=0.05 / N_TESTS, rows=rows)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for market in sorted({r["market"] for r in rows}):
        rs = [r for r in rows if r["market"] == market]
        ok = [r for r in rs if r["status"] == "ok"]
        key = "stock_cost_recomputed_profit_pct" if market.startswith("kr") else "profit_total_pct"
        ok.sort(key=lambda r: r[key], reverse=True)
        print(f"== {market}: {len(ok)} ok, {len(rs) - len(ok)} errors")
        for r in ok:
            extra = f"stockcost {r.get('stock_cost_recomputed_profit_pct', ''):>7} viol {r['fill_physics']['violations']}" if market.startswith("kr") else f"cagr {r.get('cagr_pct', ''):>7} viol {r['fill_physics']['violations']}"
            print(f"  {r['strategy']:38s} {r['timeframe']:>4s} trades {r['trades']:5d} profit {r['profit_total_pct']:8.2f}% {extra} pf {str(round(r['profit_factor'] or 0, 2)):>5s} dd {r['max_drawdown_account_pct']:6.2f}% p {str(round(r['p_value'], 4) if r['p_value'] is not None else None):>7s} mkt {r['market_change_pct']:7.2f}% {'ADVANCE' if r.get('advance') else ''}")
        for r in rs:
            if r["status"] != "ok":
                print(f"  ERROR {r['strategy']}: {r.get('error')}")
    print("saved", out_path, sha(out_path)[:8])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
