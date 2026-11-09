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


def fills_outside(st, datadir):
    import pandas as pd
    tf = st["timeframe"]
    cache, bad = {}, 0
    for t in st["trades"]:
        pair = t["pair"]
        if pair not in cache:
            df = pd.read_feather(os.path.join(datadir, pair.replace("/", "_") + f"-{tf}.feather"))
            cache[pair] = set(pd.to_datetime(df["date"], utc=True))
        for k in ("open_date", "close_date"):
            ts = pd.Timestamp(t[k])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            if ts not in cache[pair]:
                bad += 1
    return bad


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
            if market.startswith("kr"):
                row["fills_outside_input"] = fills_outside(st, KR_DATADIR)
                delta = recompute_stock_cost(st["trades"])
                row["stock_cost_recomputed_profit_pct"] = round(100 * (st["profit_total_abs"] + delta) / st["starting_balance"], 2)
                if strat == "HourBasedStrategy":
                    row["note"] = "reads the UTC clock hour; KR session is 00:00-06:30 UTC, so its hour rules see only those hours"
                row["advance"] = bool(row["stock_cost_recomputed_profit_pct"] > ADV["profit_gt"] and (row["profit_factor"] or 0) > ADV["pf_gt"] and row["trades"] >= ADV["trades_ge"] and row["fills_outside_input"] == 0)
            else:
                row["cagr_pct"] = round(100 * (st.get("cagr") or 0), 2)
            row["p_below_bonferroni"] = bool(row["p_value"] is not None and row["p_value"] < 0.05 / N_TESTS)
            rows.append(row)
    out = dict(version="sweep-summary-v1", code_sha256=sha(os.path.abspath(__file__)), root=root, stock_costs=dict(buy=BUY_FEE, sell=SELL_FEE), advance_rule=ADV, bonferroni_threshold=0.05 / N_TESTS, rows=rows)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for market in sorted({r["market"] for r in rows}):
        rs = [r for r in rows if r["market"] == market]
        ok = [r for r in rs if r["status"] == "ok"]
        key = "stock_cost_recomputed_profit_pct" if market.startswith("kr") else "profit_total_pct"
        ok.sort(key=lambda r: r[key], reverse=True)
        print(f"== {market}: {len(ok)} ok, {len(rs) - len(ok)} errors")
        for r in ok:
            extra = f"stockcost {r.get('stock_cost_recomputed_profit_pct', ''):>7} outside {r.get('fills_outside_input')}" if market.startswith("kr") else f"cagr {r.get('cagr_pct', ''):>7}"
            print(f"  {r['strategy']:38s} {r['timeframe']:>4s} trades {r['trades']:5d} profit {r['profit_total_pct']:8.2f}% {extra} pf {str(round(r['profit_factor'] or 0, 2)):>5s} dd {r['max_drawdown_account_pct']:6.2f}% p {str(round(r['p_value'], 4) if r['p_value'] is not None else None):>7s} mkt {r['market_change_pct']:7.2f}% {'ADVANCE' if r.get('advance') else ''}")
        for r in rs:
            if r["status"] != "ok":
                print(f"  ERROR {r['strategy']}: {r.get('error')}")
    print("saved", out_path, sha(out_path)[:8])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
