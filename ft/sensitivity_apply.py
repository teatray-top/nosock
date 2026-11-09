import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import fill_sensitivity as FSN
import fills_source as FS

OUT = os.path.join(HERE, "results", "conservative_fill_v1", "application_v1")
CAL_PATH = os.path.join(HERE, "session_calendar_kr_etf_v1.json")
SCALE = 10000
CAPITAL = 100_000_000
VARIANTS = {"baseline": dict(cap=False, tick=False), "tick_only": dict(cap=False, tick=True), "cap_only": dict(cap=True, tick=False), "both": dict(cap=True, tick=True)}
B_START, B_END = "2026-01-26 00:30:00", "2026-11-05"
A2_START, A2_END = "2024-01-16 02:40:00", "2025-12-30"
K200 = "user_data/data_kodex_nfi_v1_2_algo/binance/ALGO_USDT-5m.feather"
K150 = "user_data/data_kosdaq150_v1/binance/ALGO_USDT-5m.feather"
SERIES = {
    "069500_B_nfi_default": dict(kind="engine_zip", src="results/nfi_kr_v1_2/B/default", bars=K200, start=B_START, end=B_END, raw_ref=("results/holdout_229200_v1/parity_069500_B_v2_3/compare.json", "nfi_default")),
    "069500_B_nfi_single": dict(kind="engine_zip", src="results/nfi_kr_v1_2/B/single", bars=K200, start=B_START, end=B_END, raw_ref=("results/holdout_229200_v1/parity_069500_B_v2_3/compare.json", "nfi_single")),
    "229200_B_nfi_default_fillgate": dict(kind="fillgate_log", src="results/holdout_229200_v1/fillgate_B/default", bars=K150, start=B_START, end=B_END, raw_ref=("results/holdout_229200_v1/fillgate_B/compare/compare.json", "nfi_default_fillgate")),
    "229200_B_nfi_single_fillgate": dict(kind="fillgate_log", src="results/holdout_229200_v1/fillgate_B/single", bars=K150, start=B_START, end=B_END, raw_ref=("results/holdout_229200_v1/fillgate_B/compare/compare.json", "nfi_single_fillgate")),
    "069500_B_hold": dict(kind="ledger_csv", src="results/holdout_229200_v1/parity_069500_B_v2_3/ledger_hold.csv", bars=K200, start=B_START, end=B_END, raw_ref=("results/holdout_229200_v1/parity_069500_B_v2_3/compare.json", "hold")),
    "069500_B_ribbon": dict(kind="ledger_csv", src="results/holdout_229200_v1/parity_069500_B_v2_3/ledger_ribbon.csv", bars=K200, start=B_START, end=B_END, raw_ref=("results/holdout_229200_v1/parity_069500_B_v2_3/compare.json", "ribbon")),
    "229200_A2_hold": dict(kind="ledger_csv", src="results/holdout_229200_v1/compare_A2_v2_2/ledger_hold.csv", bars=K150, start=A2_START, end=A2_END, raw_ref=("results/holdout_229200_v1/compare_A2_v2_2/compare.json", "hold")),
    "229200_A2_ribbon": dict(kind="ledger_csv", src="results/holdout_229200_v1/compare_A2_v2_2/ledger_ribbon.csv", bars=K150, start=A2_START, end=A2_END, raw_ref=("results/holdout_229200_v1/compare_A2_v2_2/compare.json", "ribbon")),
    "229200_B_hold": dict(kind="ledger_csv", src="results/holdout_229200_v1/fillgate_B/compare/ledger_hold.csv", bars=K150, start=B_START, end=B_END, raw_ref=("results/holdout_229200_v1/fillgate_B/compare/compare.json", "hold")),
    "229200_B_ribbon": dict(kind="ledger_csv", src="results/holdout_229200_v1/fillgate_B/compare/ledger_ribbon.csv", bars=K150, start=B_START, end=B_END, raw_ref=("results/holdout_229200_v1/fillgate_B/compare/compare.json", "ribbon")),
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_orders(spec):
    p = os.path.join(HERE, spec["src"])
    if spec["kind"] == "engine_zip":
        orders, meta = FS.from_engine_zip(p, "NostalgiaForInfinityX8")
        start_capital = meta["starting_balance"] * SCALE
    elif spec["kind"] == "fillgate_log":
        orders, meta = FS.from_fillgate_log(os.path.join(p, "gate_log.json"), p, "NostalgiaForInfinityX8")
        start_capital = meta["starting_balance"] * SCALE
    else:
        df = pd.read_csv(p)
        orders = [dict(time=pd.Timestamp(r.time_utc), side=r.side, amount=float(r.shares), price=float(r.price_won) / SCALE, fee_rate=0.00015) for r in df.itertuples()]
        start_capital = CAPITAL
    won = [dict(time=o["time"] if o["time"].tzinfo else o["time"].tz_localize("UTC"), side=o["side"], shares=o["amount"], price=round(o["price"] * SCALE, 6)) for o in orders]
    return won, start_capital


def load_bars(path, start, end):
    b = pd.read_feather(os.path.join(HERE, path))
    b["date"] = pd.to_datetime(b["date"], utc=True)
    for c in ("open", "high", "low", "close"):
        b[c] = (b[c] * SCALE).round(6)
    day = b["date"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    return b[(b["date"] >= pd.Timestamp(start, tz="UTC")) & (day <= end)].reset_index(drop=True)


def raw_replay(orders, bars, capital):
    cash, held = float(capital), 0.0
    for o in sorted(orders, key=lambda o: (o["time"], o["side"] != "buy")):
        g = o["shares"] * o["price"]
        f = g * FSN.FEE
        if o["side"] == "buy":
            cash -= g + f
            held += o["shares"]
        else:
            cash += g - f
            held -= o["shares"]
    last = float(bars["close"].iloc[-1])
    return cash + held * last - held * last * FSN.FEE


def freeze():
    files = {"sensitivity_apply.py": sha(os.path.abspath(__file__)), "fill_sensitivity.py": sha(os.path.join(HERE, "fill_sensitivity.py")), "fills_source.py": sha(os.path.join(HERE, "fills_source.py")), "session_calendar_kr_etf_v1.json": sha(CAL_PATH)}
    for k, spec in SERIES.items():
        p = os.path.join(HERE, spec["src"])
        if spec["kind"] == "fillgate_log":
            files[spec["src"] + "/gate_log.json"] = sha(os.path.join(p, "gate_log.json"))
        if spec["kind"] in ("engine_zip", "fillgate_log"):
            import glob
            z = sorted(glob.glob(os.path.join(p, "*.zip")))[-1]
            files[os.path.relpath(z, HERE).replace("\\", "/")] = sha(z)
        else:
            files[spec["src"]] = sha(p)
        files[spec["bars"]] = sha(os.path.join(HERE, spec["bars"]))
        files[spec["raw_ref"][0]] = sha(os.path.join(HERE, spec["raw_ref"][0]))
    return files


def main():
    os.makedirs(OUT, exist_ok=False)
    inputs = freeze()
    json.dump(dict(version="sensitivity-application-v1-inputs", files=inputs), open(os.path.join(OUT, "inputs.json"), "x", encoding="utf-8"), indent=1)
    cal = json.load(open(CAL_PATH, encoding="utf-8"))["days"]
    summary = dict(version="fixed-order-fill-sensitivity-application-v1", label="fixed-order fill sensitivity: the strategy's original orders with degraded fills; baseline also applies the blocked-bar rule, so it is separate from the engine return; not a strategy re-execution and not a guarantee of executable return; bar volume is known only after the bar", capital_won=CAPITAL, fee_per_side=FSN.FEE, inputs_sha256=sha(os.path.join(OUT, "inputs.json")), series={})
    for name, spec in SERIES.items():
        orders, start_capital = load_orders(spec)
        bars = load_bars(spec["bars"], spec["start"], spec["end"])
        ref = json.load(open(os.path.join(HERE, spec["raw_ref"][0]), encoding="utf-8"))["methods"][spec["raw_ref"][1]]
        raw_nlv = raw_replay(orders, bars, start_capital)
        s = dict(window=[str(bars["date"].iloc[0]), str(bars["date"].iloc[-1])], orders=len(orders), start_capital_won=start_capital, start_capital_matches=abs(start_capital - CAPITAL) < 1e-6, raw_replay_nlv_won=round(raw_nlv), compare_raw_nlv_won=ref["net_liquidation_value_won"], raw_matches_compare=abs(round(raw_nlv) - ref["net_liquidation_value_won"]) <= 1, variants={})
        os.makedirs(os.path.join(OUT, name), exist_ok=False)
        for vname, opt in VARIANTS.items():
            r = FSN.run([dict(time=o["time"], side=o["side"], shares=o["shares"], price=o["price"]) for o in orders], bars, cal, CAPITAL, cap=opt["cap"], tick=opt["tick"])
            cons = FSN.check_conservation(r, CAPITAL)
            statuses = {}
            for a in r["accounting"]:
                statuses[a["status"]] = statuses.get(a["status"], 0) + 1
            rec = dict(series=name, variant=vname, options=opt, nlv_won=round(r["net_liquidation_value"]), return_pct=round(100 * (r["net_liquidation_value"] / CAPITAL - 1), 4), min_cash_won=round(r["min_cash"], 4), executable=r["executable"], held_end=r["held"], reserved_end=r["reserved"], notes=r["notes"], order_status_counts=statuses, conservation=cons, accounting=r["accounting"], fills=r["fills"])
            path = os.path.join(OUT, name, f"{vname}.json")
            json.dump(rec, open(path, "x", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
            s["variants"][vname] = {k: rec[k] for k in ("nlv_won", "return_pct", "min_cash_won", "executable", "held_end", "reserved_end", "order_status_counts")}
            s["variants"][vname].update(conservation_ok=all(cons.values()), buys_cut_shares=r["notes"]["buys_cut_shares"], sell_shares_cut_to_holdings=r["notes"]["sell_shares_cut_to_holdings"], sells_carried=r["notes"]["sells_carried"], fill_price_outside_ohlc=r["notes"]["fill_price_outside_ohlc"], blocked_attempts=r["notes"]["fills_blocked_zero_volume_or_auction"], file_sha256=sha(path))
        summary["series"][name] = s
        print(f"{name:32s} raw {s['raw_replay_nlv_won']:>12,} match {s['raw_matches_compare']} | " + " | ".join(f"{v} {s['variants'][v]['return_pct']:+.3f}% exe {s['variants'][v]['executable']} cons {s['variants'][v]['conservation_ok']}" for v in VARIANTS))
    p = os.path.join(OUT, "summary.json")
    json.dump(summary, open(p, "x", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("saved", p, sha(p)[:8], "runs", sum(len(s["variants"]) for s in summary["series"].values()))


if __name__ == "__main__":
    main()
