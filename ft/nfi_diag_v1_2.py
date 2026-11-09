import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

import freqtrade.data.dataprovider as DP
import freqtrade.data.history as H
import freqtrade.strategy.strategy_helper as SH
from ft_kr_merge_only import asof_merge

_orig_lph = DP.load_pair_history


def lph_nofill(*a, **k):
    k["fill_up_missing"] = False
    return _orig_lph(*a, **k)


DP.load_pair_history = lph_nofill
SH._merge_prepared_informative_pair = asof_merge

from freqtrade.configuration import Configuration, TimeRange
from freqtrade.enums import RunMode
from freqtrade.resolvers import StrategyResolver

SRC = Path("user_data/data_kodex_nfi/binance")
DATA = Path("user_data/data_kodex_nfi_v1_2/binance")
OUT = Path("results/nfi_kr_v1_2_diag")
FILES = ["BAT_USDT-5m.feather", "BAT_USDT-15m.feather", "BAT_USDT-1h.feather", "BAT_USDT-4h.feather", "BAT_USDT-1d.feather", "BTC_USDT-5m.feather", "BTC_USDT-4h.feather"]
WINDOWS = {"A2": "20240102-20251231", "B": "20260112-20261106"}
STARTUP = 800
CONFIGS = ["nfi_configs/trading_mode-spot.json", "nfi_configs/exampleconfig.json", "nfi_configs/blacklist-binance.json", "nfi_configs/pairs_kodex.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def freeze_inputs():
    DATA.mkdir(parents=True, exist_ok=False)
    out = {}
    for f in FILES:
        shutil.copyfile(SRC / f, DATA / f)
        a, b = sha(SRC / f), sha(DATA / f)
        if a != b:
            raise SystemExit(f"copy mismatch {f}")
        out[f] = b
    return out


def load_strategy(timerange):
    cfg = Configuration.from_files(CONFIGS)
    cfg["datadir"] = DATA
    cfg["user_data_dir"] = Path("user_data")
    cfg["strategy"] = "NostalgiaForInfinityX8"
    cfg["strategy_path"] = Path("user_data/strategies")
    cfg["runmode"] = RunMode.BACKTEST
    cfg["timerange"] = timerange
    strat = StrategyResolver.load_strategy(cfg)
    strat.dp = DP.DataProvider(cfg, None)
    return strat


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    inputs = freeze_inputs()
    summary = dict(version="nfi-kodex-classification-diagnostic-v1.2", scope="stored-data diagnostic only: indicators computed once per window with metadata BAT/USDT, then the same frame judged with metadata BAT/USDT and ALGO/USDT; strategy and thresholds unchanged; no market or API calls, no backtest", launcher_patches="load_pair_history fill_up_missing=False in the data provider; strategy_helper._merge_prepared_informative_pair replaced by the backward as-of merge (ft_kr_merge_only.asof_merge)", code_sha256=sha(os.path.abspath(__file__)), helper_sha256=sha("ft_kr_merge_only.py"), strategy_lf_sha256=hashlib.sha256(open("user_data/strategies/NostalgiaForInfinityX8.py", "rb").read().replace(b"\r\n", b"\n")).hexdigest(), inputs=inputs, windows={})
    for w, tr_s in WINDOWS.items():
        strat = load_strategy(tr_s)
        tr = TimeRange.parse_timerange(tr_s)
        tr_load = TimeRange.parse_timerange(tr_s)
        tr_load.subtract_start(300 * STARTUP)
        df = H.load_data(datadir=DATA, timeframe="5m", pairs=["BAT/USDT"], timerange=tr_load, fill_up_missing=False, data_format="feather")["BAT/USDT"]
        requested_load_start = pd.Timestamp(tr_load.startdt)
        first = df["date"].iloc[0]
        if first > requested_load_start:
            eff_start = df["date"].iloc[STARTUP]
            rule = "data began after the requested startup start, so the engine moves the start by 800 candles"
        else:
            eff_start = pd.Timestamp(tr.startdt)
            rule = "startup candles available before the window start"
        frame = strat.advise_indicators(df.copy(), {"pair": "BAT/USDT"})
        fpath = OUT / f"frame_{w}.feather"
        frame.reset_index(drop=True).to_feather(fpath)
        res = dict(timerange=tr_s, loaded_from=str(first), effective_start=str(eff_start), effective_start_rule=rule, end=str(df["date"].iloc[-1]), frame_file=str(fpath), frame_sha256=sha(fpath), frame_rows=len(frame), frame_columns=len(frame.columns), judged={})
        sigs = {}
        for meta in ("BAT/USDT", "ALGO/USDT"):
            coin = meta.partition("/")[0]
            e = strat.advise_entry(frame.copy(), {"pair": meta})
            win = e[(e["date"] >= eff_start) & (e["date"] <= pd.Timestamp(tr.stopdt))]
            s = win[win["enter_long"] == 1] if "enter_long" in win else win.iloc[0:0]
            tags = s["enter_tag"].astype(str).str.split().explode().value_counts().to_dict() if len(s) else {}
            spath = OUT / f"signals_{w}_{coin}.csv"
            s[["date", "enter_tag"]].to_csv(spath, index=False) if len(s) else pd.DataFrame(columns=["date", "enter_tag"]).to_csv(spath, index=False)
            sigs[coin] = set(s["date"]) if len(s) else set()
            res["judged"][meta] = dict(top_coins_mode=coin in strat.top_coins_mode_coins, grind_mode=coin in strat.grind_mode_coins, rows_in_window=len(win), enter_long_rows=int(len(s)), tag_counts=tags, tags_141_145={t: tags.get(t, 0) for t in ("141", "142", "143", "144", "145")}, signals_file=str(spath), signals_sha256=sha(spath))
        res["algo_only_signal_rows"] = len(sigs["ALGO"] - sigs["BAT"])
        res["bat_only_signal_rows"] = len(sigs["BAT"] - sigs["ALGO"])
        summary["windows"][w] = res
        print(w, json.dumps({k: v for k, v in res.items() if k != "judged"}, default=str))
        for meta, j in res["judged"].items():
            print("  ", meta, json.dumps(j, default=str))
    p = OUT / "summary.json"
    json.dump(summary, open(p, "x", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("saved", p, sha(p)[:8])


if __name__ == "__main__":
    main()
