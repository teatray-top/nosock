import json
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


def count(config_files, datadir, pair, timerange):
    cfg = Configuration.from_files(config_files)
    cfg["datadir"] = Path(datadir)
    cfg["user_data_dir"] = Path("user_data")
    cfg["strategy"] = "NostalgiaForInfinityX8"
    cfg["strategy_path"] = Path("user_data/strategies")
    cfg["runmode"] = RunMode.BACKTEST
    cfg["timerange"] = timerange
    strat = StrategyResolver.load_strategy(cfg)
    dp = DP.DataProvider(cfg, None)
    strat.dp = dp
    tr = TimeRange.parse_timerange(timerange)
    tr.subtract_start(300 * 800)
    data = H.load_data(datadir=Path(datadir), timeframe="5m", pairs=[pair], timerange=tr, fill_up_missing=False, data_format="feather")
    df = data[pair]
    out = strat.ft_advise_signals(strat.advise_indicators(df.copy(), {"pair": pair}), {"pair": pair})
    start = TimeRange.parse_timerange(timerange).startdt
    w = out[out["date"] >= pd.Timestamp(start)]
    sig = w[w.get("enter_long", 0) == 1]
    tags = sig["enter_tag"].value_counts().head(10).to_dict() if "enter_tag" in sig else {}
    btc = w["btc_roc_1_5m"].describe().to_dict() if "btc_roc_1_5m" in w else None
    return dict(pair=pair, datadir=datadir, timerange=timerange, rows=len(w), enter_long=int(len(sig)), top_tags=tags, btc_roc_1_5m_eq_100_share=float((w["btc_roc_1_5m"] == 100.0).mean()) if "btc_roc_1_5m" in w else None)


if __name__ == "__main__":
    base = ["nfi_configs/trading_mode-spot.json", "nfi_configs/exampleconfig.json", "nfi_configs/blacklist-binance.json"]
    res = []
    res.append(count(base + ["nfi_configs/pairs_kodex.json"], "user_data/data_kodex_nfi/binance", "BAT/USDT", "20240102-20251231"))
    res.append(count(base + ["nfi_configs/pairs_kodex.json"], "user_data/data_kodex_nfi/binance", "BAT/USDT", "20260112-20261106"))
    res.append(count(base + ["nfi_configs/pairs_11.json"], "user_data/data/binance", "AVAX/USDT", "20251101-20261101"))
    res.append(count(base + ["nfi_configs/pairs_11.json"], "user_data/data/binance", "BTC/USDT", "20251101-20261101"))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
