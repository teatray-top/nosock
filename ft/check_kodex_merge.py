import sys

import pandas as pd

sys.argv = ["x"]
import ft_kr_merge_only as M

base = pd.read_feather("user_data/data_kodex_nfi/binance/BAT_USDT-5m.feather")
for tf in ("1d", "4h"):
    inf = pd.read_feather(f"user_data/data_kodex_nfi/binance/BAT_USDT-{tf}.feather")
    prepared = M.SH._prepare_informative_pair(inf, "5m", tf)
    exact = M.ORIG(base.copy(), prepared, ffill=False)
    asof = M.asof_merge(base.copy(), prepared, ffill=False)
    col = f"close_{tf}"
    print(tf, "exact-merge non-null rows", int(exact[col].notna().sum()), "of", len(exact), "| as-of non-null", int(asof[col].notna().sum()))
    for day in ("2024-03-05", "2025-06-10", "2026-07-31"):
        rows = asof[asof["date"].dt.strftime("%Y-%m-%d") == day]
        first = rows.iloc[0]
        prev = inf[inf["date"] < pd.Timestamp(day, tz="UTC")].iloc[-1] if tf == "1d" else None
        if tf == "1d":
            print(f"  {day} 09:00 KST sees close_1d {first[col]} from bar {first['date_1d']} | previous daily bar {prev['date']} close {prev['close']} | match {first[col] == prev['close']}")
        else:
            r1300 = rows[rows["date"].dt.strftime("%H:%M") == "04:00"].iloc[0]
            print(f"  {day} 09:00 KST sees 4h bar {first['date_4h']}; 13:00 KST sees 4h bar {r1300['date_4h']}")
