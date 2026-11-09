import atexit
import sys

import pandas as pd

import freqtrade.data.dataprovider as DP
import freqtrade.data.history as H
import freqtrade.strategy.strategy_helper as SH

CALLS = dict(load_data=0, dataprovider_load_pair_history=0, asof_merge=0)
_orig_load_data = H.load_data
_orig_dp_lph = DP.load_pair_history


def load_data_nofill(*args, **kwargs):
    kwargs["fill_up_missing"] = False
    CALLS["load_data"] += 1
    return _orig_load_data(*args, **kwargs)


def dp_load_pair_history_nofill(*args, **kwargs):
    kwargs["fill_up_missing"] = False
    CALLS["dataprovider_load_pair_history"] += 1
    return _orig_dp_lph(*args, **kwargs)


def asof_merge(dataframe, prepared, *, ffill=True):
    CALLS["asof_merge"] += 1
    informative = prepared.dataframe
    date_merge = prepared.date_merge
    if informative.empty:
        out = pd.merge(dataframe, informative, left_on="date", right_on=date_merge, how="left")
        return out.drop(date_merge, axis=1)
    inf = informative.sort_values(date_merge).reset_index(drop=True)
    left = dataframe.reset_index(drop=True)
    inf[date_merge] = inf[date_merge].astype(left["date"].dtype)
    if not left["date"].is_monotonic_increasing:
        raise ValueError("base dataframe dates not sorted")
    out = pd.merge_asof(left, inf, left_on="date", right_on=date_merge, direction="backward")
    return out.drop(date_merge, axis=1)


H.load_data = load_data_nofill
DP.load_pair_history = dp_load_pair_history_nofill
SH._merge_prepared_informative_pair = asof_merge


@atexit.register
def _report():
    print(f"ft_kr: no-fill loaders and as-of informative merge active; calls {CALLS}", file=sys.stderr)


from freqtrade.main import main

main(sys.argv[1:])
