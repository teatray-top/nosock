import pandas as pd

import freqtrade.strategy.strategy_helper as SH

ORIG = SH._merge_prepared_informative_pair


def asof_merge(dataframe, prepared, *, ffill=True):
    informative = prepared.dataframe
    date_merge = prepared.date_merge
    inf = informative.sort_values(date_merge).reset_index(drop=True)
    left = dataframe.reset_index(drop=True)
    inf[date_merge] = inf[date_merge].astype(left["date"].dtype)
    out = pd.merge_asof(left, inf, left_on="date", right_on=date_merge, direction="backward")
    return out.drop(date_merge, axis=1)
