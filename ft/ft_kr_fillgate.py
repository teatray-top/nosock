import atexit
import json
import os
import sys

import pandas as pd

import freqtrade.data.dataprovider as DP
import freqtrade.data.history as H
import freqtrade.strategy.strategy_helper as SH
from freqtrade.optimize.backtesting import Backtesting

CALLS = dict(load_data=0, dataprovider_load_pair_history=0, asof_merge=0, fill_attempts=0, fills_blocked=0)
BLOCKED_LOG = []
FILL_LOG = []
_orig_load_data = H.load_data
_orig_dp_lph = DP.load_pair_history
_orig_try_close = Backtesting._try_close_open_order


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
    out = pd.merge_asof(left, inf, left_on="date", right_on=date_merge, direction="backward")
    return out.drop(date_merge, axis=1)


def build_blocked():
    datadir = os.environ["FILLGATE_DATADIR"]
    pairs = os.environ["FILLGATE_PAIRS"].split(",")
    cal = json.load(open(os.environ["FILLGATE_CALENDAR"], encoding="utf-8"))["days"]
    blocked = {}
    for pair in pairs:
        path = os.path.join(datadir, pair.replace("/", "_") + "-5m.feather")
        if not os.path.exists(path):
            raise SystemExit(f"fill gate: no 5m data file for {pair}: {path}")
        df = pd.read_feather(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        kst = df["date"].dt.tz_convert("Asia/Seoul")
        day = kst.dt.strftime("%Y-%m-%d")
        hm = kst.dt.strftime("%H:%M")
        missing_days = sorted(set(day) - set(cal))
        if missing_days:
            raise SystemExit(f"fill gate: session calendar lacks {len(missing_days)} data dates for {pair}, e.g. {missing_days[:3]}")
        VALID[pair] = set(df["date"])
        auction = []
        for d, t in zip(day, hm):
            last = pd.Timestamp(f"{d} {cal[d]['last']}") if d in cal else None
            buckets = {(last - pd.Timedelta(minutes=10)).strftime("%H:%M"), (last - pd.Timedelta(minutes=5)).strftime("%H:%M")} if last is not None else set()
            auction.append(t in buckets)
        mask = (df["volume"] <= 0) | pd.Series(auction, index=df.index)
        blocked[pair] = dict((ts, "zero_volume" if v <= 0 else "auction_bucket") for ts, v in zip(df.loc[mask, "date"], df.loc[mask, "volume"]))
    return blocked


VALID = {}
BLOCKED = build_blocked()


def gated_try_close(self, order, trade, current_date, row):
    if order is None:
        return _orig_try_close(self, order, trade, current_date, row)
    CALLS["fill_attempts"] += 1
    ts = pd.Timestamp(current_date)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if trade.pair not in BLOCKED:
        raise RuntimeError(f"fill gate: pair {trade.pair} is not registered in FILLGATE_PAIRS")
    if ts not in VALID[trade.pair]:
        raise RuntimeError(f"fill gate: fill attempt at {ts} is not an input bar of {trade.pair}")
    reason = BLOCKED[trade.pair].get(ts)
    if reason is not None:
        CALLS["fills_blocked"] += 1
        BLOCKED_LOG.append(dict(pair=trade.pair, bar_utc=str(ts), side=order.ft_order_side, is_entry=order.ft_order_side == trade.entry_side, price=order.ft_price, amount=order.safe_amount, tag=order.ft_order_tag, is_force_exit=order.ft_order_tag == "force_exit", reason=reason))
        return False
    filled = _orig_try_close(self, order, trade, current_date, row)
    if filled:
        is_entry = order.ft_order_side == trade.entry_side
        fee_rate = trade.fee_open if is_entry else trade.fee_close
        FILL_LOG.append(dict(pair=trade.pair, trade_open_date=str(trade.open_date_utc), bar_utc=str(ts), side=order.ft_order_side, is_entry=is_entry, price=order.ft_price, amount=order.safe_amount, fee_rate=fee_rate, fee=order.ft_price * order.safe_amount * fee_rate, tag=order.ft_order_tag, is_force_exit=order.ft_order_tag == "force_exit"))
    return filled


def open_trades_snapshot():
    from freqtrade.persistence import LocalTrade
    out = []
    for t in list(getattr(LocalTrade, "bt_trades_open", [])):
        out.append(dict(pair=t.pair, open_date=str(t.open_date_utc), amount=t.amount, open_rate=t.open_rate, filled_orders=[dict(side=o.ft_order_side, amount=o.safe_amount, price=o.ft_price, filled=str(o.order_filled_utc)) for o in t.orders if not o.ft_is_open], open_orders=[dict(side=o.ft_order_side, amount=o.safe_amount, price=o.ft_price) for o in t.orders if o.ft_is_open]))
    return out


H.load_data = load_data_nofill
DP.load_pair_history = dp_load_pair_history_nofill
SH._merge_prepared_informative_pair = asof_merge
Backtesting._try_close_open_order = gated_try_close


@atexit.register
def _report():
    log_path = os.environ.get("FILLGATE_LOG")
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(dict(calls=CALLS, blocked=BLOCKED_LOG, fills=FILL_LOG, open_trades_at_exit=open_trades_snapshot(), note="fills and open_trades_at_exit are the launcher's own ledger; the engine export can drop a trade whose final forced exit was blocked"), f, indent=1, default=str)
    print(f"ft_kr_fillgate: no-fill loaders, as-of informative merge, fill gate (zero-volume and date auction buckets); calls {CALLS}", file=sys.stderr)


from freqtrade.main import main

main(sys.argv[1:])
