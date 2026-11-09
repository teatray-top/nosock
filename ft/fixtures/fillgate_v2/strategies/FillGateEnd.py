from freqtrade.strategy import IStrategy
from pandas import DataFrame


class FillGateEnd(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    minimal_roi = {"0": 10}
    stoploss = -0.99
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count = 0

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["kst"] = df["date"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M")
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = (df["kst"] == "2026-01-05 09:00").astype(int)
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        return df
