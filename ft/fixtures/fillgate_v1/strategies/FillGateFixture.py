from freqtrade.strategy import IStrategy
from pandas import DataFrame

ENTRY_SIGNAL_KST = {"2026-01-05 09:00", "2026-01-05 15:15", "2026-01-06 09:55", "2026-11-19 15:15"}
EXIT_SIGNAL_KST = {"2026-01-05 09:55", "2026-01-05 10:00", "2026-11-19 16:15", "2026-11-19 16:20", "2026-11-19 16:25"}


class FillGateFixture(IStrategy):
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
        df["enter_long"] = df["kst"].isin(ENTRY_SIGNAL_KST).astype(int)
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = df["kst"].isin(EXIT_SIGNAL_KST).astype(int)
        return df
