import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        ts = df["timestamp"].astype(np.int64)
        unit = "ms" if ts.max() > 1e12 else "s"
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
        df = df.set_index("timestamp")
    return df.sort_index()

def compute_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df.copy()
    out["ema_s"] = EMAIndicator(close=out["close"], window=p["ema_short"]).ema_indicator()
    out["ema_m"] = EMAIndicator(close=out["close"], window=p["ema_mid"]).ema_indicator()
    out["ema_l"] = EMAIndicator(close=out["close"], window=p["ema_long"]).ema_indicator()
    out["atr"] = AverageTrueRange(high=out["high"], low=out["low"], close=out["close"], window=p["atr_period"]).average_true_range()
    return out

def in_triple_ema_up(row):
    return (row["ema_s"] > row["ema_m"]) and (row["ema_m"] > row["ema_l"])

def swing_low_stop(df_slice, atr=None, atr_mult=2.0):
    low = df_slice["low"].min()
    if atr is not None and np.isfinite(atr):
        return min(low, df_slice["close"].iloc[-1] - atr_mult * atr)
    return low

class Position:
    def __init__(self, side, qty, entry, stop):
        self.side = side  # 'long' only in this system
        self.qty = qty
        self.entry = entry
        self.stop = stop
        self.open = True

    def mark(self, price):
        return (price - self.entry) * self.qty

    def check_exit(self, candle, fee_rate=0.0):
        if not self.open: return 0.0, True
        price = candle["close"]
        # stop hit
        if candle["low"] <= self.stop:
            pnl = (self.stop - self.entry) * self.qty
            fees = (abs(self.entry) + abs(self.stop)) * fee_rate * self.qty
            self.open = False
            return pnl - fees, True
        # trend hierarchy broken
        if not in_triple_ema_up(candle):
            pnl = (price - self.entry) * self.qty
            fees = (abs(self.entry) + abs(price)) * fee_rate * self.qty
            self.open = False
            return pnl - fees, True
        return 0.0, False
