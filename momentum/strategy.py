import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, PSARIndicator
import warnings
# Suppress ta.PSAR pandas FutureWarning
warnings.filterwarnings(
    "ignore",
    message="Series.__setitem__ treating keys as positions is deprecated",
    category=FutureWarning,
)
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        ts = df["timestamp"].astype(np.int64)
        if ts.max() > 1e12:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp")
    return df.sort_index()

def slope(series: pd.Series, lookback: int):
    if len(series) < lookback + 1:
        return np.nan
    y = series.iloc[-lookback:]
    x = np.arange(len(y))
    # simple linear regression slope
    denom = (x - x.mean())
    denom = (denom*denom).sum()
    if denom == 0:
        return 0.0
    m = ((x - x.mean())*(y - y.mean())).sum() / denom
    return float(m)

def compute_indicators(df: pd.DataFrame, params: dict):
    out = df.copy()
    ema_fast = EMAIndicator(close=out["close"], window=params["ema_fast"]).ema_indicator()
    ema_slow = EMAIndicator(close=out["close"], window=params["ema_slow"]).ema_indicator()
    rsi = RSIIndicator(close=out["close"], window=params["rsi_period"]).rsi()
    macd_ind = MACD(close=out["close"],
                    window_fast=params["macd_fast"],
                    window_slow=params["macd_slow"],
                    window_sign=params["macd_signal"])
    macd = macd_ind.macd()
    macd_signal = macd_ind.macd_signal()
    macd_hist = macd_ind.macd_diff()

    obv = OnBalanceVolumeIndicator(close=out["close"], volume=out["volume"]).on_balance_volume()
    atr = AverageTrueRange(high=out["high"], low=out["low"], close=out["close"], window=params["atr_period"]).average_true_range()
    use_psar = params.get("use_psar", True)
    if use_psar:
        psar = PSARIndicator(high=out["high"], low=out["low"], close=out["close"],
                             step=params["psar_step"], max_step=params["psar_max"])
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["rsi"] = rsi
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["obv"] = obv
    out["atr"] = atr
    if params.get("use_psar", True):
        out["psar"] = psar.psar()
        out["psar_up"] = psar.psar_up()
        out["psar_down"] = psar.psar_down()
    else:
        out["psar"] = pd.Series(index=out.index, dtype=float)
        out["psar_up"] = pd.Series(index=out.index, dtype=float)
        out["psar_down"] = pd.Series(index=out.index, dtype=float)
    return out

def momentum_signal(prev, now, params):
    # Trend alignment
    bull_trend = now["ema_fast"] > now["ema_slow"]
    bear_trend = now["ema_fast"] < now["ema_slow"]

    # MACD cross confirmation (line above/below signal)
    macd_bull = now["macd"] > now["macd_signal"]
    macd_bear = now["macd"] < now["macd_signal"]

    # RSI filter
    rsi_ok_long = now["rsi"] >= params["rsi_long_min"]
    rsi_ok_short = now["rsi"] <= params["rsi_short_max"]

    # OBV slope confirmation
    # We'll compute slope on the fly using last obv_lookback bars (caller can pass rolling window slice)
    # This function expects caller to ensure slices.
    # We'll treat positive slope as > obv_slope_min for longs, negative < -obv_slope_min for shorts.
    return bull_trend and macd_bull and rsi_ok_long, bear_trend and macd_bear and rsi_ok_short

class Position:
    def __init__(self, side, qty, entry, stop=None):
        self.side = side  # 'long' or 'short'
        self.qty = qty
        self.entry = entry
        self.stop = stop
        self.open = True

    def mark(self, price):
        direction = 1 if self.side == "long" else -1
        return (price - self.entry) * direction * self.qty

    def check_exit(self, candle, params, fee_rate=0.0):
        if not self.open:
            return 0.0, True
        price = candle["close"]
        psar = candle.get("psar", np.nan)
        ema_slow = candle["ema_slow"]
        macd = candle["macd"]
        macd_sig = candle["macd_signal"]

        realized = 0.0
        # Hard stop
        if self.stop is not None:
            if self.side == "long" and candle["low"] <= self.stop:
                pnl = (self.stop - self.entry) * self.qty
                fees = (abs(self.entry) + abs(self.stop)) * fee_rate * self.qty
                self.open = False
                return pnl - fees, True
            if self.side == "short" and candle["high"] >= self.stop:
                pnl = (self.entry - self.stop) * self.qty
                fees = (abs(self.entry) + abs(self.stop)) * fee_rate * self.qty
                self.open = False
                return pnl - fees, True

        # Trend exits: PSAR flip OR price cross ema_slow OR MACD opposite cross
        if self.side == "long":
            psar_ok = (not np.isfinite(psar)) or (price < psar)
            if price < ema_slow or macd < macd_sig or psar_ok:
                pnl = (price - self.entry) * self.qty
                fees = (abs(self.entry) + abs(price)) * fee_rate * self.qty
                self.open = False
                return pnl - fees, True
        else:
            if price > ema_slow or price > psar or macd > macd_sig:
                pnl = (self.entry - price) * self.qty
                fees = (abs(self.entry) + abs(price)) * fee_rate * self.qty
                self.open = False
                return pnl - fees, True

        return 0.0, False
