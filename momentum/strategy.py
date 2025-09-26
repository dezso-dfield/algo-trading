import warnings
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, PSARIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# Silence PSAR/pandas deprecation spam
warnings.filterwarnings(
    "ignore",
    message="Series.__setitem__ treating keys as positions is deprecated",
    category=FutureWarning,
)

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set UTC datetime index from 'timestamp' (ms or s)."""
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        unit = "ms" if ts.max() > 1e12 else "s"
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
        df = df.set_index("timestamp")
    return df.sort_index()

def compute_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Compute indicators used by the momentum strategy."""
    out = df.copy()

    # EMAs
    ema_fast = EMAIndicator(close=out["close"], window=params.get("ema_fast", 20)).ema_indicator()
    ema_slow = EMAIndicator(close=out["close"], window=params.get("ema_slow", 50)).ema_indicator()

    # RSI
    rsi = RSIIndicator(close=out["close"], window=params.get("rsi_period", 14)).rsi()

    # MACD
    macd_ind = MACD(
        close=out["close"],
        window_fast=params.get("macd_fast", 12),
        window_slow=params.get("macd_slow", 26),
        window_sign=params.get("macd_signal", 9),
    )
    macd = macd_ind.macd()
    macd_signal = macd_ind.macd_signal()
    macd_hist = macd_ind.macd_diff()

    # OBV + ATR
    obv = OnBalanceVolumeIndicator(close=out["close"], volume=out["volume"]).on_balance_volume()
    atr = AverageTrueRange(
        high=out["high"], low=out["low"], close=out["close"], window=params.get("atr_period", 14)
    ).average_true_range()

    # Optional PSAR
    use_psar = params.get("use_psar", False)
    psar_vals = np.nan
    if use_psar:
        ps = PSARIndicator(
            high=out["high"], low=out["low"], close=out["close"],
            step=params.get("psar_step", 0.02), max_step=params.get("psar_max", 0.2)
        )
        psar_vals = ps.psar()

    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["rsi"] = rsi
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["obv"] = obv
    out["atr"] = atr
    out["psar"] = psar_vals
    return out

def momentum_signal(prev, now, params):
    """Return (long_ok, short_ok) based on EMA, MACD, RSI (core momentum)."""
    bull_trend = now["ema_fast"] > now["ema_slow"]
    bear_trend = now["ema_fast"] < now["ema_slow"]

    macd_bull = now["macd"] > now["macd_signal"]
    macd_bear = now["macd"] < now["macd_signal"]

    rsi_long_min = params.get("rsi_long_min", 50)
    rsi_short_max = params.get("rsi_short_max", 50)

    rsi_ok_long = now["rsi"] >= rsi_long_min
    rsi_ok_short = now["rsi"] <= rsi_short_max

    return bull_trend and macd_bull and rsi_ok_long, bear_trend and macd_bear and rsi_ok_short

def entry_filter_and_score(window: pd.DataFrame, now: pd.Series, params: dict):
    """
    Additional entry filter for LONGS and a momentum score.
    Requirements:
      - Breakout above prior N-bar high (exclude current bar)
      - Volume confirmation: vol >= vol_ma * min_ratio
      - OBV slope >= obv_slope_min
      - MACD histogram >= min_macd_hist
    Score combines MACD hist, RSI distance from 50, and OBV slope.
    """
    lb = int(params.get("breakout_lookback", 50))
    vol_ma_n = int(params.get("volume_ma", 20))
    vol_ratio_min = float(params.get("min_volume_ma_ratio", 1.1))
    obv_slope_min = float(params.get("obv_slope_min", 0.0))
    min_macd_hist = float(params.get("min_macd_hist", 0.0))
    require_breakout = bool(params.get("require_breakout", True))

    if len(window) < max(lb, vol_ma_n) + 1:
        return False, -1e9  # not enough data

    # Prior high (exclude current bar)
    prior = window.iloc[:-1]
    prior_high = float(prior["high"].tail(lb).max())
    price_ok = True if not require_breakout else float(now["close"]) > prior_high

    # Volume confirmation
    vol_ma = float(prior["volume"].tail(vol_ma_n).mean())
    vol_ok = float(now["volume"]) >= (vol_ma * vol_ratio_min) if vol_ma > 0 else False

    # OBV slope (simple)
    obv_slope = (float(window["obv"].iloc[-1]) - float(window["obv"].iloc[0])) / max(len(window), 1)
    obv_ok = obv_slope >= obv_slope_min

    # MACD hist threshold
    macd_hist_ok = float(now["macd_hist"]) >= min_macd_hist

    ok = price_ok and vol_ok and obv_ok and macd_hist_ok

    # Score: emphasize hist, then RSI distance, then OBV slope
    score = (float(now["macd_hist"]) * 1.0) + ((float(now["rsi"]) - 50.0) * 0.05) + (obv_slope * 1e-6)
    return ok, score

class Position:
    def __init__(self, side, qty, entry, stop=None):
        self.side = side  # 'long' or 'short'
        self.qty = float(qty)
        self.entry = float(entry)
        self.stop = None if stop is None else float(stop)
        self.open = True

    def mark(self, price):
        direction = 1.0 if self.side == "long" else -1.0
        return (float(price) - self.entry) * direction * self.qty

    def check_exit(self, candle, params, fee_rate=0.0):
        """Exit on (a) stop hit, (b) trend invalidation (EMA/MACD/optional PSAR)."""
        if not self.open:
            return 0.0, True

        price = float(candle["close"])
        ema_slow = float(candle["ema_slow"])
        macd = float(candle["macd"])
        macd_sig = float(candle["macd_signal"])
        psar_val = candle.get("psar", np.nan)
        use_psar = params.get("use_psar", False)

        # Hard stop
        if self.stop is not None:
            if self.side == "long" and float(candle["low"]) <= self.stop:
                pnl = (self.stop - self.entry) * self.qty
                fees = (abs(self.entry) + abs(self.stop)) * fee_rate * self.qty
                self.open = False
                return pnl - fees, True
            if self.side == "short" and float(candle["high"]) >= self.stop:
                pnl = (self.entry - self.stop) * self.qty
                fees = (abs(self.entry) + abs(self.stop)) * fee_rate * self.qty
                self.open = False
                return pnl - fees, True

        # Trend exits: EMA slow cross, MACD flip, PSAR (if enabled)
        if self.side == "long":
            psar_break = (price < float(psar_val)) if (use_psar and np.isfinite(psar_val)) else False
            if price < ema_slow or macd < macd_sig or psar_break:
                pnl = (price - self.entry) * self.qty
                fees = (abs(self.entry) + abs(price)) * fee_rate * self.qty
                self.open = False
                return pnl - fees, True
        else:
            psar_break = (price > float(psar_val)) if (use_psar and np.isfinite(psar_val)) else False
            if price > ema_slow or macd > macd_sig or psar_break:
                pnl = (self.entry - price) * self.qty
                fees = (abs(self.entry) + abs(price)) * fee_rate * self.qty
                self.open = False
                return pnl - fees, True

        return 0.0, False