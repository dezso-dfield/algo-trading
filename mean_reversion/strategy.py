import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

# ===== Helpers =====
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        ts = df["timestamp"].astype(np.int64)
        # auto-detect ms vs s
        if ts.max() > 1e12:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp")
    df = df.sort_index()
    return df

def compute_indicators(df: pd.DataFrame, ema_period=20, bb_period=20, bb_std=2.0, rsi_period=14, adx_period=14):
    ema = EMAIndicator(close=df["close"], window=ema_period).ema_indicator()
    bb = BollingerBands(close=df["close"], window=bb_period, window_dev=bb_std)
    rsi = RSIIndicator(close=df["close"], window=rsi_period).rsi()
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=adx_period).adx()

    out = df.copy()
    out["ema"] = ema
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_up"]  = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_width"] = (out["bb_up"] - out["bb_low"]) / out["close"]
    out["rsi"] = rsi
    out["adx"] = adx
    return out

def detect_entries(row_prev, row_now, adx_max=20):
    """Return 'long'/'short'/None when confirmation occurs on row_now given row_prev context."""
    # Skip high trend
    if np.isfinite(row_now.get("adx", np.nan)) and row_now["adx"] > adx_max:
        return None

    # Long: prev close below lower band & prev rsi<30, now close back inside
    cond_long_prev = (row_prev["close"] < row_prev["bb_low"]) and (row_prev["rsi"] < 30)
    cond_long_now  = (row_now["close"] >= row_now["bb_low"])  # back inside
    # Short
    cond_short_prev = (row_prev["close"] > row_prev["bb_up"]) and (row_prev["rsi"] > 70)
    cond_short_now  = (row_now["close"] <= row_now["bb_up"])

    if cond_long_prev and cond_long_now:
        return "long"
    if cond_short_prev and cond_short_now:
        return "short"
    return None

def compute_stops_and_tps(side, entry_price, recent_extreme, ema_now, stop_buffer_pct=0.003, tp1_ratio=0.5, tp2_to_mean=True):
    if side == "long":
        stop = min(recent_extreme, entry_price) * (1 - stop_buffer_pct)
        tp1  = entry_price + (ema_now - entry_price) * tp1_ratio
        tp2  = ema_now if tp2_to_mean else None
    else:
        stop = max(recent_extreme, entry_price) * (1 + stop_buffer_pct)
        tp1  = entry_price - (entry_price - ema_now) * tp1_ratio
        tp2  = ema_now if tp2_to_mean else None
    return stop, tp1, tp2

class Position:
    def __init__(self, side, qty, entry, stop, tp1=None, tp2=None):
        self.side = side
        self.qty = qty
        self.entry = entry
        self.stop = stop
        self.tp1 = tp1
        self.tp2 = tp2
        self.realized = 0.0
        self.open = True
        self.partially_exited = False

    def mark(self, price):
        direction = 1 if self.side == "long" else -1
        return (price - self.entry) * direction * self.qty

    def check_exit(self, candle, fee_rate=0.0, scale_out=True):
        """Process stop/TP hits within the candle. Returns realized pnl and whether closed."""
        if not self.open:
            return 0.0, True

        high, low, close = candle["high"], candle["low"], candle["close"]
        realized = 0.0

        def hit_level(level, is_long, hit_on_low_first):
            # conservative fill ordering to avoid lookahead
            if hit_on_low_first:
                return low <= level if is_long else high >= level
            else:
                return high >= level if is_long else low <= level

        # Decide which side gets touched first based on candle direction proxy
        hit_on_low_first = candle["open"] > close

        # Check TP1 then Stop, or Stop then TP1 depending on sequence
        if scale_out and self.tp1 is not None and not self.partially_exited:
            if hit_level(self.tp1, self.side == "long", hit_on_low_first):
                # exit half at tp1
                qty_exit = self.qty * 0.5
                pnl = (self.tp1 - self.entry) * (1 if self.side == "long" else -1) * qty_exit
                fees = (abs(self.entry) + abs(self.tp1)) * fee_rate * qty_exit
                realized += pnl - fees
                self.qty -= qty_exit
                self.partially_exited = True

        # Check Stop
        stop_hit = (low <= self.stop) if self.side == "long" else (high >= self.stop)
        if stop_hit:
            qty_exit = self.qty
            pnl = (self.stop - self.entry) * (1 if self.side == "long" else -1) * qty_exit
            fees = (abs(self.entry) + abs(self.stop)) * fee_rate * qty_exit
            realized += pnl - fees
            self.qty = 0.0
            self.open = False
            return realized, True

        # Check TP2 or close at mean if provided
        if self.tp2 is not None:
            tp2_hit = (high >= self.tp2) if self.side == "long" else (low <= self.tp2)
            if tp2_hit:
                qty_exit = self.qty
                pnl = (self.tp2 - self.entry) * (1 if self.side == "long" else -1) * qty_exit
                fees = (abs(self.entry) + abs(self.tp2)) * fee_rate * qty_exit
                realized += pnl - fees
                self.qty = 0.0
                self.open = False
                return realized, True

        return realized, False
