# strategy.py
import numpy as np
import pandas as pd

# ---------- core technicals ----------
def _ema(s: pd.Series, n: int) -> pd.Series:
    n = int(n)
    return s.ewm(span=n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    n = int(n)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    n = int(n)
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    macd_signal = _ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum()

def _rolling_z(x: pd.Series, win: int) -> pd.Series:
    win = int(win)
    m = x.rolling(win, min_periods=win).mean()
    s = x.rolling(win, min_periods=win).std(ddof=0)
    return (x - m) / (s + 1e-12)

def _slope(series: pd.Series, win: int) -> pd.Series:
    win = int(win)
    return (series - series.shift(win)) / float(win)

# ---------- IO helpers ----------
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Input must have 'timestamp' column")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        # Try ms epoch first (Binance), then ISO
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        except Exception:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()

# ---------- pipeline ----------
def compute_indicators(df_idxed: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Build a rich feature set the entry filter expects.
    Do NOT drop rows aggressively; the forward runner handles warm-up.
    """
    out = df_idxed.copy()
    p = params or {}

    # numeric dtypes
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # params
    ema_fast_n = int(p.get("ema_fast", 20))
    ema_slow_n = int(p.get("ema_slow", 50))
    rsi_n      = int(p.get("rsi_period", 14))
    atr_n      = int(p.get("atr_period", 14))
    macd_f     = int(p.get("macd_fast", 12))
    macd_s     = int(p.get("macd_slow", 26))
    macd_sig   = int(p.get("macd_signal", 9))
    vol_ma_n   = int(p.get("volume_ma", 20))
    z_win      = int(p.get("zscore_win", 200))
    regime_win = int(p.get("regime_win", 500))
    brk_lb     = int(p.get("breakout_lookback", 50))

    # core indicators
    out["ema_fast"] = _ema(out["close"], ema_fast_n)
    out["ema_slow"] = _ema(out["close"], ema_slow_n)
    out["rsi"] = _rsi(out["close"], rsi_n)
    out["atr"] = _atr(out, atr_n)
    out["macd"], out["macd_signal"], out["macd_hist"] = _macd(out["close"], macd_f, macd_s, macd_sig)
    out["obv"] = _obv(out["close"], out["volume"])

    # volume features
    out["vol_ma"] = out["volume"].rolling(vol_ma_n, min_periods=max(3, vol_ma_n // 3)).mean()
    out["vol_ratio"] = out["volume"] / (out["vol_ma"] + 1e-12)

    # prior breakout levels (exclude current bar)
    out["prior_high"] = out["high"].shift(1).rolling(brk_lb, min_periods=max(5, brk_lb // 5)).max()
    out["prior_low"]  = out["low"].shift(1).rolling(brk_lb, min_periods=max(5, brk_lb // 5)).min()

    # slopes & normalizations
    out["ema_slow_slope"] = _slope(out["ema_slow"], win=max(3, ema_slow_n // 5))
    out["atr_pct"] = out["atr"] / (out["close"].abs() + 1e-12)
    out["macd_hist_z"] = _rolling_z(out["macd_hist"], max(30, z_win))

    # ATR% percentile for regime gating
    def _percentile_of_last(s: pd.Series, w: int) -> pd.Series:
        w = int(w)
        # percentile of the last item within the rolling window
        return s.rolling(w, min_periods=w).apply(
            lambda arr: (np.searchsorted(np.sort(arr), arr[-1], side="right") / len(arr)),
            raw=True,
        )

    out["atrpct_pct"] = _percentile_of_last(out["atr_pct"].fillna(0.0), regime_win)

    # placeholder for PSAR if you later enable it
    out["psar"] = np.nan

    # sanitize but DO NOT drop rows here
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

# ---------- regime & signals ----------
def regime_ok(now: pd.Series, params: dict) -> bool:
    low_cut  = float(params.get("atrpct_pct_low", 0.15))
    high_cut = float(params.get("atrpct_pct_high", 0.95))
    pct = float(now.get("atrpct_pct", np.nan))
    if not np.isfinite(pct) or not (low_cut <= pct <= high_cut):
        return False

    slope_min = float(params.get("ema_slope_min", 0.0))
    slope = float(now.get("ema_slow_slope", np.nan))
    if not np.isfinite(slope) or abs(slope) < slope_min:
        return False
    return True

def momentum_signal(prev: pd.Series, now: pd.Series, params: dict, side: str | None = None):
    """
    Core bias: EMA fast/slow + MACD alignment + RSI threshold + slope alignment.
    Returns (long_ok, short_ok). If side is provided, the other is False.
    """
    bull = (now.get("ema_fast", np.nan) > now.get("ema_slow", np.nan))
    bear = (now.get("ema_fast", np.nan) < now.get("ema_slow", np.nan))
    macd_bull = now.get("macd", np.nan) > now.get("macd_signal", np.nan)
    macd_bear = now.get("macd", np.nan) < now.get("macd_signal", np.nan)

    rsi_long_min  = float(params.get("rsi_long_min", 50))
    rsi_short_max = float(params.get("rsi_short_max", 50))
    ema_slope_min = float(params.get("ema_slope_min", 0.0))
    slope = float(now.get("ema_slow_slope", 0.0))

    long_ok  = bool(bull and macd_bull and (now.get("rsi", 50) >= rsi_long_min)  and (slope >= +ema_slope_min))
    short_ok = bool(bear and macd_bear and (now.get("rsi", 50) <= rsi_short_max) and (slope <= -ema_slope_min))

    if side == "long":
        return long_ok, False
    if side == "short":
        return False, short_ok
    return long_ok, short_ok

def entry_filter_and_score(window: pd.DataFrame, now: pd.Series, params: dict, side: str = "long"):
    """
    Extra entry filter:
    - Regime gate
    - Breakout over prior high/low (optionally with pullback confirmation)
    - Volume confirmation (vol_ratio)
    - OBV slope sign
    - MACD histogram z-score strength
    Returns (ok, score) with a standardized score to rank candidates.
    """
    if not regime_ok(now, params):
        return False, -1e9

    lb = int(params.get("breakout_lookback", 50))
    vol_ma_n = int(params.get("volume_ma", 20))
    vol_ratio_min = float(params.get("min_volume_ma_ratio", 1.1))
    obv_slope_min = float(params.get("obv_slope_min", 0.0))
    min_macd_hist_z = float(params.get("min_macd_hist_z", 0.0))
    require_breakout = bool(params.get("require_breakout", True))
    pullback_after_breakout = bool(params.get("pullback_after_breakout", False))

    if len(window) < max(lb, vol_ma_n) + 1:
        return False, -1e9

    prior = window.iloc[:-1]
    try:
        prior_high = float(prior["high"].tail(lb).max())
        prior_low  = float(prior["low"].tail(lb).min())
    except Exception:
        return False, -1e9

    price = float(now.get("close", np.nan))
    if not np.isfinite(price):
        return False, -1e9

    # volume confirmation
    vol_ma = float(prior["volume"].tail(vol_ma_n).mean())
    vol_ok = (float(now.get("volume", 0.0)) >= vol_ma * vol_ratio_min) if vol_ma > 0 else False

    # OBV slope over the window
    try:
        obv_slope = (float(window["obv"].iloc[-1]) - float(window["obv"].iloc[0])) / max(len(window), 1)
    except Exception:
        obv_slope = np.nan
    obv_ok = (obv_slope >= obv_slope_min) if side == "long" else (obv_slope <= -obv_slope_min)

    # MACD hist z strength
    macd_hist_z = float(now.get("macd_hist_z", np.nan))
    macd_ok = (macd_hist_z >= min_macd_hist_z) if side == "long" else ((-macd_hist_z) >= min_macd_hist_z)

    # breakout logic (exclude current bar)
    price_ok = True
    if side == "long":
        if require_breakout:
            price_ok = price > prior_high
        if pullback_after_breakout and require_breakout:
            prev = window.iloc[-2]
            price_ok = bool((prev["close"] > prior_high) and (price > prior_high))
        ok = price_ok and vol_ok and obv_ok and macd_ok
        score = (macd_hist_z) + ((float(now.get("rsi", 50.0)) - 50.0) * 0.05) + (obv_slope * 1e-6 if np.isfinite(obv_slope) else 0.0)
    else:
        if require_breakout:
            price_ok = price < prior_low
        if pullback_after_breakout and require_breakout:
            prev = window.iloc[-2]
            price_ok = bool((prev["close"] < prior_low) and (price < prior_low))
        ok = price_ok and vol_ok and obv_ok and macd_ok
        score = (-macd_hist_z) + ((50.0 - float(now.get("rsi", 50.0))) * 0.05) + ((-obv_slope) * 1e-6 if np.isfinite(obv_slope) else 0.0)

    return bool(ok), float(score)

# ---------- sizing & levels ----------
def atr_position_size(equity, price, atr, target_risk_frac: float = 0.004, risk_multiple: float = 1.5):
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(price) or price <= 0:
        return 0.0
    risk_per_unit = risk_multiple * atr
    dollars_at_risk = target_risk_frac * float(equity)
    qty = dollars_at_risk / (risk_per_unit + 1e-12)
    return max(qty, 0.0)

def compute_levels(side: str, entry: float, atr: float, prior_level: float | None = None, params: dict | None = None):
    params = params or {}
    k_stop = float(params.get("k_stop_atr", 1.5))
    tp1_k  = float(params.get("tp1_atr", 1.0))
    if side == "long":
        stop = (prior_level - 0.1 * atr) if (prior_level is not None and np.isfinite(prior_level)) else entry - k_stop * atr
        tp1  = entry + tp1_k * atr
    else:
        stop = (prior_level + 0.1 * atr) if (prior_level is not None and np.isfinite(prior_level)) else entry + k_stop * atr
        tp1  = entry - tp1_k * atr
    return float(stop), float(tp1)

def update_trailing_stop(pos, bar: pd.Series, params: dict):
    trail_k = float(params.get("trail_after_tp1_atr", 0.5))
    atr = float(bar.get("atr", np.nan))
    if not np.isfinite(atr):
        return
    if pos.side == "long":
        pos.stop = max(pos.stop, pos.entry)
        pos.stop = max(pos.stop, float(bar.get("close", pos.entry)) - trail_k * atr)
    else:
        pos.stop = min(pos.stop, pos.entry)
        pos.stop = min(pos.stop, float(bar.get("close", pos.entry)) + trail_k * atr)

# ---------- exits ----------
def check_exit_conservative(pos, candle: pd.Series, params: dict, fee_rate: float = 0.0, slippage_bps: float = 0.0):
    """
    Conservative intrabar sequencing:
    - If both TP1 and Stop hit in same bar → count adverse (stop) first.
    - Otherwise apply TP1 (partial) then potential stop.
    - Trend-invalidation exit (EMA slow cross / MACD flip / optional PSAR).
    """
    high = float(candle.get("high", np.nan))
    low  = float(candle.get("low", np.nan))
    close = float(candle.get("close", np.nan))
    stop, tp1 = pos.stop, getattr(pos, "tp1", None)
    realized = 0.0
    partial = False

    def fee(px, qty): return abs(px * qty) * fee_rate

    if pos.side == "long":
        hit_stop = (stop is not None) and np.isfinite(stop) and np.isfinite(low) and (low <= stop)
        hit_tp1  = (tp1  is not None) and np.isfinite(tp1)  and np.isfinite(high) and (high >= tp1)
        if hit_stop and hit_tp1:
            qty = pos.qty
            realized += (stop - pos.entry) * qty - fee(stop, qty) - fee(pos.entry, qty)
            pos.open = False
            return realized, True, False
        if hit_tp1:
            qty1 = pos.qty * 0.5
            realized += (tp1 - pos.entry) * qty1 - fee(tp1, qty1) - fee(pos.entry, qty1)
            pos.qty -= qty1
            partial = True
        if hit_stop:
            qty = pos.qty
            realized += (stop - pos.entry) * qty - fee(stop, qty) - fee(pos.entry, qty)
            pos.open = False
            return realized, True, partial
    else:  # short
        hit_stop = (stop is not None) and np.isfinite(stop) and np.isfinite(high) and (high >= stop)
        hit_tp1  = (tp1  is not None) and np.isfinite(tp1)  and np.isfinite(low)  and (low  <= tp1)
        if hit_stop and hit_tp1:
            qty = pos.qty
            realized += (pos.entry - stop) * qty - fee(stop, qty) - fee(pos.entry, qty)
            pos.open = False
            return realized, True, False
        if hit_tp1:
            qty1 = pos.qty * 0.5
            realized += (pos.entry - tp1) * qty1 - fee(tp1, qty1) - fee(pos.entry, qty1)
            pos.qty -= qty1
            partial = True
        if hit_stop:
            qty = pos.qty
            realized += (pos.entry - stop) * qty - fee(stop, qty) - fee(pos.entry, qty)
            pos.open = False
            return realized, True, partial

    # trend invalidation (close-based)
    ema_slow = float(candle.get("ema_slow", np.nan))
    macd = float(candle.get("macd", np.nan))
    macd_sig = float(candle.get("macd_signal", np.nan))
    use_psar = bool(params.get("use_psar", False))
    psar_val = float(candle.get("psar", np.nan)) if "psar" in candle else np.nan

    if pos.side == "long":
        cond = (
            (np.isfinite(ema_slow) and close < ema_slow)
            or (np.isfinite(macd) and np.isfinite(macd_sig) and macd < macd_sig)
            or (use_psar and np.isfinite(psar_val) and close < psar_val)
        )
    else:
        cond = (
            (np.isfinite(ema_slow) and close > ema_slow)
            or (np.isfinite(macd) and np.isfinite(macd_sig) and macd > macd_sig)
            or (use_psar and np.isfinite(psar_val) and close > psar_val)
        )

    if cond:
        qty = pos.qty
        pnl = (close - pos.entry) * qty if pos.side == "long" else (pos.entry - close) * qty
        fees = abs(close * qty + pos.entry * qty) * fee_rate
        realized += (pnl - fees)
        pos.open = False
        return realized, True, partial

    return realized, False, partial

# ---------- position ----------
class Position:
    def __init__(self, side: str, qty: float, entry: float, stop: float | None = None, tp1: float | None = None):
        self.side = side  # "long" or "short"
        self.qty = float(qty)
        self.entry = float(entry)
        self.stop = float(stop) if stop is not None else stop
        self.tp1 = float(tp1) if tp1 is not None else tp1
        self.open = True
        self.bars = 0

    def check_exit(self, now: pd.Series, params: dict, fee_rate: float = 0.0, slippage_bps: float = 0.0):
        realized, closed, partial = check_exit_conservative(
            self, now, params, fee_rate=fee_rate, slippage_bps=slippage_bps
        )
        if partial:
            update_trailing_stop(self, now, params)
        if closed:
            self.open = False
        return realized, closed