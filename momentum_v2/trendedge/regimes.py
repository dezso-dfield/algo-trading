
import numpy as np
import pandas as pd

def add_regime_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    out["ema_slow_slope"] = (out["ema_slow"] - out["ema_slow"].shift(5)) / 5.0
    out["atr_pct"] = out["atr"] / out["close"]
    win = int(params.get("regime_win", 500))

    def roll_pct(s: pd.Series):
        return s.rolling(win, min_periods=win).apply(
            lambda x: np.searchsorted(np.sort(x), x[-1], side="right")/len(x), raw=True)

    out["atrpct_pct"] = roll_pct(out["atr_pct"].fillna(0))
    w = int(params.get("zscore_win", 200))
    mh = out["macd_hist"]
    out["macd_hist_z"] = (mh - mh.rolling(w).mean()) / (mh.rolling(w).std(ddof=0) + 1e-12)
    return out

def regime_ok(now: pd.Series, params: dict) -> bool:
    low_cut  = float(params.get("atrpct_pct_low", 0.15))
    high_cut = float(params.get("atrpct_pct_high", 0.95))
    pct = float(now.get("atrpct_pct", 0.5))
    if not (low_cut <= pct <= high_cut):
        return False
    slope_min = float(params.get("ema_slope_min", 0.0))
    if abs(float(now.get("ema_slow_slope", 0.0))) < slope_min:
        return False
    # optional HTF gate injected later (in data.prepare)
    return True
