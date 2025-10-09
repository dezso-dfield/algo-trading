
import numpy as np
import pandas as pd
from .regimes import regime_ok

def entry_filter_and_score(window: pd.DataFrame, now: pd.Series, params: dict, side: str = "long"):
    if not regime_ok(now, params):
        return False, -1e9

    lb = int(params.get("breakout_lookback", 50))
    vol_ma_n = int(params.get("volume_ma", 20))
    vol_ratio_min = float(params.get("min_volume_ma_ratio", 1.1))
    obv_slope_min = float(params.get("obv_slope_min", 0.0))
    min_macd_hist_z = float(params.get("min_macd_hist_z", 0.0))
    require_breakout = bool(params.get("require_breakout", True))
    pullback = bool(params.get("pullback_after_breakout", False))

    if len(window) < max(lb, vol_ma_n) + 1:
        return False, -1e9

    prior = window.iloc[:-1]
    prior_high = float(prior["high"].tail(lb).max())
    prior_low  = float(prior["low"].tail(lb).min())

    price = float(now["close"])
    vol_ma = float(prior["volume"].tail(vol_ma_n).mean())
    vol_ok = (float(now["volume"]) >= vol_ma * vol_ratio_min) if vol_ma > 0 else False

    obv_slope = (float(window["obv"].iloc[-1]) - float(window["obv"].iloc[0])) / max(len(window), 1)
    obv_ok = obv_slope >= obv_slope_min if side == "long" else obv_slope <= -obv_slope_min

    macd_hist_z = float(now.get("macd_hist_z", 0.0))
    macd_ok = macd_hist_z >= min_macd_hist_z if side == "long" else (-macd_hist_z) >= min_macd_hist_z

    if side == "long":
        price_ok = (price > prior_high) if require_breakout else True
        if pullback and price_ok:
            # require last bar (now) to close back above prior_high and previous bar to have tested it
            prev_close = float(window.iloc[-2]["close"])
            price_ok = (prev_close <= prior_high) and (price > prior_high)
        ok = price_ok and vol_ok and obv_ok and macd_ok
        score = macd_hist_z + ((float(now["rsi"]) - 50.0) * 0.05) + (obv_slope * 1e-6)
    else:
        price_ok = (price < prior_low) if require_breakout else True
        if pullback and price_ok:
            prev_close = float(window.iloc[-2]["close"])
            price_ok = (prev_close >= prior_low) and (price < prior_low)
        ok = price_ok and vol_ok and obv_ok and macd_ok
        score = (-macd_hist_z) + ((50.0 - float(now["rsi"])) * 0.05) + ((-obv_slope) * 1e-6)
    return ok, score

def momentum_signal(prev, now, params, side=None):
    bull = (now["ema_fast"] > now["ema_slow"])
    bear = (now["ema_fast"] < now["ema_slow"])
    macd_bull = now["macd"] > now["macd_signal"]
    macd_bear = now["macd"] < now["macd_signal"]

    rsi_long_min  = params.get("rsi_long_min", 50)
    rsi_short_max = params.get("rsi_short_max", 50)
    ema_slope_min = float(params.get("ema_slope_min", 0.0))

    long_ok  = bull and macd_bull and (now["rsi"] >= rsi_long_min)  and (now.get("ema_slow_slope",0.0) >= ema_slope_min)
    short_ok = bear and macd_bear and (now["rsi"] <= rsi_short_max) and (now.get("ema_slow_slope",0.0) <= -ema_slope_min)

    if side == "long":  return long_ok, False
    if side == "short": return False, short_ok
    return long_ok, short_ok
