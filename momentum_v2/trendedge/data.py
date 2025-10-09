
import pandas as pd
import numpy as np
from .indicators import ema, rsi, macd, atr as atr_fn, obv, psar
from .regimes import add_regime_features

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    for k in ["timestamp","open","high","low","close","volume"]:
        assert k in [x.lower() for x in df.columns], f"CSV missing column {k}"
    # coerce names to expected lower-case
    df.columns = [c.lower() for c in df.columns]
    # timestamp parse
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except Exception:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)

def prepare(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], int(params.get("ema_fast", 20)))
    out["ema_slow"] = ema(out["close"], int(params.get("ema_slow", 50)))
    m, s, h = macd(out["close"], int(params.get("macd_fast",12)), int(params.get("macd_slow",26)), int(params.get("macd_signal",9)))
    out["macd"] = m; out["macd_signal"] = s; out["macd_hist"] = h
    out["rsi"] = rsi(out["close"], int(params.get("rsi_period",14)))
    out["atr"] = atr_fn(out)
    out["obv"] = obv(out)
    if params.get("use_psar", False):
        out["psar"] = psar(out)
    out = add_regime_features(out, params)

    # Optional HTF bias: require EMA trend aligned, using multiple-of-timeframe via rolling downsample proxy
    if params.get("use_htf", False):
        mult = int(params.get("htf_multiple", 4))
        ema_slow = out["ema_slow"].rolling(mult).mean()
        out["htf_ema_slow"] = ema_slow.rolling(mult).mean()
        out["htf_ema_slow_slope"] = (out["htf_ema_slow"] - out["htf_ema_slow"].shift(mult)) / max(mult,1)
        out["htf_ok_long"]  = out["htf_ema_slow_slope"] >= float(params.get("htf_ema_slope_min", 0.0))
        out["htf_ok_short"] = out["htf_ema_slow_slope"] <= -float(params.get("htf_ema_slope_min", 0.0))
        # simple gate flags; entries module doesn't check directly, but users can filter via these columns if desired

    # drop signals until fully ready
    warm = int(params.get("warmup_bars", 300))
    out = out.iloc[warm:].copy()
    out.reset_index(drop=True, inplace=True)
    return out
