
import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    m = ema_fast - ema_slow
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0))
    return (sign * df["volume"]).cumsum()

def psar(df: pd.DataFrame, step=0.02, max_step=0.2):
    # Simple PSAR implementation (optional)
    high, low = df["high"].values, df["low"].values
    length = len(df)
    psar = np.zeros(length)
    bull = True
    af = step
    ep = low[0]
    psar[0] = low[0]
    for i in range(1, length):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        if bull:
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = step
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = step
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
    return pd.Series(psar, index=df.index)
