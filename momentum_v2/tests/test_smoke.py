
import pandas as pd
from trendedge import data, backtest, params as default_params

def test_smoke():
    # Create a tiny synthetic dataset
    n = 600
    ts = pd.date_range("2022-01-01", periods=n, freq="H")
    close = pd.Series(range(n), dtype=float) + 0.1*pd.Series(range(n)).diff().fillna(0).cumsum()
    df = pd.DataFrame({
        "timestamp": ts,
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0
    })
    P = dict(default_params.PARAMS)
    dfp = data.prepare(df, P)
    eq, trades, stats = backtest.run(dfp, P)
    assert "end_equity" in stats
