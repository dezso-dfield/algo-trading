
# trendedge

A portable, robust trend-following breakout strategy package.

**Core ideas**
- Trend bias (EMA fast/slow, MACD>signal, RSI threshold)
- Symmetric long/short breakout entries with regime gates (volatility, slope, optional HTF)
- Standardized scores (z-scores) for ranking
- ATR risk-unit sizing, structured stops, TP1 partials, trailing, time stop
- Conservative intrabar sequencing (adverse move first if both stop/TP hit)
- Fees + slippage modeled
- Data hygiene: warm-up, next-bar execution, completed bars only

## Quick start

```bash
pip install -e .
trendedge backtest --csv examples/BTCUSDT_1h_sample.csv
```

Or use from Python:

```python
import pandas as pd
from trendedge import data, backtest, params as default_params

df = data.load_csv("examples/BTCUSDT_1h_sample.csv")
df = data.prepare(df, default_params.PARAMS)
equity, trades, stats = backtest.run(df, default_params.PARAMS)
print(stats)
```

## CSV format

Required columns: `timestamp,open,high,low,close,volume` (UTC ISO8601 or epoch seconds).

## Parameters

See `trendedge/params.py` and override via CLI `--params path.json` or `--override key=value` (supports dotted keys).
