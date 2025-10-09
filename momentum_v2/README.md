# momentum_trendedge

Momentum (trend-following) package with:
- Symmetric long/short gate (EMA fast/slow, MACD vs signal, RSI)
- Regime gates (ATR% percentiles, EMA slow slope; HTF hook ready)
- Breakout + confirmation filter (prior high/low, vol vs MA, OBV slope, MACD-hist z-score)
- Standardized score for ranking
- ATR risk-unit sizing, TP1, trail, time stop, trend-invalidation exits
- Conservative intrabar sequencing, fees & slippage accounted
- Forward runner for COIN50 on Binance (CCXT)

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 forward_trendedge.py --universe COIN50 --exchange binance --config config.yaml --verbose
```

Backtest a CSV with columns: `timestamp,open,high,low,close,volume`:
```bash
python3 backtest.py --csv path/to/BTCUSDT_5m.csv --config config.yaml
```
