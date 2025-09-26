# Mean Reversion 5m Strategy (Crypto) — Backtest & Forward Paper Trading

This repo contains a production-ready baseline for a **5-minute mean reversion strategy** using:
- 20 EMA as mean
- Bollinger Bands (20, 2)
- RSI(14) confirmation
- ADX(14) trend filter (skip if trending; configurable)
- Optional higher-timeframe (1h) RSI filter to avoid counter-trending extremes

It includes:
- `strategy.py` — signal generation & risk/exit logic
- `backtest.py` — CSV/ccxt-based backtesting (event-driven loop)
- `forward.py` — continuous paper trading loop via ccxt polling on 5m candles
- `config.yaml` — strategy + risk config
- `data/` — place your CSV OHLCV files here for backtesting (or use ccxt fetch)
- `logs/` — trade logs & equity curves are written here

> **Note**: You need your own exchange API keys only if you want to fetch data faster or submit *real* orders. Forward tester here uses **paper** trading.


## Quick Start

### 1) Install deps
```bash
pip install -r requirements.txt
```

### 2) Configure
Edit `config.yaml` to set symbols, exchange, risk, and filters.

### 3) Backtest on CSV
Put a CSV with columns `[timestamp, open, high, low, close, volume]` (ms or s). Then:
```bash
python backtest.py --symbol BTC/USDT --csv data/BTCUSDT_5m.csv
```
Or fetch via ccxt (if your IP is allowed by exchange):
```bash
python backtest.py --symbol BTC/USDT --exchange binance --since_days 60
```

### 4) Forward paper test (continuous loop)
```bash
python forward.py --symbol BTC/USDT --exchange binance
```
The loop waits for the 5m candle to close, evaluates signals, and paper-trades with your risk settings. Trades log to `logs/forward_trades.csv` and equity to `logs/forward_equity.csv`.

---

## Strategy Rules (5m)
**Long**
1. Prior candle closes **below lower BB**.
2. RSI < 30.
3. Current candle closes **back inside** the bands (confirmation).
4. Enter at current-close.
5. TP1 = halfway to EMA; TP2 = EMA (scale out); stop = recent extreme ± buffer.

**Short**
Symmetric on upper BB & RSI > 70.

**Filters**
- ADX(14) < 20 required (skip trending). Threshold configurable.
- Optional 1h RSI filter: if RSI(1h) > 70 (for longs) or < 30 (for shorts), skip (counter-trend filter).

---

## CSV Format
```
timestamp,open,high,low,close,volume
1718068800000,68000,68100,67900,68050,123.45
...
```
`timestamp` can be in seconds or milliseconds since epoch — auto-detected.

---

## Disclaimers
This code is for **educational** use. Trading involves risk.


## Portfolio / COIN50 Mode
- The config now sets `symbols: [COIN50]`, `initial_equity: 10000`, and allocates **$1,000** per new position.
- Max concurrent positions is **10** (10 x $1k = full $10k exposure). Adjust in `config.yaml` via `risk.max_positions` and `risk.allocation_per_trade_cash`.

### Backtest portfolio
```bash
python backtest.py --universe COIN50 --since_days 60
# or specify your own list
python backtest.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT" --since_days 60
```

### Forward (paper) portfolio
```bash
python forward.py --universe COIN50 --exchange binance
# or
python forward.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT" --exchange binance
```
