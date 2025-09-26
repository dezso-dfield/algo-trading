# Momentum / Trend Following (5m) — COIN50 Portfolio, $10k, $1k per Trade

This project implements a crypto **momentum (trend following)** strategy on 5-minute candles with:
- Fast/Slow EMA trend (default 20/50)
- **MACD** bullish/bearish cross confirmation
- **RSI** filter (>50 for longs, <50 for shorts)
- **OBV** slope confirmation (volume-backed moves)
- **Parabolic SAR** / **ATR** trail for exits

**Portfolio model**
- Initial equity: **$10,000**
- Fixed allocation: **$1,000** per new position
- Max concurrent positions: **10**
- Universe: **COIN50** (top 50 USDT-quoted majors)

Scripts:
- `strategy.py` — indicators & momentum entry/exit logic
- `backtest.py` — portfolio backtest across many symbols (ccxt or CSV)
- `forward.py` — continuous paper trader across the universe
- `universe.py` — COIN50 list
- `config.yaml`, `requirements.txt`

## Install
```bash
pip install -r requirements.txt
```

## Backtest (portfolio)
```bash
python backtest.py --universe COIN50 --since_days 60
# or custom list
python backtest.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT" --since_days 60
```

## Forward (paper) trading
```bash
python forward.py --universe COIN50 --exchange binance
```

### Strategy Rules (default)
**Long**
- EMA_fast > EMA_slow (trend up) AND MACD line > signal (bullish)
- RSI > 50
- OBV slope positive over lookback (e.g., last 20 bars)
- Entry: on close when above fast EMA (or on pullback-to-EMA if enabled)
- Exit: Parabolic SAR flip OR close < EMA_slow OR MACD bear cross; ATR trail optional

**Short** — symmetric.

**Stops & trailing**
- Initial stop = EMA_slow or ATR multiple (configurable)
- Trail with Parabolic SAR; optional ATR x N trail.

> Educational only. Trading risk is real.
