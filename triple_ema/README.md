# Triple-EMA (9/21/55) + Momentum Filter — Coinbase Top 50
**Same functionality** as previous projects: portfolio backtest and continuous forward (paper) trading, COIN50 universe, logging trade **ENTRY/EXIT** events, and logging **equity once per cycle**.

## Highlights
- Universe: Coinbase Top-50 proxy (COIN50 list in `universe.py`).
- Macro filter: **Top-N momentum** (default 7-day return, configurable lookback in bars/days).
- Micro timing: **Triple EMA** (9/21/55). Long when 9 > 21 > 55; exit when hierarchy fails or stop hit.
- Risk: **risk_pct_per_trade=0.02** (2%) *or* fixed cash (e.g., $1,000). Choose via `risk.sizing_mode`.
- Stops: default **swing-low stop** over recent N bars (configurable) or ATR multiple fallback.
- Allocation cap: **max_positions** across the portfolio.
- Logging: Trade **ENTRY** + **EXIT** CSV; **equity** logged **once per full loop** cycle (forward).

## Install
```bash
pip install -r requirements.txt
```

## Backtest (portfolio)
```bash
python backtest.py --universe COIN50 --since_days 120
# or custom list
python backtest.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT" --since_days 120
```

## Forward (paper) trading
```bash
python forward.py --universe COIN50 --exchange binance
```

> Educational only. Use real historical data for validation; synthetic demo CSV is provided in `data/`.
