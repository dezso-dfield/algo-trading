#!/usr/bin/env python3
"""
Mean Reversion Crypto Strategy using FSVZO (Stochastic VZO) + RSI
=================================================================

WHAT IT DOES (high level)
-------------------------
- Indicators:
  - RSI
  - VZO (Volume Zone Oscillator)
  - FSVZO (Stochastic of VZO) — default:
      FSVZO_%K = (VZO - min(VZO, lookback)) / (max(VZO, lookback) - min(VZO, lookback)) * 100
      FSVZO_%D = EMA(FSVZO_%K, d)
    (You can switch to Fisher or Z-score modes via --fsvzo-mode.)
- Signals (mean reversion, long-only by default):
  - Enter long when RSI <= rsi_buy AND FSVZO_%K <= fsvzo_buy
  - Exit long when RSI >= rsi_sell OR  FSVZO_%K >= fsvzo_sell
  - Optional symmetric shorts with --allow-short
- Backtest over Binance OHLCV, or
- Forward paper-trading runner that:
  - Runs UNTIL STOPPED (Ctrl+C)
  - Maintains a single portfolio (cash + positions) starting from initial equity (default 10,000 USDT)
  - Equal-weights capital across all active signals
  - Logs trades, positions, and equity to CSV under --state-dir

Dependencies
------------
pip install pandas numpy ccxt python-dateutil

DISCLAIMER: Educational code. No warranty. Test before any real use.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt

# --------------------------- Utils ---------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(length).mean()
    roll_down = pd.Series(down, index=close.index).rolling(length).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def vzo(close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    # Simple VZO: EMA(signed volume)/EMA(volume) scaled to 100
    sign = np.sign(close.diff().fillna(0.0))
    signed_vol = pd.Series(sign, index=close.index) * volume
    ev = ema(signed_vol, length)
    evol = ema(volume, length) + 1e-12
    return 100 * (ev / evol)

def fisher_transform(x: pd.Series) -> pd.Series:
    clipped = x.clip(-0.999, 0.999)
    return 0.5 * np.log((1 + clipped) / (1 - clipped + 1e-12))

def rolling_stoch(series: pd.Series, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    low = series.rolling(k).min()
    high = series.rolling(k).max()
    kpct = (series - low) / (high - low + 1e-12) * 100
    dpct = ema(kpct, d)
    return kpct, dpct

def zscore(series: pd.Series, lookback: int = 100) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std(ddof=0) + 1e-12
    return (series - mean) / std

# --------------------------- Data ---------------------------

def get_binance(exchange=None):
    return exchange or ccxt.binance({'enableRateLimit': True})

def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int = 1000, exchange=None) -> pd.DataFrame:
    ex = get_binance(exchange)
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df

def fetch_top_usdt_symbols(n: int = 50, min_price: float = 0.01, exchange=None) -> List[str]:
    ex = get_binance(exchange)
    markets = ex.load_markets()
    tickers = ex.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith('/USDT'):
            continue
        m = markets.get(sym, {})
        if not m.get('spot', True):
            continue
        last = t.get('last') or t.get('close') or 0
        quote_vol = t.get('quoteVolume') or 0
        if last and last >= min_price and quote_vol:
            rows.append((sym, float(quote_vol)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s,_ in rows[:n]]

# --------------------------- Indicators & Signals ---------------------------

@dataclass
class IndicatorParams:
    rsi_len: int = 14
    vzo_len: int = 30
    fsvzo_k: int = 14
    fsvzo_d: int = 3
    fsvzo_mode: str = "stoch"  # "stoch", "fisher", "zscore"

@dataclass
class StrategyParams:
    rsi_buy: float = 30.0
    rsi_sell: float = 55.0
    fsvzo_buy: float = 20.0
    fsvzo_sell: float = 60.0
    allow_short: bool = False

def compute_indicators(df: pd.DataFrame, ip: IndicatorParams) -> pd.DataFrame:
    out = df.copy()
    out['RSI'] = rsi(out['close'], ip.rsi_len)
    out['VZO'] = vzo(out['close'], out['volume'], ip.vzo_len)
    if ip.fsvzo_mode == 'stoch':
        k, d = rolling_stoch(out['VZO'], ip.fsvzo_k, ip.fsvzo_d)
        out['FSVZO_K'] = k
        out['FSVZO_D'] = d
    elif ip.fsvzo_mode == 'fisher':
        min_v = out['VZO'].rolling(ip.fsvzo_k).min()
        max_v = out['VZO'].rolling(ip.fsvzo_k).max()
        norm = ((out['VZO'] - min_v) / (max_v - min_v + 1e-12)) * 2 - 1
        out['FSVZO_K'] = (fisher_transform(norm) + 3) / 6 * 100
        out['FSVZO_D'] = ema(out['FSVZO_K'], ip.fsvzo_d)
    elif ip.fsvzo_mode == 'zscore':
        z = zscore(out['VZO'], ip.fsvzo_k)
        out['FSVZO_K'] = (np.tanh(z) + 1) * 50
        out['FSVZO_D'] = ema(out['FSVZO_K'], ip.fsvzo_d)
    else:
        raise ValueError(f"Unknown fsvzo_mode: {ip.fsvzo_mode}")
    return out

def generate_signals(df: pd.DataFrame, sp: StrategyParams) -> pd.Series:
    rsi_ = df['RSI']
    fsk = df['FSVZO_K']

    long_entry = (rsi_ <= sp.rsi_buy) & (fsk <= sp.fsvzo_buy)
    long_exit = (rsi_ >= sp.rsi_sell) | (fsk >= sp.fsvzo_sell)

    pos = pd.Series(0, index=df.index, dtype=float)
    in_long = False
    in_short = False

    for i in range(len(df)):
        if not in_long and not in_short and long_entry.iat[i]:
            in_long = True
        elif in_long and long_exit.iat[i]:
            in_long = False
        if sp.allow_short:
            short_entry = (rsi_ >= 100 - sp.rsi_buy) & (fsk >= 100 - sp.fsvzo_buy)
            short_exit  = (rsi_ <= 100 - sp.rsi_sell) | (fsk <= 100 - sp.fsvzo_sell)
            if not in_long and not in_short and short_entry.iat[i]:
                in_short = True
            elif in_short and short_exit.iat[i]:
                in_short = False
        pos.iat[i] = (1.0 if in_long else (-1.0 if in_short else 0.0))
    return pos

# --------------------------- Backtester (unchanged from prior) ---------------------------

@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 1.0

class Backtester:
    def __init__(self, df: pd.DataFrame, signals: pd.Series, cfg: BacktestConfig):
        self.df = df.copy()
        self.signals = signals.astype(float).copy()
        self.cfg = cfg

    def run(self) -> Dict[str, pd.DataFrame]:
        price = self.df['close']
        target_pos = self.signals
        pos = target_pos.shift(1).fillna(0.0)
        trade = pos.diff().fillna(pos)
        slip = (self.cfg.slippage_bps / 1e4) * price
        exec_price = price + slip * np.sign(trade)
        ret = pos * price.pct_change().fillna(0.0)
        turnover = (abs(trade)).fillna(0.0)
        fees = turnover * (self.cfg.fee_bps / 1e4)
        net_ret = ret - fees
        eq = (1 + net_ret).cumprod() * (self.cfg.initial_equity)
        curve = pd.DataFrame({
            'price': price,
            'position': pos,
            'turnover': turnover,
            'gross_ret': ret,
            'fees': -fees,
            'net_ret': net_ret,
            'equity': eq
        }, index=self.df.index)
        trades = []
        current = 0.0
        entry_price = None
        entry_time = None
        for i, (t, tr, p, ex) in enumerate(zip(curve.index, trade, price, exec_price)):
            if tr == 0:
                continue
            if current != 0.0:
                pnl = (ex - entry_price) * np.sign(current)
                trades.append({
                    'entry_time': entry_time, 'exit_time': t,
                    'side': 'LONG' if current > 0 else 'SHORT',
                    'entry_price': float(entry_price), 'exit_price': float(ex),
                    'pnl_quote': float(pnl),
                })
                current = 0.0
            current = tr
            entry_price = ex
            entry_time = t
        trades_df = pd.DataFrame(trades)
        return {'curve': curve, 'trades': trades_df}

# --------------------------- Portfolio Paper Trader ---------------------------

@dataclass
class PaperConfig:
    initial_equity: float = 10_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 1.0
    state_dir: str = './paper_state'
    exposure_limit: float = 1.0  # 1.0 = fully investable when signals exist (no leverage)
    min_trade_notional: float = 5.0  # skip dust trades

class PortfolioTrader:
    """
    Portfolio-level forward runner:
      - Equal-weights all symbols with active target_pos != 0
      - Trades diffs vs current positions at next closed bar
      - Tracks cash, positions (qty), equity, realized PnL (through trade logs)
      - Persists state under state_dir, runs until interrupted
    """
    def __init__(self, symbols: List[str], timeframe: str, ip: IndicatorParams, sp: StrategyParams, cfg: PaperConfig):
        self.symbols = symbols
        self.timeframe = timeframe
        self.ip = ip
        self.sp = sp
        self.cfg = cfg
        os.makedirs(cfg.state_dir, exist_ok=True)
        self.exchange = get_binance()
        self.state_file = os.path.join(cfg.state_dir, 'portfolio_state.json')
        self.trades_csv = os.path.join(cfg.state_dir, 'trades.csv')
        self.equity_csv = os.path.join(cfg.state_dir, 'equity.csv')
        self.pos_file = os.path.join(cfg.state_dir, 'positions.json')  # {symbol: qty}
        self._ensure_files()

    def _ensure_files(self):
        if not os.path.exists(self.state_file):
            with open(self.state_file, 'w') as f:
                json.dump({'cash': self.cfg.initial_equity, 'last_ts': None}, f)
        if not os.path.exists(self.pos_file):
            with open(self.pos_file, 'w') as f:
                json.dump({}, f)
        if not os.path.exists(self.trades_csv):
            pd.DataFrame(columns=[
                'timestamp','symbol','action','qty','price','notional','fee','cash_after'
            ]).to_csv(self.trades_csv, index=False)
        if not os.path.exists(self.equity_csv):
            pd.DataFrame(columns=[
                'timestamp','equity','cash','holdings','num_positions','gross_exposure'
            ]).to_csv(self.equity_csv, index=False)

    def _load_state(self):
        with open(self.state_file, 'r') as f:
            st = json.load(f)
        with open(self.pos_file, 'r') as f:
            pos = json.load(f)
        return st, {k: float(v) for k,v in pos.items()}

    def _save_state(self, st, pos):
        with open(self.state_file, 'w') as f:
            json.dump(st, f)
        with open(self.pos_file, 'w') as f:
            json.dump(pos, f)

    def _append_trade(self, row: dict):
        df = pd.DataFrame([row])
        df.to_csv(self.trades_csv, mode='a', header=False, index=False)

    def _append_equity(self, row: dict):
        df = pd.DataFrame([row])
        df.to_csv(self.equity_csv, mode='a', header=False, index=False)

    def _latest_bars(self) -> Dict[str, dict]:
        out = {}
        for sym in self.symbols:
            df = fetch_ohlcv_df(sym, self.timeframe, limit=400, exchange=self.exchange)
            ind = compute_indicators(df, self.ip).dropna()
            if ind.empty:
                continue
            sig = generate_signals(ind, self.sp).iloc[-1]
            out[sym] = {
                'ts': ind.index[-1].isoformat(),
                'price': float(ind['close'].iloc[-1]),
                'signal': float(sig)
            }
        return out

    def run_once(self):
        st, pos = self._load_state()
        snapshot = self._latest_bars()
        if not snapshot:
            print("No data fetched.")
            return

        # Determine desired allocation
        actives = [s for s, info in snapshot.items() if info['signal'] != 0.0]
        cash = float(st['cash'])
        # Mark-to-market holdings value
        holdings = 0.0
        for s, qty in pos.items():
            if s in snapshot:
                holdings += qty * snapshot[s]['price']
        equity = cash + holdings

        # Target exposure (<= equity * exposure_limit)
        target_gross = equity * self.cfg.exposure_limit
        per_symbol_notional = (target_gross / max(1, len(actives))) if actives else 0.0

        # Build desired qty map
        desired_qty = {}
        for s, info in snapshot.items():
            if info['signal'] == 0.0:
                desired_qty[s] = 0.0
            else:
                desired_qty[s] = (per_symbol_notional / info['price']) * np.sign(info['signal'])

        # Trade towards desired
        ts = max(v['ts'] for v in snapshot.values())
        slip_bps = self.cfg.slippage_bps / 1e4
        fee_rate = self.cfg.fee_bps / 1e4

        for s, info in snapshot.items():
            price = info['price']
            current = float(pos.get(s, 0.0))
            target = float(desired_qty.get(s, 0.0))
            delta_qty = target - current
            notional = abs(delta_qty) * price
            if notional < self.cfg.min_trade_notional:
                continue
            # execute with slippage
            exec_price = price + (slip_bps * price * np.sign(delta_qty))
            trade_notional = exec_price * delta_qty
            fee = abs(trade_notional) * fee_rate
            # cash update (buy reduces cash, sell increases)
            cash -= trade_notional + np.sign(trade_notional) * 0.0  # trade_notional already signed
            cash -= fee
            # update position
            new_qty = current + delta_qty
            if abs(new_qty) < 1e-12:
                new_qty = 0.0
            pos[s] = new_qty
            self._append_trade({
                'timestamp': ts, 'symbol': s,
                'action': 'BUY' if delta_qty > 0 else 'SELL',
                'qty': float(delta_qty), 'price': float(exec_price),
                'notional': float(trade_notional), 'fee': float(fee),
                'cash_after': float(cash)
            })

        # End-of-bar MTM
        holdings = sum(float(q) * snapshot[s]['price'] for s, q in pos.items() if s in snapshot)
        equity = cash + holdings
        gross_exposure = sum(abs(float(q)) * snapshot[s]['price'] for s, q in pos.items() if s in snapshot)
        num_positions = sum(1 for q in pos.values() if abs(q) > 0)
        self._append_equity({
            'timestamp': ts, 'equity': float(equity), 'cash': float(cash),
            'holdings': float(holdings), 'num_positions': int(num_positions),
            'gross_exposure': float(gross_exposure)
        })
        st['cash'] = cash
        st['last_ts'] = ts
        self._save_state(st, pos)
        print(f"[{ts}] Equity={equity:.2f} Cash={cash:.2f} Holdings={holdings:.2f} Positions={num_positions}")

# --------------------------- CLI ---------------------------

def run_backtest(args):
    ex = get_binance()
    symbols = args.symbols or fetch_top_usdt_symbols(args.top_n, exchange=ex)
    print("Symbols:", symbols)

    ip = IndicatorParams(
        rsi_len=args.rsi_len, vzo_len=args.vzo_len,
        fsvzo_k=args.fsvzo_k, fsvzo_d=args.fsvzo_d, fsvzo_mode=args.fsvzo_mode
    )
    sp = StrategyParams(
        rsi_buy=args.rsi_buy, rsi_sell=args.rsi_sell,
        fsvzo_buy=args.fsvzo_buy, fsvzo_sell=args.fsvzo_sell,
        allow_short=args.allow_short
    )
    cfg = BacktestConfig(
        initial_equity=args.initial_equity, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps
    )

    all_curves = []
    all_trades = []
    for sym in symbols:
        df = fetch_ohlcv_df(sym, args.timeframe, limit=args.limit, exchange=ex)
        ind = compute_indicators(df, ip).dropna().copy()
        sig = generate_signals(ind, sp)
        bt = Backtester(ind, sig, cfg)
        res = bt.run()
        curve = res['curve']
        curve['symbol'] = sym
        all_curves.append(curve)
        trades = res['trades']
        if not trades.empty:
            trades['symbol'] = sym
            all_trades.append(trades)

    curves_df = pd.concat(all_curves).sort_index()
    trades_df = pd.concat(all_trades) if all_trades else pd.DataFrame(columns=['entry_time','exit_time','side','entry_price','exit_price','pnl_quote','symbol'])

    port = curves_df.groupby(curves_df.index).apply(lambda x: x['net_ret'].mean()).to_frame('port_ret')
    port['equity'] = (1 + port['port_ret']).cumprod() * cfg.initial_equity
    port.to_csv(args.out_prefix + "_portfolio_equity.csv")
    curves_df.to_csv(args.out_prefix + "_curves.csv")
    trades_df.to_csv(args.out_prefix + "_trades.csv", index=False)

    print("Saved:")
    print(" -", args.out_prefix + "_portfolio_equity.csv")
    print(" -", args.out_prefix + "_curves.csv")
    print(" -", args.out_prefix + "_trades.csv")

def run_forward(args):
    ex = get_binance()
    symbols = args.symbols or fetch_top_usdt_symbols(args.top_n, exchange=ex)
    print("Forward symbols:", symbols)

    ip = IndicatorParams(
        rsi_len=args.rsi_len, vzo_len=args.vzo_len,
        fsvzo_k=args.fsvzo_k, fsvzo_d=args.fsvzo_d, fsvzo_mode=args.fsvzo_mode
    )
    sp = StrategyParams(
        rsi_buy=args.rsi_buy, rsi_sell=args.rsi_sell,
        fsvzo_buy=args.fsvzo_buy, fsvzo_sell=args.fsvzo_sell,
        allow_short=args.allow_short
    )
    cfg = PaperConfig(
        initial_equity=args.initial_equity, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        state_dir=args.state_dir, exposure_limit=args.exposure_limit, min_trade_notional=args.min_trade_notional
    )
    trader = PortfolioTrader(symbols, args.timeframe, ip, sp, cfg)

    if args.once:
        trader.run_once()
        return

    print("Starting forward portfolio loop. Press Ctrl+C to stop.")
    try:
        while True:
            trader.run_once()
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Stopped.")

def parse_args():
    p = argparse.ArgumentParser(description="FSVZO + RSI Mean Reversion for Top-N Cryptos (Binance)")
    sub = p.add_subparsers(dest='mode', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--symbols', nargs='*', help="Override symbols, e.g., BTC/USDT ETH/USDT")
    common.add_argument('--top-n', type=int, default=50, help="Auto-pick top-N USDT symbols by 24h volume")
    common.add_argument('--timeframe', type=str, default='1h')
    common.add_argument('--limit', type=int, default=1000)
    common.add_argument('--rsi-len', type=int, default=14)
    common.add_argument('--vzo-len', type=int, default=30)
    common.add_argument('--fsvzo-k', type=int, default=14)
    common.add_argument('--fsvzo-d', type=int, default=3)
    common.add_argument('--fsvzo-mode', type=str, default='stoch', choices=['stoch','fisher','zscore'])
    common.add_argument('--rsi-buy', type=float, default=30.0)
    common.add_argument('--rsi-sell', type=float, default=55.0)
    common.add_argument('--fsvzo-buy', type=float, default=20.0)
    common.add_argument('--fsvzo-sell', type=float, default=60.0)
    common.add_argument('--allow-short', action='store_true')
    common.add_argument('--initial-equity', type=float, default=10_000.0)
    common.add_argument('--fee-bps', type=float, default=5.0)
    common.add_argument('--slippage-bps', type=float, default=1.0)

    # Backtest
    b = sub.add_parser('backtest', parents=[common])
    b.add_argument('--out-prefix', type=str, default='backtest_fsvzo_rsi')
    b.set_defaults(func=run_backtest)

    # Forward
    f = sub.add_parser('forward', parents=[common])
    f.add_argument('--state-dir', type=str, default='./paper_state')
    f.add_argument('--poll-seconds', type=int, default=60)
    f.add_argument('--once', action='store_true', help="Run exactly one cycle and exit (useful for crons)")
    f.add_argument('--exposure-limit', type=float, default=1.0, help="0..1 investable fraction of equity")
    f.add_argument('--min-trade-notional', type=float, default=5.0, help="Skip tiny dust trades")
    f.set_defaults(func=run_forward)

    return p.parse_args()

def main():
    args = parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
