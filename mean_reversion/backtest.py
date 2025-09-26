import argparse
import pandas as pd
import numpy as np
import os, math, time, datetime as dt
import yaml
from typing import Optional
from strategy import ensure_datetime_index, compute_indicators, detect_entries, compute_stops_and_tps, Position
from universe import COIN50

try:
    import ccxt
except Exception:
    ccxt = None

def fetch_ccxt_ohlcv(symbol, exchange="binance", timeframe="5m", since_ms=None, limit=1000, max_batches=20):
    if ccxt is None:
        raise RuntimeError("ccxt not installed.")
    ex = getattr(ccxt, exchange)()
    ex.load_markets()
    all_rows = []
    since = since_ms
    for _ in range(max_batches):
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows += batch
        since = batch[-1][0] + 1  # next ms
        # polite sleep
        time.sleep(ex.rateLimit / 1000.0)
    df = pd.DataFrame(batch for batch in all_rows)
    df.columns = ["timestamp","open","high","low","close","volume"]
    return df

def load_data(csv: Optional[str], symbol: str, exchange: str, timeframe: str, since_days: int = 60):
    if csv:
        df = pd.read_csv(csv)
    else:
        since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=since_days)).timestamp() * 1000)
        df = fetch_ccxt_ohlcv(symbol, exchange=exchange, timeframe=timeframe, since_ms=since)
    df = ensure_datetime_index(df)
    return df


def resolve_symbol_list(symbols_arg, universe_flag):
    if universe_flag == "COIN50":
        return COIN50
    if symbols_arg:
        return [s.strip() for s in symbols_arg.split(",") if s.strip()]
    return COIN50  # default

def load_multi_symbol_data(symbols, exchange, timeframe, since_days, csv_map=None):
    data_map = {}
    for sym in symbols:
        csv = None
        if csv_map and sym in csv_map:
            csv = csv_map[sym]
        df = load_data(csv, sym, exchange, timeframe, since_days)
        data_map[sym] = df
    return data_map


def run_backtest(df, config, symbol="SYMBOL"):
    s = config["strategy"]
    r = config["risk"]
    initial_equity = config["backtest"]["initial_equity"]

    df = compute_indicators(df, s["ema_period"], s["bb_period"], s["bb_std"], s["rsi_period"], s["adx_period"])
    df = df.dropna().copy()

    equity = initial_equity
    positions = []
    trade_log = []

    for i in range(1, len(df)):
        prev_row = df.iloc[i-1]
        row = df.iloc[i]
        price = row["close"]

        # Close/manage open positions
        still_open = []
        for pos in positions:
            realized, closed = pos.check_exit(row, fee_rate=r["fee_rate"], scale_out=r["scale_out"])
            equity += realized
            if not closed:
                still_open.append(pos)
            else:
                trade_log.append({
                    "timestamp": row.name.isoformat(),
                    "symbol": symbol,
                    "event": "EXIT", "side": pos.side,
                    "entry": pos.entry,
                    "exit": row["close"],
                    "realized": realized,
                    "equity": equity
                })
        positions = still_open

        # Daily loss guard (optional)
        # Skipped for backtest — add if desired

        # Entry logic (only if we can open more positions)
        if len(positions) < r["max_positions"]:
            signal = detect_entries(prev_row, row, adx_max=s["adx_max"])
            if signal:
                # optional HTF RSI filter skip
                # For backtest we skip HTF due to data simplicity. Can be added by merging HTF series.
                mean = row["ema"]
                recent_extreme = prev_row["low"] if signal=="long" else prev_row["high"]
                stop, tp1, tp2 = compute_stops_and_tps(signal, price, recent_extreme, mean,
                                                       r["stop_buffer_pct"], r["tp1_ratio"], r["tp2_to_mean"])

                # position sizing by risk_per_trade
                stop_dist = abs(price - stop)
                if stop_dist <= 0:
                    continue
                risk_cash = equity * r["risk_per_trade"]
                qty = risk_cash / stop_dist
                if qty <= 0:
                    continue

                # fees & slippage on entry
                price_eff = price * (1 + r["slippage"] * (1 if signal=="long" else -1))
                fee = price_eff * qty * r["fee_rate"]
                equity -= fee

                pos = Position(signal, qty, price_eff, stop, tp1, tp2)
                positions.append(pos)
                trade_log.append({
                    "timestamp": row.name.isoformat(),
                    "symbol": symbol,
                    "event": "ENTRY",
                    "side": signal,
                    "entry": price_eff,
                    "equity": equity
                })

    # Close at end
    for pos in positions:
        pnl = pos.mark(df.iloc[-1]["close"])
        trade_log.append({
            "timestamp": df.index[-1].isoformat(),
            "symbol": symbol,
            "event": "EXIT", "side": pos.side,
            "entry": pos.entry,
            "exit": df.iloc[-1]["close"],
            "realized": pnl,
            "equity": equity + pnl
        })
        equity += pnl

    trade_df = pd.DataFrame(trade_log)
    return equity, trade_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Single symbol e.g., BTC/USDT")
    parser.add_argument("--symbols", help="Comma-separated symbols")
    parser.add_argument("--universe", default=None, help="e.g., COIN50")
    parser.add_argument("--csv", default=None, help="Single-symbol CSV")
    parser.add_argument("--exchange", default=None, help="ccxt exchange id, e.g., binance")
    parser.add_argument("--since_days", type=int, default=60)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    exchange = args.exchange or config["exchange"]
    timeframe = config["timeframe"]

    # Resolve symbol list: CLI override > config symbols
    if args.symbol:
        symbols = [args.symbol]
    else:
        cfg_syms = config.get("symbols", [])
        use_universe = None
        if len(cfg_syms) == 1 and cfg_syms[0] in ("COIN50",):
            use_universe = cfg_syms[0]
        symbols = resolve_symbol_list(args.symbols, args.universe or use_universe)

    # Load data for each symbol (ccxt fetch unless csv provided for single symbol)
    csv_map = None
    if args.csv:
        csv_map = {symbols[0]: args.csv} if len(symbols) == 1 else None
    data_map = load_multi_symbol_data(symbols, exchange, timeframe, args.since_days, csv_map)

    # Compute indicators per symbol
    s = config["strategy"]
    for sym, df in data_map.items():
        data_map[sym] = compute_indicators(df, s["ema_period"], s["bb_period"], s["bb_std"], s["rsi_period"], s["adx_period"]).dropna()

    final_eq, trades = run_portfolio_backtest(data_map, config)

    os.makedirs("logs", exist_ok=True)
    symtag = (args.universe or "MIX").replace("/","_")
    eq_curve_path = os.path.join("logs", f"backtest_{symtag}_equity.csv")
    trades_path = os.path.join("logs", f"backtest_{symtag}_trades.csv")
    trades.to_csv(trades_path, index=False)
    pd.DataFrame([{"final_equity": final_eq}]).to_csv(eq_curve_path, index=False)
    print(f"Portfolio final equity: {final_eq:.2f}")
    print(f"Saved trades to {trades_path}")
    print(f"Saved final equity to {eq_curve_path}")
if __name__ == "__main__":
    main()


def run_portfolio_backtest(data_map, config):
    s = config["strategy"]
    r = config["risk"]
    initial_equity = config["backtest"]["initial_equity"]
    allocation_cash = r.get("allocation_per_trade_cash", 1000)

    equity = initial_equity
    positions = {sym: [] for sym in data_map.keys()}
    trade_log = []

    # Align by common timeline: iterate by each symbol's data in lockstep via timestamps union
    all_ts = sorted(set().union(*[df.index for df in data_map.values()]))
    for i in range(1, len(all_ts)):
        ts_prev, ts_now = all_ts[i-1], all_ts[i]
        # Build a per-symbol row if timestamp exists
        for sym, df in data_map.items():
            # skip if timestamp not in df (uneven histories)
            if ts_prev not in df.index or ts_now not in df.index:
                continue

            row_prev = df.loc[ts_prev]
            row = df.loc[ts_now]
            price = row["close"]

            # manage open positions for this symbol
            still_open = []
            for pos in positions[sym]:
                realized, closed = pos.check_exit(row, fee_rate=r["fee_rate"], scale_out=r["scale_out"])
                equity += realized
                if not closed:
                    still_open.append(pos)
                else:
                    trade_log.append({
                        "timestamp": ts_now.isoformat(),
                        "symbol": sym,
                        "event": "EXIT", "side": pos.side,
                        "entry": pos.entry,
                        "exit": price,
                        "realized": realized,
                        "equity": equity
                    })
            positions[sym] = still_open

            # Entry only if portfolio position cap not exceeded
            total_open = sum(len(lst) for lst in positions.values())
            if total_open >= r["max_positions"]:
                continue

            signal = detect_entries(row_prev, row, adx_max=s["adx_max"])
            if signal:
                mean = row["ema"]
                recent_extreme = row_prev["low"] if signal=="long" else row_prev["high"]
                stop, tp1, tp2 = compute_stops_and_tps(signal, price, recent_extreme, mean,
                                                       r["stop_buffer_pct"], r["tp1_ratio"], r["tp2_to_mean"])
                stop_dist = abs(price - stop)
                if stop_dist <= 0:
                    continue

                # Fixed cash allocation per trade -> qty = allocation_cash / price
                qty = allocation_cash / price
                if qty <= 0:
                    continue

                # fees & slippage on entry
                price_eff = price * (1 + r["slippage"] * (1 if signal=="long" else -1))
                fee = price_eff * qty * r["fee_rate"]
                equity -= fee

                pos = Position(signal, qty, price_eff, stop, tp1, tp2)
                positions[sym].append(pos)

    # Mark-to-market close at end
    last_ts = all_ts[-1] if all_ts else None
    if last_ts:
        for sym, df in data_map.items():
            if last_ts in df.index:
                px = df.loc[last_ts, "close"]
                for pos in positions[sym]:
                    pnl = pos.mark(px)
                    trade_log.append({
                        "timestamp": last_ts.isoformat(),
                        "symbol": sym,
                        "event": "EXIT", "side": pos.side,
                        "entry": pos.entry,
                        "exit": px,
                        "realized": pnl,
                        "equity": equity + pnl
                    })
                    equity += pnl

    trades = pd.DataFrame(trade_log)
    return equity, trades
