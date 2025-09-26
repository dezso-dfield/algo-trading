import argparse, time, os, math, datetime as dt
import pandas as pd
import numpy as np
import yaml

from strategy import ensure_datetime_index, compute_indicators, detect_entries, compute_stops_and_tps, Position
from universe import COIN50

try:
    import ccxt
except Exception:
    ccxt = None

def fetch_latest_ohlcv(ex, symbol, timeframe="5m", limit=200):
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    return ensure_datetime_index(df)

def rsi_series(close, period=14):
    # quick RSI to reduce dependency when ta not desired in live loop (we still use ta in strategy.compute_indicators normally)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -1*delta.clip(upper=0.0)
    roll_up = up.ewm(com=period-1, adjust=False).mean()
    roll_down = down.ewm(com=period-1, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100/(1+rs))

def fetch_htf_rsi(ex, symbol, timeframe="1h", period=14, limit=200):
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=period+5)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    df = ensure_datetime_index(df)
    df["rsi"] = rsi_series(df["close"], period=period)
    return df["rsi"].iloc[-1]


def fetch_latest_ohlcv_multi(ex, symbols, timeframe="5m", limit=300):
    data_map = {}
    for sym in symbols:
        try:
            df = fetch_latest_ohlcv(ex, sym, timeframe=timeframe, limit=limit)
            data_map[sym] = df
        except Exception as e:
            print("Data fetch error for", sym, ":", e)
    return data_map

def resolve_symbols(config_syms, cli_symbol=None, cli_symbols=None, cli_universe=None):
    if cli_symbol:
        return [cli_symbol]
    if cli_symbols:
        return [s.strip() for s in cli_symbols.split(",") if s.strip()]
    if len(config_syms) == 1 and config_syms[0] == "COIN50":
        return COIN50
    return config_syms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--symbols", help="Comma-separated symbols")
    parser.add_argument("--universe", default=None, help="e.g., COIN50 (if used, overrides)")
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    exchange_id = args.exchange or config["exchange"]
    timeframe = config["timeframe"]
    s = config["strategy"]
    r = config["risk"]
    fwd = config["forward"]

    allocation_cash = r.get("allocation_per_trade_cash", 1000)
    max_positions = r.get("max_positions", 10)

    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    ex = getattr(ccxt, exchange_id)()
    ex.load_markets()

    symbols = resolve_symbols(config.get("symbols", []), args.symbol, args.symbols, args.universe or None)
    if args.universe == "COIN50":
        symbols = COIN50

    os.makedirs("logs", exist_ok=True)
    tag = (args.universe or "PORTFOLIO").replace("/","_")
    trades_path = os.path.join("logs", f"forward_trades_{tag}.csv")
    equity_path = os.path.join("logs", f"forward_equity_{tag}.csv")

    equity = config["backtest"]["initial_equity"]
    positions = {sym: [] for sym in symbols}
    last_ts_by_sym = {sym: None for sym in symbols}

    print(f"Starting forward paper trading for {len(symbols)} symbols on {exchange_id} ({timeframe})...")
    while True:
        cycle_latest_ts = None
        try:
            data_map = fetch_latest_ohlcv_multi(ex, symbols, timeframe=timeframe, limit=300)
            # compute indicators
            for sym, df in data_map.items():
                data_map[sym] = compute_indicators(df, s["ema_period"], s["bb_period"], s["bb_std"], s["rsi_period"], s["adx_period"]).dropna()

            # iterate each symbol independently; act only on new candle close
            for sym, df in data_map.items():
                if len(df) < 5:
                    continue
                ts_now = df.index[-1]
                cycle_latest_ts = max(cycle_latest_ts, ts_now) if cycle_latest_ts else ts_now
                if last_ts_by_sym[sym] is not None and ts_now == last_ts_by_sym[sym]:
                    continue
                last_ts_by_sym[sym] = ts_now

                prev_row = df.iloc[-2]
                row = df.iloc[-1]
                price = row["close"]

                # optional HTF RSI filter
                if s.get("use_htf_filter", False):
                    try:
                        htf_rsi = fetch_htf_rsi(ex, sym, timeframe=s.get("htf_timeframe","1h"), period=s["rsi_period"])
                    except Exception:
                        htf_rsi = None
                else:
                    htf_rsi = None

                # manage exits
                still_open = []
                for pos in positions[sym]:
                    realized, closed = pos.check_exit(row, fee_rate=r["fee_rate"], scale_out=r["scale_out"])
                    equity += realized
                    if not closed:
                        still_open.append(pos)
                    else:
                        rec = {
                            "timestamp": row.name.isoformat(),
                            "symbol": sym,
                            "side": pos.side,
                            "entry": pos.entry,
                            "exit": price,
                            "realized": realized,
                            "equity": equity
                        }
                        pd.DataFrame([rec]).to_csv(trades_path, mode="a", header=not os.path.exists(trades_path), index=False)
                positions[sym] = still_open

                # entry if portfolio has capacity
                total_open = sum(len(lst) for lst in positions.values())
                if total_open >= max_positions:
                    continue

                signal = detect_entries(prev_row, row, adx_max=s["adx_max"])
                if signal and htf_rsi is not None:
                    if signal == "long" and htf_rsi > s["htf_rsi_overbought"]:
                        signal = None
                    if signal == "short" and htf_rsi < s["htf_rsi_oversold"]:
                        signal = None

                if signal:
                    mean = row["ema"]
                    recent_extreme = prev_row["low"] if signal=="long" else prev_row["high"]
                    stop, tp1, tp2 = compute_stops_and_tps(signal, price, recent_extreme, mean,
                                                           r["stop_buffer_pct"], r["tp1_ratio"], r["tp2_to_mean"])
                    # Fixed allocation per trade
                    qty = allocation_cash / price
                    if qty > 0:
                        price_eff = price * (1 + r["slippage"] * (1 if signal=="long" else -1))
                        fee = price_eff * qty * r["fee_rate"]
                        equity -= fee
                        pos = Position(signal, qty, price_eff, stop, tp1, tp2)
                        positions[sym].append(pos)

                # log equity each symbol's close (once per symbol per close)
                pd.DataFrame([{"timestamp": row.name.isoformat(), "equity": equity}]).to_csv(
                    equity_path, mode="a", header=not os.path.exists(equity_path), index=False
                )

                print(f"[{sym} {row.name}] px={price:.4f} open={sum(len(lst) for lst in positions.values())} eq={equity:.2f}")

            # END-CYCLE-EQUITY: log once per full iteration over symbols
            if cycle_latest_ts is not None:
                pd.DataFrame([{"timestamp": cycle_latest_ts.isoformat(), "equity": equity}]).to_csv(
                    equity_path, mode="a", header=not os.path.exists(equity_path), index=False
                )
            time.sleep(fwd["poll_seconds"])  # real sleep

        except KeyboardInterrupt:
            print("Stopping forward tester.")
            break
        except Exception as e:
            print("Error in loop:", e)
            # END-CYCLE-EQUITY: log once per full iteration over symbols
            if cycle_latest_ts is not None:
                pd.DataFrame([{"timestamp": cycle_latest_ts.isoformat(), "equity": equity}]).to_csv(
                    equity_path, mode="a", header=not os.path.exists(equity_path), index=False
                )
            time.sleep(fwd["poll_seconds"])  # real sleep

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
