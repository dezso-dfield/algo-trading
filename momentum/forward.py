import argparse
import os
import time
import pandas as pd
import numpy as np
import yaml

from strategy import (
    ensure_datetime_index,
    compute_indicators,
    momentum_signal,
    Position,
)
from universe import COIN50

try:
    import ccxt
except Exception:
    ccxt = None


def fetch_latest_ohlcv(ex, symbol, timeframe="5m", limit=300):
    """Fetch latest OHLCV and return a datetime-indexed DataFrame."""
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return ensure_datetime_index(df)


def filter_supported_symbols(ex, symbols):
    """
    Keep only symbols that exist on the selected exchange (ex.symbols).
    Prints any skipped items and returns the filtered list.
    """
    available = set(ex.symbols) if hasattr(ex, "symbols") else set()
    supported, missing = [], []
    for s in symbols:
        if s in available:
            supported.append(s)
        else:
            missing.append(s)
    if missing:
        print("Skipping unsupported symbols:", ", ".join(missing))
    return supported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--symbols", help="Comma-separated symbols")
    parser.add_argument("--universe", default="COIN50", help="e.g., COIN50")
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    exchange_id = args.exchange or config["exchange"]
    timeframe = config["timeframe"]
    p = config["strategy"]
    r = config["risk"]
    fwd = config["forward"]

    allocation_cash = r.get("allocation_per_trade_cash", 1000)
    max_positions = r.get("max_positions", 10)
    fee_rate = r.get("fee_rate", 0.0004)
    slippage = r.get("slippage", 0.0002)

    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")

    ex = getattr(ccxt, exchange_id)()
    ex.load_markets()

    # Resolve symbols from CLI or config/universe
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = COIN50 if args.universe == "COIN50" else config.get("symbols", [])

    # Filter to exchange-supported symbols (prevents "does not have market symbol" errors)
    symbols = filter_supported_symbols(ex, symbols)
    if not symbols:
        raise RuntimeError("No supported symbols after filtering. Check universe or exchange.")

    os.makedirs("logs", exist_ok=True)
    tag = (args.universe or "PORT").replace("/", "_")
    trades_path = os.path.join("logs", f"forward_trades_{tag}.csv")
    equity_path = os.path.join("logs", f"forward_equity_{tag}.csv")

    equity = r.get("initial_equity", 10000)
    positions = {sym: [] for sym in symbols}
    last_ts = {sym: None for sym in symbols}

    print(
        f"Starting momentum paper trading on {exchange_id} ({timeframe}) "
        f"with {len(symbols)} symbols..."
    )

    while True:
        try:
            cycle_latest_ts = None  # to log equity once per full pass
            # Iterate symbols
            for sym in symbols:
                # fetch & indicators
                df = fetch_latest_ohlcv(
                    ex, sym, timeframe=timeframe, limit=max(300, p.get("obv_lookback", 20) + 10)
                )
                df = compute_indicators(df, p).dropna()
                if len(df) < p.get("obv_lookback", 20) + 2:
                    continue

                ts = df.index[-1]
                # only act on new closed bar
                if last_ts[sym] is not None and ts == last_ts[sym]:
                    continue
                last_ts[sym] = ts
                cycle_latest_ts = max(cycle_latest_ts, ts) if cycle_latest_ts is not None else ts

                prev = df.iloc[-2]
                now = df.iloc[-1]
                price = now["close"]
                window = df.iloc[-p.get("obv_lookback", 20):]

                # exits first
                still = []
                for pos in positions[sym]:
                    realized, closed = pos.check_exit(now, params=p, fee_rate=fee_rate)
                    equity += realized
                    if not closed:
                        still.append(pos)
                    else:
                        rec = {
                            "timestamp": ts.isoformat(),
                            "symbol": sym,
                            "event": "EXIT",
                            "side": pos.side,
                            "entry": pos.entry,
                            "exit": price,
                            "realized": realized,
                            "equity": equity,
                        }
                        pd.DataFrame([rec]).to_csv(
                            trades_path,
                            mode="a",
                            header=not os.path.exists(trades_path),
                            index=False,
                        )
                positions[sym] = still

                # capacity
                total_open = sum(len(lst) for lst in positions.values())
                if total_open >= max_positions:
                    continue

                # entries
                long_ok, short_ok = momentum_signal(prev, now, p)
                # OBV slope confirmation (simple slope over window)
                obv_slope = (window["obv"].iloc[-1] - window["obv"].iloc[0]) / len(window)
                if long_ok and obv_slope <= p.get("obv_slope_min", 0.0):
                    long_ok = False
                if short_ok and obv_slope >= -p.get("obv_slope_min", 0.0):
                    short_ok = False

                side = "long" if long_ok else ("short" if short_ok else None)
                if not side:
                    continue

                # fixed $ allocation per trade (as requested for portfolio model)
                price_eff = price * (1 + slippage * (1 if side == "long" else -1))
                qty = allocation_cash / price_eff if price_eff > 0 else 0.0
                if qty <= 0:
                    continue

                # initial stop using ema_slow / ATR trail logic inside Position if you extended it;
                # in this momentum version, Position.stop is optional and exits also rely on trend rules.
                stop = now.get("ema_slow", np.nan)

                # entry fee impact
                equity -= price_eff * qty * fee_rate

                # open position and log ENTRY
                positions[sym].append(Position(side, qty, price_eff, stop=stop))
                rec_entry = {
                    "timestamp": ts.isoformat(),
                    "symbol": sym,
                    "event": "ENTRY",
                    "side": side,
                    "entry": price_eff,
                    "equity": equity,
                }
                pd.DataFrame([rec_entry]).to_csv(
                    trades_path, mode="a", header=not os.path.exists(trades_path), index=False
                )

                print(f"[{sym} {ts}] px={price:.4f} open={total_open+1} eq={equity:.2f}")

            # end-of-cycle: log equity once
            if cycle_latest_ts is not None:
                pd.DataFrame([{"timestamp": cycle_latest_ts.isoformat(), "equity": equity}]).to_csv(
                    equity_path, mode="a", header=not os.path.exists(equity_path), index=False
                )

            time.sleep(fwd.get("poll_seconds", 15))

        except KeyboardInterrupt:
            print("Stopping forward trader.")
            break
        except Exception as e:
            print("Loop error:", e)
            time.sleep(fwd.get("poll_seconds", 15))


if __name__ == "__main__":
    main()