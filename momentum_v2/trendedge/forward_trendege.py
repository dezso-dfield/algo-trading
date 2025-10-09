import argparse
import os
import time
import traceback
import pandas as pd
import numpy as np
import yaml

from trendedge import data, backtest, entries, params
from trendedge.entries import momentum_signal, entry_filter_and_score
from trendedge.risk import compute_levels, atr_position_size
from trendedge.execution import check_exit_conservative
from trendedge.indicators import ema, macd, rsi, obv, atr

try:
    import ccxt
except Exception:
    ccxt = None


def fetch_latest_ohlcv(ex, symbol, timeframe="5m", limit=500):
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol")
    parser.add_argument("--symbols")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exid = args.exchange or cfg.get("exchange", "binance")
    timeframe = cfg.get("timeframe", "5m")
    P = dict(params.PARAMS)
    P.update(cfg.get("strategy", {}))
    risk = cfg.get("risk", {})
    forward = cfg.get("forward", {})

    alloc_cash = float(risk.get("allocation_per_trade_cash", 1000.0))
    fee_rate = float(risk.get("fee_rate", 0.0006))
    slippage_bps = float(risk.get("slippage_bps", 1.0))
    poll_sec = int(forward.get("poll_seconds", 30))
    max_positions = int(risk.get("max_positions", 5))

    if ccxt is None:
        raise RuntimeError("Install ccxt with: pip install ccxt")

    ex = getattr(ccxt, exid)({"enableRateLimit": True})
    ex.load_markets()

    symbols = []
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])

    print(f"Starting forward trading on {exid} ({timeframe}) for {len(symbols)} symbols...")

    positions = {sym: [] for sym in symbols}
    last_ts = {sym: None for sym in symbols}
    equity = float(risk.get("initial_equity", 10000.0))

    ensure_dir("logs/forward_equity.csv")

    while True:
        try:
            candidates = []
            for sym in symbols:
                df = fetch_latest_ohlcv(ex, sym, timeframe, limit=400)
                df = data.prepare(df, P)

                ts = df["timestamp"].iloc[-1]
                if last_ts[sym] == ts:
                    continue
                last_ts[sym] = ts

                prev, now = df.iloc[-2], df.iloc[-1]
                win = df.iloc[-60:]

                # Manage open positions
                still = []
                for pos in positions[sym]:
                    realized, closed, partial = check_exit_conservative(pos, now, P, fee_rate=fee_rate, slippage_bps=slippage_bps)
                    equity += realized
                    if not closed:
                        still.append(pos)
                positions[sym] = still

                # Entry logic
                long_ok, short_ok = momentum_signal(prev, now, P)
                if long_ok:
                    ok, score = entry_filter_and_score(win, now, P, side="long")
                    if ok:
                        candidates.append((score, sym, "long", now))
                elif short_ok:
                    ok, score = entry_filter_and_score(win, now, P, side="short")
                    if ok:
                        candidates.append((score, sym, "short", now))

            # Rank and open new entries
            if candidates:
                candidates.sort(reverse=True, key=lambda x: x[0])
                for score, sym, side, now in candidates[:max_positions]:
                    entry = float(now["close"])
                    atr_val = float(now["atr"])
                    qty = atr_position_size(equity, entry, atr_val,
                                            target_risk_frac=P.get("target_risk_frac", 0.004),
                                            risk_multiple=P.get("k_stop_atr", 1.5))
                    stop, tp1 = compute_levels(side, entry, atr_val, None, P)
                    pos = type("Pos", (), {})()
                    pos.side = side; pos.qty = qty; pos.entry = entry; pos.stop = stop; pos.tp1 = tp1; pos.open = True; pos.bars = 0
                    positions[sym].append(pos)
                    print(f"[{sym}] {side.upper()} ENTRY at {entry:.2f} score={score:.3f}")

            pd.DataFrame([{"timestamp": pd.Timestamp.utcnow(), "equity": equity}]).to_csv(
                "logs/forward_equity.csv", mode="a", header=not os.path.exists("logs/forward_equity.csv"), index=False
            )

            time.sleep(poll_sec)

        except KeyboardInterrupt:
            print("Stopping forward trader.")
            break
        except Exception as e:
            print("Loop error:", e)
            traceback.print_exc()
            time.sleep(poll_sec)


if __name__ == "__main__":
    main()