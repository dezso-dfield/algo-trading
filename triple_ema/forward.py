import argparse, os, time
import pandas as pd
import numpy as np
import yaml

from strategy import ensure_datetime_index, compute_indicators, in_triple_ema_up, swing_low_stop, Position
from universe import COIN50

try:
    import ccxt
except Exception:
    ccxt = None

def fetch_latest_ohlcv(ex, symbol, timeframe="5m", limit=300):
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    return ensure_datetime_index(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol")
    parser.add_argument("--symbols")
    parser.add_argument("--universe", default="COIN50")
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

    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    ex = getattr(ccxt, exchange_id)()
    ex.load_markets()

    # symbols
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = COIN50 if args.universe == "COIN50" else config.get("symbols", [])

    os.makedirs("logs", exist_ok=True)
    tag = (args.universe or "PORT").replace("/","_")
    trades_path = os.path.join("logs", f"forward_trades_{tag}.csv")
    equity_path = os.path.join("logs", f"forward_equity_{tag}.csv")

    equity = r["initial_equity"]
    positions = {sym: [] for sym in symbols}
    last_ts = {sym: None for sym in symbols}

    # ranking cadence
    rank_interval = fwd.get("rank_interval_bars", 288)
    bar_counter = 0
    current_top = set()

    print(f"Starting triple-EMA+momentum paper trading on {exchange_id} ({timeframe}) for {len(symbols)} symbols...")
    while True:
        try:
            cycle_latest_ts = None
            bar_counter += 1

            # 1) fetch and indicators per symbol
            data = {}
            for sym in symbols:
                try:
                    df = fetch_latest_ohlcv(ex, sym, timeframe=timeframe, limit=max(300, p["momentum_lookback_bars"]+p["swing_lookback"]+5))
                    data[sym] = compute_indicators(df, p).dropna()
                except Exception as e:
                    print("Fetch error", sym, e)

            # 2) re-rank top-N periodically
            if bar_counter % rank_interval == 0:
                scores = []
                for sym, df in data.items():
                    if len(df) <= p["momentum_lookback_bars"]:
                        continue
                    ret = df["close"].iloc[-1] / df["close"].iloc[-p["momentum_lookback_bars"]] - 1.0
                    scores.append((sym, ret))
                scores.sort(key=lambda x: x[1], reverse=True)
                current_top = set([s for s,_ in scores[:p["top_n"]]])
                # close positions no longer in top set
                for sym in list(positions.keys()):
                    if sym not in current_top and positions[sym]:
                        row = data[sym].iloc[-1]
                        px = row["close"]
                        still = []
                        for pos in positions[sym]:
                            pnl = (px - pos.entry) * pos.qty
                            equity += pnl - (abs(pos.entry)+abs(px)) * r["fee_rate"] * pos.qty
                            # EXIT log
                            rec = {"timestamp": row.name.isoformat(), "symbol": sym, "event": "EXIT", "side": pos.side,
                                   "entry": pos.entry, "exit": px, "realized": pnl, "equity": equity}
                            pd.DataFrame([rec]).to_csv(trades_path, mode="a", header=not os.path.exists(trades_path), index=False)
                        positions[sym] = still

            # 3) per symbol, act on closed bar only
            for sym, df in data.items():
                if len(df) == 0: continue
                ts = df.index[-1]
                cycle_latest_ts = max(cycle_latest_ts, ts) if cycle_latest_ts else ts
                if last_ts[sym] is not None and ts == last_ts[sym]:
                    continue
                last_ts[sym] = ts

                row = df.iloc[-1]
                price = row["close"]

                # Exits first
                still = []
                for pos in positions[sym]:
                    realized, closed = pos.check_exit(row, fee_rate=r["fee_rate"])
                    equity += realized
                    if not closed:
                        still.append(pos)
                    else:
                        rec = {"timestamp": ts.isoformat(), "symbol": sym, "event": "EXIT", "side": pos.side,
                               "entry": pos.entry, "exit": price, "realized": realized, "equity": equity}
                        pd.DataFrame([rec]).to_csv(trades_path, mode="a", header=not os.path.exists(trades_path), index=False)
                positions[sym] = still

                # capacity
                total_open = sum(len(lst) for lst in positions.values())
                if total_open >= r["max_positions"]:
                    continue

                # Only long and only if sym in current top set (once ranking started)
                if current_top and sym not in current_top:
                    continue

                # Entry condition: triple EMA up
                if in_triple_ema_up(row) and len(positions[sym]) == 0:
                    # stop
                    window = df.iloc[-p["swing_lookback"]:] if len(df) >= p["swing_lookback"] else df
                    stop = swing_low_stop(window, atr=row.get("atr", np.nan), atr_mult=p["atr_mult"])
                    entry = price * (1 + r["slippage"])

                    # sizing
                    if r["sizing_mode"] == "fixed_cash":
                        qty = r["fixed_cash_per_trade"] / entry
                    else:
                        risk_cash = equity * r["risk_pct_per_trade"]
                        per_unit_risk = max(entry - stop, 1e-12)
                        qty = risk_cash / per_unit_risk

                    if qty > 0:
                        equity -= entry * qty * r["fee_rate"]
                        positions[sym].append(Position("long", qty, entry, stop))
                        # ENTRY log
                        rec = {"timestamp": ts.isoformat(), "symbol": sym, "event": "ENTRY", "side": "long",
                               "entry": entry, "equity": equity}
                        pd.DataFrame([rec]).to_csv(trades_path, mode="a", header=not os.path.exists(trades_path), index=False)

            # 4) end-of-cycle single equity log
            if cycle_latest_ts is not None:
                pd.DataFrame([{"timestamp": cycle_latest_ts.isoformat(), "equity": equity}]).to_csv(
                    equity_path, mode="a", header=not os.path.exists(equity_path), index=False
                )

            time.sleep(fwd["poll_seconds"])

        except KeyboardInterrupt:
            print("Stopping forward trader.")
            break
        except Exception as e:
            print("Loop error:", e)
            time.sleep(fwd["poll_seconds"])

if __name__ == "__main__":
    main()
