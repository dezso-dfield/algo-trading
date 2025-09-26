import argparse
import os
import time
import traceback
import pandas as pd
import numpy as np
import yaml

from strategy import (
    ensure_datetime_index,
    compute_indicators,
    momentum_signal,
    entry_filter_and_score,
    Position,
)
from universe import COIN50

try:
    import ccxt
except Exception:
    ccxt = None


def fetch_latest_ohlcv(ex, symbol, timeframe="5m", limit=300):
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return ensure_datetime_index(df)


def filter_supported_symbols(ex, symbols):
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
    parser.add_argument("--symbol")
    parser.add_argument("--symbols")
    parser.add_argument("--universe", default="COIN50")
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    exchange_id = args.exchange or config.get("exchange", "binance")
    timeframe = config.get("timeframe", "5m")
    p = config.get("strategy", {})
    r = config.get("risk", {})
    fwd = config.get("forward", {})

    allocation_cash = float(r.get("allocation_per_trade_cash", 1000.0))
    max_positions = int(r.get("max_positions", 10))
    fee_rate = float(r.get("fee_rate", 0.0004))
    slippage = float(r.get("slippage", 0.0002))
    poll_seconds = int(fwd.get("poll_seconds", 15))
    obv_lb = int(p.get("obv_lookback", 20))
    max_new = int(p.get("max_new_positions_per_cycle", 2))

    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()

    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = COIN50 if args.universe == "COIN50" else config.get("symbols", [])

    symbols = filter_supported_symbols(ex, symbols)
    if not symbols:
        raise RuntimeError("No supported symbols after filtering. Check universe or exchange.")

    os.makedirs("logs", exist_ok=True)
    tag = (args.universe or "PORT").replace("/", "_")
    trades_path = os.path.join("logs", f"forward_trades_{tag}.csv")
    equity_path = os.path.join("logs", f"forward_equity_{tag}.csv")
    status_path = os.path.join("logs", f"forward_status_{tag}.csv")

    equity = float(r.get("initial_equity", 10000.0))
    positions = {sym: [] for sym in symbols}
    last_ts = {sym: None for sym in symbols}

    print(f"Starting momentum paper trading on {exchange_id} ({timeframe}) with {len(symbols)} symbols...")

    while True:
        try:
            cycle_latest_ts = None
            candidates = []  # (score, sym, ts, now, window)
            symbols_processed = 0
            new_bars_seen = 0

            # ========== Phase 1: scan, exits, collect candidates ==========
            for sym in symbols:
                try:
                    if hasattr(ex, "rateLimit") and ex.rateLimit:
                        time.sleep(ex.rateLimit / 1000.0)

                    limit = max(300, obv_lb + max(int(p.get("breakout_lookback", 50)),
                                                  int(p.get("volume_ma", 20))) + 5)
                    df = fetch_latest_ohlcv(ex, sym, timeframe=timeframe, limit=limit)
                    symbols_processed += 1

                    if df.empty:
                        if args.verbose:
                            print(f"[{sym}] Empty OHLCV — skipping")
                        continue

                    df = compute_indicators(df, p).dropna()
                    if df.empty or len(df) < obv_lb + 2:
                        if args.verbose:
                            print(f"[{sym}] Not enough bars ({len(df)}) — need >= {obv_lb+2}")
                        continue

                    ts = df.index[-1]
                    # Act only on a new closed bar (but first time - allow)
                    is_new_bar = (last_ts[sym] is None) or (ts != last_ts[sym])
                    if not is_new_bar:
                        # No new bar for this symbol this cycle
                        continue
                    new_bars_seen += 1
                    last_ts[sym] = ts
                    cycle_latest_ts = max(cycle_latest_ts, ts) if cycle_latest_ts is not None else ts

                    prev = df.iloc[-2]
                    now = df.iloc[-1]
                    window = df.iloc[-max(obv_lb, max(int(p.get("breakout_lookback", 50)),
                                                      int(p.get("volume_ma", 20)))) - 1:]

                    # ===== Exits first =====
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
                                "exit": float(now["close"]),
                                "realized": realized,
                                "equity": equity,
                            }
                            pd.DataFrame([rec]).to_csv(
                                trades_path, mode="a", header=not os.path.exists(trades_path), index=False
                            )
                    positions[sym] = still

                    # capacity check (global)
                    total_open = sum(len(lst) for lst in positions.values())
                    if total_open >= max_positions:
                        if args.verbose:
                            print(f"[{sym} {ts}] Capacity full ({total_open}/{max_positions})")
                        continue

                    # already in a position? skip entry
                    if positions[sym]:
                        if args.verbose:
                            print(f"[{sym} {ts}] Already in position — skip entry")
                        continue

                    # ===== Core momentum (EMA/MACD/RSI) =====
                    long_ok, short_ok = momentum_signal(prev, now, p)
                    if not long_ok:
                        if args.verbose:
                            print(f"[{sym} {ts}] Core momentum not met (long_ok={long_ok})")
                        continue

                    # ===== Extra entry filter (breakout + volume + OBV + MACD hist) =====
                    ok_extra, score = entry_filter_and_score(window, now, p)
                    if not ok_extra:
                        if args.verbose:
                            print(f"[{sym} {ts}] Extra entry filter failed")
                        continue

                    candidates.append((score, sym, ts, now))

                except Exception as sym_err:
                    print(f"[ERROR] {sym}: {sym_err}")
                    traceback.print_exc()
                    continue

            # ========== Phase 2: rank & open at most max_new positions ==========
            entries_taken = 0
            if candidates:
                candidates.sort(reverse=True, key=lambda x: x[0])  # score desc
                total_open = sum(len(lst) for lst in positions.values())
                capacity = max_positions - total_open
                to_open = min(max_new, capacity, len(candidates))

                for k in range(to_open):
                    score, sym, ts, now = candidates[k]
                    price = float(now["close"])
                    price_eff = price * (1 + slippage)  # buy slippage
                    qty = (allocation_cash / price_eff) if price_eff > 0 else 0.0
                    if qty <= 0:
                        continue

                    stop_ref = float(now.get("ema_slow", np.nan)) if "ema_slow" in now else None

                    equity -= price_eff * qty * fee_rate
                    positions[sym].append(Position("long", qty, price_eff, stop=stop_ref))
                    entries_taken += 1

                    entry_rec = {
                        "timestamp": ts.isoformat(),
                        "symbol": sym,
                        "event": "ENTRY",
                        "side": "long",
                        "entry": price_eff,
                        "equity": equity,
                        "score": score,
                    }
                    pd.DataFrame([entry_rec]).to_csv(
                        trades_path, mode="a", header=not os.path.exists(trades_path), index=False
                    )
                    print(f"[{sym} {ts}] ENTRY px={price:.6f} qty={qty:.6f} score={score:.4f} open_now={total_open+1}")

            # ========== Status & equity logs every cycle ==========
            # If no new bar anywhere, timestamp with current UTC so you still get a heartbeat row
            ts_for_log = (
                cycle_latest_ts.isoformat()
                if cycle_latest_ts is not None
                else pd.Timestamp.utcnow().replace(tzinfo=pd.Timestamp.utcnow().tz).isoformat()
            )

            # Equity log once per cycle
            pd.DataFrame([{"timestamp": ts_for_log, "equity": equity}]).to_csv(
                equity_path, mode="a", header=not os.path.exists(equity_path), index=False
            )

            # Status log once per cycle
            status_row = {
                "timestamp": ts_for_log,
                "symbols_processed": symbols_processed,
                "new_bars_seen": new_bars_seen,
                "candidates": len(candidates),
                "entries": entries_taken,
                "open_positions": sum(len(v) for v in positions.values()),
                "equity": equity,
            }
            pd.DataFrame([status_row]).to_csv(
                status_path, mode="a", header=not os.path.exists(status_path), index=False
            )

            if args.verbose:
                print(
                    f"[CYCLE] ts={ts_for_log} processed={symbols_processed} newbars={new_bars_seen} "
                    f"candidates={len(candidates)} entries={entries_taken} open={status_row['open_positions']} eq={equity:.2f}"
                )

            time.sleep(poll_seconds)

        except KeyboardInterrupt:
            print("Stopping forward trader.")
            break
        except Exception as loop_err:
            print("Loop error:", loop_err)
            traceback.print_exc()
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()