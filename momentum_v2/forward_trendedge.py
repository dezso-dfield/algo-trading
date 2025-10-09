#!/usr/bin/env python3
# Forward runner for momentum_trendedge (pagination + tolerant warm-up + indicator bootstrap)

import argparse
import os, time, random, traceback
import pandas as pd
import numpy as np
import yaml

from strategy import (
    ensure_datetime_index,
    compute_indicators,         # preferred pipeline (used first)
    momentum_signal,
    entry_filter_and_score,
    Position,
    atr_position_size,
    compute_levels,
)
from universe import COIN50

try:
    import ccxt
    from ccxt.base.errors import RequestTimeout, ExchangeNotAvailable, DDoSProtection, NetworkError
except Exception:
    ccxt = None
    RequestTimeout = ExchangeNotAvailable = DDoSProtection = NetworkError = Exception

BINANCE_HOSTS = ["api.binance.com", "api1.binance.com", "api2.binance.com"]
TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}

def rotate_binance_host(i): return (i + 1) % len(BINANCE_HOSTS)

def init_exchange_ccxt(exchange_id: str, host_idx: int = 0):
    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    kwargs = {
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    }
    if exchange_id == "binance":
        kwargs["hostname"] = BINANCE_HOSTS[host_idx % len(BINANCE_HOSTS)]
    ex = getattr(ccxt, exchange_id)(kwargs)
    ex.load_markets()
    return ex

def fetch_ohlcv_paginated(ex, symbol, timeframe="5m", target_bars=3000, max_chunk=1500, max_retries=4, verbose=False):
    tf_ms = TF_MS.get(timeframe)
    if tf_ms is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'")
    now_ms = int(time.time() * 1000)
    since = now_ms - target_bars * tf_ms

    out = []
    last_err = None
    while len(out) < target_bars:
        if getattr(ex, "rateLimit", None):
            time.sleep((getattr(ex, "rateLimit", 200) or 200) / 1000.0)
        ok = False
        for attempt in range(max_retries):
            try:
                data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=max_chunk, since=since)
                ok = True
                break
            except (RequestTimeout, ExchangeNotAvailable, DDoSProtection, NetworkError) as e:
                last_err = e
                sleep_s = min(10, 2**attempt) + random.uniform(0, 0.4)
                if verbose:
                    print(f"[{symbol}] fetch_ohlcv retry {attempt+1}/{max_retries}: {type(e).__name__}: sleeping {sleep_s:.2f}s")
                time.sleep(sleep_s)
            except Exception as e:
                last_err = e
                if verbose:
                    print(f"[{symbol}] fetch_ohlcv hard error: {e}")
                    traceback.print_exc()
                return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        if not ok:
            if verbose and last_err:
                print(f"[{symbol}] pagination terminated: {last_err}")
            break
        if not data:
            break
        out.extend(data)
        since = data[-1][0] + 1
        time.sleep(0.02)

    if not out:
        if last_err and verbose:
            print(f"[{symbol}] pagination produced no data: {last_err}")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df = pd.DataFrame(out[-target_bars:], columns=["timestamp","open","high","low","close","volume"])
    return ensure_datetime_index(df)

def fetch_latest_ohlcv(ex, symbol, timeframe="5m", want_bars=3000, verbose=False):
    return fetch_ohlcv_paginated(ex, symbol, timeframe=timeframe, target_bars=want_bars, max_chunk=1500, verbose=verbose)

def filter_supported_symbols(ex, symbols):
    available = set(ex.symbols) if hasattr(ex, "symbols") else set()
    supported = [s for s in symbols if s in available]
    missing = [s for s in symbols if s not in available]
    if missing:
        print("Skipping unsupported symbols:", ", ".join(missing))
    return supported

def estimate_warmup_bars(p: dict) -> int:
    regime_win = int(p.get("regime_win", 500))
    zscore_win = int(p.get("zscore_win", 200))
    ema_slow = int(p.get("ema_slow", 50))
    macd_slow = int(p.get("macd_slow", 26))
    macd_sig = int(p.get("macd_signal", 9))
    atr_p = int(p.get("atr_period", 14))
    rsi_p = int(p.get("rsi_period", 14))
    vol_ma = int(p.get("volume_ma", 20))
    breakout = int(p.get("breakout_lookback", 50))
    base = max(regime_win, zscore_win, ema_slow * 2,
               macd_slow + macd_sig + 10, atr_p + 10, rsi_p + 10,
               vol_ma + 10, breakout + 10)
    return max(base + 50, 600)

# ---------- indicator bootstrap (used only if strategy.compute_indicators returns empty/missing) ----------
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1/n, adjust=False).mean()
    loss = down.ewm(alpha=1/n, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _atr(df, n=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _macd(close, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal
    return ema_fast, ema_slow, macd, macd_signal, macd_hist

def _obv(close, volume):
    dir_ = np.sign(close.diff().fillna(0.0))
    return (dir_ * volume.fillna(0.0)).cumsum()

REQUIRED_FOR_STRATEGY = [
    "ema_fast","ema_slow","macd","macd_signal","macd_hist",
    "rsi","atr","obv","volume"
]

def has_required_cols(df):
    return all(c in df.columns for c in REQUIRED_FOR_STRATEGY)

def bootstrap_indicators(df_raw: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Create a minimal indicator set if the strategy pipeline returns empty/missing."""
    df = df_raw.copy()
    # ensure numeric dtypes
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ema_fast_n = int(p.get("ema_fast", 20))
    ema_slow_n = int(p.get("ema_slow", 50))
    rsi_n      = int(p.get("rsi_period", 14))
    atr_n      = int(p.get("atr_period", 14))
    macd_f     = int(p.get("macd_fast", 12))
    macd_s     = int(p.get("macd_slow", 26))
    macd_sig   = int(p.get("macd_signal", 9))
    vol_ma_n   = int(p.get("volume_ma", 20))

    ema_fast, ema_slow, macd, macd_signal, macd_hist = _macd(df["close"], macd_f, macd_s, macd_sig)
    df["ema_fast"] = _ema(df["close"], ema_fast_n) if "ema_fast" not in df.columns else df["ema_fast"]
    df["ema_slow"] = _ema(df["close"], ema_slow_n) if "ema_slow" not in df.columns else df["ema_slow"]

    # prefer MACD from params
    df["macd"]        = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"]   = macd_hist

    df["rsi"] = _rsi(df["close"], rsi_n)
    df["atr"] = _atr(df, atr_n)
    df["obv"] = _obv(df["close"], df["volume"])
    df["vol_ma"] = df["volume"].rolling(vol_ma_n).mean()

    return df

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol")
    ap.add_argument("--symbols")
    ap.add_argument("--universe", default="COIN50")
    ap.add_argument("--exchange", default=None)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--fetch-bars", type=int, default=None, help="override forward.fetch_bars")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exchange_id = args.exchange or cfg.get("exchange", "binance")
    timeframe = cfg.get("timeframe", "5m")
    p = dict(cfg.get("strategy", {}))
    r = cfg.get("risk", {})
    fwd = cfg.get("forward", {})

    if args.allow_short:
        p["allow_short"] = True

    max_positions = int(r.get("max_positions", 10))
    fee_rate = float(r.get("fee_rate", 0.0006))
    slippage = float(r.get("slippage", 0.0002))
    poll_seconds = int(fwd.get("poll_seconds", 15))
    max_new = int(p.get("max_new_positions_per_cycle", 2))

    WARMUP_N = estimate_warmup_bars(p)
    FETCH_N = int(args.fetch_bars or fwd.get("fetch_bars", 3000))  # default 3000

    # Exchange init
    host_idx = 0
    for _ in range(3):
        try:
            ex = init_exchange_ccxt(exchange_id, host_idx)
            break
        except Exception as e:
            print(f"[exchange init] {type(e).__name__}: {e}")
            if exchange_id == "binance":
                host_idx = rotate_binance_host(host_idx)
                print(f"Rotating Binance host → {BINANCE_HOSTS[host_idx]}")
            time.sleep(1.0)
    else:
        raise RuntimeError("Failed to initialize exchange after retries.")

    # Universe
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(COIN50)

    symbols = filter_supported_symbols(ex, symbols)
    if not symbols:
        raise RuntimeError("No supported symbols after filtering. Check universe or exchange.")

    # Logs
    os.makedirs("logs", exist_ok=True)
    tag = (args.universe or "PORT").replace("/", "_")
    trades_path = os.path.join("logs", f"forward_trades_{tag}.csv")
    equity_path = os.path.join("logs", f"forward_equity_{tag}.csv")
    status_path = os.path.join("logs", f"forward_status_{tag}.csv")

    equity = float(r.get("initial_equity", 10000.0))
    positions = {sym: [] for sym in symbols}
    last_ts = {sym: None for sym in symbols}

    warned_once = set()

    print(f"Starting trendedge paper trading on {exchange_id} ({timeframe}) with {len(symbols)} symbols...")
    print(f"[warmup] estimated ≥ {WARMUP_N} bars initially; paginating to fetch {FETCH_N} bars per symbol.")

    while True:
        try:
            cycle_latest_ts = None
            candidates = []  # (score, sym, ts, now, side, prior_level)
            symbols_processed = 0
            new_bars_seen = 0

            for sym in symbols:
                try:
                    df_raw = fetch_latest_ohlcv(ex, sym, timeframe=timeframe, want_bars=FETCH_N, verbose=args.verbose)
                    symbols_processed += 1
                    if df_raw.empty:
                        if args.verbose:
                            print(f"[{sym}] empty OHLCV — skip")
                        continue

                    # progress heartbeat from raw bars regardless
                    ts_raw = df_raw.index[-1]
                    if (last_ts[sym] is None) or (ts_raw != last_ts[sym]):
                        last_ts[sym] = ts_raw
                        new_bars_seen += 1
                        cycle_latest_ts = ts_raw if cycle_latest_ts is None else max(cycle_latest_ts, ts_raw)

                    # Preferred pipeline
                    df = compute_indicators(df_raw, p)
                    pipe_len = len(df)
                    # Bootstrap if pipeline empty or missing columns the strategy likely needs
                    if pipe_len == 0 or not has_required_cols(df):
                        if args.verbose and sym not in warned_once:
                            missing = [c for c in REQUIRED_FOR_STRATEGY if c not in df.columns]
                            print(f"[{sym}] pipeline_len={pipe_len} → bootstrapping indicators (missing: {missing})")
                            warned_once.add(sym)
                        df = bootstrap_indicators(df_raw, p)

                    # Need at least two rows to attempt signals
                    if len(df) < 2:
                        if args.verbose:
                            print(f"[{sym}] warming up: raw_bars={len(df_raw)} pipeline_len={len(df)} need≈{WARMUP_N}")
                        continue

                    ts = df.index[-1]
                    prev = df.iloc[-2]
                    now = df.iloc[-1]

                    # exits first
                    still = []
                    for pos in positions[sym]:
                        try:
                            realized, closed = pos.check_exit(now, params=p, fee_rate=fee_rate)
                        except Exception as e:
                            if args.verbose and sym not in warned_once:
                                print(f"[{sym}] exit check error: {e}")
                                warned_once.add(sym)
                            realized, closed = 0.0, False
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
                                "exit": float(now.get("close", float('nan'))),
                                "realized": realized,
                                "equity": equity,
                            }
                            pd.DataFrame([rec]).to_csv(
                                trades_path, mode="a", header=not os.path.exists(trades_path), index=False
                            )
                    positions[sym] = still

                    # capacity / already in position
                    if sum(len(v) for v in positions.values()) >= max_positions:
                        continue
                    if positions[sym]:
                        continue

                    # momentum core
                    try:
                        long_ok, short_ok = momentum_signal(prev, now, p)
                    except Exception as e:
                        if args.verbose and sym not in warned_once:
                            print(f"[{sym}] momentum_signal failed (missing cols?): {e}")
                            warned_once.add(sym)
                        continue

                    side = "long" if long_ok else ("short" if p.get("allow_short", False) and short_ok else None)
                    if side is None:
                        if args.verbose:
                            print(f"[{sym} {ts}] momentum not ok")
                        continue

                    # extra filter + score
                    window_span = max(int(p.get("breakout_lookback", 50)), int(p.get("volume_ma", 20)), 60)
                    window = df.iloc[-window_span - 1:]
                    try:
                        ok_extra, score = entry_filter_and_score(window, now, p, side=side)
                    except Exception as e:
                        if args.verbose and sym not in warned_once:
                            print(f"[{sym}] entry_filter_and_score failed (missing cols?): {e}")
                            warned_once.add(sym)
                        continue
                    if not ok_extra:
                        if args.verbose:
                            print(f"[{sym} {ts}] extra entry filter failed")
                        continue

                    # structure stop ref if available
                    prior_level = None
                    try:
                        if p.get("structure_stops", True):
                            prior_level = float(df["low"].iloc[-2]) if side == "long" else float(df["high"].iloc[-2])
                    except Exception:
                        prior_level = None

                    candidates.append((score, sym, ts, now, side, prior_level))

                except (RequestTimeout, ExchangeNotAvailable, DDoSProtection, NetworkError) as e:
                    print(f"[{sym}] transient API error: {e}")
                    if exchange_id == "binance":
                        host_idx = rotate_binance_host(host_idx)
                        try:
                            ex = init_exchange_ccxt(exchange_id, host_idx)
                            print(f"[host switch] using {BINANCE_HOSTS[host_idx]}")
                        except Exception as init_e:
                            print(f"[host switch] re-init failed: {init_e}")
                            time.sleep(2)
                    continue
                except Exception as sym_err:
                    print(f"[ERROR] {sym}: {sym_err}")
                    traceback.print_exc()
                    continue

            # entries
            entries_taken = 0
            if candidates:
                candidates.sort(reverse=True, key=lambda x: x[0])
                total_open = sum(len(lst) for lst in positions.values())
                capacity = max_positions - total_open
                to_open = min(max_new, capacity, len(candidates))

                for k in range(to_open):
                    score, sym, ts, now, side, prior_level = candidates[k]
                    price = float(now.get("close", float("nan")))
                    if not pd.notna(price):
                        continue
                    price_eff = price * (1 + (slippage if side == "long" else -slippage))

                    atr = float(now.get("atr", float("nan")))
                    if not pd.notna(atr) or atr <= 0:
                        if args.verbose:
                            print(f"[{sym}] skip entry — ATR missing/invalid")
                        continue

                    qty = atr_position_size(
                        equity, price_eff, atr,
                        p.get("target_risk_frac", 0.004),
                        p.get("k_stop_atr", 1.5),
                    )
                    if qty <= 0:
                        continue

                    stop_ref, tp1 = compute_levels(side, price_eff, atr, prior_level=prior_level, params=p)

                    positions[sym].append(Position(side, qty, price_eff, stop_ref, tp1))
                    entries_taken += 1

                    rec = {
                        "timestamp": ts.isoformat(),
                        "symbol": sym,
                        "event": "ENTRY",
                        "side": side,
                        "entry": price_eff,
                        "qty": qty,
                        "equity": equity,
                        "score": score,
                    }
                    pd.DataFrame([rec]).to_csv(
                        trades_path, mode="a", header=not os.path.exists(trades_path), index=False
                    )
                    print(f"[{sym} {ts}] {side.upper()} ENTRY px={price:.6f} eff={price_eff:.6f} qty={qty:.6f} score={score:.4f}")

            # logs/heartbeat
            ts_for_log = (
                cycle_latest_ts.isoformat()
                if cycle_latest_ts is not None
                else pd.Timestamp.now(tz="UTC").isoformat()
            )
            pd.DataFrame([{"timestamp": ts_for_log, "equity": equity}]).to_csv(
                equity_path, mode="a", header=not os.path.exists(equity_path), index=False
            )
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