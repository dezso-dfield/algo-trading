import argparse, os, time
import pandas as pd
import numpy as np
import yaml
from typing import Optional, Dict
from strategy import ensure_datetime_index, compute_indicators, momentum_signal, Position
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
        since = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000.0)
    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    return df

def load_data(csv: Optional[str], symbol: str, exchange: str, timeframe: str, since_days: int = 60):
    if csv:
        df = pd.read_csv(csv)
    else:
        since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=since_days)).timestamp() * 1000)
        df = fetch_ccxt_ohlcv(symbol, exchange=exchange, timeframe=timeframe, since_ms=since)
    return ensure_datetime_index(df)

def resolve_symbol_list(symbols_arg, universe_flag):
    if universe_flag == "COIN50":
        return COIN50
    if symbols_arg:
        return [s.strip() for s in symbols_arg.split(",") if s.strip()]
    return COIN50

def load_multi_symbol_data(symbols, exchange, timeframe, since_days, csv_map=None):
    data_map = {}
    for sym in symbols:
        csv = None
        if csv_map and sym in csv_map:
            csv = csv_map[sym]
        df = load_data(csv, sym, exchange, timeframe, since_days)
        data_map[sym] = df
    return data_map

def run_portfolio_backtest(data_map: Dict[str, pd.DataFrame], config: dict):
    p = config["strategy"]
    r = config["risk"]
    initial_equity = r["initial_equity"]
    allocation_cash = r["allocation_per_trade_cash"]
    fee_rate = r["fee_rate"]
    slippage = r["slippage"]
    max_positions = r["max_positions"]

    # compute indicators
    for sym in data_map:
        data_map[sym] = compute_indicators(data_map[sym], p).dropna()

    equity = initial_equity
    positions = {sym: [] for sym in data_map.keys()}
    trade_log = []

    all_ts = sorted(set().union(*[df.index for df in data_map.values()]))
    lookback = p["obv_lookback"]

    for i in range(lookback+1, len(all_ts)):
        ts_prev, ts_now = all_ts[i-1], all_ts[i]
        for sym, df in data_map.items():
            if ts_prev not in df.index or ts_now not in df.index:
                continue
            window = df.loc[all_ts[i-lookback]:ts_now]
            if len(window) < lookback:
                continue
            prev = df.loc[ts_prev]
            now = df.loc[ts_now]
            price = now["close"]

            # manage exits
            still = []
            for pos in positions[sym]:
                realized, closed = pos.check_exit(now, p, fee_rate=fee_rate)
                equity += realized
                if not closed:
                    still.append(pos)
                else:
                    trade_log.append({
                        "timestamp": ts_now.isoformat(),
                        "symbol": sym,
                        "event": "EXIT",
                        "side": pos.side,
                        "entry": pos.entry,
                        "exit": price,
                        "realized": realized,
                        "equity": equity
                    })
            positions[sym] = still

            # entry capacity
            total_open = sum(len(lst) for lst in positions.values())
            if total_open >= max_positions:
                continue

            long_ok, short_ok = momentum_signal(prev, now, p)
            # OBV slope confirmation
            obv_slope = (window["obv"].iloc[-1] - window["obv"].iloc[0]) / lookback
            if long_ok:
                if obv_slope <= p["obv_slope_min"]:
                    long_ok = False
            if short_ok:
                if obv_slope >= -p["obv_slope_min"]:
                    short_ok = False

            signal = "long" if long_ok else ("short" if short_ok else None)
            if not signal:
                continue

            # entry price with slippage
            price_eff = price * (1 + slippage * (1 if signal=="long" else -1))
            qty = allocation_cash / price_eff
            # initial stop = ema_slow or ATR trail
            stop = now["ema_slow"]
            if np.isfinite(now["atr"]) and now["atr"] > 0:
                if signal == "long":
                    stop = min(stop, price - p["atr_mult"] * now["atr"])
                else:
                    stop = max(stop, price + p["atr_mult"] * now["atr"])

            fee = price_eff * qty * fee_rate
            equity -= fee
            positions[sym].append(Position(signal, qty, price_eff, stop=stop))

            # ENTRY log
            trade_log.append({
                "timestamp": ts_now.isoformat(),
                "symbol": sym,
                "event": "ENTRY",
                "side": signal,
                "entry": price_eff,
                "equity": equity
            })

    # mark to market
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
                        "event": "EXIT",
                        "side": pos.side,
                        "entry": pos.entry,
                        "exit": px,
                        "realized": pnl,
            "equity": equity + pnl
                    })
                    equity += pnl

    return equity, pd.DataFrame(trade_log)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol")
    parser.add_argument("--symbols")
    parser.add_argument("--universe", default="COIN50")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--since_days", type=int, default=60)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    exchange = args.exchange or config["exchange"]
    timeframe = config["timeframe"]

    # resolve symbols
    if args.symbol:
        symbols = [args.symbol]
    else:
        cfg_syms = config.get("symbols", [])
        use_uni = args.universe if args.universe else (cfg_syms[0] if cfg_syms and cfg_syms[0] in ("COIN50",) else None)
        symbols = resolve_symbol_list(args.symbols, use_uni)

    csv_map = {symbols[0]: args.csv} if args.csv and len(symbols)==1 else None
    data_map = load_multi_symbol_data(symbols, exchange, timeframe, args.since_days, csv_map)

    final_eq, trades = run_portfolio_backtest(data_map, config)

    os.makedirs("logs", exist_ok=True)
    tag = (args.universe or "PORT").replace("/","_")
    trades_path = os.path.join("logs", f"backtest_{tag}_trades.csv")
    eq_path = os.path.join("logs", f"backtest_{tag}_equity.csv")
    trades.to_csv(trades_path, index=False)
    pd.DataFrame([{"final_equity": final_eq}]).to_csv(eq_path, index=False)
    print(f"Portfolio final equity: {final_eq:.2f}")
    print(f"Saved trades to {trades_path}")
    print(f"Saved final equity to {eq_path}")

if __name__ == "__main__":
    main()
