import argparse, os, time
import pandas as pd
import numpy as np
import yaml
from typing import Optional, Dict

from strategy import ensure_datetime_index, compute_indicators, in_triple_ema_up, swing_low_stop, Position
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
        if not batch: break
        all_rows += batch
        since = batch[-1][0] + 1
        time.sleep(ex.rateLimit/1000.0)
    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    return df

def load_data(csv: Optional[str], symbol: str, exchange: str, timeframe: str, since_days: int = 60):
    if csv:
        df = pd.read_csv(csv)
    else:
        since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=since_days)).timestamp() * 1000)
        df = fetch_ccxt_ohlcv(symbol, exchange=exchange, timeframe=timeframe, since_ms=since)
    return ensure_datetime_index(df)

def resolve_symbols(arg_list, universe_flag):
    if universe_flag == "COIN50": return COIN50
    if arg_list: return [s.strip() for s in arg_list.split(",") if s.strip()]
    return COIN50

def rank_top_n(momentum_map: Dict[str, pd.DataFrame], lookback_bars: int, top_n: int):
    scores = []
    for sym, df in momentum_map.items():
        if len(df) <= lookback_bars: continue
        ret = df["close"].iloc[-1] / df["close"].iloc[-lookback_bars] - 1.0
        scores.append((sym, ret))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s,_ in scores[:top_n]]

def run_portfolio_backtest(data_map: Dict[str, pd.DataFrame], config: dict):
    p = config["strategy"]
    r = config["risk"]
    fee_rate, slippage = r["fee_rate"], r["slippage"]

    # indicators
    for sym in data_map:
        data_map[sym] = compute_indicators(data_map[sym], p).dropna()

    # build master timeline (daily-like step using bar index)
    all_ts = sorted(set().union(*[df.index for df in data_map.values()]))
    equity = r["initial_equity"]
    positions = {sym: [] for sym in data_map.keys()}
    trade_log = []

    lookback = p["momentum_lookback_bars"]
    top_n = p["top_n"]
    swing_n = p["swing_lookback"]

    # helper: sizing
    def compute_qty(entry_price, stop_price, equity_now):
        if r["sizing_mode"] == "fixed_cash":
            cash = r["fixed_cash_per_trade"]
            return cash / entry_price
        # risk % of equity / per-share risk
        risk_cash = equity_now * r["risk_pct_per_trade"]
        per_unit_risk = max(entry_price - stop_price, 1e-12)
        return risk_cash / per_unit_risk

    # determine re-rank points: every 'lookback' or daily cadence; for simplicity, recalc each bar after minimum history
    active_set = set()
    for i in range(max(lookback, swing_n)+1, len(all_ts)):
        ts = all_ts[i]
        # 1) recompute momentum ranking
        slice_map = {sym: data_map[sym].loc[:ts] for sym in data_map}
        top_list = rank_top_n(slice_map, lookback, top_n)

        # 2) close positions that drop out of top list or break condition
        for sym in list(positions.keys()):
            df = slice_map[sym]
            if ts not in df.index: continue
            row = df.loc[ts]
            still = []
            for pos in positions[sym]:
                realized, closed = pos.check_exit(row, fee_rate=fee_rate)
                equity += realized
                if not closed:
                    still.append(pos)
                else:
                    trade_log.append({
                        "timestamp": ts.isoformat(),
                        "symbol": sym,
                        "event": "EXIT",
                        "side": pos.side,
                        "entry": pos.entry,
                        "exit": row["close"],
                        "realized": realized,
                        "equity": equity
                    })
            positions[sym] = still

            # if symbol not in top_list -> close remaining at market
            if sym not in top_list and positions[sym]:
                # close all
                for pos in positions[sym]:
                    pnl = pos.mark(row["close"])
                    equity += pnl - (abs(pos.entry)+abs(row["close"])) * fee_rate * pos.qty
                    trade_log.append({
                        "timestamp": ts.isoformat(),
                        "symbol": sym,
                        "event": "EXIT",
                        "side": pos.side,
                        "entry": pos.entry,
                        "exit": row["close"],
                        "realized": pnl,
                        "equity": equity
                    })
                positions[sym] = []

        # 3) entries for new top_list members if hierarchy holds and capacity available
        # capacity across portfolio
        total_open = sum(len(lst) for lst in positions.values())
        capacity = r["max_positions"] - total_open
        if capacity <= 0: 
            continue

        # iterate sorted top_list; open where no position
        for sym in top_list:
            if capacity <= 0: break
            df = slice_map[sym]
            if ts not in df.index: continue
            if positions[sym]:  # already in
                continue
            row = df.loc[ts]
            if not in_triple_ema_up(row):
                continue
            # stop = swing low over last N bars
            window = df.iloc[max(0, df.index.get_loc(ts)-swing_n):df.index.get_loc(ts)+1]
            stop = swing_low_stop(window, atr=row.get("atr", np.nan), atr_mult=p["atr_mult"])
            entry = row["close"] * (1 + slippage)
            qty = compute_qty(entry, stop, equity)
            if qty <= 0: 
                continue
            # fee
            equity -= entry * qty * fee_rate

            positions[sym].append(Position("long", qty, entry, stop))
            capacity -= 1
            trade_log.append({
                "timestamp": ts.isoformat(),
                "symbol": sym,
                "event": "ENTRY",
                "side": "long",
                "entry": entry,
                "equity": equity
            })

    # mark-to-market close
    last_ts = all_ts[-1] if all_ts else None
    if last_ts:
        for sym, df in data_map.items():
            if last_ts in df.index:
                px = df.loc[last_ts, "close"]
                for pos in positions[sym]:
                    pnl = pos.mark(px)
                    equity += pnl
                    trade_log.append({
                        "timestamp": last_ts.isoformat(),
                        "symbol": sym,
                        "event": "EXIT",
                        "side": pos.side,
                        "entry": pos.entry,
                        "exit": px,
                        "realized": pnl,
                        "equity": equity
                    })

    return equity, pd.DataFrame(trade_log)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol")
    parser.add_argument("--symbols")
    parser.add_argument("--universe", default="COIN50")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--since_days", type=int, default=120)
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
        cfg = config.get("symbols", [])
        use_uni = args.universe if args.universe else (cfg[0] if cfg and cfg[0] in ("COIN50",) else None)
        symbols = resolve_symbols(args.symbols, use_uni)

    # load data
    csv_map = {symbols[0]: args.csv} if args.csv and len(symbols)==1 else None
    data_map = {}
    for sym in symbols:
        df = load_data(None if not csv_map else csv_map.get(sym), sym, exchange, timeframe, args.since_days)
        data_map[sym] = df

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
