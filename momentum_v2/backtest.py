import argparse, pandas as pd, yaml
from strategy import ensure_datetime_index, compute_indicators, momentum_signal, entry_filter_and_score, Position, atr_position_size, compute_levels

def run_backtest(df, p, fee_rate=0.0006, slippage=0.0002, initial_equity=100_000.0, allow_short=True):
    df = ensure_datetime_index(df)
    df = compute_indicators(df, p).dropna()
    equity = initial_equity
    pos = None
    trades = []
    rows = df.reset_index()

    for i in range(2, len(rows)-1):
        prev = rows.loc[i-1]; now = rows.loc[i]; nxt = rows.loc[i+1]
        if pos and pos.open:
            realized, closed = pos.check_exit(now, p, fee_rate=fee_rate)
            equity += realized
            if closed:
                trades.append({"ts": now["timestamp"], "side": pos.side, "entry": pos.entry, "exit": now["close"], "pnl": realized})
                pos = None

        if (pos is None) or (not pos.open):
            long_ok, short_ok = momentum_signal(prev, now, p)
            side = None
            if long_ok:
                side = "long"
            elif allow_short and p.get("allow_short", True) and short_ok:
                side = "short"
            else:
                continue

            w = df.iloc[max(0, i- max(60, int(p.get('breakout_lookback',50))) - 5): i+1]
            ok, score = entry_filter_and_score(w, now, p, side=side)
            if not ok: continue
            entry = float(nxt["open"]) * (1 + (slippage if side=="long" else -slippage))
            atr = float(now["atr"])
            qty = atr_position_size(equity, entry, atr, p.get("target_risk_frac",0.004), p.get("k_stop_atr",1.5))
            stop, tp1 = compute_levels(side, entry, atr, prior_level=w["low"].iloc[-2] if (side=="long" and p.get("structure_stops",True)) else (w["high"].iloc[-2] if side=="short" and p.get("structure_stops",True) else None), params=p)
            pos = Position(side, qty, entry, stop, tp1)

    return equity - initial_equity, trades

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config,"r") as f:
        cfg = yaml.safe_load(f)
    p = cfg.get("strategy", {})
    fee_rate = cfg.get("risk",{}).get("fee_rate", 0.0006)

    df = pd.read_csv(args.csv)
    pnl, trades = run_backtest(df, p, fee_rate=fee_rate)
    print(f"PNL: {pnl:.2f}, trades={len(trades)}")
