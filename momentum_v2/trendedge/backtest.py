
import numpy as np
import pandas as pd
from .entries import momentum_signal, entry_filter_and_score
from .risk import atr_position_size, compute_levels, update_trailing_stop
from .execution import check_exit_conservative

def _day_key(ts):
    # assumes ts is pandas Timestamp
    return pd.Timestamp(ts).normalize()

def run(df: pd.DataFrame, params: dict, start=None, initial_equity=100_000.0, allow_short=True):
    fee_rate = params.get("fee_rate", 0.0006)
    slippage_bps = params.get("slippage_bps", 0.0)

    start = start if start is not None else int(params.get("warmup_bars", 300))
    eq = initial_equity
    pos = None
    trades = []
    equity = []
    cooldown_left = 0
    daily_start_eq = eq
    daily_key = None

    rows = df.reset_index()
    for i in range(start, len(rows)-1):
        prev = rows.loc[i-1]; now = rows.loc[i]; nxt = rows.loc[i+1]
        ts = pd.to_datetime(now["timestamp"])
        dk = _day_key(ts)
        if daily_key is None or dk != daily_key:
            daily_key = dk
            daily_start_eq = eq

        # guardrail: daily loss
        daily_drawdown = (eq - daily_start_eq) / max(daily_start_eq, 1e-12)
        if daily_drawdown <= -float(params.get("max_daily_loss_frac", 0.03)):
            equity.append(eq); 
            if pos and getattr(pos,"open",False):
                pos.bars += 1
            continue

        # manage open position
        if pos is not None and getattr(pos, "open", False):
            realized, closed, partial = check_exit_conservative(pos, now, params, fee_rate=fee_rate, slippage_bps=slippage_bps)
            eq += realized
            if partial:
                update_trailing_stop(pos, now, params)
            # trend invalidation
            if pos.open:
                px = float(now["close"])
                ema_slow = float(now["ema_slow"])
                macd = float(now["macd"]); macd_sig = float(now["macd_signal"])
                use_psar = params.get("use_psar", False)
                psar_val = now.get("psar", np.nan)
                if (pos.side=="long" and (px<ema_slow or macd<macd_sig or (use_psar and np.isfinite(psar_val) and px<psar_val))) or                    (pos.side=="short" and (px>ema_slow or macd>macd_sig or (use_psar and np.isfinite(psar_val) and px>psar_val))):
                    qty = pos.qty
                    # market-out at close with fees+slip
                    realized = ((px - pos.entry)*qty if pos.side=="long" else (pos.entry - px)*qty)
                    fees = abs(px*qty+pos.entry*qty)*fee_rate
                    eq += (realized - fees)
                    pos.open = False
                    trades.append({"side":pos.side,"entry":pos.entry,"exit":px,"pnl":realized-fees,"trend_exit":True,"time":str(ts)})
            # time stop
            if pos and pos.open and getattr(pos, "bars", 0) >= params.get("max_bars_in_trade", 100):
                px = float(now["close"])
                qty = pos.qty
                realized = (px - pos.entry)*qty if pos.side=="long" else (pos.entry - px)*qty
                fees = abs(px*qty+pos.entry*qty)*fee_rate
                eq += (realized - fees); pos.open=False
                trades.append({"side":pos.side,"entry":pos.entry,"exit":px,"pnl":realized-fees,"time_exit":True,"time":str(ts)})

        # flat → look for entries (execute next open)
        if (pos is None) or (not getattr(pos, "open", False)):
            if cooldown_left > 0:
                cooldown_left -= 1
            else:
                long_ok, short_ok = momentum_signal(prev, now, params)
                win = df.iloc[max(0, i- max(60, params.get("breakout_lookback",50))-10): i+1]
                chosen = None
                if long_ok:
                    ok, score = entry_filter_and_score(win, now, params, side="long")
                    if ok:
                        chosen = ("long", score)
                if allow_short and short_ok:
                    ok, score_s = entry_filter_and_score(win, now, params, side="short")
                    if ok:
                        if chosen is None or score_s > chosen[1]:
                            chosen = ("short", score_s)

                if chosen is not None:
                    side = chosen[0]
                    entry = float(nxt["open"]) * (1.0 + (params.get("slippage_bps",0.0)/10000.0) * (+1 if side=="long" else -1))
                    atr = float(now["atr"])
                    qty = atr_position_size(eq, entry, atr,
                        target_risk_frac=params.get("target_risk_frac",0.004),
                        risk_multiple=params.get("k_stop_atr",1.5))
                    if qty <= 0:
                        equity.append(eq); 
                        continue
                    # structured stop reference
                    prior_level = None
                    if params.get("structure_stops",True):
                        prior_level = win["low"].iloc[-2] if side=="long" else win["high"].iloc[-2]
                    stop, tp1 = compute_levels(side, entry, atr, prior_level=prior_level, params=params)
                    pos = type("Pos", (), {})()
                    pos.side=side; pos.qty=qty; pos.entry=entry; pos.stop=stop; pos.tp1=tp1; pos.open=True; pos.bars=0
        else:
            # still open position
            pass

        equity.append(eq)
        if pos and pos.open: pos.bars += 1

        # cooldown after stop-outs (approx via last trade pnl)
        if len(trades) >= 1 and "trend_exit" not in trades[-1] and "time_exit" not in trades[-1]:
            # stop/TP handled inside check_exit; if last closed and pnl < 0, apply cooldown
            pass

    # Basic stats
    eq_series = pd.Series(equity, index=df.index[start:])
    pnl = [t.get("pnl",0.0) for t in trades]
    wins = sum(1 for x in pnl if x>0)
    stats = dict(
        start_equity=initial_equity,
        end_equity=float(eq_series.iloc[-1]) if len(eq_series)>0 else initial_equity,
        n_trades=len(trades), win_rate=(wins/len(pnl) if pnl else 0.0),
        total_pnl=sum(pnl)
    )
    return eq_series, trades, stats
