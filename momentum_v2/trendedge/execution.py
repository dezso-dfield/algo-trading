
def check_exit_conservative(pos, candle, params, fee_rate=0.0, slippage_bps=0.0):
    high, low, close = float(candle["high"]), float(candle["low"]), float(candle["close"])
    stop, tp1 = pos.stop, getattr(pos, "tp1", None)
    realized = 0.0; partial = False

    def fee(px, qty): 
        return abs(px*qty)*fee_rate
    def slip(px, side_mult):
        return px * (1.0 + side_mult * slippage_bps/10000.0)

    if pos.side == "long":
        hit_stop = low <= stop
        hit_tp1  = (tp1 is not None) and (high >= tp1)
        if hit_stop and hit_tp1:
            qty = pos.qty
            px = slip(stop, -1)  # worse fill for stop
            realized += (px - pos.entry)*qty - fee(px,qty) - fee(pos.entry,qty)
            pos.open = False; return realized, True, False
        if hit_tp1:
            qty1 = pos.qty*0.5
            px = slip(tp1, +1)
            realized += (px - pos.entry)*qty1 - fee(px,qty1) - fee(pos.entry,qty1)
            pos.qty -= qty1; partial = True
        if low <= pos.stop:
            qty = pos.qty
            px = slip(pos.stop, -1)
            realized += (px - pos.entry)*qty - fee(px,qty) - fee(pos.entry,qty)
            pos.open = False; return realized, True, partial
    else:
        hit_stop = high >= stop
        hit_tp1  = (tp1 is not None) and (low <= tp1)
        if hit_stop and hit_tp1:
            qty = pos.qty
            px = slip(stop, +1)  # worse fill for stop on shorts
            realized += (pos.entry - px)*qty - fee(px,qty) - fee(pos.entry,qty)
            pos.open = False; return realized, True, False
        if hit_tp1:
            qty1 = pos.qty*0.5
            px = slip(tp1, -1)
            realized += (pos.entry - px)*qty1 - fee(px,qty1) - fee(pos.entry,qty1)
            pos.qty -= qty1; partial = True
        if high >= pos.stop:
            qty = pos.qty
            px = slip(pos.stop, +1)
            realized += (pos.entry - px)*qty - fee(px,qty) - fee(pos.entry,qty)
            pos.open = False; return realized, True, partial

    return realized, False, partial
