
import numpy as np

def atr_position_size(equity, price, atr, target_risk_frac=0.004, risk_multiple=1.5):
    if not np.isfinite(atr) or atr <= 0 or price <= 0:
        return 0.0
    risk_per_unit = risk_multiple * atr
    dollars_at_risk = target_risk_frac * equity
    qty = dollars_at_risk / (risk_per_unit + 1e-12)
    return max(qty / price, 0.0)  # position size in *units* (e.g., contracts/coins)

def compute_levels(side, entry, atr, prior_level=None, params=None):
    params = params or {}
    k_stop = float(params.get("k_stop_atr", 1.5))
    tp1_k  = float(params.get("tp1_atr", 1.0))
    if side == "long":
        stop = (prior_level - 0.1*atr) if (prior_level is not None) else entry - k_stop*atr
        tp1  = entry + tp1_k*atr
    else:
        stop = (prior_level + 0.1*atr) if (prior_level is not None) else entry + k_stop*atr
        tp1  = entry - tp1_k*atr
    return stop, tp1

def update_trailing_stop(pos, bar, params):
    trail_k = float(params.get("trail_after_tp1_atr", 0.5))
    atr = float(bar["atr"])
    if not np.isfinite(atr):
        return
    if pos.side == "long":
        pos.stop = max(pos.stop, pos.entry)  # breakeven at least
        pos.stop = max(pos.stop, bar["close"] - trail_k*atr)
    else:
        pos.stop = min(pos.stop, pos.entry)
        pos.stop = min(pos.stop, bar["close"] + trail_k*atr)
