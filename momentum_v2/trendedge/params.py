
PARAMS = dict(
    ema_fast=20, ema_slow=50, rsi_period=14,
    macd_fast=12, macd_slow=26, macd_signal=9,
    atr_period=14,
    # regimes
    regime_win=500, zscore_win=200, ema_slope_min=0.0,
    atrpct_pct_low=0.15, atrpct_pct_high=0.95,
    # entries
    breakout_lookback=50, volume_ma=20, min_volume_ma_ratio=1.15,
    obv_slope_min=0.0, min_macd_hist_z=0.0, require_breakout=True,
    pullback_after_breakout=False,  # wait one bar retest option
    # HTF bias
    use_htf=False, htf_multiple=4, htf_ema_slow=50, htf_ema_slope_min=0.0,
    # risk/exits
    k_stop_atr=1.5, tp1_atr=1.0, trail_after_tp1_atr=0.5,
    target_risk_frac=0.004, max_bars_in_trade=80,
    structure_stops=True, use_psar=False,
    fee_rate=0.0006, slippage_bps=1.0,
    # guardrails
    max_daily_loss_frac=0.03,  # kill-switch per day
    cooldown_trades=2, cooldown_bars=20,
    max_concurrent=3, per_asset_exposure_frac=0.5,
    # hygiene
    warmup_bars=300
)
