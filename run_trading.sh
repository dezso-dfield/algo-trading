# ~/run_trading.sh
#!/usr/bin/env bash
set -e
LOG=~/logs; mkdir -p "$LOG"
ROOT=~/Desktop/dfield_project/trading_strategies

tmux new-session -d -s trading \
  "cd $ROOT/mean_reversion && source ../.venv/bin/activate && python forward.py --universe COIN50 --exchange binance --verbose 2>&1 | tee -a $LOG/mean_rev_forward.log" \; \
  split-window -h \
  "cd $ROOT/momentum && source ../.venv/bin/activate && python forward.py --universe COIN50 --exchange binance --verbose 2>&1 | tee -a $LOG/momentum_forward.log" \; \
  split-window -v \
  "cd $ROOT/triple_ema && source ../.venv/bin/activate && python forward.py --universe COIN50 --exchange binance --verbose 2>&1 | tee -a $LOG/triple_ema_forward.log" \; \
  select-pane -t 0 \; select-layout even-horizontal \; attach