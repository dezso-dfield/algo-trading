#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Get the absolute path of the directory where this script is located.
# This makes the script portable.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Define and create the log directory
LOG=~/logs
mkdir -p "$LOG"

# --- Tmux Session ---
# Create a new detached tmux session named 'trading'
tmux new-session -d -s trading \
  "cd $SCRIPT_DIR/mean_reversion && source ../.venv/bin/activate && python forward.py --universe COIN50 --exchange binance --verbose 2>&1 | tee -a $LOG/mean_reversion.log" \; \
  split-window -h \
  "cd $SCRIPT_DIR/momentum && source ../.venv/bin/activate && python forward.py --universe COIN50 --exchange binance --verbose 2>&1 | tee -a $LOG/momentum.log" \; \
  split-window -v \
  "cd $SCRIPT_DIR/momentum_v2 && source ../.venv/bin/activate && python3 forward_trendedge.py --universe COIN50 --exchange binance --config config.yaml --verbose --allow-short 2>&1 | tee -a $LOG/momentum_v2.log" \; \
  split-window -h \
  "cd $SCRIPT_DIR/triple_ema && source ../.venv/bin/activate && python forward.py --universe COIN50 --exchange binance --verbose 2>&1 | tee -a $LOG/triple_ema.log" \; \
  split-window -v \
  "cd $SCRIPT_DIR/fsvzo && source ../.venv/bin/activate && python forward.py --universe COIN50 --exchange binance --verbose 2>&1 | tee -a $LOG/fsvzo.log" \; \
  select-layout tiled \; \
  attach