
import argparse, json, sys
from . import params as default_params
from . import data, backtest
import pandas as pd

def parse_override(ovr_list):
    out = {}
    for item in ovr_list or []:
        if "=" not in item: 
            continue
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except Exception:
            try:
                out[k] = float(v) if "." in v else int(v)
            except Exception:
                out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser(prog="trendedge")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("backtest", help="Run backtest on CSV")
    b.add_argument("--csv", required=True, help="Path to CSV with timestamp,open,high,low,close,volume")
    b.add_argument("--params", help="JSON file with params")
    b.add_argument("--override", nargs="*", help='Override params e.g. --override k_stop_atr=2 target_risk_frac=0.003')

    args = ap.parse_args()
    if args.cmd == "backtest":
        P = dict(default_params.PARAMS)
        if args.params:
            with open(args.params, "r") as f:
                P.update(json.load(f))
        P.update(parse_override(args.override))
        df = data.load_csv(args.csv)
        dfp = data.prepare(df, P)
        eq, trades, stats = backtest.run(dfp, P)
        print(json.dumps(stats, indent=2))
        # optional: write trades CSV
        trades_df = pd.DataFrame(trades)
        if len(trades_df):
            out_path = args.csv.rsplit(".",1)[0] + "_trades.csv"
            trades_df.to_csv(out_path, index=False)
            print(f"Trades saved to {out_path}")
    else:
        ap.print_help()
