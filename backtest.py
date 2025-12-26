import argparse
import importlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def parse_args():
    parser = argparse.ArgumentParser(description="Nifty Backtest")
    parser.add_argument("--strategy", "-s", required=True)
    parser.add_argument("--days", "-d", type=int, default=20)
    parser.add_argument("--interval", "-i", default="15m")
    return parser.parse_args()


def load_strategy(strategy_name):
    strategy_dir = Path("./strategies")
    strategy_path = strategy_dir / f"{strategy_name}.py"
    if not strategy_path.exists():
        print(
            "Available:",
            [f.stem for f in strategy_dir.glob("*.py") if f.stem != "__init__"],
        )
        sys.exit(1)

    sys.path.insert(0, str(strategy_dir.parent))
    module = importlib.import_module(f"strategies.{strategy_name}")
    return module.run_strategy


def run_backtest(args):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # ✅ WORKING SYMBOLS
    symbols = ["^NSEI", "NIFTY50.NS"]
    nifty = None
    for symbol in symbols:
        nifty = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=args.interval,
            progress=False,
        )
        if len(nifty) > 10:
            break

    if len(nifty) == 0:
        print("❌ NO DATA - Use --days 20 --interval 15m")
        return

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [
            col[0] if isinstance(col, tuple) else col for col in nifty.columns
        ]
    nifty = nifty.dropna()

    strategy_func = load_strategy(args.strategy)
    nifty = strategy_func(nifty)

    # FIXED POSITION LOGIC
    position = 0
    positions = [0] * len(nifty)
    for i in range(1, len(nifty)):
        if nifty["Long_Signal"].iloc[i] and position != 1:
            position = 1
        elif nifty["Short_Signal"].iloc[i] and position != -1:
            position = -1
        elif (nifty["Short_Signal"].iloc[i] and position == 1) or (
            nifty["Long_Signal"].iloc[i] and position == -1
        ):
            position = 0
        positions[i] = position

    nifty["Position"] = positions
    nifty["Returns"] = nifty["Close"].pct_change()
    nifty["Strategy"] = nifty["Position"].shift(1) * nifty["Returns"]

    # FIXED METRICS
    total_trades = len(nifty[nifty["Long_Signal"]]) + len(nifty[nifty["Short_Signal"]])
    total_return = (1 + nifty["Strategy"].dropna()).prod() - 1

    print(f"\n🎉 RESULTS: {total_return:.2%} return, {total_trades} trades")

    # SIMPLE PLOT
    plt.figure(figsize=(12, 8))
    plt.plot(nifty.index[-200:], nifty["Close"].iloc[-200:])
    plt.title(f"{args.strategy} Strategy")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(args)
