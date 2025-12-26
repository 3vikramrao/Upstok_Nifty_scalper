#!/usr/bin/env python3
"""
NISO Core Bot Master - Run ALL Strategies Backtest + Comparison
"""

import argparse
import importlib.util
import os
import sys
import warnings
import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ALL Nifty Strategies")
    parser.add_argument("--days", "-d", type=int, default=20, help="Days of data")
    parser.add_argument("--symbols", "-s", nargs="+", default=["^NSEI"], 
                       help="Yahoo symbols (default: ^NSEI)")
    parser.add_argument("--show-plots", action="store_true", help="Show comparison plots")
    return parser.parse_args()


def load_strategy(strategy_name: str) -> Optional[Callable]:
    """Load strategy module dynamically."""
    try:
        strategy_dir = Path("./strategy")
        strategy_path = strategy_dir / f"{strategy_name}.py"
        
        if not strategy_path.exists():
            print(f"❌ Strategy {strategy_name} not found")
            return None
            
        spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        
        if hasattr(module, "run_strategy"):
            return module.run_strategy
            
    except ImportError as e:
        print(f"❌ Import error in {strategy_name}: {e}")
    except Exception as e:
        print(f"❌ Error loading {strategy_name}: {e}")
    
    return None


def fetch_data(days: int, interval: str = "1h", symbols: List[str] = None) -> pd.DataFrame:
    """Fetch Nifty data with multiple symbol fallback."""
    if symbols is None:
        symbols = ["^NSEI"]
    
    print("📊 Fetching data...")
    
    for symbol in symbols:
        try:
            df = yf.download(symbol, period=f"{days}d", interval=interval, progress=False)
            if not df.empty and len(df) >= 50:
                df.columns = df.columns.get_level_values(0)  # Flatten MultiIndex
                print(f"✅ {symbol}: {len(df)} candles")
                return df.reset_index()
        except Exception:
            continue
    
    raise RuntimeError("Failed to fetch data from all symbols")


def backtest_strategy(df: pd.DataFrame, strategy_func: Callable, name: str) -> Dict:
    """Run single strategy backtest."""
    try:
        result_df = strategy_func(df.copy())
        long_signals = result_df["Long_Signal"].sum() if "Long_Signal" in result_df else 0
        short_signals = result_df["Short_Signal"].sum() if "Short_Signal" in result_df else 0
        
        # Simple return calculation
        signals = result_df["Long_Signal"].fillna(False) | result_df["Short_Signal"].fillna(False)
        returns = result_df["Close"].pct_change() * signals.shift().fillna(0)
        total_return = (1 + returns.dropna()).prod() - 1
        
        return {
            "strategy": name,
            "long": int(long_signals),
            "short": int(short_signals),
            "total_signals": int(long_signals + short_signals),
            "total_return": total_return * 100,
            "win_rate": 0 if signals.sum() == 0 else (returns > 0).sum() / signals.sum() * 100
        }
    except Exception as e:
        console.print(f"❌ {name} failed: {e}", style="bold red")
        return {"strategy": name, "long": 0, "short": 0, "total_signals": 0, "total_return": 0, "win_rate": 0}


def run_all_strategies(df: pd.DataFrame, show_plots: bool = False) -> List[Dict]:
    """Run ALL available strategies."""
    strategy_dir = Path("./strategy")
    strategies = []
    
    # Load all strategy files
    for strategy_file in strategy_dir.glob("*.py"):
        if strategy_file.name.startswith("_") or strategy_file.stem == "__init__":
            continue
            
        strategy_func = load_strategy(strategy_file.stem)
        if strategy_func:
            strategies.append(strategy_func)
    
    results = []
    console.print(f"🔬 Running {len(strategies)} strategies...", style="bold cyan")
    
    for strategy_func in strategies:
        # Extract strategy name from function or use generic
        name = getattr(strategy_func, "__name__", "Unknown").replace("run_strategy", "")
        result = backtest_strategy(df, strategy_func, name)
        results.append(result)
    
    return results


def plot_comparison(results: List[Dict]):
    """Rich table dashboard comparing ALL strategies."""
    if not results:
        console.print("No results to display", style="bold yellow")
        return
    
    table = Table(title="🏆 STRATEGY COMPARISON", show_header=True, header_style="bold magenta")
    table.add_column("Strategy", style="cyan")
    table.add_column("Long", justify="right")
    table.add_column("Short", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Win %", justify="right")
    
    best_return = max(r["total_return"] for r in results)
    
    for result in sorted(results, key=lambda x: x["total_return"], reverse=True):
        return_color = "bold green" if result["total_return"] == best_return else "white"
        table.add_row(
            result["strategy"][:20],
            str(result["long"]),
            str(result["short"]),
            str(result["total_signals"]),
            f"[{return_color}]{result['total_return']:.1f}%[/]",
            f"{result['win_rate']:.0f}%"
        )
    
    console.print(table)


def main():
    """Main execution."""
    args = parse_args()
    
    # Fetch data
    df = fetch_data(args.days)
    
    # Run all strategies
    results = run_all_strategies(df, args.show_plots)
    
    # Display comparison
    plot_comparison(results)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"niso_backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
    console.print("💾 Results saved to niso_backtest_*.csv", style="bold green")


if __name__ == "__main__":
    main()

