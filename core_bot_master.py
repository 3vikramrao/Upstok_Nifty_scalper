#!/usr/bin/env python3
"""
CORE BOT master: Run ALL Nifty Strategy + Comparison Dashboard
Usage: python core_bot.py --days 20 --interval 15m
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import argparse
import importlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ALL YOUR STRATEGIES
STRATEGY_LIST = ["ema_crossover", "crt_hourly", "vp_profile", "multi_bot"]

def parse_args():
    parser = argparse.ArgumentParser(description='Run ALL Nifty Strategies')
    parser.add_argument('--days', '-d', type=int, default=20, help='Days of data')
    parser.add_argument('--interval', '-i', default='15m', choices=['5m', '15m', '1h'])
    parser.add_argument('--show-plots', action='store_true', help='Show individual charts')
    return parser.parse_args()

def load_strategy(strategy_name):
    strategy_dir = Path("./strategy")
    strategy_path = strategy_dir / f"{strategy_name}.py"
    
    if not strategy_path.exists():
        print(f"❌ Missing: {strategy_name}")
        return None
    
    if str(strategy_dir.parent) not in sys.path:
        sys.path.insert(0, str(strategy_dir.parent))
    
    try:
        module = importlib.import_module(f"strategy.{strategy_name}")
        if hasattr(module, 'run_strategy'):
            return module.run_strategy
    except:
        pass
    return None#!/usr/bin/env python3
"""
CORE BOT: Run ALL Nifty Strategies + Comparison Dashboard
Usage: python core_bot.py --days 20 --interval 15m
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import argparse
import importlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ALL YOUR STRATEGIES
STRATEGY_LIST = ["ema_crossover", "crt_hourly", "vp_profile", "multi_bot"]

def parse_args():
    parser = argparse.ArgumentParser(description='Run ALL Nifty Strategies')
    parser.add_argument('--days', '-d', type=int, default=20, help='Days of data')
    parser.add_argument('--interval', '-i', default='15m', choices=['5m', '15m', '1h'])
    parser.add_argument('--show-plots', action='store_true', help='Show individual charts')
    return parser.parse_args()

def load_strategy(strategy_name):
    strategy_dir = Path("./strategy")
    strategy_path = strategy_dir / f"{strategy_name}.py"
    
    if not strategy_path.exists():
        print(f"❌ Missing: {strategy_name}")
        return None
    
    if str(strategy_dir.parent) not in sys.path:
        sys.path.insert(0, str(strategy_dir.parent))
    
    try:
        module = importlib.import_module(f"strategy.{strategy_name}")
        if hasattr(module, 'run_strategy'):
            return module.run_strategy
    except:
        pass
    return None

def fetch_data(days, interval):
    """Fetch Nifty data with multiple symbol fallback"""
    print("📊 Fetching data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    symbols = ["^NSEI", "NIFTY50.NS", "BANKNIFTY.NS"]
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, 
                           interval=interval, progress=False)
            if len(df) > 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                df = df.dropna()
                print(f"✅ {symbol}: {len(df)} candles")
                return df
        except:
            continue
    
    print("❌ No data found!")
    sys.exit(1)

def backtest_strategy(df, strategy_func, name):
    """Run single strategy backtest"""
    try:
        df_signals = strategy_func(df.copy())
        
        # Position logic
        position = 0
        positions = [0] * len(df_signals)
        for i in range(1, len(df_signals)):
            if df_signals['Long_Signal'].iloc[i] and position != 1:
                position = 1
            elif df_signals['Short_Signal'].iloc[i] and position != -1:
                position = -1
            elif (df_signals['Short_Signal'].iloc[i] and position == 1) or \
                 (df_signals['Long_Signal'].iloc[i] and position == -1):
                position = 0
            positions[i] = position
        
        df_signals['Position'] = positions
        df_signals['Returns'] = df_signals['Close'].pct_change()
        df_signals['Strategy'] = df_signals['Position'].shift(1) * df_signals['Returns']
        
        # Metrics
        total_return = (1 + df_signals['Strategy'].dropna()).prod() - 1
        total_trades = int(df_signals['Long_Signal'].sum() + df_signals['Short_Signal'].sum())
        equity = (1 + df_signals['Strategy']).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()
        
        return {
            'name': name,
            'return': total_return,
            'trades': total_trades,
            'max_dd': max_dd,
            'win_rate': len(df_signals[df_signals['Strategy'] > 0]) / total_trades if total_trades > 0 else 0,
            'df': df_signals
        }
    except Exception as e:
        print(f"❌ {name}: {e}")
        return None

def run_all_strategy(df, show_plots=False):
    """Run ALL strategy and compare"""
    results = []
    
    print("\n🔥 RUNNING ALL STRATEGIES...\n")
    print("Strategy\t\tReturn\tTrades\tWin%\tMaxDD")
    print("-" * 50)
    
    for strategy_name in STRATEGY_LIST:
        strategy_func = load_strategy(strategy_name)
        if strategy_func:
            result = backtest_strategy(df, strategy_func, strategy_name)
            if result:
                results.append(result)
                print(f"{strategy_name:<15}\t{result['return']:.2%}\t{result['trades']:3d}\t{result['win_rate']:.1%}\t{result['max_dd']:.2%}")
                
                # Individual plot
                if show_plots and len(result['df']) > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(result['df'].index, result['df']['Close'], label='Nifty')
                    longs = result['df'][result['df']['Long_Signal']]
                    shorts = result['df'][result['df']['Short_Signal']]
                    plt.scatter(longs.index, longs['Low'], marker='^', color='green', s=50, label='Long')
                    plt.scatter(shorts.index, shorts['High'], marker='v', color='red', s=50, label='Short')
                    plt.title(f"{strategy_name} - Return: {result['return']:.2%}")
                    plt.legend()
                    plt.show()
    
    return results

def plot_comparison(results):
    """Dashboard comparing ALL strategy"""
    if not results:
        return
    
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Returns comparison
    names = [r['name'] for r in results]
    returns = [r['return'] for r in results]
    ax1.bar(names, returns)
    ax1.set_title('Total Return Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Trades
    trades = [r['trades'] for r in results]
    ax2.bar(names, trades)
    ax2.set_title('Total Trades')
    ax2.tick_params(axis='x', rotation=45)
    
    # Win Rate
    win_rates = [r['win_rate'] for r in results]
    ax3.bar(names, win_rates)
    ax3.set_title('Win Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    # Max Drawdown
    drawdowns = [abs(r['max_dd']) for r in results]
    ax4.bar(names, drawdowns)
    ax4.set_title('Max Drawdown')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # BEST STRATEGY
    best = max(results, key=lambda x: x['return'])
    print(f"\n🏆 BEST: {best['name']} - {best['return']:.2%} return")

def main():
    args = parse_args()
    print(f"🚀 CORE BOT: {args.days} days, {args.interval}")
    
    df = fetch_data(args.days, args.interval)
    results = run_all_strategy(df, args.show_plots)
    
    if results:
        plot_comparison(results)

if __name__ == '__main__':
    main()


def fetch_data(days, interval):
    """Fetch Nifty data with multiple symbol fallback"""
    print("📊 Fetching data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    symbols = ["^NSEI", "NIFTY50.NS", "BANKNIFTY.NS"]
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, 
                           interval=interval, progress=False)
            if len(df) > 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                df = df.dropna()
                print(f"✅ {symbol}: {len(df)} candles")
                return df
        except:
            continue
    
    print("❌ No data found!")
    sys.exit(1)

def backtest_strategy(df, strategy_func, name):
    """Run single strategy backtest"""
    try:
        df_signals = strategy_func(df.copy())
        
        # Position logic
        position = 0
        positions = [0] * len(df_signals)
        for i in range(1, len(df_signals)):
            if df_signals['Long_Signal'].iloc[i] and position != 1:
                position = 1
            elif df_signals['Short_Signal'].iloc[i] and position != -1:
                position = -1
            elif (df_signals['Short_Signal'].iloc[i] and position == 1) or \
                 (df_signals['Long_Signal'].iloc[i] and position == -1):
                position = 0
            positions[i] = position
        
        df_signals['Position'] = positions
        df_signals['Returns'] = df_signals['Close'].pct_change()
        df_signals['Strategy'] = df_signals['Position'].shift(1) * df_signals['Returns']
        
        # Metrics
        total_return = (1 + df_signals['Strategy'].dropna()).prod() - 1
        total_trades = int(df_signals['Long_Signal'].sum() + df_signals['Short_Signal'].sum())
        equity = (1 + df_signals['Strategy']).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()
        
        return {
            'name': name,
            'return': total_return,
            'trades': total_trades,
            'max_dd': max_dd,
            'win_rate': len(df_signals[df_signals['Strategy'] > 0]) / total_trades if total_trades > 0 else 0,
            'df': df_signals
        }
    except Exception as e:
        print(f"❌ {name}: {e}")
        return None

def run_all_strategy(df, show_plots=False):
    """Run ALL strategy and compare"""
    results = []
    
    print("\n🔥 RUNNING ALL STRATEGIES...\n")
    print("Strategy\t\tReturn\tTrades\tWin%\tMaxDD")
    print("-" * 50)
    
    for strategy_name in STRATEGY_LIST:
        strategy_func = load_strategy(strategy_name)
        if strategy_func:
            result = backtest_strategy(df, strategy_func, strategy_name)
            if result:
                results.append(result)
                print(f"{strategy_name:<15}\t{result['return']:.2%}\t{result['trades']:3d}\t{result['win_rate']:.1%}\t{result['max_dd']:.2%}")
                
                # Individual plot
                if show_plots and len(result['df']) > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(result['df'].index, result['df']['Close'], label='Nifty')
                    longs = result['df'][result['df']['Long_Signal']]
                    shorts = result['df'][result['df']['Short_Signal']]
                    plt.scatter(longs.index, longs['Low'], marker='^', color='green', s=50, label='Long')
                    plt.scatter(shorts.index, shorts['High'], marker='v', color='red', s=50, label='Short')
                    plt.title(f"{strategy_name} - Return: {result['return']:.2%}")
                    plt.legend()
                    plt.show()
    
    return results

def plot_comparison(results):
    """Dashboard comparing ALL strategy"""
    if not results:
        return
    
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Returns comparison
    names = [r['name'] for r in results]
    returns = [r['return'] for r in results]
    ax1.bar(names, returns)
    ax1.set_title('Total Return Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Trades
    trades = [r['trades'] for r in results]
    ax2.bar(names, trades)
    ax2.set_title('Total Trades')
    ax2.tick_params(axis='x', rotation=45)
    
    # Win Rate
    win_rates = [r['win_rate'] for r in results]
    ax3.bar(names, win_rates)
    ax3.set_title('Win Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    # Max Drawdown
    drawdowns = [abs(r['max_dd']) for r in results]
    ax4.bar(names, drawdowns)
    ax4.set_title('Max Drawdown')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # BEST STRATEGY
    best = max(results, key=lambda x: x['return'])
    print(f"\n🏆 BEST: {best['name']} - {best['return']:.2%} return")

def main():
    args = parse_args()
    print(f"🚀 CORE BOT: {args.days} days, {args.interval}")
    
    df = fetch_data(args.days, args.interval)
    results = run_all_strategy(df, args.show_plots)
    
    if results:
        plot_comparison(results)

if __name__ == '__main__':
    main()
