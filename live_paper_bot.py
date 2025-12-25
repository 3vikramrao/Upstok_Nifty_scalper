#!/usr/bin/env python3
"""
🚀 PRODUCTION LIVE PAPER BOT - EMA CROSSOVER ✅
100% Upstox Nifty 1min + Live Trading
"""

import pandas as pd
import numpy as np
import time
import argparse
from pathlib import Path
import sys
import importlib.util
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

try:
    from env import UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET
    print("✅ env.py loaded")
except ImportError:
    print("❌ env.py missing")
    sys.exit(1)

TOKEN_FILE = "upstox_access_token.txt"

import upstox_client
from upstox_client import Configuration, ApiClient

STRATEGY_LIST = ["ema_crossover"]

class PaperTrader:
    def __init__(self):
        print("🔐 Upstox LIVE Trading Bot...")
        with open(TOKEN_FILE, 'r') as f:
            self.access_token = f.read().strip()
        
        config = Configuration()
        config.access_token = self.access_token
        self.api_client = ApiClient(config)
        
        self.paper_positions = {}
        self.trades_log = []
        self.last_nifty_price = 0
        self.current_strategy = None
    
    def get_nifty_data(self, days_back=3):
        nifty_key = "NSE_INDEX|Nifty 50"
        history_api = upstox_client.api.HistoryApi(self.api_client)
        
        candles_response = history_api.get_historical_candle_data(
            nifty_key, "1minute", 
            (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'), 
            "v2"
        )
        
        candle_data = candles_response.data.candles
        if len(candle_data[0]) == 7:
            df = pd.DataFrame(candle_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        else:
            df = pd.DataFrame(candle_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.set_index('timestamp', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(1000)
        
        self.last_nifty_price = float(df['Close'].iloc[-1])
        return df.dropna()
    
    def paper_order(self, action, symbol, qty, price):
        trade = {
            'action': action, 'symbol': symbol, 'qty': qty, 
            'price': price, 'nifty': self.last_nifty_price,
            'time': datetime.now(), 'strategy': self.current_strategy
        }
        self.trades_log.append(trade)
        
        if action == 'BUY':
            self.paper_positions[symbol] = {'qty': qty, 'avg_price': price}
            print(f"📈 🚀 LIVE BUY {symbol}: {qty}@{price:.0f}")
        else:
            pos = self.paper_positions[symbol]
            pnl = (price - pos['avg_price']) * pos['qty'] * 100  # Lot size
            trade['pnl'] = pnl
            print(f"💰 🚀 LIVE SELL {symbol}: P&L ₹{pnl:+.0f}")
            del self.paper_positions[symbol]
    
    def run_strategy_live(self, df, strategy_name):
        self.current_strategy = strategy_name
        
        strategy_path = Path(f"./strategy/{strategy_name}.py")
        if not strategy_path.exists():
            print(f"⚠️  Create: {strategy_path}")
            return
        
        spec = importlib.util.spec_from_file_location("strategy", strategy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        signals = module.run_strategy(df.copy())
        latest = signals.iloc[-1]
        
        # 🔥 LIVE TRADE EXECUTION
        if latest['Long_Signal'] and 'NIFTY_CE' not in self.paper_positions:
            self.paper_order('BUY', 'NIFTY_CE', 25, self.last_nifty_price)
        elif latest['Short_Signal'] and 'NIFTY_PE' not in self.paper_positions:
            self.paper_order('BUY', 'NIFTY_PE', 25, self.last_nifty_price)

def main():
    parser = argparse.ArgumentParser(description='🚀 LIVE Upstox Nifty Bot')
    parser.add_argument('--strategy', '-s', required=True, choices=STRATEGY_LIST)
    parser.add_argument('--duration', '-t', default=1800, type=int)  # 30min
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║           🚀 PRODUCTION LIVE TRADING BOT             ║
║             Strategy: {args.strategy:^14}             ║
║             Duration: {args.duration//60}m             ║
╚══════════════════════════════════════════════════════╝
    """)
    
    trader = PaperTrader()
    
    start_time = time.time()
    cycle = 0
    while time.time() - start_time < args.duration:
        try:
            df = trader.get_nifty_data()
            if len(df) > 50:
                trader.run_strategy_live(df, args.strategy)
            
            print(f"⏱️  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Nifty: {trader.last_nifty_price:6.0f} | "
                  f"💼 Pos: {len(trader.paper_positions)} | "
                  f"📊 Trades: {len(trader.trades_log)}")
            
            cycle += 1
            time.sleep(60)  # 1min cycles
            
        except KeyboardInterrupt:
            print("\n⏹️  STOPPED BY USER")
            break
    
    # FINAL RESULTS
    sells = [t for t in trader.trades_log if t['action'] == 'SELL']
    total_pnl = sum(t.get('pnl', 0) for t in sells)
    print(f"\n{'='*60}")
    print(f"💎 FINAL RESULTS")
    print(f"📊 Total Trades: {len(trader.trades_log)}")
    print(f"💰 Closed Trades: {len(sells)}")
    print(f"💵 Total P&L: ₹{total_pnl:+.0f}")
    print(f"📈 Open Positions: {len(trader.paper_positions)}")
    
    if trader.trades_log:
        log_df = pd.DataFrame(trader.trades_log)
        filename = f"LIVE_{args.strategy}_{datetime.now().strftime('%y%m%d_%H%M')}.csv"
        log_df.to_csv(filename, index=False)
        print(f"💾 LOG SAVED: {filename}")

if __name__ == "__main__":
    main()
