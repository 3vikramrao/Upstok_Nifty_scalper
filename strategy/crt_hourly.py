import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, ir_streak_min=2, use_volume_filter=True):
        self.ir_streak_min = ir_streak_min
        self.use_volume_filter = use_volume_filter
        self.name = "CRT Hourly"
    
    def detect_signal(self, df):
        if len(df) < 30: 
            return 0
        
        df['time'] = pd.to_datetime(df['time'])
        df_h = df.set_index('time').resample('1H').agg({
            'open':'first', 'high':'max', 'low':'min', 
            'close':'last', 'volume':'sum'
        }).dropna().reset_index()
        
        if len(df_h) < 3: 
            return 0
        
        df_h['ir'] = (df_h['high'] < df_h['high'].shift(1)) & \
                     (df_h['low'] > df_h['low'].shift(1))
        df_h['or'] = (df_h['high'] > df_h['high'].shift(1)) & \
                     (df_h['low'] < df_h['low'].shift(1))
        df_h['ir_streak'] = df_h['ir'].rolling(3).sum()
        
        ir_streak = df_h['ir_streak'].iloc[-1]
        current_or = df_h['or'].iloc[-1]
        
        if ir_streak >= self.ir_streak_min and current_or:
            if df_h['close'].iloc[-1] > df_h['high'].shift(1).iloc[-1]:
                return 1
            elif df_h['close'].iloc[-1] < df_h['low'].shift(1).iloc[-1]:
                return -1
        
        return 0

def run_strategy(nifty):
    """✅ PERFECTLY COMPATIBLE with backtest.py - returns original column names"""
    print("🔧 CRT Strategy: Processing...")
    
    # Store original columns for restoration
    original_columns = nifty.columns.tolist()
    
    # Fix MultiIndex if present
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [col[0] if isinstance(col, tuple) else col for col in nifty.columns]
    
    # Create lowercase working copy
    df_crt = nifty.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    }).copy()
    
    # Add time column from index
    df_crt['time'] = pd.to_datetime(df_crt.index)
    
    # Generate CRT signals
    crt = Strategy(ir_streak_min=2)
    signals = []
    for i in range(30, len(df_crt)):
        signal = crt.detect_signal(df_crt.iloc[:i+1])
        signals.append(signal)
    
    # Pad signals
    signals = [0] * 30 + signals[:len(df_crt)-30]
    
    # Add signals to ORIGINAL dataframe (Title Case columns)
    nifty['Long_Signal'] = [s == 1 for s in signals]
    nifty['Short_Signal'] = [s == -1 for s in signals]
    
    print(f"✅ CRT: {sum(nifty['Long_Signal'])} Long, {sum(nifty['Short_Signal'])} Short signals")
    return nifty  # ← Returns ORIGINAL column names (Open, High, Low, Close)

CRTStrategy = Strategy
