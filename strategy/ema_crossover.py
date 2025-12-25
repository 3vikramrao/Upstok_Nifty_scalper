import pandas as pd  # ← ADD THIS
import numpy as np

def run_strategy(nifty):
    """EMA Crossover + Confirmation strategy - MultiIndex safe"""
    print("⚡ EMA Crossover: Processing...")
    
    # Fix MultiIndex columns if present (yfinance issue)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [col[0] if isinstance(col, tuple) else col for col in nifty.columns]
        print("✅ MultiIndex fixed")
    
    # Calculate EMAs
    fast_ema = nifty['Close'].ewm(span=9).mean()
    slow_ema = nifty['Close'].ewm(span=21).mean()
    nifty['Fast_EMA'] = fast_ema
    nifty['Slow_EMA'] = slow_ema
    
    # Crossover detection
    crossover_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    crossover_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
    
    # Confirmation: Next candle direction
    confirmation_long = (nifty['Close'] > nifty['Open'])  # Green candle
    confirmation_short = (nifty['Close'] < nifty['Open'])  # Red candle
    
    # Entry on confirmation candle AFTER crossover
    nifty['Long_Signal'] = crossover_up.shift(1) & confirmation_long
    nifty['Short_Signal'] = crossover_down.shift(1) & confirmation_short
    
    long_signals = nifty['Long_Signal'].sum()
    short_signals = nifty['Short_Signal'].sum()
    print(f"✅ EMA Cross: {long_signals} Long, {short_signals} Short signals generated")
    
    return nifty
