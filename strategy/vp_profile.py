import pandas as pd
import numpy as np
import talib  # pip install TA-Lib

def calculate_volume_profile(df, rows=24, value_area_pct=0.7):
    """Calculate Volume Profile: POC, VAH, VAL"""
    if len(df) < 50:
        return df
    
    # Price range for VP bins
    price_min = df['Low'].min()
    price_max = df['High'].max()
    bin_size = (price_max - price_min) / rows
    
    # Create price bins
    bins = np.arange(price_min, price_max + bin_size, bin_size)
    df['price_bin'] = pd.cut(df['Close'], bins=bins, labels=bins[:-1])
    
    # Volume at each price level
    vp = df.groupby('price_bin')['Volume'].sum()
    
    # POC = highest volume node
    poc_price = vp.idxmax()
    poc_volume = vp.max()
    
    # Value Area (70% of total volume)
    total_volume = vp.sum()
    target_volume = total_volume * value_area_pct
    sorted_vp = vp.sort_values(ascending=False)
    
    value_area_volume = 0
    value_area_prices = []
    for price, volume in sorted_vp.items():
        value_area_volume += volume
        value_area_prices.append(price)
        if value_area_volume >= target_volume:
            break
    
    vah = max(value_area_prices)
    val = min(value_area_prices)
    
    df['POC'] = poc_price
    df['VAH'] = vah
    df['VAL'] = val
    df['Bin_Size'] = bin_size
    
    print(f"📊 VP Profile: POC={poc_price:.1f}, VAH={vah:.1f}, VAL={val:.1f}")
    return df

def run_strategy(nifty):
    """Volume Profile Scalping: POC bounces + VAH/VAL rejections"""
    print("📈 VP Profile Scalper Starting...")
    
    # Fix MultiIndex
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [col[0] if isinstance(col, tuple) else col for col in nifty.columns]
    
    # Ensure Volume column exists
    if 'Volume' not in nifty.columns:
        nifty['Volume'] = 1000  # Dummy volume for backtest
    
    # Calculate rolling Volume Profile (session-based)
    nifty = calculate_volume_profile(nifty, rows=24, value_area_pct=0.7)
    
    # VP Signals
    # LONG: Price tests VAL + volume surge + close > VAL
    volume_surge = nifty['Volume'] > nifty['Volume'].rolling(20).mean() * 1.2
    nifty['Long_Signal'] = (
        (nifty['Low'] <= nifty['VAL']) &  # Tests VAL support
        (nifty['Close'] > nifty['VAL']) &  # Bounce confirmed
        volume_surge  # Volume confirmation
    )
    
    # SHORT: Price tests VAH + volume surge + close < VAH  
    nifty['Short_Signal'] = (
        (nifty['High'] >= nifty['VAH']) &  # Tests VAH resistance
        (nifty['Close'] < nifty['VAH']) &  # Rejection confirmed
        volume_surge  # Volume confirmation
    )
    
    # POC Mean Reversion (extra filter)
    poc_bounce_long = (
        (nifty['Low'] <= nifty['POC']) &
        (nifty['Close'] > nifty['POC']) &
        (nifty['Volume'] > nifty['Volume'].rolling(10).mean())
    )
    poc_bounce_short = (
        (nifty['High'] >= nifty['POC']) &
        (nifty['Close'] < nifty['POC']) &
        (nifty['Volume'] > nifty['Volume'].rolling(10).mean())
    )
    
    nifty['Long_Signal'] |= poc_bounce_long
    nifty['Short_Signal'] |= poc_bounce_short
    
    long_count = nifty['Long_Signal'].sum()
    short_count = nifty['Short_Signal'].sum()
    print(f"✅ VP Profile: {long_count} Long, {short_count} Short signals")
    
    return nifty
