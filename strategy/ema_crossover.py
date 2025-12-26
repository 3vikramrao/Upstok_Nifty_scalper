"""EMA Crossover strategy with candle confirmation."""

import pandas as pd


def run_strategy(nifty):
    """EMA Crossover + confirmation - MultiIndex safe."""
    print("⚡ EMA Crossover: Processing...")

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [
            col[0] if isinstance(col, tuple) else col for col in nifty.columns
        ]
        print("✅ MultiIndex fixed")

    fast_ema = nifty["Close"].ewm(span=9).mean()
    slow_ema = nifty["Close"].ewm(span=21).mean()
    nifty["Fast_EMA"] = fast_ema
    nifty["Slow_EMA"] = slow_ema

    crossover_up = (fast_ema > slow_ema) & (
        fast_ema.shift(1) <= slow_ema.shift(1)
    )
    crossover_down = (fast_ema < slow_ema) & (
        fast_ema.shift(1) >= slow_ema.shift(1)
    )

    confirmation_long = nifty["Close"] > nifty["Open"]
    confirmation_short = nifty["Close"] < nifty["Open"]

    nifty["Long_Signal"] = crossover_up.shift(1) & confirmation_long
    nifty["Short_Signal"] = crossover_down.shift(1) & confirmation_short

    long_signals = nifty["Long_Signal"].sum()
    short_signals = nifty["Short_Signal"].sum()
    print(
        f"✅ EMA Cross: {long_signals} Long, "
        f"{short_signals} Short signals generated"
    )

    return nifty
