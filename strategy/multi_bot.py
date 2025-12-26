"""Multi-strategy ensemble for backtesting with dynamic loading."""

import pandas as pd

STRATEGIES = ["crt_hourly", "vp_profile", "ema_crossover"]


def load_strategy(strategy_name):
    """Load strategy modules dynamically."""
    import importlib
    import sys
    from pathlib import Path

    strategy_dir = Path("./strategies")
    if str(strategy_dir.parent) not in sys.path:
        sys.path.insert(0, str(strategy_dir.parent))

    module = importlib.import_module(f"strategies.{strategy_name}")
    return module.run_strategy


def ensemble_signal(df, strategies=STRATEGIES):
    """Vote-based ensemble for full backtest."""
    print(f"🤖 MultiBot: Ensemble of {len(strategies)} strategies")

    all_long_signals = pd.Series(False, index=df.index)
    all_short_signals = pd.Series(False, index=df.index)

    for name in strategies:
        try:
            strat_func = load_strategy(name)
            temp_df = strat_func(df.copy())
            all_long_signals |= temp_df['Long_Signal']
            all_short_signals |= temp_df['Short_Signal']
            print(
                f"  ✅ {name}: {temp_df['Long_Signal'].sum()}L "
                f"{temp_df['Short_Signal'].sum()}S"
            )
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    df['Long_Signal'] = all_long_signals
    df['Short_Signal'] = all_short_signals

    print(
        f"✅ MultiBot: {all_long_signals.sum()}L "
        f"{all_short_signals.sum()}S total"
    )

    return df


def run_strategy(nifty):
    """Backtest.py compatible entry point."""
    print("🤖 MultiBot Ensemble Strategy Starting...")

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in nifty.columns
        ]

    nifty = ensemble_signal(nifty, STRATEGIES)

    print("🎯 MultiBot Complete!")
    return nifty
