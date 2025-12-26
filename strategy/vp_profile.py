"""Volume Profile trading strategy with POC, VAH, VAL levels."""

import numpy as np
import pandas as pd
import talib  # noqa: F401


def calculate_volume_profile(df, rows=24, value_area_pct=0.7):
    """Calculate Volume Profile: POC, VAH, VAL."""
    if len(df) < 50:
        return df

    price_min = df["Low"].min()
    price_max = df["High"].max()
    bin_size = (price_max - price_min) / rows

    bins = np.arange(price_min, price_max + bin_size, bin_size)
    df["price_bin"] = pd.cut(df["Close"], bins=bins, labels=bins[:-1])

    vp = df.groupby("price_bin")["Volume"].sum()

    poc_price = vp.idxmax()
    _ = vp.max()  # Unused POC volume

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

    df["POC"] = poc_price
    df["VAH"] = vah
    df["VAL"] = val
    df["Bin_Size"] = bin_size

    print(f"📊 VP Profile: POC={poc_price:.1f}, VAH={vah:.1f}, VAL={val:.1f}")
    return df


def run_strategy(nifty):
    """Volume Profile Scalping: POC bounces + VAH/VAL rejections."""
    print("📈 VP Profile Scalper Starting...")

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [
            col[0] if isinstance(col, tuple) else col for col in nifty.columns
        ]

    if "Volume" not in nifty.columns:
        nifty["Volume"] = 1000

    nifty = calculate_volume_profile(nifty, rows=24, value_area_pct=0.7)

    volume_surge = nifty["Volume"] > nifty["Volume"].rolling(20).mean() * 1.2

    nifty["Long_Signal"] = (
        (nifty["Low"] <= nifty["VAL"]) & (nifty["Close"] > nifty["VAL"]) & volume_surge
    )

    nifty["Short_Signal"] = (
        (nifty["High"] >= nifty["VAH"]) & (nifty["Close"] < nifty["VAH"]) & volume_surge
    )

    poc_bounce_long = (
        (nifty["Low"] <= nifty["POC"])
        & (nifty["Close"] > nifty["POC"])
        & (nifty["Volume"] > nifty["Volume"].rolling(10).mean())
    )

    poc_bounce_short = (
        (nifty["High"] >= nifty["POC"])
        & (nifty["Close"] < nifty["POC"])
        & (nifty["Volume"] > nifty["Volume"].rolling(10).mean())
    )

    nifty["Long_Signal"] |= poc_bounce_long
    nifty["Short_Signal"] |= poc_bounce_short

    long_count = nifty["Long_Signal"].sum()
    short_count = nifty["Short_Signal"].sum()
    print(f"✅ VP Profile: {long_count} Long, {short_count} Short signals")

    return nifty
