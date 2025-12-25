#!/usr/bin/env python3
"""
NISO – NIFTY Options Volume Profile Scalper with Paper Logging
Replaces Supertrend with Volume Profile POC/VAH/VAL signals.
"""

import os
import time
import csv
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta  # noqa: F401
import datetime as dt
import re  # noqa: F401


# ======================================================================
# CONFIG & CREDENTIALS
# ======================================================================
from env import (
    UPSTOX_CLIENT_KEY,
    UPSTOX_CLIENT_SECRET,
    UPSTOX_REDIRECT_URI,
)

PAPER = True
CLIENT_ID = UPSTOX_CLIENT_KEY
CLIENT_SECRET = UPSTOX_CLIENT_SECRET
REDIRECT_URI = UPSTOX_REDIRECT_URI

if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
    raise RuntimeError(
        "Set UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, "
        "UPSTOX_REDIRECT_URI in env.py"
    )

ACCESS_TOKEN_FILE = "upstox_access_token.txt"
BASE_REST = "https://api.upstox.com/v2"
BASE_HFT = "https://api-hft.upstox.com/v2"

QTY = 150
PRODUCT = "I"  # Intraday
TAG = "niso-volume-profile-bot"

# VP + Risk parameters
SL_PCT = 0.20
TG_PCT = 0.30
TRAIL_ATR_MULT = 1.5
VP_ROWS = 24
VA_PERCENTILE = 0.7

PAPER_LOG_DIR = Path("paper_logs")
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
UPSTOX_API_VERSION = "2.0"


def get_access_token():
    """Read Upstox access token from file."""
    if not os.path.exists(ACCESS_TOKEN_FILE):
        raise RuntimeError(
            "Run Upstox auth script to create upstox_access_token.txt"
        )
    with open(ACCESS_TOKEN_FILE, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        raise RuntimeError("upstox_access_token.txt is empty")
    return token


def api_headers():
    """REST API headers."""
    return {"Authorization": f"Bearer {get_access_token()}", "accept": "application/json"}


def hft_headers():
    """HFT API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def upstox_headers():
    """Upstox API headers with version."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Api-Version": UPSTOX_API_VERSION,
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def now_iso():
    """Current ISO timestamp."""
    return datetime.now().isoformat(timespec="seconds")


def ensure_log_files():
    """Create paper log files."""
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    lifetime_file = PAPER_LOG_DIR / "lifetime-vp_log.csv"
    today_file = PAPER_LOG_DIR / f"{datetime.now().date()}_vp-trades.csv"

    header = [
        "timestamp",
        "side",
        "symbol",
        "instrument_key",
        "strike",
        "qty",
        "entry_price",
        "sl_price",
        "tgt_price",
        "reason",
        "pnl",
    ]

    for f in (lifetime_file, today_file):
        if not f.exists():
            with f.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)

    return lifetime_file, today_file


def get_nifty_option_contracts(expiry_date=None, verbose=True):
    """Fetch NIFTY option contracts."""
    params = {"instrument_key": "NSE_INDEX|Nifty 50"}
    if expiry_date:
        params["expiry_date"] = expiry_date
    url = f"{BASE_REST}/option/contract"
    r = requests.get(url, headers=api_headers(), params=params)
    if verbose:
        print("Contracts status:", r.status_code)
        print("Contracts body (truncated):", r.text[:300], "...")
    r.raise_for_status()
    j = r.json()
    return j.get("data", [])


def instrument_info_from_key(instrument_key: str, contracts=None):
    """Get strike/type/expiry from instrument_key."""
    if contracts is None:
        try:
            contracts = get_nifty_option_contracts(verbose=False)
        except Exception:
            contracts = []

    for c in contracts:
        if c.get("instrument_key") == instrument_key:
            return {
                "strike": c.get("strike_price"),
                "type": c.get("instrument_type"),
                "expiry": c.get("expiry"),
                "trading_symbol": c.get("trading_symbol"),
            }
    return None


def log_trade_row(row):
    """Log trade to CSV files."""
    lifetime_file, today_file = ensure_log_files()
    instrument_key = row[3]
    info = instrument_info_from_key(instrument_key)

    if info and info.get("strike") is not None:
        strike_val = float(info["strike"])
        opt_type = info.get("type") or ""
        expiry = info.get("expiry") or ""
        strike_display = f"{strike_val:.1f} {opt_type} {expiry}".strip()
    else:
        strike_display = "N/A"

    log_row = row[:4] + [strike_display] + row[4:]

    for f in (lifetime_file, today_file):
        with f.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(log_row)


def get_nifty_ltp():
    """Get NIFTY 50 spot LTP."""
    req_key = "NSE_INDEX|Nifty 50"
    url = f"{BASE_REST}/market-quote/ltp"
    params = {"instrument_key": req_key}
    r = requests.get(url, headers=api_headers(), params=params)
    print("LTP status:", r.status_code)
    print("LTP body:", r.text[:200], "...")
    r.raise_for_status()
    j = r.json()
    data_block = j.get("data", {})
    if not data_block:
        raise RuntimeError(f"No LTP data: {j}")
    key = list(data_block.keys())[0]
    item = data_block[key]
    return item.get("last_price") or item.get("ltp")


def get_nifty_intraday_candles(minutes_back: int):
    """Get NIFTY 1min candles."""
    end_time = dt.datetime.now()
    start_time = end_time - dt.timedelta(minutes=minutes_back)

    url = (
        "https://api.upstox.com/v2/historical-candle/intraday/"
        f"{NIFTY_SPOT_INSTRUMENT}/1minute"
    )
    params = {
        "to_date": end_time.isoformat(timespec="seconds"),
        "from_date": start_time.isoformat(timespec="seconds"),
    }
    r = requests.get(url, headers=upstox_headers(), params=params)
    print("Hist status:", r.status_code)
    print("Hist body (truncated):", r.text[:200], "...")
    r.raise_for_status()
    data = r.json()
    rows = []
    for c in data["data"]["candles"]:
        rows.append(
            dict(
                time=dt.datetime.fromisoformat(c[0].replace("Z", "+00:00")),
                open=c[1],
                high=c[2],
                low=c[3],
                close=c[4],
                volume=c[5],
            )
        )
    return pd.DataFrame(rows)


def build_strike_maps(contracts):
    """Build CE/PE strike maps for nearest expiry."""
    df = pd.DataFrame(contracts)
    if df.empty:
        return {}, {}, None
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    nearest_expiry = df["expiry"].min()
    df = df[df["expiry"] == nearest_expiry]
    ce_df = df[df["instrument_type"] == "CE"]
    pe_df = df[df["instrument_type"] == "PE"]
    ce_map = dict(zip(ce_df["strike_price"], ce_df["instrument_key"]))
    pe_map = dict(zip(pe_df["strike_price"], pe_df["instrument_key"]))
    return ce_map, pe_map, nearest_expiry


def pick_near_atm_strikes(spot, strike_list, n=5):
    """Pick N nearest ATM strikes."""
    if not strike_list:
        return []
    sorted_strikes = sorted(strike_list, key=lambda k: abs(k - spot))
    return sorted_strikes[:n]


def calculate_volume_profile(df, rows=VP_ROWS):
    """Build Volume Profile with POC, VAH, VAL."""
    if len(df) < 20:
        return None

    price_min = df['low'].min()
    price_max = df['high'].max()
    bin_size = (price_max - price_min) / rows

    bins = np.arange(price_min, price_max + bin_size, bin_size)
    df['price_bin'] = np.digitize(df[['low', 'high']].mean(axis=1), bins)

    vp = df.groupby('price_bin')['volume'].sum().sort_index()
    if vp.empty:
        return None

    poc_bin = vp.idxmax()
    poc_price = bins[poc_bin]

    total_vol = vp.sum()
    target_vol = total_vol * VA_PERCENTILE
    sorted_vp = vp.sort_values(ascending=False)

    cumulative_vol = 0
    value_area_bins = [poc_bin]
    for bin_idx in sorted_vp.index:
        if bin_idx in value_area_bins:
            continue
        cumulative_vol += sorted_vp[bin_idx]
        value_area_bins.append(bin_idx)
        if cumulative_vol >= target_vol:
            break

    vah_price = bins[max(value_area_bins)]
    val_price = bins[min(value_area_bins)]

    return {
        'poc': poc_price,
        'vah': vah_price,
        'val': val_price,
        'total_vol': total_vol,
        'bin_size': bin_size
    }


def detect_vp_signal(df):
    """Detect VP signals: 1=CALL, -1=PUT, 0=no signal."""
    if df.empty or len(df) < 30
