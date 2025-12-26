#!/usr/bin/env python3
"""
NISO – NIFTY Options Scalper v1 with Paper Logging

Requirements:
- env.py with: UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, UPSTOX_REDIRECT_URI
- upstox_access_token.txt created by your auth script
- Folder paper_logs/ will be auto-created with: lifetime-rsi-div-dc_log.csv,
  YYYY-MM-DD-rsi-div-dc-trades.csv
"""

import csv
import datetime as dt
import os
import re  # noqa: F401
from datetime import datetime, timedelta  # noqa: F401
from pathlib import Path

import pandas as pd
import requests

# ======================================================================
# CONFIG & CREDENTIALS
# ======================================================================
from env import (
    UPSTOX_CLIENT_KEY,
    UPSTOX_CLIENT_SECRET,
    UPSTOX_REDIRECT_URI,
)

PAPER = True  # False = LIVE, True = paper-trade with logging
CLIENT_ID = UPSTOX_CLIENT_KEY
CLIENT_SECRET = UPSTOX_CLIENT_SECRET
REDIRECT_URI = UPSTOX_REDIRECT_URI

if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
    raise RuntimeError(
        "Set UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, " "UPSTOX_REDIRECT_URI in env.py"
    )

ACCESS_TOKEN_FILE = "upstox_access_token.txt"
BASE_REST = "https://api.upstox.com/v2"
BASE_HFT = "https://api-hft.upstox.com/v2"

QTY = 150
PRODUCT = "I"  # Intraday
TAG = "niso-rsi-div-dc"

SL_PCT = 0.20
TG_PCT = 0.30
TRAIL_ATR_MULT = 1.5

PAPER_LOG_DIR = Path("paper_logs")
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
UPSTOX_API_VERSION = "2.0"


def get_access_token():
    """Read access token from file."""
    if not os.path.exists(ACCESS_TOKEN_FILE):
        raise RuntimeError(
            "Run your Upstox auth script once to create " "upstox_access_token.txt"
        )
    with open(ACCESS_TOKEN_FILE, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        raise RuntimeError("upstox_access_token.txt is empty")
    return token


def upstox_headers():
    """Return Upstox API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Api-Version": UPSTOX_API_VERSION,
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def api_headers():
    """REST API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
    }


def hft_headers():
    """HFT API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def now_iso():
    """Current timestamp in ISO 8601 with seconds."""
    return datetime.now().isoformat(timespec="seconds")


def ensure_log_files():
    """Create paper_logs dir and CSVs with header if missing."""
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    lifetime_file = PAPER_LOG_DIR / "lifetime-rsi-div-dc_log.csv"
    today_file = PAPER_LOG_DIR / f"{datetime.now().date()}_rsi-div-dc-trades.csv"

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


def instrument_info_from_key(instrument_key: str, contracts=None):
    """Return strike/type/expiry/trading_symbol for instrument_key."""
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
    """Append row to lifetime + daily CSV."""
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
    """NIFTY index LTP via LTP endpoint."""
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
        raise RuntimeError(f"No data in LTP response: {j}")
    key = list(data_block.keys())[0]
    item = data_block[key]
    return item.get("last_price") or item.get("ltp")


def get_nifty_intraday_candles(minutes_back: int):
    """Get NIFTY 1min candles."""
    end_time = dt.datetime.now()
    start_time = end_time - dt.timedelta(minutes=minutes_back)

    url = (
        f"https://api.upstox.com/v2/historical-candle/intraday/"
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


def get_nifty_option_contracts(expiry_date=None, verbose=True):
    """NIFTY option contracts via option/contract endpoint."""
    params = {"instrument_key": "NSE_INDEX|Nifty 50"}
    if expiry_date:
        params["expiry_date"] = expiry_date
