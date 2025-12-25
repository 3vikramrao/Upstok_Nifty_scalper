#!/usr/bin/env python3
"""
NISO – NIFTY Options Scalper v1 with Paper Logging
"""

import os
import time
import csv
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import datetime as dt

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
        "Set UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, "
        "UPSTOX_REDIRECT_URI in env.py"
    )

ACCESS_TOKEN_FILE = "upstox_access_token.txt"
BASE_REST = "https://api.upstox.com/v2"
BASE_HFT = "https://api-hft.upstox.com/v2"

QTY = 150
PRODUCT = "I"  # Intraday
TAG = "niso-vwap-bot"

# SL/TGT parameters
SL_PCT = 0.20  # 20% stop loss
TG_PCT = 0.30  # 30% target
TRAIL_ATR_MULT = 1.5  # trailing SL ATR multiple

# Paper log config
PAPER_LOG_DIR = Path("paper_logs")
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
UPSTOX_API_VERSION = "2.0"


def upstox_headers():
    """Return Upstox API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Api-Version": UPSTOX_API_VERSION,
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def ensure_log_files():
    """Create paper_logs dir and CSVs with header if missing."""
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    lifetime_file = PAPER_LOG_DIR / "lifetime-vwap_log.csv"
    today_file = PAPER_LOG_DIR / f"{datetime.now().date()}_vwap-trades.csv"

    header = [
        "timestamp",
        "side",
        "symbol",
        "instrument_key",
        "strike",  # e.g. '26200 CE'
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
    """Parse strike/type/expiry from instrument_key."""
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


def now_iso():
    """Current timestamp in ISO 8601 with seconds."""
    return datetime.now().isoformat(timespec="seconds")


def get_access_token():
    """Read access token from file."""
    if not os.path.exists(ACCESS_TOKEN_FILE):
        raise RuntimeError(
            "Run your Upstox auth script once to create "
            "upstox_access_token.txt"
        )
    with open(ACCESS_TOKEN_FILE, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        raise RuntimeError("upstox_access_token.txt is empty")
    return token


def api_headers():
    """Standard REST API headers."""
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


def add_vwap(df):
    """Add running VWAP to NIFTY candles."""
    if df.empty:
        return df
    tpv = (df["high"] + df["low"] + df["close"]) / 3.0 * df["volume"]
    df["vwap"] = tpv.cumsum() / df["volume"].cumsum()
    return df


def find_vwap_cross_trigger(df):
    """Find VWAP cross direction and trigger price."""
    if df.empty or len(df) < 2 or "vwap" not in df.columns:
        return 0, None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_close, prev_vwap = prev["close"], prev["vwap"]
    close, vwap = curr["close"], curr["vwap"]

    if prev_close <= prev_vwap and close > vwap:
        print(f"VWAP BULL CROSS: close {close:.2f} > vwap {vwap:.2f}")
        return 1, curr["high"]

    if prev_close >= prev_vwap and close < vwap:
        print(f"VWAP BEAR CROSS: close {close:.2f} < vwap {vwap:.2f}")
        return -1, curr["low"]

    return 0, None


def detect_trend_vwap(df):
    """Detect trend from VWAP cross."""
    df = add_vwap(df)
    trend, trigger_price = find_vwap_cross_trigger(df)
    return trend, trigger_price


def should_enter_from_trigger(opt_type, trigger_price, current_spot):
    """Check if spot broke trigger level."""
    if trigger_price is None:
        return False
    if opt_type == "CE":
        return current_spot >= trigger_price
    if opt_type == "PE":
        return current_spot <= trigger_price
    return False


def pick_instrument_for_trend(trend, spot, ce_map, pe_map):
    """Pick ATM CE/PE based on trend."""
    if trend == 0:
        return None, None, None
    if trend == 1:
        strikes = pick_near_atm_strikes(spot, list(ce_map.keys()), n=5)
        if not strikes:
            return None, None, None
        strike = strikes[0]
        return ce_map[strike], "CE", strike
    else:
        strikes = pick_near_atm_strikes(spot, list(pe_map.keys()), n=5)
        if not strikes:
            return None, None, None
        strike = strikes[0]
        return pe_map[strike], "PE", strike


def info_from_instrument_key(contracts, instrument_key: str):
    """Get strike/type/expiry from instrument_key."""
    for c in contracts:
        if c.get("instrument_key") == instrument_key:
            return {
                "strike": c.get("strike_price"),
                "type": c.get("instrument_type"),
                "expiry": c.get("expiry"),
                "trading_symbol": c.get("trading_symbol"),
            }
    return None


def get_option_ltp(instrument_key):
    """Get option LTP."""
    url = f"{BASE_REST}/market-quote/ltp"
    params = {"instrument_key": instrument_key}
    r = requests.get(url, headers=api_headers(), params=params)
    r.raise_for_status()
    j = r.json()
    data_block = j.get("data", {})
    key = list(data_block.keys())[0]
    item = data_block[key]
    return item.get("last_price") or item.get("ltp")


def place_hft_market_order(instrument_token, quantity, side):
    """Place HFT market order."""
    url = f"{BASE_HFT}/order/place"
    payload = {
        "quantity": quantity,
        "product": PRODUCT,
        "validity": "DAY",
        "price": 0,
        "tag": TAG,
        "instrument_token": instrument_token,
        "order_type": "MARKET",
        "transaction_type": side.upper(),
        "disclosed_quantity": 0,
        "trigger_price": 0,
        "is_amo": False,
    }
    if PAPER:
        print("[PAPER] MARKET:", payload)
        return f"PAPER-MKT-{side}-{int(time.time())}"
    r = requests.post(url, headers=hft_headers(), json=payload)
    print("Order status:", r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json()["data"]["order_id"]


def place_hft_limit_order(instrument_token, quantity, side, price):
    """Place HFT limit order."""
    payload = {
        "quantity": quantity,
        "product": PRODUCT,
        "validity": "DAY",
        "price": float(price),
        "tag": TAG,
        "instrument_token": instrument_token,
        "order_type": "LIMIT",
        "transaction_type": side.upper(),
        "disclosed_quantity": 0,
        "trigger_price": 0,
        "is_amo": False
