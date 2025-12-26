#!/usr/bin/env python3
"""
NISO – NIFTY Options Scalper v2 with EMA 9/15 + RSI 9 + SuperTrend + Parabolic SAR + VWAP + ADX + MACD
FIXED DASHBOARD - Live market ready
"""

import csv
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import talib  # pip install TA-Lib

# ======================================================================
# CONFIG & CREDENTIALS
# ======================================================================
from env import UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, UPSTOX_REDIRECT_URI

PAPER = True  # False = LIVE, True = paper-trade with logging
CLIENT_ID = UPSTOX_CLIENT_KEY
CLIENT_SECRET = UPSTOX_CLIENT_SECRET
REDIRECT_URI = UPSTOX_REDIRECT_URI

if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
    raise RuntimeError(
        "Set UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, UPSTOX_REDIRECT_URI in env.py"
    )

ACCESS_TOKEN_FILE = "upstox_access_token.txt"
BASE_REST = "https://api.upstox.com/v2"
BASE_HFT = "https://api-hft.upstox.com/v2"

QTY = 150
PRODUCT = "I"
TAG = "niso-multi-beast-bot"

# Enhanced SL/TGT parameters
SL_PCT = 0.20
TG_PCT = 0.30
TRAIL_ATR_MULT = 1.5

PAPER_LOG_DIR = Path("paper_logs")
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
UPSTOX_API_VERSION = "2.0"

# Indicator Parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 2.0

# Silence FutureWarning
warnings.filterwarnings("ignore")


# ======================================================================
# TECHNICAL INDICATORS (CLEANED & FIXED)
# ======================================================================
def ema(series, n):
    """Calculate EMA."""
    return series.ewm(span=n, adjust=False).mean()


def calculate_vwap_signals(df):
    """Calculate VWAP and signals."""
    if len(df) < 2:
        df["vwap"] = np.nan
        df["vwap_bull"] = False
        df["vwap_bear"] = False
        return df

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df[
        "volume"
    ].cumsum()
    df["price_above_vwap"] = df["close"] > df["vwap"]
    df["price_below_vwap"] = df["close"] < df["vwap"]
    df["vwap_rising"] = df["vwap"] > df["vwap"].shift(1)
    df["vwap_bull"] = df["price_above_vwap"] & df["vwap_rising"]
    df["vwap_bear"] = df["price_below_vwap"] & (~df["vwap_rising"])
    return df


def add_supertrend(df, atr_period=10, multiplier=3.0):
    """SuperTrend implementation - CLEAN SINGLE VERSION."""
    if len(df) < atr_period + 1:
        df["supertrend"] = np.nan
        return df

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    atr = talib.ATR(high, low, close, timeperiod=atr_period)
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = np.full(len(df), np.nan)
    in_uptrend = [True]
    supertrend[0] = lowerband[0]

    for i in range(1, len(df)):
        if close[i] > upperband[i - 1]:
            in_uptrend.append(True)
        elif close[i] < lowerband[i - 1]:
            in_uptrend.append(False)
        else:
            in_uptrend.append(in_uptrend[-1])

        if in_uptrend[-1]:
            supertrend[i] = lowerband[i]
            if supertrend[i] > supertrend[i - 1] and not np.isnan(
                supertrend[i - 1]
            ):
                supertrend[i] = supertrend[i - 1]
        else:
            supertrend[i] = upperband[i]
            if supertrend[i] < supertrend[i - 1] and not np.isnan(
                supertrend[i - 1]
            ):
                supertrend[i] = supertrend[i - 1]

    df["supertrend"] = supertrend
    return df


def calculate_indicators(df):
    """7-Indicator Powerhouse - ROBUST VERSION."""
    print(f"🎯 Calculating indicators on {len(df)} candles...")

    if len(df) < 30:
        print("⚠️ Insufficient data (<30 candles)")
        return df

    try:
        # 1. EMAs (always safe)
        df["ema9"] = ema(df["close"], 9)
        df["ema15"] = ema(df["close"], 15)
        print("✅ EMAs calculated")

        # 2. RSI9 (safe)
        rsi = talib.RSI(df["close"].values, timeperiod=9)
        df["rsi9"] = pd.Series(rsi, index=df.index)
        print("✅ RSI9 calculated")

        # 3. SuperTrend
        df = add_supertrend(df)
        print("✅ SuperTrend calculated")

        # 4. Parabolic SAR
        high_arr = df["high"].fillna(method="ffill").values
        low_arr = df["low"].fillna(method="ffill").values
        sar = talib.SAR(high_arr, low_arr)
        df["parabolic_sar"] = pd.Series(sar, index=df.index)
        print("✅ Parabolic SAR calculated")

        # 5. VWAP
        df = calculate_vwap_signals(df)
        print("✅ VWAP calculated")

        # 6. ADX (14)
        adx = talib.ADX(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            timeperiod=14,
        )
        df["adx"] = pd.Series(adx, index=df.index)
        df["adx_strong"] = df["adx"] > 25
        print(f"✅ ADX calculated: latest={df['adx'].iloc[-1]:.1f}")

        # 7. MACD (12,26,9)
        macd, macd_signal, macd_hist = talib.MACD(df["close"].values)
        df["macd"] = pd.Series(macd, index=df.index)
        df["macd_signal"] = pd.Series(macd_signal, index=df.index)
        df["macd_hist"] = pd.Series(macd_hist, index=df.index)
        print(f"✅ MACD calculated: latest={df['macd'].iloc[-1]:.3f}")

        print("🎉 ALL 7 INDICATORS SUCCESS!")
        return df

    except Exception as e:
        print(f"❌ Indicator calc FAILED: {e}")
        print(f"Data shape: {df.shape}, NaNs: {df.isna().sum().sum()}")
        return df


def detect_multi_trend(df):
    """7-Indicator Consensus (needs 4/7 for entry)."""
    if df.empty or len(df) < 30:
        return 0

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signals = []

    # 1. EMA Crossover
    ema_bull = (
        latest["ema9"] > latest["ema15"] and prev["ema9"] <= prev["ema15"]
    )
    if ema_bull:
        signals.append(1)
    elif latest["ema9"] < latest["ema15"] and prev["ema9"] >= prev["ema15"]:
        signals.append(-1)

    # 2. RSI9 Momentum
    if latest["rsi9"] > 50 and latest["rsi9"] > prev["rsi9"]:
        signals.append(1)
    elif latest["rsi9"] < 50 and latest["rsi9"] < prev["rsi9"]:
        signals.append(-1)

    # 3. SuperTrend
    if (
        latest["close"] > latest["supertrend"]
        and prev["close"] <= prev["supertrend"]
    ):
        signals.append(1)
    elif (
        latest["close"] < latest["supertrend"]
        and prev["close"] >= prev["supertrend"]
    ):
        signals.append(-1)

    # 4. Parabolic SAR
    if (
        latest["close"] > latest["parabolic_sar"]
        and prev["close"] <= prev["parabolic_sar"]
    ):
        signals.append(1)
    elif (
        latest["close"] < latest["parabolic_sar"]
        and prev["close"] >= prev["parabolic_sar"]
    ):
        signals.append(-1)

    # 5. VWAP
    if latest["vwap_bull"]:
        signals.append(1)
    elif latest["vwap_bear"]:
        signals.append(-1)

    # 6. ADX (Strong trend filter)
    if latest["adx_strong"] and latest["close"] > latest["ema9"]:
        signals.append(1)
    elif latest["adx_strong"] and latest["close"] < latest["ema9"]:
        signals.append(-1)

    # 7. MACD (crossover + histogram)
    macd_bull = (
        latest["macd"] > latest["macd_signal"]
        and prev["macd"] <= prev["macd_signal"]
        and latest["macd_hist"] > 0
    )
    macd_bear = (
        latest["macd"] < latest["macd_signal"]
        and prev["macd"] >= prev["macd_signal"]
        and latest["macd_hist"] < 0
    )
    if macd_bull:
        signals.append(1)
    elif macd_bear:
        signals.append(-1)

    # Consensus: 4/7 required
    bull_signals = sum(1 for s in signals if s == 1)
    bear_signals = sum(1 for s in signals if s == -1)

    print(f"🔥 7-SIGNAL POWER: Bull={bull_signals}/7, Bear={bear_signals}/7")
    print(
        f"ADX: {latest['adx']:.1f} {'🟢STRONG' if latest['adx_strong'] else '⚪WEAK'}"
    )
    print(
        f"MACD: {latest['macd']:.3f}/{latest['macd_signal']:.3f} | Hist: {latest['macd_hist']:.3f}"
    )

    if bull_signals >= 4:
        return 1
    elif bear_signals >= 4:
        return -1
    return 0


# ======================================================================
# UPSTOX HELPERS
# ======================================================================
def get_access_token():
    """Get access token from file."""
    if not os.path.exists(ACCESS_TOKEN_FILE):
        raise RuntimeError(
            "Run your Upstox auth script once to create upstox_access_token.txt"
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
    """Return REST API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
    }


def hft_headers():
    """Return HFT API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


# ======================================================================
# LOGGING HELPERS
# ======================================================================
def ensure_log_files():
    """Create paper_logs dir and CSVs with header if missing."""
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    lifetime_file = PAPER_LOG_DIR / "lifetime-multi-beast_log.csv"
    today_file = (
        PAPER_LOG_DIR / f"{datetime.now().date()}_multi-beast-trades.csv"
    )

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


def now_iso():
    """Current timestamp in ISO 8601 with seconds."""
    return datetime.now().isoformat(timespec="seconds")


def instrument_info_from_key(instrument_key: str, contracts=None):
    """
    Return strike/type/expiry/trading_symbol for a numeric Upstox instrument_key
    like 'NSE_FO|57013' by looking into option contracts list.
    """
    if contracts is None:
        try:
            contracts = get_nifty_option_contracts(verbose=False)
        except Exception:
            contracts = []

    for c in contracts:
        if c.get("instrument_key") == instrument_key:
            return {
                "strike": c.get("strike_price"),
                "type": c.get("instrument_type"),  # CE or PE
                "expiry": c.get("expiry"),
                "trading_symbol": c.get("trading_symbol"),
            }
    return None


def log_trade_row(row):
    """
    Append a row to lifetime + daily CSV.

    Expected input row (no strike included):
    [timestamp, side, symbol, instrument_key, qty,
     entry_price, sl_price, tgt_price, reason, pnl]
    """
    lifetime_file, today_file = ensure_log_files()

    instrument_key = row[3]
    info = instrument_info_from_key(instrument_key)

    if info and info.get("strike") is not None:
        strike_val = float(info["strike"])
        opt_type = info.get("type") or ""
        expiry = info.get("expiry") or ""
        # Final display: 26200.0 CE 2025-12-23
        strike_display = f"{strike_val:.1f} {opt_type} {expiry}".strip()
    else:
        strike_display = "N/A"

    # Insert strike just after instrument_key
    log_row = row[:4] + [strike_display] + row[4:]

    for f in (lifetime_file, today_file):
        with f.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(log_row)


# ======================================================================
# FIXED NIFTY DATA FUNCTIONS
# ======================================================================
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
    """FIXED: Correct Upstox v2 intraday endpoint."""
    try:
        # Upstox intraday gets current day data only
        url = (
            f"https://api.upstox.com/v2/historical-candle/intraday/"
            f"{NIFTY_SPOT_INSTRUMENT}/1minute"
        )
        r = requests.get(url, headers=upstox_headers())

        if r.status_code != 200:
            print(f"❌ Candle fetch failed: {r.status_code}")
            return pd.DataFrame()

        data = r.json()
        if "data" not in data or not data["data"].get("candles"):
            print("❌ No candle data in response")
            return pd.DataFrame()

        rows = []
        for c in data["data"]["candles"]:
            # Handle Upstox timestamp format +05:30
            timestamp = c[0].replace("Z", "+00:00")
            rows.append(
                {
                    "time": datetime.fromisoformat(timestamp),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                }
            )
        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        print(f"✅ Got {len(df)} candles")
        return df
    except Exception as e:
        print(f"❌ Candle fetch error: {e}")
        return pd.DataFrame()


# ======================================================================
# OPTION CONTRACTS & STRIKE SELECTION
# ======================================================================
def get_nifty_option_contracts(expiry_date=None, verbose=True):
    """NIFTY option contracts via option/contract endpoint."""
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
    """Fixed: Handle actual Upstox option contract response format."""
    if not contracts:
        print("No option contracts returned")
        return {}, {}, None

    print(f"Found {len(contracts)} option contracts")
    df = pd.DataFrame(contracts)

    # Debug: Print available columns
    print("Available columns:", df.columns.tolist())

    # Check required columns exist
    required_cols = [
        "expiry",
        "instrument_key",
        "instrument_type",
        "strike_price",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        print("Sample contract:", contracts[0] if contracts else "None")
        return {}, {}, None

    # Parse expiry dates
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    nearest_expiry = df["expiry"].min()
    print(f"Nearest expiry: {nearest_expiry}")

    # Filter nearest expiry
    df = df[df["expiry"] == nearest_expiry]
    print(f"Contracts for nearest expiry: {len(df)}")

    # Split CE/PE using correct field name
    ce_df = df[df["instrument_type"] == "CE"]
    pe_df = df[df["instrument_type"] == "PE"]

    print(f"CE contracts: {len(ce_df)}, PE contracts: {len(pe_df)}")

    # Build strike maps
    ce_map = dict(
        zip(ce_df["strike_price"].astype(float), ce_df["instrument_key"])
    )
    pe_map = dict(
        zip(pe_df["strike_price"].astype(float), pe_df["instrument_key"])
    )

    return ce_map, pe_map, nearest_expiry


def pick_near_atm_strikes(spot, strike_list, n=5):
    """Pick n nearest ATM strikes."""
    if not strike_list:
        return []
    sorted_strikes = sorted(strike_list, key=lambda k: abs(k - spot))
    return sorted_strikes[:n]


def pick_instrument_for_trend(trend, spot, ce_map, pe_map):
    """Pick one ATM CE or PE instrument_key based on trend."""
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
    """Get instrument info from key."""
    for c in contracts:
        if c.get("instrument_key") == instrument_key:
            return {
                "strike": c.get("strike_price"),
                "type": c.get("instrument_type"),
                "expiry": c.get("expiry"),
                "trading_symbol": c.get("trading_symbol"),
            }
    return None


# ======================================================================
# ORDER FUNCTIONS
# ======================================================================
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
        "is_amo": False,
    }
    if PAPER:
        print("[PAPER] LIMIT:", payload)
        return f"PAPER-LMT-{side}-{int(time.time())}"
    r = requests.post(
        f"{BASE_HFT}/order/place", headers=hft_headers(), json=payload
    )
    print("LIMIT status:", r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json()["data"]["order_id"]


def place_hft_sl_order(instrument_token, quantity, side, price, trigger_price):
    """Place HFT SL order."""
    payload = {
        "quantity": quantity,
        "product": PRODUCT,
        "validity": "DAY",
        "price": float(price),
        "tag": TAG,
        "instrument_token": instrument_token,
        "order_type": "SL",
        "transaction_type": side.upper(),
        "disclosed_quantity": 0,
        "trigger_price": float(trigger_price),
        "is_amo": False,
    }
    if PAPER:
        print("[PAPER] SL:", payload)
        return f"PAPER-SL-{side}-{int(time.time())}"
    r = requests.post(
        f"{BASE_HFT}/order/place", headers=hft_headers(), json=payload
    )
    print("SL status:", r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json()["data"]["order_id"]


# ======================================================================
# POSITION MANAGEMENT
# ======================================================================
open_positions = {}
daily_pnl = 0.0


class Position:
    """Position tracking class."""

    def __init__(self, entry_oid, instrument_key, qty, entry_price):
        self.entry_oid = entry_oid
        self.instrument_key = instrument_key
        self.qty = qty
        self.entry_price = entry_price

        self.sl_price = entry_price * (1 - SL_PCT)
        self.tgt_price = entry_price * (1 + TG_PCT)
        self.trail_price = self.sl_price

        # Place SL + TGT orders
        self.sl_oid = place_hft_sl_order(
            instrument_key, qty, "SELL", self.sl_price, self.sl_price * 1.02
        )
        self.tgt_oid = place_hft_limit_order(
            instrument_key, qty, "SELL", self.tgt_price
        )

        print(
            f"Position: entry={entry_price:.2f}, "
            f"SL={self.sl_price:.2f}, TGT={self.tgt_price:.2f}"
        )

        # Log entry (paper or live) into CSVs (strike/type added in log_trade_row)
        row = [
            now_iso(),
            "BUY",
            TAG,
            self.instrument_key,
            self.qty,
            round(self.entry_price, 2),
            round(self.sl_price, 2),
            round(self.tgt_price, 2),
            "ENTRY",
            0.0,  # pnl
        ]
        log_trade_row(row)


def update_trailing_sl(pos):
    """Update trailing stop loss."""
    ltp = get_option_ltp(pos.instrument_key)
    atr_proxy = ltp * 0.10
    new_trail = ltp - TRAIL_ATR_MULT * atr_proxy
    if new_trail > pos.trail_price:
        pos.trail_price = new_trail
        print(f"Trail updated to {pos.trail_price:.2f} (LTP={ltp:.2f})")


def check_exit_conditions(pos):
    """Check if position should exit."""
    ltp = get_option_ltp(pos.instrument_key)
    if ltp <= pos.trail_price:
        print(f"TRAIL HIT: {ltp:.2f} <= {pos.trail_price:.2f}")
        return "TRAIL"
    elif ltp >= pos.tgt_price:
        print(f"TARGET HIT: {ltp:.2f} >= {pos.tgt_price:.2f}")
        return "TARGET"
    return None


def monitor_positions():
    """Monitor all open positions."""
    global open_positions, daily_pnl
    for entry_oid, pos in list(open_positions.items()):
        update_trailing_sl(pos)
        reason = check_exit_conditions(pos)
        if reason:
            print(f"EXIT {entry_oid}: {reason}")

            # realized P&L at exit
            exit_ltp = get_option_ltp(pos.instrument_key)
            pnl = (exit_ltp - pos.entry_price) * pos.qty
            daily_pnl += pnl

            if reason == "TRAIL":
                r = "EXIT_TRAIL"
            elif reason == "TARGET":
                r = "EXIT_TARGET"
            else:
                r = "EXIT"

            row = [
                now_iso(),
                "SELL",
                TAG,
                pos.instrument_key,
                pos.qty,
                None,
                round(pos.sl_price, 2),
                round(pos.tgt_price, 2),
                r,
                round(pnl, 2),  # pnl for this round-trip
            ]
            log_trade_row(row)

            del open_positions[entry_oid]
    print("Monitor done; open positions:", len(open_positions))


def dashboard():
    """Display clean dashboard."""
    os.system("cls" if os.name == "nt" else "clear")
    print("🎯 HA-V2 TREND BREAKOUT BEAST –", "PAPER" if PAPER else "LIVE")
    print(
        f"Time: {datetime.now().strftime('%H:%M:%S')} | "
        f"Open: {len(open_positions)}"
    )

    if open_positions:
        total_mtm = 0
        contracts = get_nifty_option_contracts(verbose=False)
        for oid, pos in open_positions.items():
            try:
                ltp = get_option_ltp(pos.instrument_key)
                mtm = (ltp - pos.entry_price) * pos.qty
                total_mtm += mtm

                info = info_from_instrument_key(contracts, pos.instrument_key)
                if info:
                    print(
                        f"  {pos.instrument_key[:15]}... "
                        f"({info['strike']} {info['type']}) "
                        f"E:{pos.entry_price:.1f} "
                        f"LTP:{ltp:.1f} "
                        f"SL:{pos.trail_price:.1f} "
                        f"MTM:{mtm:.0f}"
                    )
                else:
                    print(
                        f"  {pos.instrument_key[:15]}... "
                        f"E:{pos.entry_price:.1f} "
                        f"LTP:{ltp:.1f} MTM:{mtm:.0f}"
                    )
            except Exception:
                print(f"  {pos.instrument_key[:15]}... LTP fetch failed")
        print(f"Open MTM: {total_mtm:.0f}")
    else:
        print("Open MTM: 0")

    print(f"Daily P&L: {daily_pnl:.0f}")
    print("=" * 50)


# ======================================================================
# ENHANCED ONE-SHOT ENTRY
# ======================================================================
def run_once():
    """Enhanced trend detection with all indicators."""
    spot = get_nifty_ltp()
    print(f"NIFTY LTP: {spot}")

    df = get_nifty_intraday_candles(1440)
    print(f"Candles received: {len(df)}")
    if df.empty:
        print("🛌 MARKET CLOSED - No candle data. Sleeping...")
        return

    df = calculate_indicators(df)
    trend = detect_multi_trend(df)
    print(
        f"\n*** TREND SIGNAL: "
        f"{'BULLISH' if trend == 1 else 'BEARISH' if trend == -1 else 'NEUTRAL'} ***"
    )

    if trend == 0:
        print("No clear multi-indicator trend.")
        return

    contracts = get_nifty_option_contracts()
    ce_map, pe_map, expiry = build_strike_maps(contracts)
    print("Using expiry:", expiry)

    inst_key, opt_type, strike = pick_instrument_for_trend(
        trend, spot, ce_map, pe_map
    )
    if not inst_key:
        print("No option instrument selected.")
        return

    print(f"🚀 BUY {opt_type} {strike} ({inst_key})")

    opt_ltp = get_option_ltp(inst_key)
    print(f"Option LTP: {opt_ltp}")

    entry_oid = place_hft_market_order(inst_key, QTY, "BUY")
    pos = Position(entry_oid, inst_key, QTY, opt_ltp)
    open_positions[entry_oid] = pos
    print("✅ Trade placed with SL/TGT and logged.")


# ======================================================================
# AUTO LOOP
# ======================================================================
def auto_loop(
    max_trades_per_day=3,
    monitor_interval_sec=30,
    start_time=datetime.time(9, 15),
    end_time=datetime.time(15, 15),
):
    """Main auto trading loop."""
    global open_positions
    trades_done = 0
    print("Starting multi-indicator auto loop. Stop with Ctrl+C.")
    while True:
        now = datetime.now().time()

        if now < start_time or now > end_time:
            print("Outside trading window, sleeping 60s...")
            time.sleep(60)
            continue

        if not open_positions and trades_done < max_trades_per_day:
            print("\n=== 🔍 SCANNING FOR MULTI-INDICATOR SIGNAL ===")
            run_once()
            if open_positions:
                trades_done += 1
        else:
            print("\n=== 📊 MONITORING POSITIONS ===")
            monitor_positions()

        dashboard()

        now = datetime.now().time()
        if (now > end_time and not open_positions) or (
            trades_done >= max_trades_per_day and not open_positions
        ):
            print("Stopping auto loop: time window over or max trades done.")
            break

        time.sleep(monitor_interval_sec)


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    try:
        auto_loop(
            max_trades_per_day=3,
            monitor_interval_sec=30,
            start_time=datetime.time(9, 15),
            end_time=datetime.time(15, 15),
        )
    except KeyboardInterrupt:
        print("\nNISO-MULTI-BEAST stopped by user.")
