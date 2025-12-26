#!/usr/bin/env python3
"""
NISO – NIFTY Options Scalper v1 with Paper Logging

Requirements:
- env.py with:
    UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, UPSTOX_REDIRECT_URI
- upstox_access_token.txt created by your auth script
- Folder paper_log/ will be auto-created with:
    - lifetime-ema_log.csv
    - YYYY-MM-DD-ema_trades.csv
"""

import csv
import datetime as dt
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

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
PRODUCT = "I"  # Intraday
TAG = "niso-ema-bot"

# SL/TGT parameters
SL_PCT = 0.20  # 20% stop loss
TG_PCT = 0.30  # 30% target
TRAIL_ATR_MULT = 1.5  # trailing SL ATR multiple

# Paper log config
PAPER_LOG_DIR = Path("paper_logs")

# NIFTY 50 index instrument key for candles
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
UPSTOX_API_VERSION = "2.0"


def upstox_headers():
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Api-Version": UPSTOX_API_VERSION,
        "accept": "application/json",
        "Content-Type": "application/json",
    }


# ======================================================================
# LOGGING HELPERS
# ======================================================================
def ensure_log_files():
    """Create paper_logs dir and CSVs with header if missing."""
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    lifetime_file = PAPER_LOG_DIR / "lifetime-ema915rsi9_log.csv"
    today_file = PAPER_LOG_DIR / f"{datetime.now().date()}_ema915rsi9-trades.csv"

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
    """
    Return strike/type/expiry/trading_symbol for a numeric Upstox instrument_key
    like 'NSE_FO|57013' by looking into option contracts list.[web:7]
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


def now_iso():
    """Current timestamp in ISO 8601 with seconds."""
    return datetime.now().isoformat(timespec="seconds")


# ======================================================================
# AUTH & HEADERS
# ======================================================================
def get_access_token():
    if not os.path.exists(ACCESS_TOKEN_FILE):
        raise RuntimeError(
            "Run your Upstox auth script once to create upstox_access_token.txt"
        )
    with open(ACCESS_TOKEN_FILE, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        raise RuntimeError("upstox_access_token.txt is empty")
    return token


def api_headers():
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
    }


def hft_headers():
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


# ======================================================================
# NIFTY SPOT LTP & CANDLES
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


# ======================================================================
# OPTION CONTRACTS, STRIKES & TREND
# ======================================================================
def get_nifty_option_contracts(expiry_date=None, verbose=True):
    """NIFTY option contracts via option/contract endpoint.[web:7]"""
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
    if not strike_list:
        return []
    sorted_strikes = sorted(strike_list, key=lambda k: abs(k - spot))
    return sorted_strikes[:n]


def detect_trend(df):
    """Return 1 for bullish (EMA9>EMA15 + RSI bullish), -1 for bearish, 0 otherwise."""
    if df.empty or len(df) < 21:
        print("❌ Need 21+ candles")
        return 0

    # EMA trend - FIXED function names
    ema9 = ema(df["close"], 9).iloc[-1]  # ✅ FIXED: ema()
    ema15 = ema(df["close"], 15).iloc[-1]  # ✅ FIXED: ema()
    ema_bull = ema9 > ema15
    ema_bear = ema9 < ema15

    # RSI confirmation
    rsi_bull = df["mom_up"].iloc[-1] or df["mom_up_strong"].iloc[-1]
    rsi_bear = df["mom_dn"].iloc[-1] or df["mom_dn_strong"].iloc[-1]

    print(f"📊 EMA9={ema9:.1f} EMA15={ema15:.1f} | RSI={df['rsi'].iloc[-1]:.1f}")
    print(
        f"🎯 EMA: {'🟢' if ema_bull else '🔴' if ema_bear else '⚪'} | RSI: {'🟢' if rsi_bull else '🔴' if rsi_bear else '⚪'}"
    )

    if ema_bull and rsi_bull:
        print("✅ BULLISH: BUY CE")
        return 1
    elif ema_bear and rsi_bear:
        print("✅ BEARISH: BUY PE")
        return -1
    else:
        print("❌ No alignment - EMA+RSI must agree")
        return 0


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


def ema(series, n):
    """EMA calculation helper."""
    return series.ewm(span=n, adjust=False).mean()


def info_from_instrument_key(contracts, instrument_key: str):
    """
    Given the full contracts list from get_nifty_option_contracts(),
    return strike, type, expiry, trading_symbol for a specific instrument_key.
    """
    for c in contracts:
        if c.get("instrument_key") == instrument_key:
            return {
                "strike": c.get("strike_price"),
                "type": c.get("instrument_type"),
                "expiry": c.get("expiry"),
                "trading_symbol": c.get("trading_symbol"),
            }
    return None


def compute_momentum_bars(
    df: pd.DataFrame, mo_bars_on: bool = True
) -> pd.DataFrame:  # 2️⃣ SECOND
    """Compute RSI(9) + momentum signals - FIXED for NaN issues."""
    close = df["close"]

    # RSI(9) calculation
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    n_rsi = 9
    alpha = 1 / n_rsi
    up_rma = up.ewm(alpha=alpha, adjust=False).mean()
    down_rma = down.ewm(alpha=alpha, adjust=False).mean()

    rsi = np.where(
        down_rma == 0,
        100,
        np.where(up_rma == 0, 0, 100 - 100 / (1 + up_rma / down_rma)),
    )
    df["rsi"] = rsi

    # FIXED WMA(21) - no NaN problems
    n_wma = 21
    weights = np.arange(1, n_wma + 1)

    def wma_func(x):
        return np.dot(x, weights[: len(x)]) / weights[: len(x)].sum()

    df["rsi_wma21"] = (
        df["rsi"]
        .rolling(window=n_wma, min_periods=5)
        .apply(wma_func, raw=True)
        .fillna(method="ffill")
        .fillna(df["rsi"])
    )

    df["rsi_ema3"] = df["rsi"].ewm(span=3, adjust=False).mean()

    # Momentum signals
    rsi_series = df["rsi"]
    wma = df["rsi_wma21"]
    ema_rsi = df["rsi_ema3"]

    df["mom_up"] = ((rsi_series > ema_rsi) & (rsi_series > wma)).fillna(False)
    df["mom_up_strong"] = ((rsi_series > wma) & (rsi_series > 50)).fillna(False)
    df["mom_dn"] = ((rsi_series < ema_rsi) & (rsi_series < wma)).fillna(False)
    df["mom_dn_strong"] = ((rsi_series < wma) & (rsi_series < 40)).fillna(False)

    bar_color = np.where(df["mom_up"], "green", np.where(df["mom_dn"], "red", "white"))
    df["bar_color"] = bar_color if mo_bars_on else None

    return df


# ======================================================================
# OPTION LTP & HFT ORDER HELPERS
# ======================================================================
def get_option_ltp(instrument_key):
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
    r = requests.post(f"{BASE_HFT}/order/place", headers=hft_headers(), json=payload)
    print("LIMIT status:", r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json()["data"]["order_id"]


def place_hft_sl_order(instrument_token, quantity, side, price, trigger_price):
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
    r = requests.post(f"{BASE_HFT}/order/place", headers=hft_headers(), json=payload)
    print("SL status:", r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json()["data"]["order_id"]


# ======================================================================
# POSITION CLASS & TRAILING LOGIC
# ======================================================================
open_positions = {}
daily_pnl = 0.0


class Position:
    def __init__(self, entry_oid, instrument_key, qty, entry_price):
        self.entry_oid = entry_oid
        self.instrument_key = instrument_key
        self.qty = qty
        self.entry_price = entry_price

        self.sl_price = entry_price * (1 - SL_PCT)
        self.tgt_price = entry_price * (1 + TG_PCT)
        self.trail_price = self.sl_price

        # Place paper/live SL + TGT orders
        self.sl_oid = place_hft_sl_order(
            instrument_key, qty, "SELL", self.sl_price, self.sl_price * 1.02
        )
        self.tgt_oid = place_hft_limit_order(
            instrument_key, qty, "SELL", self.tgt_price
        )

        print(
            f"Position: entry={entry_price:.2f}, SL={self.sl_price:.2f}, TGT={self.tgt_price:.2f}"
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


def update_trailing_sl(pos: Position):
    ltp = get_option_ltp(pos.instrument_key)
    atr_proxy = ltp * 0.10
    new_trail = ltp - TRAIL_ATR_MULT * atr_proxy
    if new_trail > pos.trail_price:
        pos.trail_price = new_trail
        print(f"Trail updated to {pos.trail_price:.2f} (LTP={ltp:.2f})")


def check_exit_conditions(pos: Position):
    ltp = get_option_ltp(pos.instrument_key)
    if ltp <= pos.trail_price:
        print(f"TRAIL HIT: {ltp:.2f} <= {pos.trail_price:.2f}")
        return "TRAIL"
    elif ltp >= pos.tgt_price:
        print(f"TARGET HIT: {ltp:.2f} >= {pos.tgt_price:.2f}")
        return "TARGET"
    return None


def monitor_positions():
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
    os.system("cls" if os.name == "nt" else "clear")
    print(
        "==========  NIFTY EMA SCALPER  –",
        "PAPER" if PAPER else "LIVE",
        "MODE  ==========",
    )
    print("Time   :", dt.datetime.now().strftime("%H:%M:%S"))
    print("Open   :", len(open_positions))

    # fetch contracts once for strike lookup, without debug spam
    contracts = get_nifty_option_contracts(verbose=False)

    total_mtm = 0.0
    for oid, pos in open_positions.items():
        ltp = get_option_ltp(pos.instrument_key)
        mtm = (ltp - pos.entry_price) * pos.qty
        total_mtm += mtm

        info = info_from_instrument_key(contracts, pos.instrument_key)
        if info:
            print(
                f"  {pos.instrument_key} "
                f"({info['strike']} {info['type']} {info['expiry']})  "
                f"ENTRY {pos.entry_price:.2f}  "
                f"LTP {ltp:.2f}  "
                f"SL {pos.trail_price:.2f}  "
                f"MTM {mtm:.0f}"
            )
        else:
            print(
                f"  {pos.instrument_key}  "
                f"ENTRY {pos.entry_price:.2f}  "
                f"LTP {ltp:.2f}  "
                f"SL {pos.trail_price:.2f}  "
                f"MTM {mtm:.0f}"
            )

    print(f"Open MTM : {total_mtm:.0f}")
    print(f"Daily P&L: {daily_pnl:.0f}")
    print("====================================================")


# ======================================================================
# ONE-SHOT ENTRY (run_once)
# ======================================================================
def run_once():
    """Detect trend using BOTH EMA + RSI, pick CE/PE, place BUY + SL + TGT."""
    spot = get_nifty_ltp()
    print(f"NIFTY LTP: {spot}")

    df = get_nifty_intraday_candles(120)
    if df.empty:
        print("No candle data.")
        return

    # Compute RSI momentum FIRST (critical for new detect_trend)
    df = compute_momentum_bars(df, mo_bars_on=True)

    trend = detect_trend(df)  # Now uses BOTH indicators
    if trend == 0:
        print("No aligned trend (EMA + RSI required).")
        return

    contracts = get_nifty_option_contracts()
    ce_map, pe_map, expiry = build_strike_maps(contracts)
    print("Using expiry:", expiry)

    inst_key, opt_type, strike = pick_instrument_for_trend(trend, spot, ce_map, pe_map)
    if not inst_key:
        print("No option instrument selected.")
        return

    print(f"BUY {opt_type} {strike} ({inst_key})")

    opt_ltp = get_option_ltp(inst_key)
    print(f"Option LTP: {opt_ltp}")

    entry_oid = place_hft_market_order(inst_key, QTY, "BUY")
    pos = Position(entry_oid, inst_key, QTY, opt_ltp)

    open_positions[entry_oid] = pos
    print("Trade placed with SL/TGT and logged.")


# ======================================================================
# AUTO LOOP
# ======================================================================
def auto_loop(
    max_trades_per_day=3,
    monitor_interval_sec=30,
    start_time=dt.time(9, 20),
    end_time=dt.time(15, 15),
):
    global open_positions
    trades_done = 0
    print("Starting auto loop. Stop with Ctrl+C.")
    while True:
        now = dt.datetime.now().time()

        if now < start_time or now > end_time:
            print("Outside trading window, sleeping 60s...")
            time.sleep(60)
            continue

        if not open_positions and trades_done < max_trades_per_day:
            print("\n=== ENTERING NEW TRADE ===")
            run_once()
            if open_positions:
                trades_done += 1
        else:
            print("\n=== MONITORING POSITIONS ===")
            monitor_positions()

        # always show dashboard here
        dashboard()

        now = dt.datetime.now().time()
        if (now > end_time and not open_positions) or (
            trades_done >= max_trades_per_day and not open_positions
        ):
            print(
                "Stopping auto loop: time window over or max trades done, and no open positions."
            )
            break

        time.sleep(monitor_interval_sec)


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    try:
        # For a single manual trade, comment auto_loop() and use:
        # run_once()

        # For automated intraday loop:
        auto_loop(
            max_trades_per_day=3,
            monitor_interval_sec=30,
            start_time=dt.time(9, 15),
            end_time=dt.time(15, 15),
        )
    except KeyboardInterrupt:
        print("\nNISOEMA stopped by user.")
