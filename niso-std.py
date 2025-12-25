# %% [markdown]
# Cell 1 – Imports and basic config

# %%
#!/usr/bin/env python3
"""
NISO – NIFTY Options Supertrend Scalper with Paper Logging (Notebook version)
"""

import os
import time
import csv
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import datetime as dt
import re



# %% [markdown]
# Cell 2 – Env variables and global constants

# %%
# CONFIG & CREDENTIALS

from env import UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, UPSTOX_REDIRECT_URI

PAPER = True  # False = LIVE, True = paper-trade with logging

CLIENT_ID = UPSTOX_CLIENT_KEY
CLIENT_SECRET = UPSTOX_CLIENT_SECRET
REDIRECT_URI = UPSTOX_REDIRECT_URI

if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
    raise RuntimeError("Set UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, UPSTOX_REDIRECT_URI in env.py")

ACCESS_TOKEN_FILE = "upstox_access_token.txt"

BASE_REST = "https://api.upstox.com/v2"
BASE_HFT = "https://api-hft.upstox.com/v2"

QTY = 150
PRODUCT = "I"      # Intraday
TAG = "niso-supertrend-bot"

# SL/TGT parameters
SL_PCT = 0.20         # 20% stop loss
TG_PCT = 0.30         # 30% target
TRAIL_ATR_MULT = 1.5  # trailing SL ATR multiple

# Paper log config
PAPER_LOG_DIR = Path("paper_logs")

# NIFTY 50 index instrument key for candles
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
UPSTOX_API_VERSION = "2.0"



# %% [markdown]
# Cell 3 – Auth helpers and headers

# %%
def get_access_token():
    if not os.path.exists(ACCESS_TOKEN_FILE):
        raise RuntimeError("Run your Upstox auth script once to create upstox_access_token.txt")
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


def upstox_headers():
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Api-Version": UPSTOX_API_VERSION,
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def now_iso():
    """Current timestamp in ISO 8601 with seconds."""
    return datetime.now().isoformat(timespec="seconds")



# %% [markdown]
# Cell 4 – Logging helpers (CSV + strike / P&L)

# %%
def ensure_log_files():
    """Create paper_logs dir and CSVs with header if missing."""
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    lifetime_file = PAPER_LOG_DIR / "lifetime-std_log.csv"
    today_file = PAPER_LOG_DIR / f"{datetime.now().date()}_std-trades.csv"

    header = [
        "timestamp",
        "side",
        "symbol",
        "instrument_key",
        "strike",        # e.g. '26200.0 CE 2025-12-23'
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
                "type": c.get("instrument_type"),   # CE or PE
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
        # Example: 26200.0 CE 2025-12-23
        strike_display = f"{strike_val:.1f} {opt_type} {expiry}".strip()
    else:
        strike_display = "N/A"

    # Insert strike just after instrument_key
    log_row = row[:4] + [strike_display] + row[4:]

    for f in (lifetime_file, today_file):
        with f.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(log_row)


# %% [markdown]
# Cell 5 – Market data: NIFTY LTP and candles

# %%
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


# %% [markdown]
# Cell 6 – Option contracts utilities and trend logic

# %%
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


# ======================================================================
# Supertrend(10,2) strategy utilities
# ======================================================================

def pick_near_atm_strikes(spot, strike_list, n=5):
    if not strike_list:
        return []
    sorted_strikes = sorted(strike_list, key=lambda k: abs(k - spot))
    return sorted_strikes[:n]


def supertrend_10_2(df):
    """
    Supertrend with period=10, multiplier=2 implemented in pure pandas.
    Adds columns: 'st_10_2' and 'st_dir_10_2' (1 uptrend, -1 downtrend).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR(10) – simple moving average
    atr = tr.rolling(window=10, min_periods=10).mean()

    # Basic bands
    hl2 = (high + low) / 2
    upperband = hl2 + 2 * atr
    lowerband = hl2 - 2 * atr

    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    for i in range(1, len(df)):
        # Final upper band
        if close.iloc[i - 1] <= final_upperband.iloc[i - 1]:
            final_upperband.iloc[i] = min(upperband.iloc[i], final_upperband.iloc[i - 1])
        else:
            final_upperband.iloc[i] = upperband.iloc[i]

        # Final lower band
        if close.iloc[i - 1] >= final_lowerband.iloc[i - 1]:
            final_lowerband.iloc[i] = max(lowerband.iloc[i], final_lowerband.iloc[i - 1])
        else:
            final_lowerband.iloc[i] = lowerband.iloc[i]

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = final_upperband.iloc[i]
            direction.iloc[i] = -1
        else:
            if st.iloc[i - 1] == final_upperband.iloc[i - 1]:
                if close.iloc[i] <= final_upperband.iloc[i]:
                    st.iloc[i] = final_upperband.iloc[i]
                    direction.iloc[i] = -1
                else:
                    st.iloc[i] = final_lowerband.iloc[i]
                    direction.iloc[i] = 1
            else:
                if close.iloc[i] >= final_lowerband.iloc[i]:
                    st.iloc[i] = final_lowerband.iloc[i]
                    direction.iloc[i] = 1
                else:
                    st.iloc[i] = final_upperband.iloc[i]
                    direction.iloc[i] = -1

    df["st_10_2"] = st
    df["st_dir_10_2"] = direction
    return df


def detect_trend(df):
    """
    Use Supertrend(10,2) for trend.
    Returns 1 for fresh bullish flip, -1 for fresh bearish flip, 0 for no flip/insufficient data.
    """
    if df.empty or len(df) < 20:
        return 0

    df = supertrend_10_2(df)
    last_dir = df["st_dir_10_2"].iloc[-1]
    prev_dir = df["st_dir_10_2"].iloc[-2]

    print(f"Supertrend(10,2) dir(prev) = {prev_dir}, dir(last) = {last_dir}")
    
    # Only trade on direction flip
    if last_dir == 1 and prev_dir != 1:
        return 1  # Fresh bullish flip
    elif last_dir == -1 and prev_dir != -1:
        return -1  # Fresh bearish flip
    return 0

def pick_instrument_for_trend(trend, spot, ce_map, pe_map):
    """Pick one ATM CE or PE instrument_key based on Supertrend direction."""
    if trend == 0:
        return None, None, None
    if trend == 1:  # bullish -> CE
        strikes = pick_near_atm_strikes(spot, list(ce_map.keys()), n=5)
        if not strikes:
            return None, None, None
        strike = strikes[0]
        return ce_map[strike], "CE", strike
    else:          # bearish -> PE
        strikes = pick_near_atm_strikes(spot, list(pe_map.keys()), n=5)
        if not strikes:
            return None, None, None
        strike = strikes[0]
        return pe_map[strike], "PE", strike



def info_from_instrument_key(contracts, instrument_key: str):
    """
    Given the full contracts list, return strike, type, expiry, trading_symbol.
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


# %% [markdown]
# Cell 7 – Option LTP and order placement helpers

# %%
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


# %% [markdown]
# Cell 8 – Position class, trailing SL, monitoring, dashboard

# %%
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

        # Place SL + TGT
        self.sl_oid = place_hft_sl_order(
            instrument_key, qty, "SELL",
            self.sl_price,
            self.sl_price * 1.02
        )
        self.tgt_oid = place_hft_limit_order(
            instrument_key, qty, "SELL",
            self.tgt_price
        )

        print(f"Position: entry={entry_price:.2f}, SL={self.sl_price:.2f}, TGT={self.tgt_price:.2f}")

        # Log entry
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
            0.0,   # pnl
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
                exit_ltp,
                round(pos.sl_price, 2),
                round(pos.tgt_price, 2),
                r,
                round(pnl, 2),
            ]
            log_trade_row(row)

            del open_positions[entry_oid]
    print("Monitor done; open positions:", len(open_positions))


def dashboard():
    os.system("cls" if os.name == "nt" else "clear")
    print("==========  NIFTY SUPER-TREND SCALPER  –", "PAPER" if PAPER else "LIVE", "MODE  ==========")
    print("Time   :", dt.datetime.now().strftime("%H:%M:%S"))
    print("Open   :", len(open_positions))

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


# %% [markdown]
# Cell 9 – Single trade run helper

# %%
def run_once():
    """Detect trend, pick CE/PE, place BUY + SL + TGT (one-shot)."""
    spot = get_nifty_ltp()
    print(f"NIFTY LTP: {spot}")

    df = get_nifty_intraday_candles(120)
    if df.empty:
        print("No candle data.")
        return

    trend = detect_trend(df)
    if trend == 0:
        print("No clear trend.")
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



# %% [markdown]
# Cell 10 – Auto loop (main loop you run in the notebook)

# %%
def auto_loop(
    max_trades_per_day=3,
    monitor_interval_sec=30,
    start_time=dt.time(9, 20),
    end_time=dt.time(15, 15),
):
    global open_positions
    trades_done = 0
    print("Starting auto loop. Stop with Kernel Interrupt.")
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

        dashboard()

        now = dt.datetime.now().time()
        if (now > end_time and not open_positions) or (
            trades_done >= max_trades_per_day and not open_positions
        ):
            print("Stopping auto loop: time window over or max trades done, and no open positions.")
            break

        time.sleep(monitor_interval_sec)
# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    try:
        # For a single manual trade, use this instead of auto loop:
        # run_once()

        # For automated intraday loop with dashboard:
        auto_loop(
            max_trades_per_day=3,
            monitor_interval_sec=30,
            start_time=dt.time(9, 20),
            end_time=dt.time(15, 15),
        )
    except KeyboardInterrupt:
        print("\nNISOSTD stopped by user.")



