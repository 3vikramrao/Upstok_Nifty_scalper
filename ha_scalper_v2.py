#!/usr/bin/env python3
"""
HA-SCALPER V2: TREND BREAKOUT BEAST 
HA + CCI + Keltner + ADX + TSI (4/5)
"""

import os
import time
import csv
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import datetime as dt
import talib
import warnings
warnings.filterwarnings("ignore")

# ======================================================================
# CONFIG
# ======================================================================
try:
    from env import UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, UPSTOX_REDIRECT_URI
except ImportError:
    UPSTOX_CLIENT_KEY = os.getenv("UPSTOX_CLIENT_KEY")
    UPSTOX_CLIENT_SECRET = os.getenv("UPSTOX_CLIENT_SECRET")
    UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI")

PAPER = True
ACCESS_TOKEN_FILE = "upstox_access_token.txt"
BASE_REST = "https://api.upstox.com/v2"
BASE_HFT = "https://api-hft.upstox.com/v2"

QTY = 150
PRODUCT = "I"
TAG = "ha-v2-breakout"

SL_PCT, TG_PCT, TRAIL_ATR_MULT = 0.20, 0.30, 1.5
PAPER_LOG_DIR = Path("paper_logs")
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"

open_positions = {}
daily_pnl = 0.0

# ======================================================================
# HEIKIN ASHI + V2 INDICATORS
# ======================================================================
def heikin_ashi(df):
    """Convert to Heikin Ashi candles"""
    ha = df.copy()
    ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    ha.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2
    ha['ha_open'] = ha['ha_open'].fillna(method='bfill').fillna(ha['ha_close'])  # Fixed deprecated bfill
    ha['ha_high'] = pd.concat([ha['high'], ha['ha_open'], ha['ha_close']], axis=1).max(axis=1)
    ha['ha_low'] = pd.concat([ha['low'], ha['ha_open'], ha['ha_close']], axis=1).min(axis=1)
    return ha

def v2_indicators(df):
    """HA + CCI + Keltner + ADX + TSI"""
    df = heikin_ashi(df)
    
    # 1. HA Trend
    df['ha_green'] = df['ha_close'] > df['ha_open']
    
    # 2. CCI (20)
    df['cci'] = talib.CCI(df['ha_high'].values, df['ha_low'].values, df['ha_close'].values, 20)
    
    # 3. Keltner Channels (20, 2.0)
    kc_middle = df['ha_close'].ewm(span=20).mean()
    atr = talib.ATR(df['ha_high'].values, df['ha_low'].values, df['ha_close'].values, 20)
    df['kc_upper'] = kc_middle + 2.0 * atr
    df['kc_lower'] = kc_middle - 2.0 * atr
    
    # 4. ADX + DI (14) - Fixed: use ADX instead of ADXR, unpack properly [web:24]
    df['adx'], df['plus_di'], df['minus_di'] = talib.ADX(
        df['ha_high'].values, df['ha_low'].values, df['ha_close'].values, 14
    )
    
    # 5. TSI (13,25,7) - Fixed: TSI returns single array [web:14]
    df['tsi'] = talib.TSI(df['ha_close'].values, 13, 25, 7)
    
    return df

def detect_ha_v2_trend(df):
    """V2: 4/5 Trend signals"""
    if len(df) < 30: 
        return 0
    
    latest, prev = df.iloc[-1], df.iloc[-2]
    signals = []
    
    # 1. HA Trend
    signals.append(1 if latest['ha_green'] else -1)
    
    # 2. CCI
    signals.append(1 if latest['cci'] > 100 else -1 if latest['cci'] < -100 else 0)
    
    # 3. Keltner Channel
    signals.append(1 if latest['ha_close'] > latest['kc_upper'] else -1 if latest['ha_close'] < latest['kc_lower'] else 0)
    
    # 4. ADX + DI
    adx_strong = latest['adx'] > 25
    signals.append(1 if adx_strong and latest['plus_di'] > latest['minus_di'] else 
                   -1 if adx_strong and latest['minus_di'] > latest['plus_di'] else 0)
    
    # 5. TSI
    signals.append(1 if latest['tsi'] > 0 and latest['tsi'] > prev['tsi'] else 
                   -1 if latest['tsi'] < 0 and latest['tsi'] < prev['tsi'] else 0)
    
    bull, bear = sum(1 for s in signals if s == 1), sum(1 for s in signals if s == -1)
    
    print(f"🔥 V2 TREND: Bull={bull}/5 Bear={bear}/5 | CCI={latest['cci']:.0f} ADX={latest['adx']:.1f}")
    
    return 1 if bull >= 4 else -1 if bear >= 4 else 0

# ======================================================================
# UPSTOX API
# ======================================================================
def get_access_token():
    with open(ACCESS_TOKEN_FILE, "r") as f: 
        return f.read().strip()

def api_headers(): 
    return {"Authorization": f"Bearer {get_access_token()}", "accept": "application/json"}

def upstox_headers():
    return {**api_headers(), "Content-Type": "application/json"}  # Fixed: removed deprecated Api-Version [web:23]
# ======================================================================
# LOGGING HELPERS
# ======================================================================
def ensure_log_files():
    """Create paper_logs dir and CSVs with header if missing."""
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    lifetime_file = PAPER_LOG_DIR / "lifetime-ha_scalper_v2_log.csv"
    today_file = PAPER_LOG_DIR / f"{datetime.now().date()}_ha_scalper_v2-trades.csv"

    header = [
        "timestamp",
        "side",
        "symbol",
        "instrument_key",
        "strike",        # e.g. '26200 CE'
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
#=================================================================
def hft_headers():
    return {**api_headers(), "Content-Type": "application/json"}  # Added missing hft_headers function

def get_nifty_ltp():
    r = requests.get(f"{BASE_REST}/market-quote/ltp", headers=api_headers(), 
                    params={"instrument_key": NIFTY_SPOT_INSTRUMENT})
    r.raise_for_status()
    return r.json()["data"][NIFTY_SPOT_INSTRUMENT]["ltp"]  # Fixed key format [web:27]

def get_nifty_intraday_candles(minutes_back):
    """Fixed historical candle endpoint parameters [web:1]"""
    end_time = datetime.now().strftime('%Y-%m-%d')
    start_time = (datetime.now() - timedelta(minutes=minutes_back)).strftime('%Y-%m-%d')
    
    url = f"{BASE_REST}/historical-candle/intraday/{NIFTY_SPOT_INSTRUMENT}/1minute"
    params = {
        "to": end_time,
        "from": start_time
    }
    
    r = requests.get(url, headers=upstox_headers(), params=params)
    r.raise_for_status()
    data = r.json()["data"]["candles"]
    
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def get_option_ltp(instrument_key):
    """Get LTP for option instrument [web:27]"""
    r = requests.get(f"{BASE_REST}/market-quote/ltp", headers=api_headers(), 
                    params={"instrument_key": instrument_key})
    r.raise_for_status()
    return r.json()["data"][instrument_key]["ltp"]

# ======================================================================
# OPTION CONTRACTS
# ======================================================================
def get_nifty_option_contracts(expiry_date=None):
    """Get NIFTY option contracts [web:12]"""
    params = {"instrument_key": NIFTY_SPOT_INSTRUMENT}
    if expiry_date:
        params["expiry_date"] = expiry_date
    url = f"{BASE_REST}/option/contract"
    r = requests.get(url, headers=api_headers(), params=params)
    r.raise_for_status()
    return r.json().get("data", [])

def build_strike_maps(contracts):
    if not contracts:
        return {}, {}, None
    df = pd.DataFrame(contracts)
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

# ======================================================================
# ORDER FUNCTIONS
# ======================================================================
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
# TRADING LOGIC
# ======================================================================
def now_iso():
    return datetime.now().isoformat()

def log_trade_row(row):
    PAPER_LOG_DIR.mkdir(exist_ok=True)
    filename = PAPER_LOG_DIR / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
    file_exists = filename.exists()
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["time", "side", "tag", "instrument", "qty", "price", "sl", "tgt", "type", "pnl"])
        writer.writerow(row)

class Position:
    def __init__(self, entry_oid, instrument_key, qty, entry_price):
        self.entry_oid = entry_oid
        self.instrument_key = instrument_key
        self.qty = qty
        self.entry_price = entry_price
        self.sl_price = entry_price * (1 - SL_PCT/100)
        self.tgt_price = entry_price * (1 + TG_PCT/100)
        self.trail_price = self.sl_price

        # Place paper/live SL + TGT orders
        self.sl_oid = place_hft_sl_order(
            instrument_key, qty, "SELL",
            self.sl_price, self.sl_price * 1.02
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
            0.0,
        ]
        log_trade_row(row)

def update_trailing_sl(pos):
    """Simple trailing SL logic"""
    current_ltp = get_option_ltp(pos.instrument_key)
    new_trail = current_ltp * (1 - SL_PCT/100)
    if new_trail > pos.trail_price:
        pos.trail_price = new_trail
        pos.sl_price = new_trail

def check_exit_conditions(pos):
    """Check if position should exit"""
    current_ltp = get_option_ltp(pos.instrument_key)
    
    if current_ltp <= pos.sl_price:
        return "SL_HIT"
    elif current_ltp >= pos.tgt_price:
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

            r = {"SL_HIT": "EXIT_SL", "TARGET": "EXIT_TARGET"}.get(reason, "EXIT")

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
                round(pnl, 2),
            ]
            log_trade_row(row)

            del open_positions[entry_oid]
    print("Monitor done; open positions:", len(open_positions))

# ======================================================================
# MAIN LOGIC
# ======================================================================
def dashboard():
    os.system("cls" if os.name == "nt" else "clear")
    print("🎯 HA-V2 TREND BREAKOUT BEAST –", "PAPER" if PAPER else "LIVE")
    print(f"Time: {dt.datetime.now().strftime('%H:%M:%S')} | Open: {len(open_positions)}")
    mtm = sum((get_option_ltp(p.instrument_key)-p.entry_price)*p.qty 
              for p in open_positions.values())
    print(f"Open MTM: {mtm:.0f}")
    print(f"Daily P&L: {daily_pnl:.0f}")
    print("="*50)

def run_once():
    try:
        spot = get_nifty_ltp()
        print(f"Nifty Spot: {spot:.2f}")
        
        df = get_nifty_intraday_candles(480)  # 8hrs
        if df.empty: 
            print("No data"); 
            return
        
        df = v2_indicators(df)
        trend = detect_ha_v2_trend(df)
        
        if trend != 0 and len(open_positions) == 0:  # Only trade if no position
            print(f"🚀 HA-V2 SIGNAL: {'🟢 CE' if trend == 1 else '🔴 PE'}")
            
            # Get option contracts
            contracts = get_nifty_option_contracts()
            ce_map, pe_map, expiry = build_strike_maps(contracts)
            
            instrument_key, opt_type, strike = pick_instrument_for_trend(trend, spot, ce_map, pe_map)
            if instrument_key:
                print(f"Trading {opt_type} {strike}: {instrument_key}")
                ltp = get_option_ltp(instrument_key)
                entry_oid = place_hft_market_order(instrument_key, QTY, "BUY")
                open_positions[entry_oid] = Position(entry_oid, instrument_key, QTY, ltp)
            else:
                print("No suitable option contract found")
        else:
            print("No signal or position exists")
            
    except Exception as e:
        print(f"Error in run_once: {e}")

# ======================================================================
# AUTO LOOP
# ======================================================================
def auto_loop(
    max_trades_per_day=3,
    monitor_interval_sec=30,
    start_time=dt.time(9, 15),
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
            print("Stopping auto loop: time window over or max trades done, and no open positions.")
            break

        time.sleep(monitor_interval_sec)
# ======================================================================
# MAIN (UNCHANGED)
# ======================================================================
if __name__ == "__main__":
    try:
        auto_loop(
            max_trades_per_day=3,
            monitor_interval_sec=30,
            start_time=dt.time(9, 15),
            end_time=dt.time(15, 15),
        )
    except KeyboardInterrupt:
        print("\n🎯HA_SCALPER_V2 stopped by user.")
