#!/usr/bin/env python3
"""
HA-SCALPER V1: MEAN REVERSION BEAST
HA + Williams %R + Z-Score + Aroon + Stoch (4/5)
"""

import csv
import datetime as dt
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import talib

# ======================================================================
# CONFIG
# ======================================================================

PAPER = True
ACCESS_TOKEN_FILE = "upstox_access_token.txt"
BASE_REST = "https://api.upstox.com/v2"
BASE_HFT = "https://api-hft.upstox.com/v2"
QTY = 150
PRODUCT = "I"
TAG = "ha-v1-reversal"
SL_PCT, TG_PCT, TRAIL_ATR_MULT = 0.20, 0.30, 1.5
PAPER_LOG_DIR = Path("paper_logs")
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
UPSTOX_API_VERSION = "2.0"

open_positions, daily_pnl = {}, 0.0


# ======================================================================
# HEIKIN ASHI + V1 INDICATORS
# ======================================================================
def heikin_ashi(df):
    """Convert to Heikin Ashi candles"""
    ha = df.copy()
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha["ha_open"] = (df["open"].shift(1) + df["close"].shift(1)) / 2
    ha.loc[0, "ha_open"] = (df.loc[0, "open"] + df.loc[0, "close"]) / 2
    ha["ha_open"] = ha["ha_open"].fillna(method="bfill")
    ha["ha_high"] = pd.concat([ha["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(
        axis=1
    )
    ha["ha_low"] = pd.concat([ha["low"], ha["ha_open"], ha["ha_close"]], axis=1).min(
        axis=1
    )
    return ha


def v1_indicators(df):
    """HA + Williams %R + Z-Score + Aroon + Stoch"""
    df = heikin_ashi(df)

    # 1. HA Trend
    df["ha_green"] = df["ha_close"] > df["ha_open"]

    # 2. Williams %R (14)
    df["williams_r"] = talib.WILLR(
        df["ha_high"].values, df["ha_low"].values, df["ha_close"].values, 14
    )

    # 3. Z-Score (20)
    ma20 = df["ha_close"].rolling(20).mean()
    std20 = df["ha_close"].rolling(20).std()
    df["zscore"] = (df["ha_close"] - ma20) / std20

    # 4. Aroon (14)
    df["aroon_up"], df["aroon_down"] = talib.AROON(
        df["ha_high"].values, df["ha_low"].values, 14
    )

    # 5. Stochastic (14,3,3)
    df["stoch_k"], df["stoch_d"] = talib.STOCH(
        df["ha_high"].values, df["ha_low"].values, df["ha_close"].values
    )

    return df


def detect_ha_v1_trend(df):
    """V1: 4/5 Mean Reversion signals"""
    if len(df) < 30:
        return 0

    latest, _prev = df.iloc[-1], df.iloc[-2]
    signals = []

    # 1. HA Trend
    signals.append(1 if latest["ha_green"] else -1)

    # 2. Williams %R
    signals.append(
        1 if latest["williams_r"] > -20 else -1 if latest["williams_r"] < -80 else 0
    )

    # 3. Z-Score (reversal)
    signals.append(
        1 if latest["zscore"] < -1.0 else -1 if latest["zscore"] > 1.0 else 0
    )

    # 4. Aroon
    signals.append(
        1 if latest["aroon_up"] > 70 else -1 if latest["aroon_down"] > 70 else 0
    )

    # 5. Stochastic
    stoch_bull = latest["stoch_k"] > latest["stoch_d"] and latest["stoch_k"] > 80
    stoch_bear = latest["stoch_k"] < latest["stoch_d"] and latest["stoch_k"] < 20
    signals.append(1 if stoch_bull else -1 if stoch_bear else 0)

    bull, bear = sum(1 for s in signals if s == 1), sum(1 for s in signals if s == -1)

    print(
        f"🎯 V1 REVERSAL: Bull={bull}/5 Bear={bear}/5 | Z={latest['zscore']:.2f} %R={latest['williams_r']:.0f}"
    )

    return 1 if bull >= 4 else -1 if bear >= 4 else 0


# ======================================================================
# UPSTOX API (Same as Beast)
# ======================================================================
def get_access_token():
    with open(ACCESS_TOKEN_FILE, "r") as f:
        return f.read().strip()


def api_headers():
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
    }


def hft_headers():
    """HFT order headers - REQUIRED for Upstox HFT"""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def upstox_headers():
    return {
        **api_headers(),
        "Api-Version": "2.0",
        "Content-Type": "application/json",
    }


# ======================================================================
# LOGGING HELPERS
# ======================================================================
def ensure_log_files():
    """Create paper_logs dir and CSVs with header if missing."""
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    lifetime_file = PAPER_LOG_DIR / "lifetime-ha_scalper_v1.csv"
    today_file = PAPER_LOG_DIR / f"{datetime.now().date()}_ha_scalper_v1.csv"

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


# ========================================================================
def get_nifty_ltp():
    r = requests.get(
        f"{BASE_REST}/market-quote/ltp",
        headers=api_headers(),
        params={"instrument_key": "NSE_INDEX|Nifty 50"},
    )
    r.raise_for_status()
    return r.json()["data"]["NSE_INDEX:Nifty 50"]["last_price"]


def get_nifty_intraday_candles(minutes_back):
    end_time, start_time = dt.datetime.now(), dt.datetime.now() - dt.timedelta(
        minutes=minutes_back
    )
    r = requests.get(
        f"https://api.upstox.com/v2/historical-candle/intraday/{NIFTY_SPOT_INSTRUMENT}/1minute",
        headers=upstox_headers(),
        params={
            "to_date": end_time.isoformat(timespec="seconds"),
            "from_date": start_time.isoformat(timespec="seconds"),
        },
    )
    r.raise_for_status()
    data = r.json()["data"]["candles"]
    return pd.DataFrame(
        [
            {
                "time": dt.datetime.fromisoformat(c[0].replace("Z", "+00:00")),
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            }
            for c in data
        ]
    )


# ======================================================================
# TRADING FUNCTIONS (Same as Beast - abbreviated)
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


def log_trade_row(row):
    """Complete logging - FIXED"""
    lifetime_file, today_file = ensure_log_files()
    instrument_key = row[3]
    info = instrument_info_from_key(instrument_key)

    if info and info.get("strike"):
        strike_val = float(info["strike"])
        strike_display = (
            f"{strike_val:.1f} {info.get('type', '')} {info.get('expiry', '')}".strip()
        )
    else:
        strike_display = "N/A"

    log_row = row[:4] + [strike_display] + row[4:]

    for f in (lifetime_file, today_file):
        with f.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(log_row)


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


def get_option_ltp(instrument_key):
    """Get option LTP - REQUIRED"""
    url = f"{BASE_REST}/market-quote/ltp"
    params = {"instrument_key": instrument_key}
    r = requests.get(url, headers=api_headers(), params=params)
    r.raise_for_status()
    j = r.json()
    data_block = j.get("data", {})
    key = list(data_block.keys())[0]
    return data_block[key].get("last_price") or data_block[key].get("ltp")


def update_trailing_sl(pos):
    """Trailing stop logic - REQUIRED"""
    ltp = get_option_ltp(pos.instrument_key)
    atr_proxy = ltp * 0.10
    new_trail = ltp - TRAIL_ATR_MULT * atr_proxy
    if new_trail > pos.trail_price:
        pos.trail_price = new_trail
        print(f"Trail updated: {pos.trail_price:.2f}")


def check_exit_conditions(pos):
    """Exit conditions - REQUIRED"""
    ltp = get_option_ltp(pos.instrument_key)
    if ltp <= pos.trail_price:
        return "TRAIL"
    elif ltp >= pos.tgt_price:
        return "TARGET"
    return None


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


def dashboard():  # V1 version
    os.system("cls" if os.name == "nt" else "clear")
    print("🎯 HA-V1 MEAN REVERSION –", "PAPER" if PAPER else "LIVE")
    print(
        f"Time: {dt.datetime.now().strftime('%H:%M:%S')} | Open: {len(open_positions)}"
    )
    print(
        f"Open MTM: {sum((get_option_ltp(p.instrument_key)-p.entry_price)*p.qty for p in open_positions.values()):.0f}"
    )
    print(f"Daily P&L: {daily_pnl:.0f}")
    print("=" * 50)


# Simplified run_once for V1
def run_once():
    get_nifty_ltp()
    df = get_nifty_intraday_candles(480)  # 8hrs

    if df.empty:
        print("No data")
        return

    df = v1_indicators(df)
    trend = detect_ha_v1_trend(df)

    if trend != 0:
        print(f"🚀 HA-V1 SIGNAL: {'🟢 CE' if trend == 1 else '🔴 PE'}")
        # Place trade logic here [same as beast]


# ======================================================================
# AUTO LOOP
# ======================================================================
def auto_loop(
    max_trades_per_day=3,
    monitor_interval_sec=30,
    start_time=dt.time(9, 15),
    end_time=dt.time(15, 15),
):
    """HA-V1 Auto trading loop - FIXED"""
    global open_positions
    trades_done = 0

    print("🎯 HA-V1 MEAN REVERSION BEAST starting... Ctrl+C to stop")
    print("Trading window:", start_time, "to", end_time)

    while True:
        now = dt.datetime.now().time()

        # Outside trading hours - sleep longer
        if now < start_time or now > end_time:
            print(f"⏰ Outside trading hours ({now}), sleeping 60s...")
            time.sleep(60)
            continue

        # No position + under trade limit → Scan for signal
        if not open_positions and trades_done < max_trades_per_day:
            print("\n=== 🔍 SCANNING FOR REVERSAL SIGNAL ===")
            run_once()
            if open_positions:
                trades_done += 1
                print(f"✅ Trades done today: {trades_done}/{max_trades_per_day}")
        else:
            print("\n=== 📊 MONITORING POSITIONS ===")
            monitor_positions()

        # Always show dashboard
        dashboard()

        # Stop conditions
        now = dt.datetime.now().time()
        if (now > end_time and not open_positions) or (
            trades_done >= max_trades_per_day and not open_positions
        ):
            print("🏁 Trading session complete!")
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
        print("\n🎯 HA-V1 stopped by user.")
