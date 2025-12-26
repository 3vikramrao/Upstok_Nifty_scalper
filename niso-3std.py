#!/usr/bin/env python3
"""
NISO – NIFTY Options Supertrend Scalper with Paper Logging (Notebook version)
"""

import csv
import datetime as dt
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# CONFIG & CREDENTIALS
try:
    from env import (
        UPSTOX_CLIENT_KEY,
        UPSTOX_CLIENT_SECRET,
        UPSTOX_REDIRECT_URI,
    )
except ImportError as e:
    raise RuntimeError(
        "Set UPSTOX_CLIENT_KEY, UPSTOX_CLIENT_SECRET, "
        "UPSTOX_REDIRECT_URI in env.py"
    ) from e

PAPER = True  # False = LIVE, True = paper-trade with logging

CLIENT_ID = UPSTOX_CLIENT_KEY
CLIENT_SECRET = UPSTOX_CLIENT_SECRET
REDIRECT_URI = UPSTOX_REDIRECT_URI

if not all([CLIENT_ID, CLIENT_SECRET, REDIRECT_URI]):
    raise RuntimeError("Missing Upstox credentials. Check env.py")

ACCESS_TOKEN_FILE = "upstox_access_token.txt"

BASE_REST = "https://api.upstox.com/v2"
BASE_HFT = "https://api-hft.upstox.com/v2"

QTY = 750
PRODUCT = "I"  # Intraday
TAG = "niso-triple-supertrend-bot"

RUPEE_TARGET = 10000  # Exit ALL when total MTM >= ₹10k
RUPEE_SL = -3500  # Exit ALL when total MTM <= -₹3.5k (1:2.85 R:R)

# Paper log config
PAPER_LOG_DIR = Path("paper_logs")

# NIFTY 50 index instrument key for candles
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
UPSTOX_API_VERSION = "2.0"


def get_access_token() -> str:
    """Get access token from file."""
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


def api_headers() -> dict[str, str]:
    """Return REST API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
    }


def hft_headers() -> dict[str, str]:
    """Return HFT API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def upstox_headers() -> dict[str, str]:
    """Return Upstox historical API headers."""
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Api-Version": UPSTOX_API_VERSION,
        "accept": "application/json",
        "Content-Type": "application/json",
    }


def now_iso() -> str:
    """Current timestamp in ISO 8601 with seconds."""
    return datetime.now().isoformat(timespec="seconds")


def ensure_log_files() -> tuple[Path, Path]:
    """Create paper_logs dir and CSVs with header if missing."""
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    lifetime_file = PAPER_LOG_DIR / "lifetime-3std_log.csv"
    today_file = PAPER_LOG_DIR / f"{datetime.now().date()}_3std-trades.csv"

    header = [
        "timestamp",
        "side",
        "symbol",
        "instrument_key",
        "strike",  # e.g. '26200.0 CE 2025-12-23'
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


def get_nifty_option_contracts(
    expiry_date: str | None = None, verbose: bool = True
) -> list[dict]:
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


def instrument_info_from_key(
    instrument_key: str, contracts: list[dict] | None = None
) -> dict[str, str | float | None] | None:
    """
    Return strike/type/expiry/trading_symbol for a numeric Upstox
    instrument_key like 'NSE_FO|57013' by looking into option contracts list.
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


def log_trade_row(row: list) -> None:
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


def get_nifty_ltp() -> float:
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


def get_nifty_intraday_candles(minutes_back: int) -> pd.DataFrame:
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


def build_strike_maps(
    contracts: list[dict],
) -> tuple[dict[float, str], dict[float, str], datetime.date | None]:
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


def pick_near_atm_strikes(
    spot: float, strike_list: list[float], n: int = 5
) -> list[float]:
    if not strike_list:
        return []
    sorted_strikes = sorted(strike_list, key=lambda k: abs(k - spot))
    return sorted_strikes[:n]


def supertrend(
    df: pd.DataFrame, period: int, multiplier: float, prefix: str
) -> pd.DataFrame:
    """
    Generic Supertrend calculator. Adds 'st_{prefix}' and 'st_dir_{prefix}'
    columns.
    period: ATR window (10,11,12), multiplier: ATR multiple (1,2,3)
    """
    high, low, close = df["high"], df["low"], df["close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # ATR
    atr = tr.rolling(window=period, min_periods=period).mean()

    # Basic bands
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    # Final bands (ratcheting logic)
    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    for i in range(1, len(df)):
        # Upper band: can't rise after downtrend starts
        final_upper.iloc[i] = (
            min(upperband.iloc[i], final_upper.iloc[i - 1])
            if close.iloc[i - 1] <= final_upper.iloc[i - 1]
            else upperband.iloc[i]
        )

        # Lower band: can't fall after uptrend starts
        final_lower.iloc[i] = (
            max(lowerband.iloc[i], final_lower.iloc[i - 1])
            if close.iloc[i - 1] >= final_lower.iloc[i - 1]
            else lowerband.iloc[i]
        )

    # Supertrend line & direction
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = final_upper.iloc[i]
            direction.iloc[i] = -1
        else:
            prev_st = st.iloc[i - 1]
            prev_upper = final_upper.iloc[i - 1]
            final_lower.iloc[i - 1]

            if prev_st == prev_upper:  # Was in downtrend
                if close.iloc[i] <= final_upper.iloc[i]:
                    st.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i] = -1
                else:
                    st.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i] = 1
            else:  # Was in uptrend
                if close.iloc[i] >= final_lower.iloc[i]:
                    st.iloc[i] = final_lower.iloc[i]
                    direction.iloc[i] = 1
                else:
                    st.iloc[i] = final_upper.iloc[i]
                    direction.iloc[i] = -1

    # Add to dataframe
    df[f"st_{prefix}"] = st
    df[f"st_dir_{prefix}"] = direction
    return df


# Triple Supertrend factory functions
def supertrend_10_1(df: pd.DataFrame) -> pd.DataFrame:
    return supertrend(df, 10, 1.0, "10_1")


def supertrend_11_2(df: pd.DataFrame) -> pd.DataFrame:
    return supertrend(df, 11, 2.0, "11_2")


def supertrend_12_3(df: pd.DataFrame) -> pd.DataFrame:
    return supertrend(df, 12, 3.0, "12_3")


def detect_trend(df: pd.DataFrame) -> int:
    """
    Triple Supertrend FIRST ALIGNMENT detection.
    Returns 1 (first bullish), -1 (first bearish), 0 (no new signal).
    """
    if df.empty or len(df) < 20:
        return 0

    df = supertrend_10_1(df)
    df = supertrend_11_2(df)
    df = supertrend_12_3(df)

    # Latest directions
    dir_10_1 = df["st_dir_10_1"].iloc[-1]
    dir_11_2 = df["st_dir_11_2"].iloc[-1]
    dir_12_3 = df["st_dir_12_3"].iloc[-1]

    # Check if ALL 3 aligned (current logic)
    all_bull = dir_10_1 == 1 and dir_11_2 == 1 and dir_12_3 == 1
    all_bear = dir_10_1 == -1 and dir_11_2 == -1 and dir_12_3 == -1

    # Previous candle directions (if exists)
    if len(df) > 1:
        prev_10_1 = df["st_dir_10_1"].iloc[-2]
        prev_11_2 = df["st_dir_11_2"].iloc[-2]
        prev_12_3 = df["st_dir_12_3"].iloc[-2]

        prev_all_bull = prev_10_1 == 1 and prev_11_2 == 1 and prev_12_3 == 1
        prev_all_bear = prev_10_1 == -1 and prev_11_2 == -1 and prev_12_3 == -1

        # FIRST ALIGNMENT: current align + previous NOT aligned
        if all_bull and not prev_all_bull:
            print(
                f"FIRST BULLISH: 10,1={dir_10_1} 11,2={dir_11_2} "
                f"12,3={dir_12_3}"
            )
            return 1
        elif all_bear and not prev_all_bear:
            print(
                f"FIRST BEARISH: 10,1={dir_10_1} 11,2={dir_11_2} "
                f"12,3={dir_12_3}"
            )
            return -1
    else:
        # First candle ever - treat as signal
        if all_bull:
            return 1
        elif all_bear:
            return -1

    print(
        f"Triple ST aligned but not FIRST: 10,1={dir_10_1} "
        f"11,2={dir_11_2} 12,3={dir_12_3}"
    )
    return 0


def pick_instrument_for_trend(
    trend: int,
    spot: float,
    ce_map: dict[float, str],
    pe_map: dict[float, str],
) -> tuple[str | None, str | None, float | None]:
    """Pick one ATM CE or PE instrument_key based on Supertrend direction."""
    if trend == 0:
        return None, None, None
    if trend == 1:  # bullish -> CE
        strikes = pick_near_atm_strikes(spot, list(ce_map.keys()), n=5)
        if not strikes:
            return None, None, None
        strike = strikes[0]
        return ce_map[strike], "CE", strike
    else:  # bearish -> PE
        strikes = pick_near_atm_strikes(spot, list(pe_map.keys()), n=5)
        if not strikes:
            return None, None, None
        strike = strikes[0]
        return pe_map[strike], "PE", strike


def info_from_instrument_key(
    contracts: list[dict], instrument_key: str
) -> dict[str, str | float | None] | None:
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


def get_option_ltp(instrument_key: str) -> float:
    url = f"{BASE_REST}/market-quote/ltp"
    params = {"instrument_key": instrument_key}
    r = requests.get(url, headers=api_headers(), params=params)
    r.raise_for_status()
    j = r.json()
    data_block = j.get("data", {})
    key = list(data_block.keys())[0]
    item = data_block[key]
    return item.get("last_price") or item.get("ltp")


def place_hft_market_order(
    instrument_token: str, quantity: int, side: str
) -> str:
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


def place_hft_limit_order(
    instrument_token: str, quantity: int, side: str, price: float
) -> str:
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


def place_hft_sl_order(
    instrument_token: str,
    quantity: int,
    side: str,
    price: float,
    trigger_price: float,
) -> str:
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


def get_account_margin() -> float:
    """Get available margin from Upstox profile."""
    url = f"{BASE_REST}/profile/user"
    r = requests.get(url, headers=api_headers())
    r.raise_for_status()
    data = r.json()["data"]
    return float(data.get("day_limit", 0))  # Available intraday margin


def get_option_margin_estimate(instrument_key: str, qty: int) -> float:
    """Rough estimate: option premium × qty × 1.5 margin."""
    ltp = get_option_ltp(instrument_key)
    return ltp * qty * 1.5  # Conservative 1.5x estimate


def check_margin_for_order(instrument_key: str) -> bool:
    """Check if enough margin for QTY (750) of this option."""
    margin_needed = get_option_margin_estimate(instrument_key, QTY)
    available = get_account_margin()

    print(
        f"Margin check: Available ₹{available:,.0f} | Needed ₹{margin_needed:,.0f}"
    )

    if available < margin_needed * 1.1:  # 10% buffer
        print("❌ INSUFFICIENT MARGIN")
        return False
    return True


open_positions: dict[str, "Position"] = {}
daily_pnl = 0.0


class Position:
    def __init__(
        self, entry_oid: str, instrument_key: str, qty: int, entry_price: float
    ):
        self.entry_oid = entry_oid
        self.instrument_key = instrument_key
        self.qty = qty
        self.entry_price = entry_price

        # Log entry (SL/TGT columns = 0 for portfolio exits)
        row = [
            now_iso(),
            "BUY",
            TAG,
            self.instrument_key,
            self.qty,
            round(self.entry_price, 2),
            0.0,  # No individual SL
            0.0,  # No individual TGT
            "ENTRY",
            0.0,  # pnl
        ]
        log_trade_row(row)
        print(f"Position opened: {self.instrument_key} @ ₹{entry_price:.2f}")


def exit_all_positions(reason: str, final_mtm: float) -> None:
    global open_positions, daily_pnl
    for oid, pos in list(open_positions.items()):
        exit_ltp = get_option_ltp(pos.instrument_key)
        pnl = (exit_ltp - pos.entry_price) * pos.qty
        daily_pnl += pnl

        row = [
            now_iso(),
            "SELL",
            TAG,
            pos.instrument_key,
            pos.qty,
            exit_ltp,
            0,
            0,
            reason,
            round(pnl, 2),
        ]
        log_trade_row(row)
        del open_positions[oid]
    print(f"All positions exited: {reason} | Final MTM: ₹{final_mtm:.0f}")


def monitor_positions() -> None:
    global open_positions, daily_pnl

    # Portfolio MTM check ONLY
    total_mtm = 0.0
    for oid, pos in open_positions.items():
        ltp = get_option_ltp(pos.instrument_key)
        mtm = (ltp - pos.entry_price) * pos.qty
        total_mtm += mtm

    print(f"Portfolio MTM: ₹{total_mtm:.0f}")

    if total_mtm >= RUPEE_TARGET:
        print("🎯 ₹10K TARGET HIT!")
        exit_all_positions("RUPEE_TARGET", total_mtm)
        return
    elif total_mtm <= RUPEE_SL:
        print("🛑 ₹3.5K SL HIT!")
        exit_all_positions("RUPEE_SL", total_mtm)
        return

    print("Monitoring... no exits triggered")


def dashboard() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print(
        "==========  NIFTY TRIPLE-SUPER-TREND SCALPER  –",
        "PAPER" if PAPER else "LIVE",
        "MODE  ==========",
    )
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
                f"PORTFOLIO ₹{RUPEE_TARGET:.0f}/₹{RUPEE_SL:.0f}  "
                f"MTM {mtm:.0f}"
            )
        else:
            print(
                f"  {pos.instrument_key}  "
                f"ENTRY {pos.entry_price:.2f}  "
                f"LTP {ltp:.2f}  "
                f"PORTFOLIO ₹{RUPEE_TARGET:.0f}/₹{RUPEE_SL:.0f}  "
                f"MTM {mtm:.0f}"
            )

    print(f"Open MTM : {total_mtm:.0f}")
    print(f"Daily P&L: {daily_pnl:.0f}")
    print(
        "======================================================================"
    )


def run_once() -> None:
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

    inst_key, opt_type, strike = pick_instrument_for_trend(
        trend, spot, ce_map, pe_map
    )
    if not inst_key:
        print("No option instrument selected.")
        return
    # ADD MARGIN CHECK HERE
    if not check_margin_for_order(inst_key):
        print("Margin insufficient - skipping trade")
        return
    print(f"BUY {opt_type} {strike} ({inst_key})")

    opt_ltp = get_option_ltp(inst_key)
    print(f"Option LTP: {opt_ltp}")

    entry_oid = place_hft_market_order(inst_key, QTY, "BUY")
    pos = Position(entry_oid, inst_key, QTY, opt_ltp)

    open_positions[entry_oid] = pos
    print("Trade placed with SL/TGT and logged.")


def auto_loop(
    max_trades_per_day: int = 3,
    monitor_interval_sec: int = 30,
    start_time: dt.time = dt.time(9, 20),
    end_time: dt.time = dt.time(15, 15),
) -> None:
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
            print(
                "Stopping auto loop: time window over or max trades done, "
                "and no open positions."
            )
            break

        time.sleep(monitor_interval_sec)


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
        print("\nNISO-3STD stopped by user.")
