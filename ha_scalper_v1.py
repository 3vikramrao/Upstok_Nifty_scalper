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
