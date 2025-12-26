                    "instrument",
                    "qty",
                    "price",
                    "sl",
                    "tgt",
                    "type",
                    "pnl",
                ]
            )
        writer.writerow(row)


class Position:
    def __init__(self, entry_oid, instrument_key, qty, entry_price):
        self.entry_oid = entry_oid
        self.instrument_key = instrument_key
        self.qty = qty
        self.entry_price = entry_price
        self.sl_price = entry_price * (1 - SL_PCT / 100)
        self.tgt_price = entry_price * (1 + TG_PCT / 100)
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
    new_trail = current_ltp * (1 - SL_PCT / 100)
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
    print(
        f"Time: {dt.datetime.now().strftime('%H:%M:%S')} | Open: {len(open_positions)}"
    )
    mtm = sum(
        (get_option_ltp(p.instrument_key) - p.entry_price) * p.qty
        for p in open_positions.values()
    )
    print(f"Open MTM: {mtm:.0f}")
    print(f"Daily P&L: {daily_pnl:.0f}")
    print("=" * 50)


def run_once():
    try:
        spot = get_nifty_ltp()
        print(f"Nifty Spot: {spot:.2f}")

        df = get_nifty_intraday_candles(480)  # 8hrs
        if df.empty:
            print("No data")
            return

        df = v2_indicators(df)
        trend = detect_ha_v2_trend(df)

        if trend != 0 and len(open_positions) == 0:  # Only trade if no position
            print(f"🚀 HA-V2 SIGNAL: {'🟢 CE' if trend == 1 else '🔴 PE'}")

            # Get option contracts
            contracts = get_nifty_option_contracts()
            ce_map, pe_map, expiry = build_strike_maps(contracts)

            instrument_key, opt_type, strike = pick_instrument_for_trend(
                trend, spot, ce_map, pe_map
            )
            if instrument_key:
                print(f"Trading {opt_type} {strike}: {instrument_key}")
                ltp = get_option_ltp(instrument_key)
                entry_oid = place_hft_market_order(instrument_key, QTY, "BUY")
                open_positions[entry_oid] = Position(
                    entry_oid, instrument_key, QTY, ltp
                )
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
            print(
                "Stopping auto loop: time window over or max trades done, and no open positions."
            )
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
