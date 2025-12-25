import os, json, time, requests, pandas as pd
from datetime import datetime, timedelta

from env import (
    UPSTOX_API_KEY,
    UPSTOX_API_SECRET,
    UPSTOX_REDIRECT_URI,
)


class BrokerClient:
    """
    Upstox REST wrapper for data (historical candles, LTP) using access_token.
    """

    def __init__(self, access_token_path="upstox_access_token.txt"):
        self.api_key  = UPSTOX_API_KEY
        self.secret   = UPSTOX_API_SECRET
        self.redirect = UPSTOX_REDIRECT_URI

        if not self.api_key or not self.secret or not self.redirect:
            raise RuntimeError(
                "Set UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_REDIRECT_URI in env.py"
            )

        if not os.path.exists(access_token_path):
            raise RuntimeError(
                f"{access_token_path} not found. Run upstox_auth.py to generate access token."
            )
        self.access_tok = open(access_token_path, "r", encoding="utf-8").read().strip()

    # ---------- headers ---------------------------------------------------
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.access_tok}",
            "Api-Version": "2.0",
            "accept": "application/json",
        }

    # ---------- historic candles (1 min) ----------------------------------
    def historical_data(self, instrument_key, interval="1minute", days=6):
        """
        Uses v2 historical-candle-data endpoint.
        instrument_key: e.g. 'NSE_INDEX|Nifty 50'
        interval: '1minute' -> I1, '5minute' -> I5 etc.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        interval_map = {
            "1minute": "I1",
            "5minute": "I5",
            "15minute": "I15",
        }
        upstox_interval = interval_map.get(interval, "I1")

        url = "https://api.upstox.com/v2/historical-candle-data"
        params = {
            "instrument_key": instrument_key,
            "interval": upstox_interval,
            "to_date": end_time.isoformat(timespec="seconds"),
            "from_date": start_time.isoformat(timespec="seconds"),
        }
        r = requests.get(url, headers=self._headers(), params=params)
        r.raise_for_status()
        data = r.json()
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        rows = []
        for c in data["data"]["candles"]:
            rows.append(c)
        df = pd.DataFrame(rows, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ---------- LTP -------------------------------------------------------
    def get_ltp(self, instrument_key):
        """
        Uses v2 LTP endpoint.
        instrument_key: e.g. 'NSE_INDEX|Nifty 50'
        """
        url = "https://api.upstox.com/v2/market-quote/ltp"
        params = {"instrument_key": instrument_key}
        r = requests.get(url, headers=self._headers(), params=params)
        r.raise_for_status()
        data = r.json()
        return data["data"][instrument_key]["ltp"]

    # ---------- order helpers (REAL, if you go live) ----------------------
    def place_order(self, *args, **kwargs):
        raise NotImplementedError("Real order placement not used in PAPER mode via BrokerClient.")

    def positions(self):
        raise NotImplementedError("Positions not needed in PAPER mode via BrokerClient.")

    # ---------- option symbol helper --------------------------------------
    def get_option_symbol(self, underlying, expiry, strike, right):
        """
        For now return a dummy instrument_key for paper trading.
        In live mode, look up real instrument_key via /v2/option/contract or instrument master.
        """
        return f"{underlying}|{expiry.strftime('%Y%m%d')}|{strike}{right}"


class PaperBroker:
    """
    Simple in-memory paper broker to simulate orders without hitting Upstox.
    Use this in your strategy when PAPER = True.
    """

    def __init__(self):
        self.orderbook = {}
        self._oid = 1

    def _next_oid(self):
        oid = f"PAPER-{self._oid}"
        self._oid += 1
        return oid

    def place_order(self, instrument_token, quantity, side):
        side = side.upper()
        oid = self._next_oid()
        print(f"[PAPER] Placing {side} {quantity} of {instrument_token}, oid={oid}")
        self.orderbook[oid] = {
            "symbol": instrument_token,
            "qty": quantity,
            "side": side,
            "time": datetime.now(),
        }
        return oid

    def exit_position(self, oid):
        if oid not in self.orderbook:
            print(f"[PAPER] No such oid in orderbook: {oid}")
            return
        pos = self.orderbook.pop(oid)
        exit_side = "SELL" if pos["side"] == "BUY" else "BUY"
        print(f"[PAPER] Exiting {pos['symbol']} {pos['qty']} with {exit_side}")
        exit_oid = self._next_oid()
        return exit_oid
