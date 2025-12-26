#!/usr/bin/env python3
"""
NISO – NIFTY Options Supertrend Scalper with Paper Logging.
"""

import os
import re  # noqa: F401
from datetime import datetime, timedelta  # noqa: F401
from pathlib import Path

# ======================================================================
# CONFIG & CREDENTIALS
# ======================================================================
from env import (
    UPSTOX_CLIENT_KEY,
    UPSTOX_CLIENT_SECRET,
)

PAPER = True
CLIENT_ID = UPSTOX_CLIENT_KEY
CLIENT_SECRET = UPSTOX_CLIENT_SECRET
REDIRECT_URI = UPSTOX_CLIENT_SECRET

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
TAG = "niso-supertrend-bot"

SL_PCT = 0.20
TG_PCT = 0.30
TRAIL_ATR_MULT = 1.5

PAPER_LOG_DIR = Path("paper_logs")
NIFTY_SPOT_INSTRUMENT = "NSE_INDEX|Nifty 50"
UPSTOX_API_VERSION = "2.0"


def get_access_token():
    """Read Upstox access token."""
    if not os.path.exists(ACCESS_TOKEN_FILE):
        raise RuntimeError(
            "Run Upstox auth script to create upstox_access_token.txt"
        )
    with open(ACCESS_TOKEN_FILE, "r", encoding="utf-8"):
        token
