#!/usr/bin/env python3
"""
One-time (per day) auth for Upstox → saves access-token locally.
Run this when your access token expires.
"""

import os
import webbrowser
import requests

# ---------- CONFIG LOADING ----------

# Option A: from env.py (Python module in same folder)
# Create env.py with:
# UPSTOX_API_KEY = "your_key_here"
# UPSTOX_API_SECRET = "your_secret_here"
# REDIRECT_URI = "http://localhost:5000/"
# import your secrets from env.py (same folder)
from env import (
    UPSTOX_CLIENT_KEY,
    UPSTOX_CLIENT_SECRET,
    UPSTOX_REDIRECT_URI,
)
# this imports env.py from the current directory

CLIENT_ID = UPSTOX_CLIENT_KEY
CLIENT_SECRET = UPSTOX_CLIENT_SECRET
REDIRECT_URI = UPSTOX_REDIRECT_URI

# If you prefer environment variables instead of env.py, use this instead:
# API_KEY = os.getenv("UPSTOX_API_KEY")
# API_SECRET = os.getenv("UPSTOX_API_SECRET")
# REDIRECT_URI = os.getenv("REDIRECT_URI")

if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
    raise RuntimeError(
        "Missing UPSTOX_API_KEY, UPSTOX_API_SECRET, or REDIRECT_URI. "
        "Set them in env.py or environment variables."
    )

# ---------- STEP 1: OPEN LOGIN PAGE ----------

auth_url = (
    "https://api.upstox.com/v2/login/authorization/dialog"
    f"?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
)
print("Opening browser (login to Upstox, approve, then copy the full URL from the address bar)...")
print("If the browser does not open, paste this into your browser manually:\n")
print(auth_url)
print()

webbrowser.open(auth_url)

# ---------- STEP 2: READ REDIRECT URL AND EXTRACT CODE ----------

auth_code = None
while True:
    url = input("Paste the FULL redirect URL here:\n").strip()
    if "code=" in url:
        try:
            auth_code = url.split("code=")[1].split("&")[0]
        except IndexError:
            auth_code = None
    if auth_code:
        break
    print("URL does not contain a valid code= parameter - please copy again.\n")

print(f"Using auth code: {auth_code}")

# ---------- STEP 3: EXCHANGE CODE FOR TOKENS ----------

token_url = "https://api.upstox.com/v2/login/authorization/token"

headers = {
    "accept": "application/json",
    "Api-Version": "2.0",
    "Content-Type": "application/x-www-form-urlencoded",
}

data = {
    "code": auth_code,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri": REDIRECT_URI,
    "grant_type": "authorization_code",
}

resp = requests.post(token_url, headers=headers, data=data)

print("\nToken endpoint status:", resp.status_code)
print("Token endpoint body:", resp.text, "\n")

resp.raise_for_status()
tokens = resp.json()

# Typical response includes access_token / expires_at etc., but NOT refresh_token.[web:12][web:70][web:65]
access_token = tokens.get("access_token")
if not access_token:
    raise RuntimeError("No access_token in response; check the printed JSON above for error details.")

# ---------- STEP 4: SAVE ACCESS TOKEN LOCALLY ----------

OUT_FILE = "upstox_access_token.txt"
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(access_token)

print(f"✅ Access token saved to {OUT_FILE}.")
print("=======================================================")
print("Remember: Upstox access tokens expire daily (around 3:30 AM), so rerun this script when needed.")
print("Now you can run your trading script that uses BrokerClient.")
print("=======================================================")