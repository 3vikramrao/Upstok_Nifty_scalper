#!/usr/bin/env python3
"""
Auto-login auth for Upstox API v2.
Flow:
1. Reads API key/secret, redirect URL, username, TOTP secret, PIN from env.py.
2. Uses Selenium to login (mobile/email -> TOTP -> PIN) on Upstox OAuth page.
3. Grabs ?code= from redirect URL.
4. Calls Upstox /login/authorization/token to get access_token.
5. Saves access_token into upstox_access_token.txt.
Run once per day before starting your trading bot.
"""
import urllib.parse
import time
import requests
from env import (
    UPSTOX_CLIENT_KEY,
    UPSTOX_CLIENT_SECRET,
    UPSTOX_REDIRECT_URI,
    UPSTOX_USERNAME,
    UPSTOX_TOTP_SECRET,
    UPSTOX_PIN,
)
import pyotp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
# ---------- BASIC VALIDATION ----------
CLIENT_ID = UPSTOX_CLIENT_KEY
CLIENT_SECRET = UPSTOX_CLIENT_SECRET
REDIRECT_URI = UPSTOX_REDIRECT_URI
USERNAME = UPSTOX_USERNAME
TOTP_SECRET = UPSTOX_TOTP_SECRET
PIN_CODE = UPSTOX_PIN
if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
    raise RuntimeError("Missing UPSTOX_CLIENT_KEY/SECRET/REDIRECT_URI in env.py")
if not USERNAME or not TOTP_SECRET or not PIN_CODE:
    raise RuntimeError("Missing UPSTOX_USERNAME / UPSTOX_TOTP_SECRET / UPSTOX_PIN in env.py")
# ---------- STEP 1: SELENIUM LOGIN TO GET CODE ----------
def get_auth_code():
    browser = None
    try:
        chrome_options = Options()
        # For debugging, keep window visible; later you can use headless if it works for you.
        # chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        service = Service(ChromeDriverManager().install())
        browser = webdriver.Chrome(service=service, options=chrome_options)
        browser.delete_all_cookies()
        auth_url = (
            "https://api.upstox.com/v2/login/authorization/dialog"
            f"?response_type=code&client_id={CLIENT_ID}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
        )
        print("Opening Upstox OAuth URL...")
        print(auth_url)
        browser.get(auth_url)
        wait = WebDriverWait(browser, 30)
        # 1) Enter mobile/email
        try:
            user_field = wait.until(
                EC.presence_of_element_located((By.ID, "mobileNum"))
            )
        except TimeoutException:
            user_field = wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, "//input[@type='text' or @inputmode='numeric']")
                )
            )
        user_field.clear()
        user_field.send_keys(USERNAME)
        print("Username/mobile entered")
        # 2) Click Continue/Get OTP
        cont_btn = wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[contains(.,'Continue') or contains(.,'Get OTP') or @id='getOtp']",
                )
            )
        )
        cont_btn.click()
        print("Clicked Continue / Get OTP")
        # 3) Enter TOTP (OTP)
        totp = pyotp.TOTP(TOTP_SECRET).now()
        time.sleep(2)
        otp_field = wait.until(
            EC.presence_of_element_located((By.ID, "otpNum"))
        )
        otp_field.clear()
        otp_field.send_keys(totp)
        print("TOTP entered")
        cont_otp_btn = wait.until(
            EC.element_to_be_clickable((By.ID, "continueBtn"))
        )
        cont_otp_btn.click()
        print("Clicked Continue after OTP")
        # 4) Enter PIN
        time.sleep(2)
        pin_field = wait.until(
            EC.presence_of_element_located((By.ID, "pinCode"))
        )
        pin_field.clear()
        pin_field.send_keys(PIN_CODE)
        print("PIN entered")
        pin_continue_btn = wait.until(
            EC.element_to_be_clickable((By.ID, "pinContinueBtn"))
        )
        pin_continue_btn.click()
        print("Clicked Continue after PIN")
        # 5) Wait for redirect with ?code=
        WebDriverWait(browser, 180).until(lambda d: "code=" in d.current_url)
        codelink = browser.current_url
        print("Redirect URL:", codelink)
        parsed = urllib.parse.urlparse(codelink)
        query = urllib.parse.parse_qs(parsed.query)
        if "code" not in query:
            raise RuntimeError("Authorization code not found in redirect URL")
        code = query["code"][0]
        print("Authorization Code:", code)
        return code
    except (TimeoutException, NoSuchElementException, WebDriverException) as e:
        print("❌ Selenium error while logging in:", repr(e))
        raise
    finally:
        if browser is not None:
            try:
                browser.quit()
            except Exception:
                pass
# ---------- STEP 2: EXCHANGE CODE FOR ACCESS TOKEN ----------
def exchange_code_for_access_token(auth_code: str) -> str:
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
    access_token = tokens.get("access_token")
    if not access_token:
        raise RuntimeError("No access_token in response; check the printed JSON above for error details.")
    return access_token
# ---------- STEP 3: MAIN ----------
def main():
    print("Starting auto Upstox auth (v2)...")
    auth_code = get_auth_code()
    access_token = exchange_code_for_access_token(auth_code)
    out_file = "upstox_access_token.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(access_token)
    print(f"✅ Access token saved to {out_file}.")
    print("=" * 55)
print(
    "Remember: Upstox access tokens expire daily, "
    "so rerun this script when needed."
)
print("=" * 55)
if __name__ == "__main__":
    main()
