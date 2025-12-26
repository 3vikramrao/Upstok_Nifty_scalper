# Upstox Nifty Scalper Bot

Automated scalping framework for Nifty options using Upstox API with 15+ strategies including HA Scalper, NISO variants, and multi-bot systems.

## 🚀 Features

- **15+ Scalping Strategies**: HA Scalper v1/v2, NISO-3STD, EMA 5/20, RSI divergence, VWAP, Volume Profile
- **Paper Trading**: Full backtest + live paper mode with CSV logs
- **Multi-Bot Master**: Run multiple strategies simultaneously
- **Upstox API Integration**: Auto-authentication and order execution
- **Batch Scripts**: Windows `.bat` files for start/kill scalpers


## 📁 File Structure

```
Upstok_Nifty_scalper/
├── core_bot.py           # Main bot engine
├── core_bot_master.py    # Multi-strategy runner
├── ha_scalper_v1.py      # Heikin Ashi scalper
├── ha_scalper_v2.py      # Advanced HA strategy
├── niso-*.py             # NISO variants (3std, ema520, rsi9, etc.)
├── strategy/             # Strategy modules
│   ├── ema_crossover.py
│   ├── vp_profile.py
│   └── multi_bot.py
├── broker_client.py      # Upstox API wrapper
├── env.py                # Environment config
├── requirements.txt      # Dependencies
├── *.bat                 # Windows launchers
└── config.yaml           # Trading parameters
```


## ⚙️ Quick Setup

```bash
git clone https://github.com/3vikramrao/Upstok_Nifty_scalper.git
cd Upstok_Nifty_scalper

# Install dependencies
pip install -r requirements.txt

# Setup Upstox API credentials in .env
# UPSTOX_API_KEY=your_key
# UPSTOX_ACCESS_TOKEN=your_token

# Paper trading (no real money)
python live_paper_bot.py

# Live trading
start_scalpers.bat
```


## 📊 Strategies Overview

| Strategy | Indicators | Timeframe | Risk Profile |
| :-- | :-- | :-- | :-- |
| HA Scalper v1/v2 | Heikin Ashi | 1-3min | Low-Medium |
| NISO-3STD | Bollinger + StdDev | 1min | Medium |
| EMA 9/15 + RSI9 | EMA Cross + RSI | 1min | Medium |
| VWAP Scalper | VWAP + Volume | 3min | Low |
| Volume Profile | VP + Price Action | 1min | Medium-High |

## 🛡️ Security Notes

- ✅ Credentials removed (use `.env` file)
- ✅ No API tokens committed
- ✅ Paper logs excluded from repo


## 📈 Paper Trading Results

Backtest logs show performance across multiple Nifty sessions. Check `paper_logs/` locally for CSV analysis.

## 🔧 Usage

```bash
# Kill all running bots
kill_scalpers.bat

# Start specific strategy
python niso-3std.py

# Master mode (all strategies)
python core_bot_master.py
```


## 📞 Support

- **Issues**: Create GitHub issue
- **Discord/Telegram**: Check README updates
- **Upstox API Docs**: [Upstox Developer](https://upstox.com/developer/api-documentation)

***

**⚠️ Disclaimer**: Trading involves risk. Test thoroughly in paper mode. Past performance ≠ future results. Not financial advice.

***


