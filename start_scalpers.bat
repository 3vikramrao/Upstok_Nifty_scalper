@echo off
cd /d "C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper"
title Scalper Launcher
color 0A
echo ========================================
echo    SCALPER BOTS LAUNCHER (Post-Auth)
echo ========================================
echo 🚀 LAUNCHING 9 SCALPER BOTS...
echo Make sure you ran: python auth.py first!
echo.
start "HA-V1" cmd /k "cd /d C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper && title HA-SCALPER V1 && python ha_scalper_v1.py"
start "HA-V2" cmd /k "cd /d C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper && title HA-SCALPER V2 && python ha_scalper_v2.py"
start "niso-ema520" cmd /k "cd /d C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper && title NISO Scalper && python niso-ema520.py"
start "niso-std" cmd /k "cd /d C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper && title niso-std && python niso-std.py"
start "niso-3std" cmd /k "cd /d C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper && title niso-3std && python niso-3std.py"
start "niso-ema915rsi9" cmd /k "cd /d C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper && title niso-ema915rsi9 && python niso-ema915rsi9.py"
start "niso-ema915rsi9std" cmd /k "cd /d C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper && title niso-ema915rsi9 && python niso-ema915rsi9std.py"
start "niso-multi" cmd /k "cd /d C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper && title niso-multi && python niso-multi.py"
start "niso-multi-beast" cmd /k "cd /d C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper && title niso-multi-beast && python niso-multi-beast.py"
echo.
echo ✅ ALL 9 BOTS LAUNCHED!
echo ========================================
echo Close this window anytime.
pause
