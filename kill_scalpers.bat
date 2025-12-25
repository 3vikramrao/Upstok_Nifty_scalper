@echo off
cd /d "C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper"
title Scalper KILLER
color 0C
echo ========================================
echo      ⚠️  KILL ALL 9 SCALPER BOTS ⚠️
echo ========================================
echo.

taskkill /fi "windowtitle eq HA-SCALPER V1*" /f
taskkill /fi "windowtitle eq HA-SCALPER V2*" /f
taskkill /fi "windowtitle eq NISO Scalper*" /f
taskkill /fi "windowtitle eq niso-std*" /f
taskkill /fi "windowtitle eq niso-3std*" /f
taskkill /fi "windowtitle eq niso-ema915rsi9*" /f
taskkill /fi "windowtitle eq niso-multi*" /f
taskkill /fi "windowtitle eq niso-multi-beast*" /f
taskkill /fi "windowtitle eq niso-ema520*" /f

echo.
echo ✅ ALL 9 SCALPER BOTS TERMINATED!
echo ========================================
echo Check Task Manager if any python.exe remain.
pause
