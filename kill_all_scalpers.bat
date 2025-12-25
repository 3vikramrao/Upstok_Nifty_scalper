@echo off
cd /d "C:\Users\Dell\Documents\Kimi_Upstok_Nifty_scalper"
title Scalper KILLER
color 0C
echo ========================================
echo      💀 KILL ALL 9 SCALPER BOTS 💀
echo ========================================
echo.

REM Kill by exact window titles (cmd + python)
taskkill /fi "windowtitle eq HA-SCALPER V1" /f >nul 2>&1
taskkill /fi "windowtitle eq HA-SCALPER V2" /f >nul 2>&1
taskkill /fi "windowtitle eq NISO Scalper" /f >nul 2>&1
taskkill /fi "windowtitle eq niso-std" /f >nul 2>&1
taskkill /fi "windowtitle eq niso-3std" /f >nul 2>&1
taskkill /fi "windowtitle eq niso-ema915rsi9" /f >nul 2>&1
taskkill /fi "windowtitle eq niso-multi" /f >nul 2>&1
taskkill /fi "windowtitle eq niso-multi-beast" /f >nul 2>&1
taskkill /fi "windowtitle eq niso-ema520" /f >nul 2>&1

REM Nuclear option - kill ALL python processes (backup)
echo Killing remaining python.exe processes...
taskkill /im python.exe /f >nul 2>&1
taskkill /im pythonw.exe /f >nul 2>&1

REM Final cleanup - kill any lingering cmd windows with "Scalper" in title
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq cmd.exe" /fo csv ^| findstr Scalper') do (
    taskkill /pid %%i /f >nul 2>&1
)

echo.
echo ✅ ALL WINDOWS + PROCESSES KILLED!
echo ========================================
timeout /t 2 /nobreak >nul
echo Check Task Manager: Ctrl+Shift+Esc
pause
