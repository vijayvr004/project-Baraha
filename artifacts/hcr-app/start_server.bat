@echo off
title Baraha -- Handwriting Recognition Server
cls
echo ============================================================
echo  BARAHA  Handwritten Document Recognition
echo  Engine: Google Gemini Vision (95%+ accuracy)
echo ============================================================
echo.

set GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
set GROQ_API_KEY=YOUR_GROQ_API_KEY_HERE
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HCR_PYTHON=C:\Users\varsh\.conda\envs\hcr-env\python.exe

echo  [OK] API keys configured
echo.

echo  [*] Stopping any existing server on port 8008...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8008 "') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo  [*] Starting server with hcr-env Python 3.10...
echo  [*] Open browser at: http://127.0.0.1:8008
echo.
"%HCR_PYTHON%" main.py
pause
