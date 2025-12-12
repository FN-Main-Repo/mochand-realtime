@echo off
REM Quick start script for Google Meet Voice Agent
echo.
echo ================================================
echo   Google Meet Voice Agent - Quick Start
echo ================================================
echo.

REM Check if venv is activated
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call venv\Scripts\activate.ps1
)

echo.
echo [1/4] Checking ngrok...
where ngrok >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: ngrok not found!
    echo Please install ngrok from https://ngrok.com/download
    echo.
    pause
    exit /b 1
)

echo [2/4] Starting ngrok in background...
start "ngrok" cmd /c "ngrok http 8765"
timeout /t 3 /nobreak >nul

echo [3/4] Waiting for ngrok to start...
timeout /t 2 /nobreak >nul

echo.
echo ================================================
echo   IMPORTANT: UPDATE .env.local
echo ================================================
echo.
echo 1. Check the ngrok window that just opened
echo 2. Copy the 'Forwarding' URL (https://xxxxx.ngrok.io)
echo 3. Open .env.local file
echo 4. Set WEBSOCKET_PUBLIC_URL=wss://xxxxx.ngrok.io
echo    (Change https:// to wss://)
echo.
echo Press any key once you've updated .env.local...
pause >nul

echo.
echo [4/4] Ready to start!
echo.
echo Usage:
echo   python src\meet_agent.py https://meet.google.com/your-meeting-code
echo.
echo Example:
echo   python src\meet_agent.py https://meet.google.com/abc-defg-hij
echo.
pause
