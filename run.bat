@echo off
REM ============================================================
REM  Quant & Data Science Master Portfolio — Windows Launcher
REM ============================================================
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║  Quant ^& Data Science Master Portfolio           ║
echo  ║  Self-Configuring Windows Launcher               ║
echo  ╚══════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

REM Check for python
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PY_CMD=py
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set PY_CMD=python
    ) else (
        echo [ERROR] Python is not found in PATH. Please install Python 3.10+ from python.org
        pause
        exit /b 1
    )
)

REM Setup virtual environment if not present
if not exist ".venv" (
    echo [INFO] Creating virtual environment (.venv)...
    %PY_CMD% -m venv .venv
)

REM Activate virtualenv
call .venv\Scripts\activate.bat

REM Install dependencies
echo [INFO] Installing / updating requirements...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

REM Launch Streamlit and open browser
echo [INFO] Starting Streamlit server...
start http://localhost:8501
streamlit run app.py --server.headless true --server.port 8501

pause
