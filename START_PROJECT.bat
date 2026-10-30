@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"

echo ======================================================
echo    FLIGHT DELAY RISK SERVICE  -  Askar Turanazarov
echo ======================================================
echo.

:: 1. FIND PYTHON 3.10
set "PY_CMD="

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :python_found
)

py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.10"
    goto :python_found
)

if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
    set "PY_CMD="%LocalAppData%\Programs\Python\Python310\python.exe""
    goto :python_found
)

echo [ERROR] Python not found. Please install Python 3.10 (the model file requires it).
pause
exit /b

:python_found
echo [INFO] Using interpreter: %PY_CMD%

:: 2. VIRTUAL ENVIRONMENT
if not exist "venv" (
    echo [INFO] Creating virtual environment 'venv'...
    %PY_CMD% -m venv venv
    call venv\Scripts\activate
    echo [INFO] Installing dependencies from requirements.txt...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo [OK] Environment ready.
) else (
    echo [INFO] Virtual environment 'venv' found, skipping installation.
    call venv\Scripts\activate
)

echo.
echo [INFO] Model: models\model_pipeline_main.pkl if present,
echo [INFO]        otherwise downloaded from Hugging Face Hub on first start.
echo ======================================================

:: 3. FIND A FREE PORT (starting from 8505)
set /a PORT=8505

:check_port
netstat -ano | findstr :%PORT% >nul
if %errorlevel% equ 0 (
    echo [INFO] Port %PORT% is busy, trying the next one...
    set /a PORT=%PORT%+1
    goto :check_port
)

echo [INFO] Starting Streamlit on http://127.0.0.1:%PORT%
echo ======================================================

streamlit run src/app.py --server.port=%PORT% --server.address=127.0.0.1

pause
