@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"

echo ======================================================
echo                Askar Turanazarov
echo ======================================================
echo    СИСТЕМА ПРОГНОЗИРОВАНИЯ ЗАДЕРЖЕК АВИАРЕЙСОВ
echo ======================================================
echo.

:: 1. ПОИСК PYTHON
set "PY_CMD="

:: Проверка 1: Команда python (PATH)
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :python_found
)

:: Проверка 2: Команда py (в CMD Windows)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py"
    goto :python_found
)

:: Проверка 3: Прямой путь в AppData
if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
    set "PY_CMD="%LocalAppData%\Programs\Python\Python310\python.exe""
    goto :python_found
)

:: Если ничего не нашли
echo [ОШИБКА] Python не найден! 
echo Пожалуйста, убедитесь, что Python 3.10 установлен.
pause
exit /b

:python_found
echo [INFO] Используется интерпретатор: %PY_CMD%

:: 2. РАБОТА С ВИРТУАЛЬНЫМ ОКРУЖЕНИЕМ
if not exist "venv" (
    echo [INFO] Папка 'venv' не найдена. Создание нового окружения...
    %PY_CMD% -m venv venv
    
    echo [INFO] Активация среды и установка библиотек...
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo [SUCCESS] Окружение создано и библиотеки установлены.
) else (
    echo [INFO] Виртуальное окружение 'venv' уже существует. 
    echo [INFO] Пропускаем установку, переходим к запуску...
    call venv\Scripts\activate
)

echo.
echo [INFO] Запуск веб-интерфейса Streamlit...
echo [INFO] "Аналитическая система прогнозирования 
echo [INFO] авиарисков - риск задержки авиарейса"
echo [INFO]                 by Askar Turanazarov
echo ======================================================

:: 3. ПОИСК СВОБОДНОГО ПОРТА (начиная с 8505)
set /a PORT=8505

:check_port
netstat -ano | findstr :%PORT% >nul
if %errorlevel% equ 0 (
    echo [INFO] Порт %PORT% занят, пробую %PORT%+1...
    set /a PORT=%PORT%+1
    goto :check_port
)

echo [INFO] Запуск веб-интерфейса на порту: %PORT%
echo ======================================================

:: ЗАПУСК
streamlit run src/app.py --server.port=%PORT% --server.address=127.0.0.1

pause