@echo off
setlocal

cd /d "%~dp0"

REM === CONFIGURATION ===
set VENV_DIR=..\.venv
set PYTHON=%VENV_DIR%\Scripts\python.exe
set FRONTEND_FILE=app.py
set BACKEND_APP=main:app
set PORT=8000

if not exist "%PYTHON%" (
    echo ❌ Python introuvable dans "%PYTHON%"
    pause
    exit /b
)

REM === Lancer FastAPI ===
echo 🚀 Lancement de FastAPI...
start "API" cmd /k "%PYTHON%" -m uvicorn %BACKEND_APP% --reload --port %PORT%

REM === Attente de quelques secondes ===
timeout /t 5 > nul

REM === Lancer Streamlit ===
echo 🌐 Lancement de Streamlit...
start "Streamlit" cmd /k "%PYTHON%" -m streamlit run %FRONTEND_FILE%

REM === Ouvrir le navigateur ===
timeout /t 2 > nul
start http://localhost:8501

endlocal
