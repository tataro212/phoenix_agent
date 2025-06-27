@echo off
title Phoenix Agent Interactive Launcher
echo.
echo ========================================
echo   Phoenix Agent Interactive Launcher
echo ========================================
echo.
echo Starting Phoenix Agent in interactive mode...
echo This will open file dialogs to help you select:
echo - Input PDF file to translate
echo - Output directory for results  
echo - Target language for translation
echo.
echo ========================================
echo.

REM Try to run with Python
python run_interactive.py

REM If that fails, try python3
if errorlevel 1 (
    echo Trying with python3...
    python3 run_interactive.py
)

REM If that also fails, show error
if errorlevel 1 (
    echo.
    echo ========================================
    echo   ERROR: Python not found or not working
    echo ========================================
    echo.
    echo Please make sure Python is installed and in your PATH.
    echo You can download Python from: https://python.org
    echo.
    echo Also make sure all required dependencies are installed:
    echo pip install -r requirements.txt
    echo.
)

echo.
echo Press any key to exit...
pause >nul 