@echo off
echo ========================================
echo Phoenix Agent - Clean Environment Setup
echo ========================================
echo.

echo Step 1: Creating clean conda environment...
conda create --name phoenix_prod python=3.10 -y

if %errorlevel% neq 0 (
    echo ❌ Failed to create conda environment
    pause
    exit /b 1
)

echo ✅ Conda environment created successfully
echo.

echo Step 2: Activating environment...
call conda activate phoenix_prod

if %errorlevel% neq 0 (
    echo ❌ Failed to activate conda environment
    pause
    exit /b 1
)

echo ✅ Environment activated
echo.

echo Step 3: Installing requirements...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

echo ✅ Requirements installed successfully
echo.

echo Step 4: Testing Nougat functionality...
python test_nougat_station1.py

if %errorlevel% neq 0 (
    echo ❌ Nougat test failed
    echo.
    echo 💡 Troubleshooting tips:
    echo    1. Check that all packages installed correctly
    echo    2. Try running: pip install --upgrade pip
    echo    3. Check your internet connection for model downloads
    pause
    exit /b 1
)

echo.
echo ========================================
echo 🎉 Setup completed successfully!
echo ========================================
echo.
echo 💡 Next steps:
echo    1. Try running the full pipeline:
echo       python phoenix_orchestrator.py --interactive
echo    2. Or test with a specific PDF:
echo       python phoenix_orchestrator.py your_document.pdf
echo.
echo 💡 To activate this environment in the future:
echo    conda activate phoenix_prod
echo.
pause 