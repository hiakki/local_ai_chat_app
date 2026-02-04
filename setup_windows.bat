@echo off
setlocal enabledelayedexpansion

echo.
echo =====================================================
echo   Llama 3.3 70B Chat - Windows Setup (NVIDIA CUDA)
echo =====================================================
echo.

:: Check Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check Node.js
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Node.js not found!
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

:: Check NVIDIA GPU
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: nvidia-smi not found. CUDA may not be available.
    echo Make sure you have NVIDIA drivers installed.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i "!CONTINUE!" neq "y" exit /b 1
) else (
    echo NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo.
)

echo ========================================
echo   Setting up Backend...
echo ========================================
cd backend

:: Create virtual environment
if exist venv (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

:: Activate venv and install dependencies
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing llama-cpp-python with CUDA support...
echo This may take 5-15 minutes depending on your setup.
echo.

:: Try pre-built wheel first (faster)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 2>nul
if %ERRORLEVEL% neq 0 (
    echo Pre-built wheel not available, building from source...
    set CMAKE_ARGS=-DGGML_CUDA=on
    pip install llama-cpp-python --force-reinstall --no-cache-dir
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Failed to install llama-cpp-python
    echo.
    echo Please ensure you have:
    echo   1. CUDA Toolkit 12.x installed
    echo   2. Visual Studio Build Tools with C++ workload
    echo   3. CMake installed
    echo.
    pause
    exit /b 1
)

echo Installing other dependencies...
pip install -r requirements.txt

call deactivate
cd ..

echo.
echo ========================================
echo   Setting up Frontend...
echo ========================================
cd frontend

echo Installing npm dependencies...
call npm install

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install frontend dependencies
    pause
    exit /b 1
)

cd ..

echo.
echo =====================================================
echo   Setup Complete!
echo =====================================================
echo.
echo To start the application:
echo   .\start_windows.bat
echo.
echo To use a specific model:
echo   set MODEL_PATH=C:\path\to\models
echo   set QUANT=IQ2_XXS
echo   set GPU_LAYERS=20
echo   .\start_windows.bat
echo.
echo See README.md for GPU-specific recommendations.
echo.

pause
