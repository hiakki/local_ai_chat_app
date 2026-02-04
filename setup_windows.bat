@echo off
setlocal enabledelayedexpansion

echo.
echo =====================================================
echo   Llama 3.3 70B Chat - Windows Setup (NVIDIA CUDA)
echo =====================================================
echo.

:: ========================================
:: Check Prerequisites
:: ========================================
echo Checking prerequisites...
echo.

:: Check Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [X] Python - NOT FOUND
    echo     Install from: https://www.python.org/downloads/
    echo     Make sure to check "Add Python to PATH" during installation
    set MISSING_DEPS=1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do echo [OK] Python %%i
)

:: Check Node.js
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [X] Node.js - NOT FOUND
    echo     Install from: https://nodejs.org/
    set MISSING_DEPS=1
) else (
    for /f "tokens=1" %%i in ('node --version 2^>^&1') do echo [OK] Node.js %%i
)

:: Check NVIDIA Driver
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [X] NVIDIA Driver - NOT FOUND
    echo     Install from: https://www.nvidia.com/Download/index.aspx
    set MISSING_DEPS=1
) else (
    for /f "tokens=3" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>^&1') do echo [OK] NVIDIA Driver
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

:: Check CUDA Toolkit (nvcc compiler)
nvcc --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [X] CUDA Toolkit - NOT FOUND
    echo     Install CUDA 12.x from: https://developer.nvidia.com/cuda-downloads
    echo     After install, add to PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
    set MISSING_CUDA=1
) else (
    for /f "tokens=5" %%i in ('nvcc --version 2^>^&1 ^| findstr release') do echo [OK] CUDA Toolkit %%i
)

:: Check CMake
cmake --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [X] CMake - NOT FOUND
    echo     Install from: https://cmake.org/download/
    echo     Or run: winget install Kitware.CMake
    set MISSING_DEPS=1
) else (
    for /f "tokens=3" %%i in ('cmake --version 2^>^&1 ^| findstr version') do echo [OK] CMake %%i
)

:: Check Visual Studio Build Tools (cl.exe)
where cl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [X] Visual Studio Build Tools - NOT FOUND or not in PATH
    echo     Install "Desktop development with C++" from:
    echo     https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo     IMPORTANT: After installing, run this script from
    echo     "Developer Command Prompt for VS" or "x64 Native Tools Command Prompt"
    set MISSING_VS=1
) else (
    echo [OK] Visual Studio Build Tools ^(cl.exe found^)
)

echo.

:: Exit if missing critical dependencies
if defined MISSING_DEPS (
    echo =====================================================
    echo   ERROR: Missing required dependencies!
    echo   Please install the missing components above.
    echo =====================================================
    pause
    exit /b 1
)

if defined MISSING_VS (
    echo =====================================================
    echo   WARNING: Visual Studio Build Tools not in PATH
    echo.
    echo   You have two options:
    echo   1. Run this script from "x64 Native Tools Command Prompt"
    echo      ^(Search for it in Start Menu after installing VS Build Tools^)
    echo.
    echo   2. Try installing pre-built wheel ^(no compilation needed^)
    echo      We'll attempt this first...
    echo =====================================================
    echo.
    set TRY_PREBUILT=1
)

if defined MISSING_CUDA (
    echo =====================================================
    echo   WARNING: CUDA Toolkit not found
    echo   Will try pre-built CUDA wheel instead...
    echo =====================================================
    echo.
    set TRY_PREBUILT=1
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
echo ========================================
echo Installing llama-cpp-python with CUDA...
echo ========================================
echo.

:: Try pre-built CUDA wheel first (no compilation needed!)
echo Attempting to install pre-built CUDA wheel...
echo This is the easiest method - no build tools required!
echo.

pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
if %ERRORLEVEL%==0 (
    echo.
    echo [OK] Successfully installed pre-built CUDA wheel!
    goto :install_success
)

echo.
echo Pre-built wheel failed. Trying alternative wheel sources...
echo.

:: Try CUDA 12.1 wheel
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
if %ERRORLEVEL%==0 (
    echo.
    echo [OK] Successfully installed pre-built CUDA 12.1 wheel!
    goto :install_success
)

:: Try jllllll's wheels (community builds)
echo.
echo Trying community pre-built wheels...
pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu124
if %ERRORLEVEL%==0 (
    echo.
    echo [OK] Successfully installed community CUDA wheel!
    goto :install_success
)

:: If pre-built wheels all failed, try building from source
echo.
echo =====================================================
echo Pre-built wheels not available for your configuration.
echo Attempting to build from source...
echo =====================================================
echo.

if defined MISSING_VS (
    echo ERROR: Cannot build from source without Visual Studio Build Tools!
    echo.
    echo Please install Visual Studio Build Tools:
    echo   1. Go to: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo   2. Download and run the installer
    echo   3. Select "Desktop development with C++" workload
    echo   4. Install and restart your computer
    echo   5. Open "x64 Native Tools Command Prompt for VS 2022"
    echo   6. Navigate to this directory and run setup_windows.bat again
    echo.
    pause
    exit /b 1
)

if defined MISSING_CUDA (
    echo ERROR: Cannot build from source without CUDA Toolkit!
    echo.
    echo Please install CUDA Toolkit 12.x:
    echo   1. Go to: https://developer.nvidia.com/cuda-downloads
    echo   2. Select Windows ^> x86_64 ^> Your Windows version
    echo   3. Download and install
    echo   4. Add CUDA to PATH:
    echo      set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin;%%PATH%%
    echo   5. Restart command prompt and run setup_windows.bat again
    echo.
    pause
    exit /b 1
)

echo Building llama-cpp-python from source with CUDA...
echo This will take 10-20 minutes...
echo.

set CMAKE_ARGS=-DGGML_CUDA=on
set FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --no-cache-dir --verbose

if %ERRORLEVEL% neq 0 (
    echo.
    echo =====================================================
    echo   ERROR: Failed to build llama-cpp-python
    echo =====================================================
    echo.
    echo Common fixes:
    echo.
    echo 1. Run from "x64 Native Tools Command Prompt for VS 2022"
    echo    ^(Search for it in Start Menu^)
    echo.
    echo 2. Make sure CUDA is in PATH:
    echo    set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%%PATH%%
    echo.
    echo 3. Try installing without CUDA ^(CPU-only, slower^):
    echo    pip install llama-cpp-python
    echo.
    echo 4. Check GitHub issues:
    echo    https://github.com/abetlen/llama-cpp-python/issues
    echo.
    pause
    exit /b 1
)

:install_success
echo.
echo [OK] llama-cpp-python installed successfully!

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
