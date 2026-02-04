@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo   Llama 3.3 70B Chat - Windows Startup
echo ========================================
echo.

:: Set defaults if not provided
if "%QUANT%"=="" set QUANT=Q4_K_M
if "%CTX%"=="" set CTX=2048
if "%GPU_LAYERS%"=="" set GPU_LAYERS=-1
if "%BATCH_SIZE%"=="" set BATCH_SIZE=512

:: Display configuration
echo Configuration:
echo   Quantization: %QUANT%
echo   Context:      %CTX% tokens
echo   GPU Layers:   %GPU_LAYERS%
echo   Batch Size:   %BATCH_SIZE%
if defined MODEL_PATH echo   Model Path:   %MODEL_PATH%
echo.

:: Check if backend venv exists
if not exist "backend\venv\Scripts\activate.bat" (
    echo ERROR: Backend virtual environment not found!
    echo Please run setup first. See README.md for instructions.
    echo.
    pause
    exit /b 1
)

:: Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo ERROR: Frontend dependencies not installed!
    echo Run: cd frontend ^&^& npm install
    echo.
    pause
    exit /b 1
)

:: Start backend server
echo Starting backend server...
cd backend
start "Llama Backend" cmd /c "call venv\Scripts\activate.bat && python -m uvicorn server:app --host 0.0.0.0 --port 8000"
cd ..

:: Wait for backend to be ready
echo Waiting for backend to initialize...
set BACKEND_READY=0
set ATTEMPTS=0
set MAX_ATTEMPTS=120

:wait_loop
if %ATTEMPTS% geq %MAX_ATTEMPTS% (
    echo ERROR: Backend failed to start within 5 minutes
    echo Check the backend window for errors.
    pause
    exit /b 1
)

timeout /t 3 /nobreak >nul
set /a ATTEMPTS+=1

:: Check if backend is responding
curl -s http://localhost:8000/api/models >nul 2>&1
if %ERRORLEVEL%==0 (
    set BACKEND_READY=1
    goto backend_ready
)

:: Show progress
set /a ELAPSED=ATTEMPTS*3
echo   Still loading... (%ELAPSED%s elapsed)
goto wait_loop

:backend_ready
echo.
echo Backend is ready!
echo.

:: Start frontend
echo Starting frontend...
cd frontend
start "Llama Frontend" cmd /c "npm run dev"
cd ..

echo.
echo ========================================
echo   Application Started!
echo ========================================
echo.
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo.
echo   Press Ctrl+C in each window to stop.
echo.

pause
