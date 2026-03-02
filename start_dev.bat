@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo  Starting Akshara OCR Development Environment
echo ===================================================

:: Check prerequisites
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker could not be found. Please install Docker Desktop.
    exit /b 1
)

where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: npm could not be found. Please install Node.js.
    exit /b 1
)

:: Stop any existing running containers
echo Stopping any existing orphaned containers...
docker-compose down

:: Start Backend Services
echo Starting Backend Services (API, Postgres, Redis, Celery, Model^)...
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

:: Wait for services to initialize
echo Waiting for services to initialize...
timeout /t 5 /nobreak >nul

:: Show status of backend containers
echo.
echo --- Backend Service Status ---
docker-compose ps
echo ------------------------------
echo.

:: Start Frontend Dev Server
echo Starting Frontend Development Server...
if exist "frontend" (
    cd frontend
    
    :: Check if node_modules exists, if not, install dependencies
    if not exist "node_modules" (
        echo Installing frontend dependencies...
        call npm install
    )
    
    :: Run dev server
    echo Frontend server starting. Access the application at: http://localhost:5173
    call npm run dev
) else (
    echo ERROR: Frontend directory not found!
    exit /b 1
)
