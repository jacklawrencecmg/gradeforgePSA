@echo off
REM GradeForge Deployment Script for Windows
REM This script deploys to Fly.io automatically

echo.
echo ========================================
echo   GradeForge Fly.io Deployment
echo ========================================
echo.

REM Check if flyctl is installed
where flyctl >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: flyctl (Fly.io CLI) not found!
    echo.
    echo Please install it first:
    echo https://github.com/superfly/flyctl/releases
    echo.
    echo Then run this script again.
    pause
    exit /b 1
)

echo [1/5] Checking Fly.io login status...
flyctl auth whoami >nul 2>nul
if %errorlevel% neq 0 (
    echo [2/5] Logging in to Fly.io...
    flyctl auth login
    if %errorlevel% neq 0 (
        echo ERROR: Failed to login to Fly.io
        pause
        exit /b 1
    )
) else (
    echo [2/5] Already logged in ✓
)

echo [3/5] Deploying to Fly.io...
flyctl deploy
if %errorlevel% neq 0 (
    echo ERROR: Deployment failed!
    pause
    exit /b 1
)

echo.
echo [4/5] Getting app URL...
for /f "tokens=*" %%i in ('flyctl info --json ^| findstr "hostname"') do set APP_URL=%%i
echo App deployed successfully!
echo.

echo [5/5] Next steps:
echo.
echo 1. Add Stripe keys (optional):
echo    flyctl secrets set STRIPE_SECRET_KEY=sk_test_...
echo    flyctl secrets set STRIPE_WEBHOOK_SECRET=whsec_...
echo    flyctl deploy
echo.
echo 2. Check logs for admin setup code:
echo    flyctl logs
echo.
echo 3. Visit your app and register admin account
echo.

echo ========================================
echo Deployment Complete! 🚀
echo ========================================
pause
