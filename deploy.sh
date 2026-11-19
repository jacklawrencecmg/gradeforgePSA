#!/bin/bash
# GradeForge Deployment Script for Mac/Linux
# This script deploys to Fly.io automatically

echo ""
echo "========================================"
echo "  GradeForge Fly.io Deployment"
echo "========================================"
echo ""

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "ERROR: flyctl (Fly.io CLI) not found!"
    echo ""
    echo "Please install it first:"
    echo "curl -L https://fly.io/install.sh | sh"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "[1/5] Checking Fly.io login status..."
if ! flyctl auth whoami &> /dev/null; then
    echo "[2/5] Logging in to Fly.io..."
    flyctl auth login
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to login to Fly.io"
        exit 1
    fi
else
    echo "[2/5] Already logged in ✓"
fi

echo "[3/5] Deploying to Fly.io..."
flyctl deploy
if [ $? -ne 0 ]; then
    echo "ERROR: Deployment failed!"
    exit 1
fi

echo ""
echo "[4/5] Getting app URL..."
APP_URL=$(flyctl info --json | grep -o '"hostname":"[^"]*' | cut -d'"' -f4)

echo ""
echo "========================================"
echo "✅ Deployment Complete! 🚀"
echo "========================================"
echo ""
echo "App URL: https://$APP_URL"
echo ""
echo "[5/5] Next steps:"
echo ""
echo "1. Add Stripe keys (optional):"
echo "   flyctl secrets set STRIPE_SECRET_KEY=sk_test_..."
echo "   flyctl secrets set STRIPE_WEBHOOK_SECRET=whsec_..."
echo "   flyctl deploy"
echo ""
echo "2. Check logs for admin setup code:"
echo "   flyctl logs"
echo ""
echo "3. Visit your app and register admin account:"
echo "   https://$APP_URL"
echo ""
