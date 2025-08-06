#!/bin/bash

echo "🚀 TESTING RAILWAY DEPLOYMENT DIRECTLY"
echo "======================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Enter your Railway URL (or press Enter to skip):"
echo "Example: https://chatvla-production-xxxx.up.railway.app"
read -r RAILWAY_URL

if [ -z "$RAILWAY_URL" ]; then
    echo ""
    echo -e "${YELLOW}No URL provided. Here's how to get it:${NC}"
    echo ""
    echo "1. Go to: https://railway.app/dashboard"
    echo "2. Click your 'chatVLA' project"
    echo "3. Click on your service"
    echo "4. The URL should be at the top, or:"
    echo "   - Click Variables tab"
    echo "   - Add: PORT = 8080"
    echo "   - Save and wait for redeploy"
    echo "   - URL will appear at top of service"
    echo ""
    echo "5. Run this script again with your URL"
    exit 0
fi

echo ""
echo "Testing Railway backend at: $RAILWAY_URL"
echo "----------------------------------------"

# Test health endpoint
echo -n "Testing /health endpoint... "
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "$RAILWAY_URL/health" 2>/dev/null)
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
BODY=$(echo "$HEALTH_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✅ SUCCESS!${NC}"
    echo "Response: $BODY"
    echo ""
    echo -e "${GREEN}🎉 YOUR RAILWAY BACKEND IS WORKING!${NC}"
    echo ""
    echo "Now update your frontend to use this URL:"
    echo "./update_backend_url.sh $RAILWAY_URL"
elif [ "$HTTP_CODE" = "000" ]; then
    echo -e "${RED}❌ FAILED - Cannot connect${NC}"
    echo ""
    echo "This means either:"
    echo "1. The URL is wrong"
    echo "2. Railway deployment crashed"
    echo "3. No public domain was generated"
    echo ""
    echo "Check Railway dashboard for deployment status"
else
    echo -e "${YELLOW}⚠️  HTTP $HTTP_CODE${NC}"
    echo "Response: $BODY"
    echo ""
    echo "Backend is responding but may have issues"
fi

echo ""
echo "----------------------------------------"
echo "NEXT STEPS:"
echo ""
echo "If test was SUCCESSFUL:"
echo "  ./update_backend_url.sh $RAILWAY_URL"
echo ""
echo "If test FAILED:"
echo "  1. Check Railway dashboard"
echo "  2. Look at deployment logs"
echo "  3. Make sure PORT=8080 is in Variables"
echo "  4. Ensure deployment status is 'Active'"