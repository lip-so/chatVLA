#!/bin/bash

echo "🚀 IMMEDIATE BACKEND DEPLOYMENT FIX"
echo "==================================="
echo ""

# Kill any existing servers
pkill -f "python backend_server.py" 2>/dev/null
pkill -f "python start.py" 2>/dev/null
sleep 1

echo "Starting local backend server..."
PORT=8080 python backend_server.py &
SERVER_PID=$!
sleep 2

# Test local server
echo ""
echo "Testing local backend..."
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo "✅ Backend is running locally on port 8080"
else
    echo "❌ Failed to start backend"
    exit 1
fi

echo ""
echo "==================================="
echo "OPTION 1: Deploy to Glitch (FREE & INSTANT)"
echo "==================================="
echo ""
echo "1. Go to: https://glitch.com/edit/#!/import/github/lip-so/chatVLA"
echo "2. Wait for import (30 seconds)"
echo "3. Click 'Show' → 'In a New Window'"
echo "4. Copy the URL (e.g., https://YOUR-PROJECT.glitch.me)"
echo "5. Run: ./update_backend_url.sh https://YOUR-PROJECT.glitch.me"
echo ""

echo "==================================="
echo "OPTION 2: Deploy to Replit (FREE & INSTANT)"
echo "==================================="
echo ""
echo "1. Go to: https://replit.com/github/lip-so/chatVLA"
echo "2. Click 'Import from GitHub'"
echo "3. Click 'Run' button"
echo "4. Copy the URL from the webview"
echo "5. Run: ./update_backend_url.sh YOUR_REPLIT_URL"
echo ""

echo "==================================="
echo "OPTION 3: Use Temporary Local Backend"
echo "==================================="
echo ""
echo "For immediate testing, update config.js to use localhost:"
echo ""
echo "1. Edit: frontend/js/config.js"
echo "2. Line 26: return 'http://localhost:8080';"
echo "3. Open: http://localhost:8080/health"
echo "4. Your backend is working!"
echo ""

echo "==================================="
echo "OPTION 4: Fix Railway Domain"
echo "==================================="
echo ""
echo "If Railway is deployed but no domain:"
echo ""
echo "1. Go to: https://railway.app/dashboard"
echo "2. Click your project → Click service"
echo "3. Go to Settings → Networking"
echo "4. If you see 'TCP Proxy', DELETE it"
echo "5. Click 'Generate Domain'"
echo "6. Copy the URL"
echo "7. Run: ./update_backend_url.sh YOUR_URL"
echo ""

echo "Your local backend is running on http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

# Keep running
wait $SERVER_PID