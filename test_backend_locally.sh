#!/bin/bash

echo "🧪 Testing Backend Locally"
echo "=========================="
echo ""

# Kill any existing Python servers on common ports
echo "Cleaning up old processes..."
pkill -f "python start.py" 2>/dev/null
pkill -f "python wsgi.py" 2>/dev/null
sleep 1

# Start the backend
echo "Starting backend on port 5000..."
PORT=5000 python start.py &
SERVER_PID=$!

sleep 3

# Test the backend
echo ""
echo "Testing health endpoint..."
HEALTH=$(curl -s http://localhost:5000/health)

if [[ $HEALTH == *"healthy"* ]]; then
    echo "✅ Backend is working locally!"
    echo "Response: $HEALTH"
    echo ""
    echo "You can test DataBench locally by:"
    echo "1. Opening: http://localhost:5000/health"
    echo "2. The backend is running on http://localhost:5000"
    echo ""
    echo "To connect your GitHub Pages site to Railway:"
    echo "1. Get your Railway URL from https://railway.app/dashboard"
    echo "2. Run: ./update_backend_url.sh YOUR_RAILWAY_URL"
    echo ""
    echo "Press Ctrl+C to stop the local server"
    
    # Keep server running
    wait $SERVER_PID
else
    echo "❌ Backend failed to start"
    kill $SERVER_PID 2>/dev/null
fi