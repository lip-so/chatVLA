#!/bin/bash

echo "🧪 Testing LeRobot Installation Flow"
echo "===================================="
echo ""

# Kill any existing bridge
echo "🔄 Restarting local installer bridge..."
pkill -f "python.*local_installer_bridge.py" 2>/dev/null
sleep 1

# Start fresh bridge
cd ~/chatVLA
python3 local_installer_bridge.py &
BRIDGE_PID=$!

# Wait for it to start
echo "⏳ Waiting for bridge to start..."
sleep 3

# Check if it's running
if curl -s http://localhost:7777/health > /dev/null 2>&1; then
    echo "✅ Bridge is running!"
else
    echo "❌ Bridge failed to start"
    exit 1
fi

# Open the website
echo "🌐 Opening website..."
open "https://tunerobotics.xyz/frontend/pages/plug-and-play-databench-style.html"

echo ""
echo "===================================="
echo "📋 TEST INSTRUCTIONS:"
echo "===================================="
echo "1. Click 'Install LeRobot'"
echo "2. Watch it jump to 100% (already installed)"
echo "3. ✅ Should AUTO-SCROLL to port detection"
echo "4. ✅ Port detection should START automatically"
echo ""
echo "If both happen → Installation flow is WORKING!"
echo "===================================="
echo ""
echo "Press Ctrl+C to stop the bridge when done testing."

# Keep running
wait $BRIDGE_PID