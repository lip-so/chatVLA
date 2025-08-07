#!/bin/bash

# One command to enable REAL LeRobot installation from the website

echo "🤖 LeRobot Web Installer"
echo "========================"
echo ""

# Check if chatVLA is already cloned
if [ ! -d "$HOME/chatVLA" ]; then
    echo "📦 Downloading chatVLA..."
    cd ~
    git clone https://github.com/lip-so/chatVLA.git
fi

cd ~/chatVLA

# Install dependencies
echo "📚 Installing dependencies..."
pip3 install flask flask-cors flask-socketio pyserial eventlet --quiet

# Start the local bridge
echo "🚀 Starting installation bridge..."
python3 local_installer_bridge.py &
BRIDGE_PID=$!

# Wait for it to start
sleep 3

# Open the website
echo "🌐 Opening website..."
open "https://tunerobotics.xyz/frontend/pages/plug-and-play-databench-style.html"

echo ""
echo "========================================="
echo "✅ READY! Click 'Install LeRobot' on the website!"
echo "========================================="
echo ""
echo "The website can now install LeRobot on your computer."
echo "Keep this terminal open until installation completes."
echo ""

# Keep running
wait $BRIDGE_PID