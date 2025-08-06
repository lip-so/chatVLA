#!/bin/bash

echo "🤖 LeRobot Web Installer Bridge"
echo "================================"
echo ""
echo "This enables REAL LeRobot installation from the website!"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "Please install Python 3 first"
    exit 1
fi

# Check for required packages
echo "Checking dependencies..."
python3 -c "import flask" 2>/dev/null || {
    echo "Installing Flask..."
    pip3 install flask flask-cors flask-socketio pyserial
}

# Start the local installer bridge
echo ""
echo "✅ Starting local installer bridge on port 7777..."
echo ""
echo "============================================"
echo "KEEP THIS TERMINAL OPEN!"
echo "The website can now install LeRobot on your computer"
echo "============================================"
echo ""

cd "$(dirname "$0")"
python3 local_installer_bridge.py