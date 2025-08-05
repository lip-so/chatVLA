#!/usr/bin/env python3
"""
Railway-compatible Flask app with full LeRobot backend
"""
import os
import sys
from pathlib import Path

# Set up paths for imports
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import the full working API
from backend.plug_and_play.working_api import app, socketio

# For Railway compatibility
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Tune Robotics full backend on port {port}")
    
    # Use socketio.run for full SocketIO + Flask functionality
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=port,
        debug=False,
        use_reloader=False
    )