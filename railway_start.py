#!/usr/bin/env python3
"""
Railway startup script - simpler and more reliable
"""
import os
import sys
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variables
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)
os.environ['FLASK_ENV'] = 'production'

# Import and run the application
from simple_deploy import app, socketio

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Tune Robotics on port {port}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Working directory: {os.getcwd()}")
    
    # Use socketio.run for proper WebSocket support
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=port,
        debug=False,
        use_reloader=False
    )