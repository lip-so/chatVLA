"""
WSGI entry point for main Flask app production deployment
Includes DataBench, Plug & Play, and Firebase Authentication
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('PYTHONPATH', str(PROJECT_ROOT))

# Import the main app and socketio
from backend.api.main import app, socketio

# Create the application instance for WSGI servers
application = app

if __name__ == "__main__":
    # This is only for development
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)