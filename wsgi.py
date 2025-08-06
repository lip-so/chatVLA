#!/usr/bin/env python3
"""
WSGI entry point for production deployment
This file is referenced by Gunicorn in Procfile and deployment configurations
"""

import os
import sys

# Import the initialized app and socketio from simple_deploy
from simple_deploy import app, socketio, application

# Ensure the application is properly exported for WSGI servers
# Gunicorn expects 'app' or 'application' as the callable
if socketio:
    # Use SocketIO's WSGI app for WebSocket support
    app = application
else:
    # Fallback to regular Flask app if SocketIO isn't available
    app = app if app else application

# Development server (not used in production)
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    if socketio:
        # Run with SocketIO for WebSocket support
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    else:
        # Run standard Flask app
        app.run(host='0.0.0.0', port=port, debug=False)