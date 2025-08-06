#!/usr/bin/env python3
"""
WSGI entry point for production deployment
"""

from simple_deploy import app, socketio

if __name__ == "__main__":
    # For development
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)