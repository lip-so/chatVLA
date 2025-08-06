#!/usr/bin/env python3
"""
WSGI entry point for production deployment
This file is referenced by Gunicorn in Procfile and deployment configurations
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Import the application from simple_deploy
    from simple_deploy import app as flask_app, socketio
    
    # Create the WSGI application
    # For Gunicorn with eventlet, we just need the Flask app
    # The eventlet worker will handle WebSocket support
    logger.info("Imported Flask application successfully")
    app = flask_app
    
    # Ensure app is callable
    if not callable(app):
        logger.error("App is not callable, creating fallback")
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            return "Application is running in minimal mode", 200
        
        @app.route('/health')
        def health():
            return {"status": "ok", "mode": "minimal"}, 200
            
except Exception as e:
    logger.error(f"Failed to import application: {e}")
    # Create minimal fallback app
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return f"Application failed to start: {str(e)}", 500
    
    @app.route('/health')
    def health():
        return {"status": "error", "message": str(e)}, 500

# Make sure app is the exported WSGI callable
application = app

# Development server (not used in production)
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)