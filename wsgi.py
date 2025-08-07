#!/usr/bin/env python3
"""
Simplified WSGI entry point for Railway deployment
"""

import os
import sys
import logging

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get port from environment
PORT = int(os.environ.get('PORT', 8080))
logger.info(f"Starting on port {PORT}")

try:
    # Import the proper LeRobot backend
    from backend.api.main import app, socketio
    logger.info("Successfully imported LeRobot backend from backend.api.main")
    application = app
    
except Exception as e:
    logger.error(f"Failed to import backend.api.main: {e}")
    
    try:
        # Fallback to simple deploy
        from simple_deploy import app
        logger.info("Fallback: imported app from simple_deploy")
        application = app
        
    except Exception as e2:
        logger.error(f"Also failed to import simple_deploy: {e2}")
        
        # Create a minimal working app
        from flask import Flask, jsonify, request
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app, origins=["*"])
        
        @app.route('/')
        def index():
            return jsonify({
                "status": "running",
                "message": "ChatVLA Backend - Minimal Mode",
                "error": f"Backend import failed: {e}, Simple deploy failed: {e2}"
            })
        
        @app.route('/health')
        def health():
            return jsonify({"status": "healthy", "mode": "minimal"})
        
        @app.route('/api/databench/evaluate', methods=['POST', 'OPTIONS'])
        def databench_stub():
            if request.method == 'OPTIONS':
                # Handle CORS preflight
                return '', 204
            return jsonify({
                "error": "DataBench not available in minimal mode",
                "message": "Backend is running but full features not loaded"
            }), 503
        
        application = app

# For direct execution
if __name__ == "__main__":
    logger.info(f"Starting Flask app on port {PORT}")
    application.run(host='0.0.0.0', port=PORT, debug=False)