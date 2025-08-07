#!/usr/bin/env python3
"""
LeRobot Backend Server - UPDATED FOR PRODUCTION
This file was causing the deployment issue!
"""

import os
import sys
import logging

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.dirname(__file__))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("🚀 Loading LeRobot backend from backend_server.py...")
    from backend.api.main import app, socketio
    logger.info("✅ LeRobot backend loaded successfully from backend_server.py")
    
except Exception as e:
    logger.error(f"❌ Failed to load LeRobot backend from backend_server.py: {e}")
    import traceback
    traceback.print_exc()
    
    # Create fallback app
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])
    
    @app.route('/')
    def index():
        return jsonify({
            "status": "error",
            "message": "LeRobot backend failed to load from backend_server.py",
            "version": "2.0.0-error",
            "error": str(e)
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        })
    
    socketio = None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting LeRobot backend server on port {port}")
    
    if socketio:
        logger.info("Starting with SocketIO support")
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    else:
        logger.info("Starting without SocketIO (fallback mode)")
        app.run(host='0.0.0.0', port=port, debug=False)