#!/usr/bin/env python3
"""
LeRobot Backend Entry Point for Railway Production
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
    logger.info("🚀 Loading LeRobot backend...")
    from backend.api.main import app, socketio
    logger.info("✅ LeRobot backend loaded successfully")
    
except Exception as e:
    logger.error(f"❌ Failed to load LeRobot backend: {e}")
    import traceback
    traceback.print_exc()
    
    # Create minimal fallback app
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app, origins=["*"])
    
    @app.route('/')
    def index():
        return jsonify({
            "status": "error",
            "message": "LeRobot backend failed to load",
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
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    
    if socketio:
        logger.info("Starting with SocketIO support")
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    else:
        logger.info("Starting without SocketIO (fallback mode)")
        app.run(host='0.0.0.0', port=port, debug=False)