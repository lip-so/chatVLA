#!/usr/bin/env python3
"""
Main entry point for Railway deployment
"""

import os
import sys
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Importing LeRobot backend...")
    from backend.api.main import app, socketio
    logger.info("✅ Successfully imported LeRobot backend")
    
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting LeRobot backend on port {port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    
except Exception as e:
    logger.error(f"Failed to import LeRobot backend: {e}")
    
    try:
        logger.info("Falling back to simple_deploy...")
        from simple_deploy import app
        logger.info("✅ Fallback successful")
        
        if __name__ == '__main__':
            port = int(os.environ.get('PORT', 5000))
            logger.info(f"Starting fallback backend on port {port}")
            app.run(host='0.0.0.0', port=port, debug=False)
            
    except Exception as e2:
        logger.error(f"Fallback also failed: {e2}")
        
        # Last resort: minimal app
        from flask import Flask, jsonify
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            return jsonify({
                "status": "error",
                "message": "Backend failed to load",
                "errors": [str(e), str(e2)]
            })
            
        @app.route('/health')
        def health():
            return jsonify({"status": "error", "message": "Backend failed to load"})
            
        if __name__ == '__main__':
            port = int(os.environ.get('PORT', 5000))
            app.run(host='0.0.0.0', port=port, debug=False)
