#!/usr/bin/env python3
"""
Production Entry Point for ChatVLA Backend
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

def create_app():
    """Create and configure the Flask app"""
    try:
        logger.info("🚀 Loading ChatVLA backend...")
        from backend.api.main import app, socketio
        logger.info("✅ ChatVLA backend loaded successfully")
        return app, socketio
        
    except ImportError as e:
        logger.warning(f"⚠️ Main backend not available: {e}")
        logger.info("🔄 Creating minimal fallback app...")
        
        # Create minimal fallback app
        from flask import Flask, jsonify
        from flask_cors import CORS
        
        app = Flask(__name__, 
                   static_folder='frontend',
                   static_url_path='')
        CORS(app, origins=["*"])
        
        @app.route('/')
        def index():
            return jsonify({
                "status": "success", 
                "message": "ChatVLA Backend Running (Minimal Mode)",
                "version": "3.0.0-minimal",
                "timestamp": "2025-01-08"
            })
        
        @app.route('/health')
        def health():
            return jsonify({
                "status": "healthy",
                "mode": "minimal"
            })
            
        @app.route('/api/test')
        def api_test():
            return jsonify({
                "status": "success",
                "message": "✅ API Test Endpoint Working (Minimal Mode)",
                "timestamp": "2025-01-08"
            })
        
        # Add plug & play endpoints for minimal mode
        @app.route('/api/plugplay/list-ports', methods=['GET'])
        def list_ports():
            return jsonify({
                "success": True,
                "ports": [
                    {"port": "/dev/ttyUSB0", "description": "USB Serial Device", "role": "leader"},
                    {"port": "/dev/ttyUSB1", "description": "USB Serial Device", "role": "follower"}
                ],
                "message": "Simulated port detection (minimal mode)"
            })
        
        # Serve frontend files
        @app.route('/<path:filename>')
        def serve_frontend(filename):
            try:
                return app.send_static_file(filename)
            except Exception:
                # If file not found, return index for SPA routing
                return app.send_static_file('index.html')
        
        return app, None
        
    except Exception as e:
        logger.error(f"❌ Failed to create app: {e}")
        import traceback
        traceback.print_exc()
        raise

# Create the app
app, socketio = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}")
    
    if socketio:
        logger.info("Starting with SocketIO support")
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    else:
        logger.info("Starting in basic Flask mode")
        app.run(host='0.0.0.0', port=port, debug=False)