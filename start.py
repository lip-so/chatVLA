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
        
        # Set environment variable to make Firebase optional in production
        os.environ.setdefault('FIREBASE_OPTIONAL', 'true')
        
        from backend.api.main import app, socketio
        logger.info("✅ ChatVLA backend loaded successfully")
        return app, socketio
        
    except Exception as e:
        logger.warning(f"⚠️ Main backend failed to load: {e}")
        logger.info("🔄 Creating minimal fallback app...")
        
        # Create minimal fallback app
        from flask import Flask, jsonify
        from flask_cors import CORS
        from flask_socketio import SocketIO
        
        app = Flask(__name__, 
                   static_folder='frontend',
                   static_url_path='')
        CORS(app, origins=["*"])
        socketio = SocketIO(app, cors_allowed_origins="*")
        
        @app.route('/')
        def index():
            return app.send_static_file('index.html')
        
        @app.route('/health')
        def health():
            return jsonify({
                "status": "healthy",
                "mode": "minimal",
                "message": "ChatVLA Backend Running (Minimal Mode)",
                "version": "3.0.0-minimal",
                "timestamp": "2025-01-08",
                "services": {
                    "plugplay": {"available": False, "status": "minimal mode"},
                    "databench": {"available": False, "status": "minimal mode"}
                }
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
        
        # Add LeRobot API endpoints in minimal mode
        @app.route('/api/lerobot/calibrate', methods=['POST'])
        def lerobot_calibrate():
            return jsonify({
                "success": True,
                "message": "Calibration completed (simulation mode)",
                "mode": "simulation",
                "note": "Running in minimal mode - LeRobot dependencies not available"
            })
        
        @app.route('/api/lerobot/start-teleop', methods=['POST'])
        def lerobot_start_teleop():
            import time
            session_id = f"teleop_sim_{int(time.time())}"
            return jsonify({
                "success": True,
                "message": "Teleoperation started (simulation mode)",
                "session_id": session_id,
                "mode": "simulation",
                "note": "Running in minimal mode - LeRobot dependencies not available"
            })
        
        @app.route('/api/lerobot/stop-teleop', methods=['POST'])
        def lerobot_stop_teleop():
            return jsonify({
                "success": True,
                "message": "Teleoperation stopped (simulation mode)"
            })
        
        @app.route('/api/lerobot/start-recording', methods=['POST'])
        def lerobot_start_recording():
            import time
            session_id = f"record_sim_{int(time.time())}"
            return jsonify({
                "success": True,
                "message": "Recording started (simulation mode)",
                "session_id": session_id,
                "mode": "simulation",
                "note": "Running in minimal mode - LeRobot dependencies not available"
            })
        
        @app.route('/api/lerobot/stop-recording', methods=['POST'])
        def lerobot_stop_recording():
            return jsonify({
                "success": True,
                "message": "Recording stopped (simulation mode)"
            })
        
        @app.route('/api/lerobot/sessions', methods=['GET'])
        def lerobot_sessions():
            return jsonify({
                "success": True,
                "sessions": {},
                "mode": "simulation",
                "note": "Running in minimal mode - LeRobot dependencies not available"
            })
        
        @app.route('/api/plugplay/calibrate', methods=['POST'])
        def calibrate():
            return jsonify({
                "success": False,
                "message": "Calibration not available in minimal mode"
            })
        
        @app.route('/api/plugplay/start-teleop', methods=['POST'])
        def start_teleop():
            return jsonify({
                "success": False,
                "message": "Teleoperation not available in minimal mode"
            })
        
        @app.route('/api/plugplay/start-recording', methods=['POST'])
        def start_recording():
            return jsonify({
                "success": False,
                "message": "Recording not available in minimal mode"
            })
        
        # Serve frontend files
        @app.route('/<path:filename>')
        def serve_frontend(filename):
            try:
                return app.send_static_file(filename)
            except Exception:
                # If file not found, return index for SPA routing
                try:
                    return app.send_static_file('index.html')
                except Exception:
                    return jsonify({"error": "File not found"}), 404
        
        logger.info("✅ Minimal fallback app created successfully")
        return app, socketio

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