#!/usr/bin/env python3
"""
Railway-compatible Flask app with full LeRobot backend
"""
import os
import sys
from pathlib import Path

# Set up paths for imports
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Python path: {sys.path[:3]}")
print(f"Current dir: {os.getcwd()}")
print(f"Files in current dir: {os.listdir('.')[:10]}")

try:
    # Use the simple_deploy backend
    print("Importing simple deployment backend...")
    from simple_deploy import app, socketio
    print("✅ Successfully imported simple backend")
except ImportError as e:
    print(f"❌ Comprehensive backend import error: {e}")
    try:
        # Fallback to cloud_deploy
        print("Trying cloud_deploy fallback...")
        from cloud_deploy import app, socketio
        print("✅ Successfully imported cloud_deploy")
    except ImportError as e2:
        print(f"❌ Cloud deploy import error: {e2}")
        # Create minimal fallback app
        from flask import Flask, jsonify
        app = Flask(__name__)
        
        @app.route('/')
        def hello():
            return "Fallback app running - import failed"
        
        @app.route('/health')
        def health():
            return jsonify({'status': 'healthy', 'mode': 'fallback'})
        
        @app.route('/api/plugplay/start-installation', methods=['POST'])
        def install():
            return jsonify({'success': False, 'error': 'Backend import failed'})
        
        socketio = None

# For Railway compatibility
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Tune Robotics backend on port {port}")
    
    if socketio:
        # Use socketio.run for full SocketIO + Flask functionality
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=port,
            debug=False,
            use_reloader=False
        )
    else:
        # Fallback to basic Flask
        app.run(host='0.0.0.0', port=port, debug=False)