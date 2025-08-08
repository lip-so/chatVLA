"""
WSGI entry point for production deployment
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('PYTHONPATH', str(PROJECT_ROOT))

try:
    # Import the main app and socketio
    from backend.api.main import app, socketio
    
    # Create the application instance for WSGI servers
    application = app
    
except Exception as e:
    # Fallback to minimal app if main backend fails
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return jsonify({
            "status": "error",
            "message": "Backend failed to load",
            "error": str(e)
        })
        
    @app.route('/health')
    def health():
        return jsonify({"status": "error", "message": "Backend failed to load"})
    
    application = app

if __name__ == "__main__":
    # This is only for development
    port = int(os.environ.get('PORT', 5000))
    if 'socketio' in locals():
        socketio.run(application, host='0.0.0.0', port=port)
    else:
        application.run(host='0.0.0.0', port=port)