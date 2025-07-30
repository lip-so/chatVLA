#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from flask import Flask, Blueprint, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from datetime import datetime

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import the API modules
from databench.api import DataBenchEvaluator, METRIC_CODES, METRIC_NAMES
from plug_and_play.api import (
    PlugPlayInstallationManager, 
    USBPortDetector,
    DATABENCH_AVAILABLE,
    SERIAL_AVAILABLE
)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../../frontend',
            static_url_path='')
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize managers
databench_evaluator = DataBenchEvaluator()
installation_manager = PlugPlayInstallationManager()
usb_detector = USBPortDetector()

# Create blueprints
databench_bp = Blueprint('databench', __name__, url_prefix='/api/databench')
plugplay_bp = Blueprint('plugplay', __name__, url_prefix='/api/plugplay')

# ============================================================================
# Main Routes
# ============================================================================

@app.route('/')
def serve_index():
    """Serve the main index.html"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/pages/<path:filename>')
def serve_page(filename):
    """Serve pages from the pages directory"""
    return send_from_directory(os.path.join(app.static_folder, 'pages'), filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files"""
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files"""
    return send_from_directory(os.path.join(app.static_folder, 'js'), filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve asset files"""
    return send_from_directory(os.path.join(app.static_folder, 'assets'), filename)

@app.route('/health')
def health():
    """Unified health check endpoint"""
    return jsonify({
        "status": "healthy",
        "services": {
            "databench": {
                "available": DATABENCH_AVAILABLE,
                "status": "ready" if DATABENCH_AVAILABLE else "unavailable"
            },
            "plug_and_play": {
                "available": True,
                "serial_available": SERIAL_AVAILABLE
            }
        },
        "timestamp": datetime.now().isoformat()
    })

# ============================================================================
# DataBench Routes
# ============================================================================

@databench_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get available DataBench metrics"""
    metrics = {}
    for code, name in METRIC_NAMES.items():
        metrics[code] = {
            "name": name,
            "code": METRIC_CODES[code],
            "description": f"Evaluate {name.lower()} of robotics datasets"
        }
    return jsonify({"metrics": metrics})

@databench_bp.route('/evaluate', methods=['POST'])
def evaluate_dataset():
    """Run DataBench evaluation"""
    try:
        from flask import request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        result, status_code = databench_evaluator.run_evaluation(data)
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# Plug & Play Routes
# ============================================================================

@plugplay_bp.route('/system-info', methods=['GET'])
def system_info():
    """Get system information"""
    return jsonify({
        "os": sys.platform,
        "python_version": sys.version,
        "capabilities": {
            "usb_detection": SERIAL_AVAILABLE,
            "installation": True
        }
    })

@plugplay_bp.route('/list-ports', methods=['GET'])
def list_ports():
    """List available USB ports"""
    if not SERIAL_AVAILABLE:
        return jsonify({
            "error": "USB port detection not available",
            "ports": []
        }), 503
        
    ports = usb_detector.scan_ports()
    return jsonify({"ports": ports})

@plugplay_bp.route('/start-installation', methods=['POST'])
def start_installation():
    """Start LeRobot installation"""
    try:
        from flask import request
        data = request.get_json()
        
        result = installation_manager.start_installation(
            install_path=data.get('installPath', './lerobot'),
            selected_port=data.get('selectedPort')
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@plugplay_bp.route('/installation-status', methods=['GET'])
def installation_status():
    """Get installation status"""
    status = installation_manager.get_status()
    return jsonify(status)

@plugplay_bp.route('/cancel-installation', methods=['POST'])
def cancel_installation():
    """Cancel installation"""
    return jsonify({
        "success": True,
        "message": "Installation cancelled",
        "status": "cancelled"
    })

# Register blueprints
app.register_blueprint(databench_bp)
app.register_blueprint(plugplay_bp)

# ============================================================================
# WebSocket Events (for Plug & Play)
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Tune Robotics server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('request_installation_update')
def handle_installation_update():
    """Send installation status update"""
    status = installation_manager.get_status()
    emit('installation_update', status)

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print(f"""
    🚀 Tune Robotics Unified API Server
    =====================================
    Services:
    - DataBench: {'Available' if DATABENCH_AVAILABLE else 'Unavailable'}
    - Plug & Play: Available
    - USB Detection: {'Available' if SERIAL_AVAILABLE else 'Limited'}
    
    Starting server on http://localhost:{port}
    """)
    
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        socketio.run(app, host='0.0.0.0', port=port)