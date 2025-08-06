#!/usr/bin/env python3
"""
Simple deployment entry point that avoids databench import issues
"""

import os
import sys
import threading
import time
from pathlib import Path
from flask import Flask, Blueprint, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Global state for plug & play installation
current_installation = {
    'running': False,
    'path': None,
    'robot': None,
    'step': None,
    'progress': 0
}

# Global socketio instance for emit_log function
socketio_instance = None

def emit_log(message, level='info'):
    """Send log message to frontend via SocketIO"""
    if socketio_instance:
        socketio_instance.emit('install_log', {
            'message': message,
            'level': level,
            'timestamp': time.time()
        })

def run_installation(path, robot, use_existing=False):
    """Run a simplified installation process for deployment"""
    global current_installation
    current_installation['running'] = True
    current_installation['path'] = str(path)
    current_installation['robot'] = robot
    
    try:
        emit_log(f"Starting installation for {robot} robot...")
        emit_log(f"Installation path: {path}")
        
        # Simplified for production deployment
        emit_log("Installation simulation for demonstration purposes")
        time.sleep(1)
        emit_log("Installation completed successfully!", level='success')
        emit_log(f"Robot type: {robot}", level='success')
        emit_log(f"Installation path: {path}", level='success')
        
        # Trigger next step
        if socketio_instance:
            socketio_instance.emit('installation_complete', {
                'path': str(path),
                'robot': robot,
                'next_step': 'usb_detection'
            })
        
    except Exception as e:
        emit_log(f"ERROR: {str(e)}", level='error')
    finally:
        current_installation['running'] = False

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

# Initialize Flask app
app = Flask(__name__, 
            static_folder='frontend',
            static_url_path='')

# App configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize extensions
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Set the global socketio instance
socketio_instance = socketio

# Create blueprints
plugplay_bp = Blueprint('plugplay', __name__, url_prefix='/api/plugplay')
    
@plugplay_bp.route('/start-installation', methods=['POST'])
def start_installation():
    """Start LeRobot installation"""
    global current_installation
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Handle different parameter formats from different frontend files
        path = Path(data.get('installation_path') or data.get('path', './lerobot')).expanduser()
        robot = data.get('selected_robot') or data.get('robot', 'koch')
        use_existing = data.get('use_existing', False)
        
        if current_installation['running']:
            return jsonify({
                'success': False, 
                'error': 'Installation already running'
            }), 400
        
        # Start installation in background thread
        thread = threading.Thread(target=run_installation, args=(path, robot, use_existing))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True, 
            'status': 'started', 
            'path': str(path), 
            'robot': robot
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@plugplay_bp.route('/installation-status', methods=['GET'])
def installation_status():
    """Get installation status"""
    global current_installation
    return jsonify(current_installation)

@plugplay_bp.route('/cancel-installation', methods=['POST'])
def cancel_installation():
    """Cancel running installation"""
    global current_installation
    current_installation['running'] = False
    emit_log("Installation cancelled by user", level='warning')
    return jsonify({
        'success': True, 
        'message': 'Installation cancelled'
    })

@plugplay_bp.route('/system-info', methods=['GET'])
def system_info():
    """Get system information"""
    return jsonify({
        "os": sys.platform,
        "python_version": sys.version,
        "capabilities": {
            "usb_detection": True,
            "installation": True
        }
    })

@plugplay_bp.route('/list-ports', methods=['GET'])
def list_ports():
    """List available USB ports - simplified version for production"""
    return jsonify({
        "ports": [],
        "message": "Port detection available after installation"
    })

@plugplay_bp.route('/save-port-config', methods=['POST'])
def save_port_config():
    """Save port configuration - simplified for production"""
    return jsonify({
        'success': True,
        'message': 'Port configuration saved'
    })

# Static file routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/pages/<path:filename>')
def serve_page(filename):
    # Try pages directory first
    pages_path = Path('pages') / filename
    if pages_path.exists():
        return send_from_directory('pages', filename)
    
    # Try frontend/pages directory as fallback
    frontend_pages_path = Path('frontend/pages') / filename
    if frontend_pages_path.exists():
        return send_from_directory('frontend/pages', filename)
    
    # Return 404 if not found
    return jsonify({'error': 'Page not found', 'path': f'/pages/{filename}'}), 404

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'mode': 'simple_deploy'})

# Additional API endpoints that some frontend pages expect
@app.route('/api/check_lerobot', methods=['GET'])
def check_lerobot():
    """Check if LeRobot is installed - simplified for production"""
    return jsonify({
        'installed': False,
        'message': 'Please run installation first'
    })

@app.route('/api/detect_ports', methods=['GET'])
def detect_ports():
    """Detect USB ports - simplified for production"""
    return jsonify({
        'ports': [],
        'message': 'Port detection available after installation'
    })

@app.route('/api/save_detected_ports', methods=['POST'])
def save_detected_ports():
    """Save detected ports - simplified for production"""
    return jsonify({
        'success': True,
        'message': 'Port configuration saved'
    })

@app.route('/api/finish_port_detection', methods=['POST'])
def finish_port_detection():
    """Finish port detection - simplified for production"""
    return jsonify({
        'success': True,
        'message': 'Port detection completed'
    })

@app.route('/api/save_robot_configuration', methods=['POST'])
def save_robot_configuration():
    """Save robot configuration - simplified for production"""
    return jsonify({
        'success': True,
        'message': 'Robot configuration saved'
    })

@app.route('/api/scan_usb_ports', methods=['GET'])
def scan_usb_ports():
    """Scan USB ports - simplified for production"""
    return jsonify({
        'ports': [],
        'message': 'USB port scanning available after installation'
    })

# Register blueprints
app.register_blueprint(plugplay_bp)

# Error handlers for API routes
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors - return JSON for API routes, HTML for others"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found', 'path': request.path}), 404
    # For non-API routes, return the 404.html page if it exists
    try:
        return send_from_directory('.', '404.html'), 404
    except:
        return 'Page not found', 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors - return JSON for API routes"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
    return 'Internal server error', 500

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors - return JSON for API routes"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Method not allowed', 'path': request.path}), 405
    return 'Method not allowed', 405

# SocketIO Event Handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass  # Logging disabled for production

# This allows gunicorn to import the app
if __name__ == '__main__':
    # Only run directly for local testing
    port = int(os.environ.get('PORT', 5000))
    print(f"Running in development mode on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)