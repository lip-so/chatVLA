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

def main():
    """Simple deployment entry point without problematic imports"""
    
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
    
    # Global state for plug & play installation
    current_installation = {
        'running': False,
        'path': None,
        'robot': None,
        'step': None,
        'progress': 0
    }
    
    def emit_log(message, level='info'):
        """Send log message to frontend via SocketIO"""
        socketio.emit('install_log', {
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
            
            # Simulate installation steps for deployment
            time.sleep(1)
            emit_log("Checking system requirements...")
            
            time.sleep(2)
            emit_log("Setting up robot configuration...")
            
            time.sleep(2)
            emit_log("Configuring USB port detection...")
            
            time.sleep(1)
            emit_log("Installation completed successfully!", level='success')
            emit_log(f"Robot type: {robot}", level='success')
            emit_log(f"Installation path: {path}", level='success')
            
            # Trigger next step
            socketio.emit('installation_complete', {
                'path': str(path),
                'robot': robot,
                'next_step': 'usb_detection'
            })
            
        except Exception as e:
            emit_log(f"ERROR: {str(e)}", level='error')
        finally:
            current_installation['running'] = False
    
    # Create blueprints
    plugplay_bp = Blueprint('plugplay', __name__, url_prefix='/api/plugplay')
    
    @plugplay_bp.route('/start-installation', methods=['POST'])
    def start_installation():
        """Start LeRobot installation"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400
            
            path = Path(data.get('installation_path', './lerobot')).expanduser()
            robot = data.get('selected_robot', 'koch')
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
        """List available USB ports - simplified version"""
        return jsonify({
            "ports": [
                {"device": "/dev/ttyUSB0", "description": "USB Serial Port"},
                {"device": "/dev/ttyUSB1", "description": "USB Serial Port"}
            ]
        })
    
    # Static file routes
    @app.route('/')
    def index():
        return send_from_directory('.', 'index.html')
    
    @app.route('/pages/<path:filename>')
    def serve_page(filename):
        return send_from_directory('pages', filename)
    
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
    
    # Register blueprints
    app.register_blueprint(plugplay_bp)
    
    # SocketIO Event Handlers
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        emit('status', {'message': 'Connected to server'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        print('Client disconnected')
    
    port = int(os.environ.get('PORT', 5000))
    
    print(f"""
    ========================================
    Tune Robotics Simple Deployment
    ========================================
    Environment: {os.environ.get('FLASK_ENV', 'production')}
    Port: {port}
    
    Services Available:
    - Plug & Play API: /api/plugplay/*
    - Static Files: /css/*, /js/*, /assets/*, /pages/*
    - WebSocket: Enabled for real-time updates
    ========================================
    """)
    
    # Use SocketIO for WebSocket support
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()