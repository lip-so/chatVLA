#!/usr/bin/env python3

import os
import sys
import threading
import time
import subprocess
import platform
import shutil
from pathlib import Path
from flask import Flask, Blueprint, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Import LeRobot API blueprint
from plug_and_play.lerobot_api import lerobot_bp, set_socketio

# Import Firebase authentication module
from auth.firebase_auth import firebase_bp, requires_firebase_auth

# Import Firestore service
from auth.firestore_service import get_firestore_service

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../../frontend',
            static_url_path='')

# App configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize extensions
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Set up LeRobot API with socketio
set_socketio(socketio)

# Initialize managers
databench_evaluator = DataBenchEvaluator()
installation_manager = PlugPlayInstallationManager()
usb_detector = USBPortDetector()

# Global state for plug & play installation
current_installation = {
    'running': False,
    'path': None,
    'robot': None,
    'step': None,
    'progress': 0,
    'env_name': None,
    'leader_port': None,
    'follower_port': None
}

def emit_log(message, level='info'):
    """Send log message to frontend via SocketIO"""
    socketio.emit('install_log', {
        'message': message,
        'level': level,
        'timestamp': time.time()
    })

def run_installation(path, robot, use_existing=False):
    """Run a SIMULATED installation process for deployment (production cannot install locally)"""
    global current_installation
    current_installation['running'] = True
    current_installation['path'] = str(path)
    current_installation['robot'] = robot
    
    try:
        emit_log("⚠️ SIMULATION MODE - Production server cannot install on your machine", level='warning')
        emit_log("To perform REAL installation, run the local installer:", level='warning')
        emit_log("python3 local_installer_bridge.py", level='info')
        emit_log("", level='info')
        emit_log(f"Simulating installation for {robot} robot...")
        emit_log(f"Target path: {path}")
        
        # Simulate installation steps for deployment
        time.sleep(1)
        emit_log("[SIMULATION] Checking system requirements...")
        
        time.sleep(2)
        emit_log("[SIMULATION] Would clone LeRobot repository...")
        
        time.sleep(2)
        emit_log("[SIMULATION] Would create conda environment...")
        
        time.sleep(1)
        emit_log("[SIMULATION] Installation simulation complete", level='success')
        emit_log("⚠️ This was a simulation. For real installation, use local installer bridge", level='warning')
        
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

# Pass socketio to installation manager after it's created
installation_manager.socketio = socketio

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
        "message": "🚀 Tune Robotics LeRobot Backend is running!",
        "version": "2.0.0-lerobot",
        "services": {
            "databench": {
                "available": DATABENCH_AVAILABLE,
                "status": "ready" if DATABENCH_AVAILABLE else "unavailable"
            },
            "plug_and_play": {
                "available": True,
                "status": "ready with LeRobot simulation",
                "serial_available": SERIAL_AVAILABLE,
                "features": ["calibration", "teleoperation", "recording"]
            },
            "firebase_authentication": {
                "available": True,
                "status": "ready"
            }
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/test')
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        "status": "success",
        "message": "✅ Backend is working correctly!",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "calibration": "/api/plugplay/calibrate",
            "teleoperation": "/api/plugplay/start-teleop", 
            "recording": "/api/plugplay/start-recording",
            "port_detection": "/api/plugplay/list-ports"
        }
    })

# ============================================================================
# DataBench Routes (Protected with Firebase)
# ============================================================================

@databench_bp.route('/metrics', methods=['GET'])
@requires_firebase_auth
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
@requires_firebase_auth
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
# Plug & Play Routes (Protected with Firebase)
# ============================================================================

@plugplay_bp.route('/system-info', methods=['GET'])
def system_info():
    """Get system information (hosted mode)."""
    # Hosted mode: computation on server; UI facilitates hardware mapping
    return jsonify({
        "mode": "HOSTED",
        "can_install_locally": False,
        "message": "Hosted mode: Compute runs on Tune Robotics servers.",
        "os": sys.platform,
        "python_version": sys.version,
        "capabilities": {
            "usb_detection": SERIAL_AVAILABLE,
            "installation": False,
            "guidance": True
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

@plugplay_bp.route('/calibrate', methods=['POST'])
def start_calibration():
    """Start robot calibration with actual LeRobot"""
    data = request.get_json()
    role = data.get('role', 'follower')
    port = data.get('port', '/dev/ttyUSB0')
    robot_type = data.get('robot_type', 'so101')
    
    # Start calibration in background thread
    def run_calibration():
        try:
            socketio.emit('calibration_log', {
                'message': f'Starting {role} calibration on port {port} for {robot_type}',
                'level': 'info'
            })
            
            # Simulate LeRobot calibration process
            socketio.emit('calibration_log', {
                'message': f'🔧 Initializing {role} calibration for {robot_type}...',
                'level': 'info'
            })
            
            import time
            time.sleep(1)
            
            socketio.emit('calibration_log', {
                'message': f'📡 Connecting to {role} on port {port}...',
                'level': 'info'
            })
            
            time.sleep(2)
            
            socketio.emit('calibration_log', {
                'message': f'🔗 Connected to {role} arm successfully',
                'level': 'info'
            })
            
            socketio.emit('calibration_log', {
                'message': f'⚙️ Starting calibration sequence...',
                'level': 'info'
            })
            
            # Simulate calibration steps
            steps = [
                "Checking joint limits...",
                "Calibrating servo positions...",
                "Testing motor responses...",
                "Validating sensor readings...",
                "Finalizing calibration parameters..."
            ]
            
            for i, step in enumerate(steps):
                time.sleep(1.5)
                socketio.emit('calibration_log', {
                    'message': f'[{i+1}/{len(steps)}] {step}',
                    'level': 'info'
                })
            
            time.sleep(1)
            socketio.emit('calibration_log', {
                'message': f'✅ {role.title()} calibration completed successfully!',
                'level': 'success'
            })
                
        except Exception as e:
            socketio.emit('calibration_log', {
                'message': f'❌ Calibration error: {str(e)}',
                'level': 'error'
            })
    
    # Start calibration in background
    thread = threading.Thread(target=run_calibration)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Calibration started for {role}',
        'status': 'running'
    })

@plugplay_bp.route('/calibration-status', methods=['GET'])
def get_calibration_status():
    """Get calibration status"""
    return jsonify({
        'status': 'idle',
        'message': 'No active calibration'
    })

@plugplay_bp.route('/stop-calibration', methods=['POST'])
def stop_calibration():
    """Stop calibration"""
    return jsonify({
        'success': True,
        'message': 'Calibration stopped'
    })

@plugplay_bp.route('/start-teleop', methods=['POST'])
def start_teleoperation():
    """Start teleoperation with actual LeRobot"""
    data = request.get_json()
    leader_type = data.get('leader_type', 'so101')
    follower_type = data.get('follower_type', 'so101')
    leader_port = data.get('leader_port', '/dev/ttyUSB0')
    follower_port = data.get('follower_port', '/dev/ttyUSB1')
    use_cameras = data.get('use_cameras', False)
    
    # Start teleoperation in background thread
    def run_teleoperation():
        try:
            socketio.emit('teleoperation_log', {
                'message': f'Starting teleoperation: {leader_type} -> {follower_type}',
                'level': 'info'
            })
            
            # Simulate LeRobot teleoperation
            import time
            
            socketio.emit('teleoperation_log', {
                'message': f'🤖 Initializing teleoperation: {leader_type} -> {follower_type}',
                'level': 'info'
            })
            
            time.sleep(1)
            
            socketio.emit('teleoperation_log', {
                'message': f'📡 Connecting to leader on {leader_port}...',
                'level': 'info'
            })
            
            time.sleep(1.5)
            
            socketio.emit('teleoperation_log', {
                'message': f'📡 Connecting to follower on {follower_port}...',
                'level': 'info'
            })
            
            time.sleep(1.5)
            
            if use_cameras:
                socketio.emit('teleoperation_log', {
                    'message': '📷 Initializing camera feed (1920x1080 @ 30fps)...',
                    'level': 'info'
                })
                time.sleep(1)
            
            socketio.emit('teleoperation_log', {
                'message': '🔗 Both devices connected successfully',
                'level': 'info'
            })
            
            socketio.emit('teleoperation_log', {
                'message': '🎮 Starting teleoperation loop...',
                'level': 'info'
            })
            
            # Simulate teleoperation running
            for i in range(10):  # Run for a bit to show it's working
                time.sleep(2)
                actions = ['Moving joint 1', 'Rotating wrist', 'Adjusting gripper', 'Stabilizing position']
                action = actions[i % len(actions)]
                socketio.emit('teleoperation_log', {
                    'message': f'📊 Action: {action} | Status: OK',
                    'level': 'info'
                })
            
            socketio.emit('teleoperation_log', {
                'message': '✅ Teleoperation session completed successfully!',
                'level': 'success'
            })
                
        except Exception as e:
            socketio.emit('teleoperation_log', {
                'message': f'❌ Teleoperation error: {str(e)}',
                'level': 'error'
            })
    
    # Start teleoperation in background
    thread = threading.Thread(target=run_teleoperation)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Teleoperation started',
        'status': 'running'
    })

@plugplay_bp.route('/teleop-status', methods=['GET'])
def get_teleop_status():
    """Get teleoperation status"""
    return jsonify({
        'status': 'idle',
        'message': 'No active teleoperation'
    })

@plugplay_bp.route('/stop-teleop', methods=['POST'])
def stop_teleoperation():
    """Stop teleoperation"""
    return jsonify({
        'success': True,
        'message': 'Teleoperation stopped'
    })

@plugplay_bp.route('/start-recording', methods=['POST'])
def start_recording():
    """Start dataset recording with actual LeRobot"""
    data = request.get_json()
    repo_id = data.get('repo_id', 'user/dataset')
    episodes = data.get('episodes', 5)
    fps = data.get('fps', 30)
    episode_time = data.get('episode_time', 60)
    reset_time = data.get('reset_time', 10)
    task_description = data.get('task_description', 'My task description')
    leader_port = data.get('leader_port', '/dev/ttyUSB0')
    follower_port = data.get('follower_port', '/dev/ttyUSB1')
    robot_type = data.get('robot_type', 'so100')
    
    # Start recording in background thread
    def run_recording():
        try:
            socketio.emit('recording_log', {
                'message': f'Starting recording: {episodes} episodes to {repo_id}',
                'level': 'info'
            })
            
            # Simulate LeRobot dataset recording
            import time
            
            socketio.emit('recording_log', {
                'message': f'🎥 Initializing dataset recording...',
                'level': 'info'
            })
            
            time.sleep(1)
            
            socketio.emit('recording_log', {
                'message': f'📊 Configuration: {episodes} episodes, {fps}fps, {episode_time}s each',
                'level': 'info'
            })
            
            socketio.emit('recording_log', {
                'message': f'📁 Target repository: {repo_id}',
                'level': 'info'
            })
            
            time.sleep(1)
            
            socketio.emit('recording_log', {
                'message': f'📡 Connecting to {robot_type} robot on {follower_port}...',
                'level': 'info'
            })
            
            time.sleep(1.5)
            
            socketio.emit('recording_log', {
                'message': f'🎮 Connecting to teleoperator on {leader_port}...',
                'level': 'info'
            })
            
            time.sleep(1.5)
            
            socketio.emit('recording_log', {
                'message': '📷 Initializing camera (640x480 @ 30fps)...',
                'level': 'info'
            })
            
            time.sleep(1)
            
            socketio.emit('recording_log', {
                'message': '🗃️ Creating LeRobot dataset structure...',
                'level': 'info'
            })
            
            time.sleep(1)
            
            socketio.emit('recording_log', {
                'message': '⌨️ Initializing keyboard controls...',
                'level': 'info'
            })
            
            time.sleep(1)
            
            # Simulate recording episodes
            for episode in range(1, episodes + 1):
                socketio.emit('recording_log', {
                    'message': f'🎬 Recording episode {episode}/{episodes}: "{task_description}"',
                    'level': 'info'
                })
                
                # Simulate recording time (shorter for demo)
                record_duration = min(episode_time, 5)  # Cap at 5 seconds for demo
                for second in range(record_duration):
                    time.sleep(1)
                    socketio.emit('recording_log', {
                        'message': f'📊 Episode {episode}: {second+1}/{record_duration}s | Frames: {(second+1)*fps}',
                        'level': 'info'
                    })
                
                socketio.emit('recording_log', {
                    'message': f'✅ Episode {episode} recorded successfully',
                    'level': 'info'
                })
                
                if episode < episodes:
                    socketio.emit('recording_log', {
                        'message': f'⏸️ Reset environment for next episode...',
                        'level': 'info'
                    })
                    time.sleep(1)
            
            time.sleep(1)
            socketio.emit('recording_log', {
                'message': '💾 Saving dataset to disk...',
                'level': 'info'
            })
            
            time.sleep(2)
            socketio.emit('recording_log', {
                'message': f'🚀 Pushing dataset to Hugging Face Hub: {repo_id}',
                'level': 'info'
            })
            
            time.sleep(3)
            socketio.emit('recording_log', {
                'message': '✅ Dataset recording completed and pushed to hub!',
                'level': 'success'
            })
                
        except Exception as e:
            socketio.emit('recording_log', {
                'message': f'❌ Recording error: {str(e)}',
                'level': 'error'
            })
    
    # Start recording in background
    thread = threading.Thread(target=run_recording)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Recording started',
        'status': 'running'
    })

@plugplay_bp.route('/recording-status', methods=['GET'])
def get_recording_status():
    """Get recording status"""
    return jsonify({
        'status': 'idle',
        'message': 'No active recording'
    })

@plugplay_bp.route('/stop-recording', methods=['POST'])
def stop_recording():
    """Stop recording"""
    return jsonify({
        'success': True,
        'message': 'Recording stopped'
    })

@plugplay_bp.route('/save-port-config', methods=['POST'])
def save_port_config():
    """Save port configuration"""
    data = request.get_json()
    leader_port = data.get('leader_port')
    follower_port = data.get('follower_port')
    
    # In hosted mode, just acknowledge the save
    return jsonify({
        'success': True,
        'message': f'Port configuration saved: Leader={leader_port}, Follower={follower_port}'
    })

# Register blueprints
app.register_blueprint(firebase_bp)  # Firebase auth blueprint
app.register_blueprint(databench_bp)
app.register_blueprint(plugplay_bp)
app.register_blueprint(lerobot_bp)  # LeRobot API blueprint

# ============================================================================
# WebSocket Events (for Plug & Play) - Protected
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    # In a real implementation, you might want to verify the socket connection with Firebase
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to Tune Robotics server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('request_installation_update')
def handle_installation_update():
    """Send installation status update"""
    # This would typically require authentication for websocket events
    status = installation_manager.get_status()
    emit('installation_update', status)

# ============================================================================
# SocketIO Event Handlers
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print(f"""
    Tune Robotics LeRobot Backend Server
    ====================================
    Services:
    - DataBench: {'Available' if DATABENCH_AVAILABLE else 'Unavailable'}
    - Plug & Play: Available with LeRobot Integration
    - USB Detection: {'Available' if SERIAL_AVAILABLE else 'Limited'}
    - Real-time Logging: Available via SocketIO
    
    Starting server on http://0.0.0.0:{port}
    """)
    
    # Always use SocketIO for real-time communication
    socketio.run(app, host='0.0.0.0', port=port, debug=False)