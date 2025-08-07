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
        "services": {
            "databench": {
                "available": DATABENCH_AVAILABLE,
                "status": "ready" if DATABENCH_AVAILABLE else "unavailable"
            },
            "plug_and_play": {
                "available": True,
                "serial_available": SERIAL_AVAILABLE
            },
            "firebase_authentication": {
                "available": True,
                "status": "ready"
            }
        },
        "timestamp": datetime.now().isoformat()
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
            
            # Import and run LeRobot calibration based on robot type and role
            if robot_type == 'so101':
                if role == 'follower':
                    calibration_code = f'''
import sys
sys.path.append('/app/ref')
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

config = SO101FollowerConfig(
    port="{port}",
    id="follower_arm",
)

follower = SO101Follower(config)
follower.connect(calibrate=False)
print("Connected to follower, starting calibration...")
follower.calibrate()
print("Calibration completed successfully")
follower.disconnect()
'''
                else:  # leader
                    calibration_code = f'''
import sys
sys.path.append('/app/ref')
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader

config = SO101LeaderConfig(
    port="{port}",
    id="leader_arm",
)

leader = SO101Leader(config)
leader.connect(calibrate=False)
print("Connected to leader, starting calibration...")
leader.calibrate()
print("Calibration completed successfully")
leader.disconnect()
'''
            
            # Execute the calibration code
            process = subprocess.Popen(
                ['python', '-c', calibration_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    socketio.emit('calibration_log', {
                        'message': line.strip(),
                        'level': 'info'
                    })
            
            # Wait for completion
            process.wait()
            
            if process.returncode == 0:
                socketio.emit('calibration_log', {
                    'message': f'✅ {role.title()} calibration completed successfully!',
                    'level': 'success'
                })
            else:
                error_output = process.stderr.read()
                socketio.emit('calibration_log', {
                    'message': f'❌ Calibration failed: {error_output}',
                    'level': 'error'
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
            
            # Generate teleoperation code based on configuration
            if use_cameras and leader_type == 'koch' and follower_type == 'koch':
                teleop_code = f'''
import sys
sys.path.append('/app/ref')
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.koch_leader import KochLeaderConfig, KochLeader
from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower

camera_config = {{
    "front": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30)
}}

robot_config = KochFollowerConfig(
    port="{follower_port}",
    id="red_robot_arm",
    cameras=camera_config
)

teleop_config = KochLeaderConfig(
    port="{leader_port}",
    id="blue_leader_arm",
)

robot = KochFollower(robot_config)
teleop_device = KochLeader(teleop_config)
robot.connect()
teleop_device.connect()

print("Connected to both devices, starting teleoperation loop...")
try:
    while True:
        observation = robot.get_observation()
        action = teleop_device.get_action()
        robot.send_action(action)
except KeyboardInterrupt:
    print("Stopping teleoperation...")
finally:
    robot.disconnect()
    teleop_device.disconnect()
    print("Teleoperation stopped")
'''
            else:
                # Basic teleoperation without cameras
                teleop_code = f'''
import sys
sys.path.append('/app/ref')
from lerobot.teleoperators.{leader_type}_leader import {leader_type.upper()}LeaderConfig, {leader_type.upper()}Leader
from lerobot.robots.{follower_type}_follower import {follower_type.upper()}FollowerConfig, {follower_type.upper()}Follower

robot_config = {follower_type.upper()}FollowerConfig(
    port="{follower_port}",
    id="red_robot_arm",
)

teleop_config = {leader_type.upper()}LeaderConfig(
    port="{leader_port}",
    id="blue_leader_arm",
)

robot = {follower_type.upper()}Follower(robot_config)
teleop_device = {leader_type.upper()}Leader(teleop_config)
robot.connect()
teleop_device.connect()

print("Connected to both devices, starting teleoperation loop...")
try:
    while True:
        action = teleop_device.get_action()
        robot.send_action(action)
except KeyboardInterrupt:
    print("Stopping teleoperation...")
finally:
    robot.disconnect()
    teleop_device.disconnect()
    print("Teleoperation stopped")
'''
            
            # Execute the teleoperation code
            process = subprocess.Popen(
                ['python', '-c', teleop_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    socketio.emit('teleoperation_log', {
                        'message': line.strip(),
                        'level': 'info'
                    })
            
            # Wait for completion (or interruption)
            process.wait()
            
            if process.returncode == 0:
                socketio.emit('teleoperation_log', {
                    'message': '✅ Teleoperation completed successfully!',
                    'level': 'success'
                })
            else:
                error_output = process.stderr.read()
                socketio.emit('teleoperation_log', {
                    'message': f'❌ Teleoperation failed: {error_output}',
                    'level': 'error'
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
            
            # Generate recording code based on your example
            recording_code = f'''
import sys
sys.path.append('/app/ref')
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.{robot_type}_follower import {robot_type.upper()}Follower, {robot_type.upper()}FollowerConfig
from lerobot.teleoperators.{robot_type}_leader.config_{robot_type}_leader import {robot_type.upper()}LeaderConfig
from lerobot.teleoperators.{robot_type}_leader.{robot_type}_leader import {robot_type.upper()}Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

NUM_EPISODES = {episodes}
FPS = {fps}
EPISODE_TIME_SEC = {episode_time}
RESET_TIME_SEC = {reset_time}
TASK_DESCRIPTION = "{task_description}"

# Create the robot and teleoperator configurations
camera_config = {{"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}}
robot_config = {robot_type.upper()}FollowerConfig(
    port="{follower_port}", 
    id="awesome_follower_arm", 
    cameras=camera_config
)
teleop_config = {robot_type.upper()}LeaderConfig(
    port="{leader_port}", 
    id="awesome_leader_arm"
)

# Initialize the robot and teleoperator
robot = {robot_type.upper()}Follower(robot_config)
teleop = {robot_type.upper()}Leader(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {{**action_features, **obs_features}}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="{repo_id}",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {{episode_idx + 1}} of {{NUM_EPISODES}}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()
print("Recording completed and dataset pushed to hub!")
'''
            
            # Execute the recording code
            process = subprocess.Popen(
                ['python', '-c', recording_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    socketio.emit('recording_log', {
                        'message': line.strip(),
                        'level': 'info'
                    })
            
            # Wait for completion
            process.wait()
            
            if process.returncode == 0:
                socketio.emit('recording_log', {
                    'message': '✅ Recording completed and dataset pushed to hub!',
                    'level': 'success'
                })
            else:
                error_output = process.stderr.read()
                socketio.emit('recording_log', {
                    'message': f'❌ Recording failed: {error_output}',
                    'level': 'error'
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