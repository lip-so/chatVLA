#!/usr/bin/env python3

"""
Local Demo Server for Plug & Play
Simplified version for local testing
"""

import os
import sys
import time
import logging
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Import port detector
from plug_and_play.port_detector import USBPortDetector, detect_cameras

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')
app.config['SECRET_KEY'] = 'demo-secret-key'

# Initialize extensions
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize port detector
usb_detector = USBPortDetector()

# ============================================================================
# Static File Routes
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

# ============================================================================
# API Routes
# ============================================================================

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "🚀 Local Plug & Play Demo Server",
        "version": "1.0.0-demo",
        "services": {
            "plug_and_play": {
                "available": True,
                "status": "ready",
                "mode": "simulation"
            }
        }
    })

@app.route('/api/plugplay/list-ports', methods=['GET'])
def list_ports():
    """List available USB ports"""
    try:
        ports = usb_detector.scan_ports()
        return jsonify({
            "success": True,
            "ports": ports,
            "serial_available": usb_detector.serial_available
        })
    except Exception as e:
        logger.error(f"Port listing failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "ports": []
        }), 500

@app.route('/api/plugplay/list-cameras', methods=['GET'])
def list_cameras():
    """List available cameras"""
    try:
        cameras = detect_cameras()
        return jsonify({
            "success": True,
            "cameras": cameras
        })
    except Exception as e:
        logger.error(f"Camera listing failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "cameras": []
        }), 500

@app.route('/api/plugplay/calibrate', methods=['POST'])
def start_calibration():
    """Start robot calibration (demo mode)"""
    data = request.get_json()
    role = data.get('role', 'follower')
    port = data.get('port', '/dev/ttyUSB0')
    robot_type = data.get('robot_type', 'so101')
    
    def run_demo_calibration():
        try:
            socketio.emit('calibration_log', {
                'message': f'🔧 [DEMO] Starting {role} calibration on port {port} for {robot_type}',
                'level': 'info'
            })
            
            time.sleep(1)
            socketio.emit('calibration_log', {
                'message': f'📡 [DEMO] Connecting to {role} on port {port}...',
                'level': 'info'
            })
            
            time.sleep(2)
            socketio.emit('calibration_log', {
                'message': f'🔗 [DEMO] Connected to {role} arm successfully',
                'level': 'info'
            })
            
            socketio.emit('calibration_log', {
                'message': f'⚙️ [DEMO] Starting calibration sequence...',
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
                    'message': f'[DEMO] [{i+1}/{len(steps)}] {step}',
                    'level': 'info'
                })
            
            time.sleep(1)
            socketio.emit('calibration_log', {
                'message': f'✅ [DEMO] {role.title()} calibration completed successfully!',
                'level': 'success'
            })
                
        except Exception as e:
            socketio.emit('calibration_log', {
                'message': f'❌ [DEMO] Calibration error: {str(e)}',
                'level': 'error'
            })
    
    # Start calibration in background
    import threading
    thread = threading.Thread(target=run_demo_calibration, daemon=True)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Demo calibration started for {role}',
        'status': 'running'
    })

@app.route('/api/plugplay/start-teleop', methods=['POST'])
def start_teleoperation():
    """Start teleoperation (demo mode)"""
    data = request.get_json()
    leader_type = data.get('leader_type', 'so101')
    follower_type = data.get('follower_type', 'so101')
    leader_port = data.get('leader_port', '/dev/ttyUSB0')
    follower_port = data.get('follower_port', '/dev/ttyUSB1')
    use_cameras = data.get('use_cameras', False)
    
    def run_demo_teleoperation():
        try:
            socketio.emit('teleoperation_log', {
                'message': f'🤖 [DEMO] Initializing teleoperation: {leader_type} -> {follower_type}',
                'level': 'info'
            })
            
            time.sleep(1)
            socketio.emit('teleoperation_log', {
                'message': f'📡 [DEMO] Connecting to leader on {leader_port}...',
                'level': 'info'
            })
            
            time.sleep(1.5)
            socketio.emit('teleoperation_log', {
                'message': f'📡 [DEMO] Connecting to follower on {follower_port}...',
                'level': 'info'
            })
            
            time.sleep(1.5)
            
            if use_cameras:
                socketio.emit('teleoperation_log', {
                    'message': '📷 [DEMO] Initializing camera feed (1920x1080 @ 30fps)...',
                    'level': 'info'
                })
                time.sleep(1)
            
            socketio.emit('teleoperation_log', {
                'message': '🔗 [DEMO] Both devices connected successfully',
                'level': 'info'
            })
            
            socketio.emit('teleoperation_log', {
                'message': '🎮 [DEMO] Starting teleoperation loop...',
                'level': 'info'
            })
            
            # Simulate teleoperation running
            for i in range(10):
                time.sleep(2)
                actions = ['Moving joint 1', 'Rotating wrist', 'Adjusting gripper', 'Stabilizing position']
                action = actions[i % len(actions)]
                socketio.emit('teleoperation_log', {
                    'message': f'📊 [DEMO] Action: {action} | Status: OK',
                    'level': 'info'
                })
            
            socketio.emit('teleoperation_log', {
                'message': '✅ [DEMO] Teleoperation session completed successfully!',
                'level': 'success'
            })
                
        except Exception as e:
            socketio.emit('teleoperation_log', {
                'message': f'❌ [DEMO] Teleoperation error: {str(e)}',
                'level': 'error'
            })
    
    # Start teleoperation in background
    import threading
    thread = threading.Thread(target=run_demo_teleoperation, daemon=True)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Demo teleoperation started',
        'status': 'running'
    })

@app.route('/api/plugplay/start-recording', methods=['POST'])
def start_recording():
    """Start dataset recording (demo mode)"""
    data = request.get_json()
    repo_id = data.get('repo_id', 'user/dataset')
    episodes = data.get('episodes', 3)  # Shorter for demo
    fps = data.get('fps', 30)
    task_description = data.get('task_description', 'Demo task')
    
    def run_demo_recording():
        try:
            socketio.emit('recording_log', {
                'message': f'🎥 [DEMO] Starting recording: {episodes} episodes to {repo_id}',
                'level': 'info'
            })
            
            time.sleep(1)
            socketio.emit('recording_log', {
                'message': f'📊 [DEMO] Configuration: {episodes} episodes, {fps}fps',
                'level': 'info'
            })
            
            socketio.emit('recording_log', {
                'message': f'📁 [DEMO] Target repository: {repo_id}',
                'level': 'info'
            })
            
            time.sleep(1)
            socketio.emit('recording_log', {
                'message': '🗃️ [DEMO] Creating dataset structure...',
                'level': 'info'
            })
            
            # Simulate recording episodes
            for episode in range(1, episodes + 1):
                socketio.emit('recording_log', {
                    'message': f'🎬 [DEMO] Recording episode {episode}/{episodes}: "{task_description}"',
                    'level': 'info'
                })
                
                # Simulate recording time (shorter for demo)
                for second in range(3):  # 3 seconds per episode for demo
                    time.sleep(1)
                    socketio.emit('recording_log', {
                        'message': f'📊 [DEMO] Episode {episode}: {second+1}/3s | Frames: {(second+1)*fps}',
                        'level': 'info'
                    })
                
                socketio.emit('recording_log', {
                    'message': f'✅ [DEMO] Episode {episode} recorded successfully',
                    'level': 'info'
                })
                
                if episode < episodes:
                    socketio.emit('recording_log', {
                        'message': '⏸️ [DEMO] Reset environment for next episode...',
                        'level': 'info'
                    })
                    time.sleep(1)
            
            time.sleep(1)
            socketio.emit('recording_log', {
                'message': '💾 [DEMO] Saving dataset to disk...',
                'level': 'info'
            })
            
            time.sleep(2)
            socketio.emit('recording_log', {
                'message': f'🚀 [DEMO] Pushing dataset to Hugging Face Hub: {repo_id}',
                'level': 'info'
            })
            
            time.sleep(2)
            socketio.emit('recording_log', {
                'message': '✅ [DEMO] Dataset recording completed and pushed to hub!',
                'level': 'success'
            })
                
        except Exception as e:
            socketio.emit('recording_log', {
                'message': f'❌ [DEMO] Recording error: {str(e)}',
                'level': 'error'
            })
    
    # Start recording in background
    import threading
    thread = threading.Thread(target=run_demo_recording, daemon=True)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Demo recording started',
        'status': 'running'
    })

@app.route('/api/plugplay/stop-calibration', methods=['POST'])
def stop_calibration():
    """Stop calibration"""
    return jsonify({
        'success': True,
        'message': 'Demo calibration stopped'
    })

@app.route('/api/plugplay/stop-teleop', methods=['POST'])
def stop_teleoperation():
    """Stop teleoperation"""
    return jsonify({
        'success': True,
        'message': 'Demo teleoperation stopped'
    })

@app.route('/api/plugplay/stop-recording', methods=['POST'])
def stop_recording():
    """Stop recording"""
    return jsonify({
        'success': True,
        'message': 'Demo recording stopped'
    })

# ============================================================================
# WebSocket Events
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to demo server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    port = 5000
    
    print(f"""
    🚀 Plug & Play Demo Server
    ==========================
    
    🌐 Frontend: http://localhost:{port}
    📄 Plug & Play: http://localhost:{port}/pages/plug-and-play.html
    🔍 Port Detection: Available with mock data
    🤖 Robot Operations: Demo/simulation mode
    📡 Real-time Logs: Available via WebSocket
    
    Starting server on http://localhost:{port}
    """)
    
    socketio.run(app, host='localhost', port=port, debug=True)
