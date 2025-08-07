#!/usr/bin/env python3
"""
Local Installer Bridge for Tune Robotics
Enables REAL local installation of LeRobot on the user's machine.
Run this script locally to enable actual installation functionality.
"""

import os
import sys
import subprocess
import shutil
import json
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'local-installer-secret'
CORS(app, origins=["*"])  # Allow all origins for local development
socketio = SocketIO(app, cors_allowed_origins="*")

# Installation state
installation_state = {
    'is_running': False,
    'progress': 0,
    'message': 'Ready to install',
    'status': 'idle',
    'install_path': None
}

# Runtime task/process state
runtime_state = {
    'calibration': {
        'active': False,
        'process': None,
        'params': None,
        'last_message': None,
    },
    'teleoperation': {
        'active': False,
        'process': None,
        'params': None,
        'last_message': None,
    },
    'recording': {
        'active': False,
        'process': None,
        'params': None,
        'last_message': None,
    },
}

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify local installer is running"""
    return jsonify({
        'status': 'online',
        'local': True,
        'can_install': True,
        'message': 'Local installer bridge is running',
        'capabilities': ['install', 'list-ports', 'calibration', 'teleoperation', 'recording']
    })

@app.route('/api/plugplay/system-info', methods=['GET'])
def system_info():
    """Get system information and check prerequisites"""
    # Check for Git
    git_available = shutil.which('git') is not None
    git_version = 'Not installed'
    if git_available:
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            git_version = result.stdout.strip()
        except:
            pass
    
    # Check for Conda
    conda_available = shutil.which('conda') is not None
    conda_version = 'Not installed'
    if conda_available:
        try:
            result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
            conda_version = result.stdout.strip()
        except:
            pass
    
    # Check for existing LeRobot installations
    found_installations = []
    common_paths = [
        Path.home() / 'lerobot',
        Path.home() / 'LeRobot',
        Path.home() / 'Documents' / 'lerobot',
        Path.home() / 'Projects' / 'lerobot',
        Path.cwd() / 'lerobot'
    ]
    
    for path in common_paths:
        if path.exists() and (path / '.git').exists():
            found_installations.append(str(path))
    
    return jsonify({
        'git_available': git_available,
        'git_version': git_version,
        'conda_available': conda_available,
        'conda_version': conda_version,
        'python_version': sys.version,
        'found_installations': found_installations,
        'platform': sys.platform,
        'can_install': git_available and conda_available
    })

@app.route('/api/plugplay/start-installation', methods=['POST'])
def start_installation():
    """Start the REAL LeRobot installation"""
    global installation_state
    
    if installation_state['is_running']:
        return jsonify({
            'success': False,
            'error': 'Installation already in progress'
        }), 400
    
    data = request.get_json()
    install_path = data.get('installation_path', '~/lerobot')
    selected_robot = data.get('selected_robot', 'unknown')
    use_existing = data.get('use_existing', False)
    
    # Expand the path
    install_path = str(Path(install_path).expanduser().resolve())
    
    # Start installation in background thread
    installation_state['is_running'] = True
    installation_state['status'] = 'running'
    installation_state['progress'] = 0
    installation_state['message'] = 'Starting installation...'
    installation_state['install_path'] = install_path
    
    thread = threading.Thread(
        target=run_real_installation, 
        args=(install_path, selected_robot, use_existing)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'status': 'started',
        'message': 'Installation started',
        'install_path': install_path,
        'selected_robot': selected_robot
    })

def run_real_installation(install_path, robot_type, use_existing):
    """Actually install LeRobot on the user's machine"""
    global installation_state
    
    try:
        print(f"🚀 Starting REAL installation to {install_path}")
        
        # Step 1: Check prerequisites
        update_progress(10, "Checking prerequisites...")
        
        # Check Git
        if not shutil.which('git'):
            raise Exception("Git is not installed. Please install Git first.")
        emit_progress("✓ Git found")
        
        # Check Conda
        if not shutil.which('conda'):
            raise Exception("Conda is not installed. Please install Miniconda or Anaconda first.")
        emit_progress("✓ Conda found")
        
        if not use_existing:
            # Step 2: Clone repository
            update_progress(20, "Cloning LeRobot repository from GitHub...")
            
            install_dir = Path(install_path)
            if install_dir.exists():
                emit_progress(f"Removing existing directory: {install_dir}")
                shutil.rmtree(install_dir)
            
            # Clone the repository
            cmd = ['git', 'clone', 'https://github.com/huggingface/lerobot.git', str(install_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to clone repository: {result.stderr}")
            
            emit_progress(f"✓ Repository cloned to {install_path}")
            update_progress(40, "Repository cloned successfully")
            
            # Step 3: Create conda environment
            update_progress(50, "Creating conda environment 'lerobot'...")
            
            cmd = ['conda', 'create', '-y', '-n', 'lerobot', 'python=3.10']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Environment might already exist, try to update it
                emit_progress("Environment might exist, continuing...")
            else:
                emit_progress("✓ Conda environment created")
            
            # Step 4: Install dependencies
            update_progress(70, "Installing LeRobot package (this may take a few minutes)...")
            
            # Install in editable mode
            cmd = ['conda', 'run', '-n', 'lerobot', 'pip', 'install', '-e', '.']
            result = subprocess.run(cmd, cwd=install_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                emit_progress(f"Warning: Some dependencies might have failed: {result.stderr[:500]}")
            else:
                emit_progress("✓ LeRobot package installed")
            
            # Step 5: Install pyserial for USB detection
            update_progress(90, "Installing USB port detection tools...")
            
            cmd = ['conda', 'run', '-n', 'lerobot', 'pip', 'install', 'pyserial']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                emit_progress("✓ USB tools installed")
            else:
                emit_progress("Warning: pyserial installation failed, USB detection may not work")
        
        else:
            # Using existing installation
            update_progress(50, f"Using existing installation at {install_path}")
            emit_progress("Verifying installation...")
            
            install_dir = Path(install_path)
            if not install_dir.exists():
                raise Exception(f"Installation path does not exist: {install_path}")
            
            if not (install_dir / '.git').exists():
                raise Exception(f"Not a valid LeRobot installation: {install_path}")
            
            emit_progress("✓ Existing installation verified")
        
        # Complete!
        update_progress(100, f"🎉 LeRobot successfully installed at {install_path}!")
        
        installation_state['status'] = 'completed'
        installation_state['is_running'] = False
        
        # Emit completion event
        socketio.emit('installation_complete', {
            'success': True,
            'install_path': install_path,
            'robot_type': robot_type,
            'message': 'Installation completed successfully!'
        })
        
        print("\n" + "="*60)
        print("✅ INSTALLATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Location: {install_path}")
        print("🐍 Environment: lerobot")
        print("\nTo use LeRobot:")
        print("  conda activate lerobot")
        print(f"  cd {install_path}")
        print("="*60 + "\n")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Installation failed: {error_msg}")
        installation_state['status'] = 'failed'
        installation_state['message'] = error_msg
        installation_state['is_running'] = False
        
        socketio.emit('installation_error', {
            'error': error_msg
        })

def update_progress(progress, message):
    """Update installation progress"""
    global installation_state
    installation_state['progress'] = progress
    installation_state['message'] = message
    
    # Emit progress via WebSocket
    socketio.emit('installation_progress', {
        'progress': progress,
        'message': message,
        'status': installation_state['status'],
        'install_path': installation_state['install_path']
    })
    
    print(f"[{progress}%] {message}")

def emit_progress(message):
    """Emit a progress message"""
    socketio.emit('install_log', {
        'message': message,
        'level': 'info',
        'timestamp': time.time()
    })
    print(f"  {message}")

def _spawn_conda_python(code: str, env_name: str = 'lerobot'):
    """Spawn a background python process inside a conda env running provided code.

    Returns Popen handle with stdout/stderr piped for log streaming.
    """
    import subprocess
    process = subprocess.Popen(
        ['conda', 'run', '-n', env_name, 'python', '-u', '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return process

def _stream_process_logs(kind: str, process):
    """Read process stdout and emit Socket.IO logs until process exits."""
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        msg = line.rstrip()
        runtime_state[kind]['last_message'] = msg
        socketio.emit(f'{kind}_log', {'message': msg})
    process.wait()
    runtime_state[kind]['active'] = False
    runtime_state[kind]['process'] = None
    socketio.emit(f'{kind}_status', {'active': False, 'returncode': process.returncode})

@app.route('/api/plugplay/list-ports', methods=['GET'])
def list_ports():
    """List available USB ports"""
    try:
        import serial.tools.list_ports
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid
            })
        return jsonify({'ports': ports})
    except ImportError:
        return jsonify({'ports': [], 'error': 'pyserial not installed'})

# --------------------------
# Calibration API
# --------------------------

@app.route('/api/plugplay/calibrate', methods=['POST'])
def start_calibration():
    """Start calibration for a robot or teleoperator using LeRobot API.

    Expected JSON body:
      - role: 'follower' | 'leader'
      - robot_type: e.g. 'so101' | 'so100' | 'koch' (base type without _follower/_leader suffix)
      - port: serial device
      - device_id: optional friendly id
      - env_name: optional conda env name (default 'lerobot')
    """
    if runtime_state['calibration']['active']:
        return jsonify({'success': False, 'error': 'Calibration already running'}), 400

    data = request.get_json() or {}
    role = (data.get('role') or 'follower').lower()
    robot_type = (data.get('robot_type') or 'so101').lower()
    port = data.get('port')
    device_id = data.get('device_id') or f'my_{role}_{robot_type}'
    env_name = data.get('env_name') or 'lerobot'

    if not port:
        return jsonify({'success': False, 'error': 'Missing port'}), 400

    # Map to LeRobot module/class
    if role == 'follower':
        module = f"lerobot.robots.{robot_type}_follower"
        cfg_cls = f"{robot_type.upper()}FollowerConfig"
        dev_cls = f"{robot_type.upper()}Follower"
    else:
        module = f"lerobot.teleoperators.{robot_type}_leader"
        cfg_cls = f"{robot_type.upper()}LeaderConfig"
        dev_cls = f"{robot_type.upper()}Leader"

    # Python code to run calibration
    py_code = f"""
import sys, time
print('Starting calibration...')
from {module} import {cfg_cls} as Cfg, {dev_cls} as Dev
cfg = Cfg(port={port!r}, id={device_id!r})
dev = Dev(cfg)
dev.connect(calibrate=False)
print('Connected. Running calibrate()...')
dev.calibrate()
print('Calibration complete. Disconnecting...')
dev.disconnect()
print('Done.')
"""

    try:
        proc = _spawn_conda_python(py_code, env_name)
        runtime_state['calibration']['active'] = True
        runtime_state['calibration']['process'] = proc
        runtime_state['calibration']['params'] = data

        # Stream logs in background thread
        import threading
        t = threading.Thread(target=_stream_process_logs, args=('calibration', proc), daemon=True)
        t.start()
        socketio.emit('calibration_status', {'active': True})
        return jsonify({'success': True, 'message': 'Calibration started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/plugplay/calibration-status', methods=['GET'])
def calibration_status():
    st = runtime_state['calibration']
    return jsonify({'active': st['active'], 'params': st['params'], 'last_message': st['last_message']})

@app.route('/api/plugplay/stop-calibration', methods=['POST'])
def stop_calibration():
    st = runtime_state['calibration']
    if not st['active'] or not st['process']:
        return jsonify({'success': False, 'error': 'No calibration running'}), 400
    try:
        st['process'].terminate()
    except Exception:
        pass
    st['active'] = False
    st['process'] = None
    socketio.emit('calibration_status', {'active': False})
    return jsonify({'success': True})

# --------------------------
# Teleoperation API
# --------------------------

@app.route('/api/plugplay/start-teleop', methods=['POST'])
def start_teleop():
    """Start teleoperation loop between leader and follower.

    Expected JSON body:
      - robot_type: 'so100' | 'so101' | 'koch' (follower)
      - leader_type: defaults to same base type as robot_type
      - follower_port, leader_port: device paths
      - cameras: optional dict mapping name-> {index_or_path,width,height,fps}
      - env_name: optional conda env name (default 'lerobot')
    """
    if runtime_state['teleoperation']['active']:
        return jsonify({'success': False, 'error': 'Teleoperation already running'}), 400

    data = request.get_json() or {}
    robot_type = (data.get('robot_type') or 'so101').lower()
    leader_type = (data.get('leader_type') or robot_type).lower()
    f_port = data.get('follower_port')
    l_port = data.get('leader_port')
    cameras = data.get('cameras') or {}
    env_name = data.get('env_name') or 'lerobot'
    follower_id = data.get('follower_id') or f'my_{robot_type}_follower'
    leader_id = data.get('leader_id') or f'my_{leader_type}_leader'

    if not f_port or not l_port:
        return jsonify({'success': False, 'error': 'Missing leader_port or follower_port'}), 400

    follower_module = f"lerobot.robots.{robot_type}_follower"
    follower_cfg = f"{robot_type.upper()}FollowerConfig"
    follower_cls = f"{robot_type.upper()}Follower"

    leader_module = f"lerobot.teleoperators.{leader_type}_leader"
    leader_cfg = f"{leader_type.upper()}LeaderConfig"
    leader_cls = f"{leader_type.upper()}Leader"

    camera_code_lines = ["camera_config = {}"]
    if cameras:
        camera_code_lines = [
            "from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig",
            "camera_config = {}",
        ]
        for name, cfg in cameras.items():
            idx = cfg.get('index_or_path', 0)
            w = cfg.get('width', 640)
            h = cfg.get('height', 480)
            fps = cfg.get('fps', 30)
            camera_code_lines.append(
                f"camera_config[{name!r}] = OpenCVCameraConfig(index_or_path={idx!r}, width={w}, height={h}, fps={fps})"
            )

    py_code = f"""
import time, sys
from {follower_module} import {follower_cfg} as FCfg, {follower_cls} as FDev
from {leader_module} import {leader_cfg} as LCfg, {leader_cls} as LDev
{chr(10).join(camera_code_lines)}
f_cfg = FCfg(port={f_port!r}, id={follower_id!r}, cameras=camera_config if camera_config else None)
l_cfg = LCfg(port={l_port!r}, id={leader_id!r})
robot = FDev(f_cfg)
leader = LDev(l_cfg)
robot.connect()
leader.connect()
print('Teleoperation started. Press Ctrl+C to stop.')
try:
    while True:
        action = leader.get_action()
        robot.send_action(action)
except KeyboardInterrupt:
    print('Stopping teleop...')
finally:
    try:
        leader.disconnect()
    except Exception: pass
    try:
        robot.disconnect()
    except Exception: pass
    print('Teleoperation stopped.')
"""

    try:
        proc = _spawn_conda_python(py_code, env_name)
        runtime_state['teleoperation']['active'] = True
        runtime_state['teleoperation']['process'] = proc
        runtime_state['teleoperation']['params'] = data

        import threading
        t = threading.Thread(target=_stream_process_logs, args=('teleoperation', proc), daemon=True)
        t.start()
        socketio.emit('teleoperation_status', {'active': True})
        return jsonify({'success': True, 'message': 'Teleoperation started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/plugplay/stop-teleop', methods=['POST'])
def stop_teleop():
    st = runtime_state['teleoperation']
    if not st['active'] or not st['process']:
        return jsonify({'success': False, 'error': 'No teleoperation running'}), 400
    try:
        st['process'].terminate()
    except Exception:
        pass
    st['active'] = False
    st['process'] = None
    socketio.emit('teleoperation_status', {'active': False})
    return jsonify({'success': True})

@app.route('/api/plugplay/teleop-status', methods=['GET'])
def teleop_status():
    st = runtime_state['teleoperation']
    return jsonify({'active': st['active'], 'params': st['params'], 'last_message': st['last_message']})

# --------------------------
# Recording API
# --------------------------

@app.route('/api/plugplay/start-recording', methods=['POST'])
def start_recording():
    """Start dataset recording loop using LeRobot record API.

    Expected JSON body (minimal):
      - robot_type: base type (e.g., 'so100')
      - leader_type: optional, default same as robot_type
      - follower_port, leader_port
      - repo_id: '<hf_username>/<dataset_repo_id>'
      - fps: int
      - num_episodes: int
      - episode_time_sec: int
      - reset_time_sec: int
      - task_description: str
      - cameras: optional dict name->config
      - env_name: optional conda env
    """
    if runtime_state['recording']['active']:
        return jsonify({'success': False, 'error': 'Recording already running'}), 400

    data = request.get_json() or {}
    robot_type = (data.get('robot_type') or 'so100').lower()
    leader_type = (data.get('leader_type') or robot_type).lower()
    f_port = data.get('follower_port')
    l_port = data.get('leader_port')
    repo_id = data.get('repo_id')
    fps = int(data.get('fps') or 30)
    num_episodes = int(data.get('num_episodes') or 5)
    episode_time = int(data.get('episode_time_sec') or 60)
    reset_time = int(data.get('reset_time_sec') or 10)
    task_desc = data.get('task_description') or 'Task'
    cameras = data.get('cameras') or {}
    env_name = data.get('env_name') or 'lerobot'

    if not (f_port and l_port and repo_id):
        return jsonify({'success': False, 'error': 'Missing follower_port, leader_port, or repo_id'}), 400

    follower_module = f"lerobot.robots.{robot_type}_follower"
    follower_cfg = f"{robot_type.upper()}FollowerConfig"
    follower_cls = f"{robot_type.upper()}Follower"
    leader_module = f"lerobot.teleoperators.{leader_type}_leader"
    leader_cfg = f"{leader_type.upper()}LeaderConfig"
    leader_cls = f"{leader_type.upper()}Leader"

    camera_code_lines = ["camera_config = {}"]
    if cameras:
        camera_code_lines = [
            "from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig",
            "camera_config = {}",
        ]
        for name, cfg in cameras.items():
            idx = cfg.get('index_or_path', 0)
            w = cfg.get('width', 640)
            h = cfg.get('height', 480)
            fps_c = cfg.get('fps', fps)
            camera_code_lines.append(
                f"camera_config[{name!r}] = OpenCVCameraConfig(index_or_path={idx!r}, width={w}, height={h}, fps={fps_c})"
            )

    py_code = f"""
import time
from {follower_module} import {follower_cfg} as FCfg, {follower_cls} as FDev
from {leader_module} import {leader_cfg} as LCfg, {leader_cls} as LDev
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
{chr(10).join(camera_code_lines)}
FPS = {fps}
NUM_EPISODES = {num_episodes}
EPISODE_TIME_SEC = {episode_time}
RESET_TIME_SEC = {reset_time}
TASK_DESCRIPTION = {task_desc!r}
robot = FDev(FCfg(port={f_port!r}, id='rec_follower', cameras=camera_config if camera_config else None))
teleop = LDev(LCfg(port={l_port!r}, id='rec_leader'))
action_features = hw_to_dataset_features(robot.action_features, 'action')
obs_features = hw_to_dataset_features(robot.observation_features, 'observation')
dataset_features = dict(action_features)
dataset_features.update(obs_features)
dataset = LeRobotDataset.create(repo_id={repo_id!r}, fps=FPS, features=dataset_features, robot_type=robot.name, use_videos=True, image_writer_threads=4)
robot.connect(); teleop.connect()
events = dict(stop_recording=False, exit_early=False, rerecord_episode=False)
episode_idx = 0
try:
    while episode_idx < NUM_EPISODES:
        print(f'Recording episode {{episode_idx+1}}/{{NUM_EPISODES}}')
        record_loop(robot=robot, events=events, fps=FPS, teleop=teleop, dataset=dataset, control_time_s=EPISODE_TIME_SEC, single_task=TASK_DESCRIPTION, display_data=False)
        dataset.save_episode()
        if episode_idx < NUM_EPISODES - 1:
            record_loop(robot=robot, events=events, fps=FPS, teleop=teleop, control_time_s=RESET_TIME_SEC, single_task=TASK_DESCRIPTION, display_data=False)
        episode_idx += 1
    print('Pushing dataset to hub...')
    dataset.push_to_hub()
finally:
    try: teleop.disconnect()
    except Exception: pass
    try: robot.disconnect()
    except Exception: pass
    print('Recording finished')
"""

    try:
        proc = _spawn_conda_python(py_code, env_name)
        runtime_state['recording']['active'] = True
        runtime_state['recording']['process'] = proc
        runtime_state['recording']['params'] = data
        import threading
        t = threading.Thread(target=_stream_process_logs, args=('recording', proc), daemon=True)
        t.start()
        socketio.emit('recording_status', {'active': True})
        return jsonify({'success': True, 'message': 'Recording started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/plugplay/stop-recording', methods=['POST'])
def stop_recording():
    st = runtime_state['recording']
    if not st['active'] or not st['process']:
        return jsonify({'success': False, 'error': 'No recording running'}), 400
    try:
        st['process'].terminate()
    except Exception:
        pass
    st['active'] = False
    st['process'] = None
    socketio.emit('recording_status', {'active': False})
    return jsonify({'success': True})

@app.route('/api/plugplay/recording-status', methods=['GET'])
def recording_status():
    st = runtime_state['recording']
    return jsonify({'active': st['active'], 'params': st['params'], 'last_message': st['last_message']})

@app.route('/api/plugplay/save-port-config', methods=['POST'])
def save_port_config():
    """Save detected port configuration"""
    data = request.get_json()
    leader_port = data.get('leader_port')
    follower_port = data.get('follower_port')
    
    if installation_state['install_path']:
        config_file = Path(installation_state['install_path']) / 'port_config.json'
        config = {
            'leader_port': leader_port,
            'follower_port': follower_port,
            'detected_at': time.time()
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✓ Port configuration saved to {config_file}")
            return jsonify({'success': True, 'message': 'Configuration saved'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'No installation path set'}), 400

@app.route('/api/plugplay/status', methods=['GET'])
def get_status():
    """Get current installation status"""
    return jsonify(installation_state)

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {
        'message': 'Connected to local installer',
        'can_install': True
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 TUNE ROBOTICS LOCAL INSTALLER BRIDGE")
    print("="*60)
    print("This enables REAL installation of LeRobot on your machine.")
    print("\nThe installer is now running at: http://localhost:7777")
    print("\nGo to: https://tunerobotics.xyz/frontend/pages/plug-and-play-databench-style.html")
    print("The website will detect this local installer automatically.")
    print("\nPress Ctrl+C to stop the installer.")
    print("="*60 + "\n")
    
    # Run the local server
    socketio.run(app, host='0.0.0.0', port=7777, debug=False)
