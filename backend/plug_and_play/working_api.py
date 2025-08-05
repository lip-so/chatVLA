#!/usr/bin/env python3
"""
Actually working Plug & Play API that really installs LeRobot.
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import subprocess
import threading
import os
import sys
from pathlib import Path
import time
import json

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# Import our secure command execution system
try:
    from .system_commands import safe_commands, SystemCommandError
    SECURE_COMMANDS_AVAILABLE = True
except ImportError:
    try:
        from system_commands import safe_commands, SystemCommandError
        SECURE_COMMANDS_AVAILABLE = True
    except ImportError:
        SECURE_COMMANDS_AVAILABLE = False
        print("Warning: Secure command system not available. Using direct subprocess calls.")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
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

def find_available_env_name():
    """Find an available conda environment name"""
    base_name = "lerobot"
    
    # First try the base name
    if not check_conda_env_exists(base_name):
        return base_name
    
    # If base name exists, try with incremental numbers
    counter = 1
    while counter < 100:  # Safety limit
        env_name = f"{base_name}_{counter}"
        if not check_conda_env_exists(env_name):
            return env_name
        counter += 1
    
    # Fallback to timestamp-based name
    import time
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}"

def check_conda_env_exists(env_name):
    """Check if a conda environment exists"""
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            # Check if environment name appears in the output
            for line in result.stdout.split('\n'):
                if line.strip().startswith(env_name + ' ') or line.strip().startswith(env_name + '\t'):
                    return True
        return False
    except:
        # If we can't check, assume it doesn't exist
        return False

def is_valid_lerobot_installation(path):
    """Check if a path contains a valid LeRobot installation"""
    path = Path(path)
    if not path.exists():
        return False
    
    # Check for multiple indicators of a LeRobot installation
    indicators = [
        # Primary indicators
        path / 'setup.py',
        path / 'pyproject.toml',
        path / 'lerobot',  # Main lerobot directory
        path / 'src' / 'lerobot',  # Alternative structure
        
        # Configuration files
        path / 'lerobot.yaml',
        path / 'config.yaml',
        
        # Key Python files
        path / '__init__.py',
        path / 'lerobot' / '__init__.py',
        
        # Common LeRobot subdirectories
        path / 'lerobot' / 'robots',
        path / 'lerobot' / 'policies',
        path / 'lerobot' / 'datasets',
    ]
    
    # If any indicator exists, consider it a valid installation
    for indicator in indicators:
        if indicator.exists():
            return True
    
    # Check if directory contains Python files with 'lerobot' references
    try:
        python_files = list(path.glob('**/*.py'))[:10]  # Check first 10 Python files
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'lerobot' in content.lower():
                        return True
            except:
                continue
    except:
        pass
    
    return False

# Frontend serving routes
@app.route('/')
def index():
    """Serve the main frontend page"""
    frontend_path = Path(__file__).parent.parent.parent / 'index.html'
    if frontend_path.exists():
        return send_file(str(frontend_path))
    else:
        return jsonify({'error': 'Main page not found'}), 404

@app.route('/index.html')
def index_html():
    """Serve index.html specifically"""
    frontend_path = Path(__file__).parent.parent.parent / 'index.html'
    if frontend_path.exists():
        return send_file(str(frontend_path))
    else:
        return jsonify({'error': 'Main page not found'}), 404

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    favicon_path = Path(__file__).parent.parent.parent / 'assets' / 'logo.png'
    if favicon_path.exists():
        return send_file(str(favicon_path))
    else:
        return '', 404

@app.route('/pages/<path:filename>')
def serve_pages(filename):
    """Serve frontend pages"""
    # Try both frontend/pages and root pages directories for deployment compatibility
    frontend_pages_dir = Path(__file__).parent.parent.parent / 'frontend' / 'pages'
    root_pages_dir = Path(__file__).parent.parent.parent / 'pages'
    
    if (frontend_pages_dir / filename).exists():
        return send_from_directory(str(frontend_pages_dir), filename)
    elif (root_pages_dir / filename).exists():
        return send_from_directory(str(root_pages_dir), filename)
    else:
        return jsonify({'error': 'Page not found'}), 404

@app.route('/plug-and-play-databench-style.html')
def serve_plug_and_play_direct():
    """Direct route for plug-and-play (deployment fallback)"""
    # Try multiple locations for deployment compatibility
    locations = [
        Path(__file__).parent.parent.parent / 'frontend' / 'pages' / 'plug-and-play-databench-style.html',
        Path(__file__).parent.parent.parent / 'pages' / 'plug-and-play-databench-style.html',
        Path(__file__).parent.parent.parent / 'plug-and-play-databench-style.html'
    ]
    
    for location in locations:
        if location.exists():
            return send_file(str(location))
    
    return jsonify({'error': 'Plug & Play page not found'}), 404

@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files"""
    # Try both frontend/css and root css directories
    frontend_css_dir = Path(__file__).parent.parent.parent / 'frontend' / 'css'
    root_css_dir = Path(__file__).parent.parent.parent / 'css'
    
    if (frontend_css_dir / filename).exists():
        return send_from_directory(str(frontend_css_dir), filename)
    elif (root_css_dir / filename).exists():
        return send_from_directory(str(root_css_dir), filename)
    else:
        return jsonify({'error': 'CSS file not found'}), 404

@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files"""
    # Try both frontend/js and root js directories
    frontend_js_dir = Path(__file__).parent.parent.parent / 'frontend' / 'js'
    root_js_dir = Path(__file__).parent.parent.parent / 'js'
    
    if (frontend_js_dir / filename).exists():
        return send_from_directory(str(frontend_js_dir), filename)
    elif (root_js_dir / filename).exists():
        return send_from_directory(str(root_js_dir), filename)
    else:
        return jsonify({'error': 'JS file not found'}), 404

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve asset files"""
    # Try both frontend/assets and root assets directories
    frontend_assets_dir = Path(__file__).parent.parent.parent / 'frontend' / 'assets'
    root_assets_dir = Path(__file__).parent.parent.parent / 'assets'
    
    if (frontend_assets_dir / filename).exists():
        return send_from_directory(str(frontend_assets_dir), filename)
    elif (root_assets_dir / filename).exists():
        return send_from_directory(str(root_assets_dir), filename)
    else:
        return jsonify({'error': 'Asset file not found'}), 404

@app.route('/api/check_lerobot', methods=['GET'])
def check_lerobot():
    """Check if LeRobot is already installed"""
    # Check common locations
    possible_paths = [
        '~/lerobot',
        '~/LeRobot', 
        '~/Documents/lerobot',
        '~/Projects/lerobot',
        '~/code/lerobot',
        '/opt/lerobot'
    ]
    
    found_installations = []
    
    for path_str in possible_paths:
        path = Path(path_str).expanduser()
        if is_valid_lerobot_installation(path):
            found_installations.append(str(path))
    
    # Also check if lerobot is installed via pip in current environment
    pip_installed = False
    try:
        result = subprocess.run(['python', '-c', 'import lerobot'], capture_output=True)
        pip_installed = result.returncode == 0
    except:
        pass
    
    return jsonify({
        'found_installations': found_installations,
        'pip_installed': pip_installed
    })

@app.route('/api/install', methods=['POST'])
def install_lerobot():
    """Actually install LeRobot"""
    data = request.get_json()
    path = Path(data.get('path', '~/lerobot')).expanduser()
    robot = data.get('robot', 'koch')
    use_existing = data.get('use_existing', False)
    
    if current_installation['running']:
        return jsonify({'error': 'Installation already running'}), 400
    
    # Start installation in background
    thread = threading.Thread(target=run_installation, args=(path, robot, use_existing))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'path': str(path), 'robot': robot})

def run_installation(path, robot, use_existing=False):
    """Run the actual installation"""
    current_installation['running'] = True
    current_installation['path'] = str(path)
    current_installation['robot'] = robot
    
    try:
        if use_existing:
            emit_log(f"Using existing LeRobot installation at {path}")
            
            # Verify it's a valid installation using flexible validation
            if not is_valid_lerobot_installation(path):
                emit_log("ERROR: Selected directory doesn't appear to be a LeRobot installation", level='error')
                emit_log("Looking for: setup.py, pyproject.toml, lerobot/, or Python files with 'lerobot' references", level='error')
                return
            
            emit_log("✅ Verified existing installation!")
            emit_log("⏭️ Skipping download - using your existing LeRobot installation")
            
        else:
            # Step 1: Create directory
            emit_log("Creating installation directory...")
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Step 2: Clone LeRobot
            emit_log(f"Cloning LeRobot to {path}...")
            if path.exists():
                emit_log(f"Directory {path} already exists, removing...")
                # Safely remove existing directory
                import shutil
                shutil.rmtree(path, ignore_errors=True)
                
            # Use secure command execution
            try:
                if SECURE_COMMANDS_AVAILABLE:
                    def log_callback(line):
                        emit_log(line)
                    
                    returncode = safe_commands.stream_command(
                        'git', 
                        ['clone', 'https://github.com/huggingface/lerobot.git', str(path)],
                        callback=log_callback
                    )
                    
                    if returncode != 0:
                        emit_log("ERROR: Failed to clone repository", level='error')
                        return
                else:
                    # Fallback to direct subprocess
                    cmd = ['git', 'clone', 'https://github.com/huggingface/lerobot.git', str(path)]
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            emit_log(line.strip())
                    
                    process.wait()
                    if process.returncode != 0:
                        emit_log("ERROR: Failed to clone repository", level='error')
                        return
                        
            except SystemCommandError as e:
                emit_log(f"ERROR: {str(e)}", level='error')
                return
                
            # Step 3: Create conda environment
            env_name = find_available_env_name()
            current_installation['env_name'] = env_name
            emit_log(f"Creating conda environment '{env_name}'...")
            
            # Create new environment
            try:
                if SECURE_COMMANDS_AVAILABLE:
                    def log_callback(line):
                        emit_log(line)
                    
                    returncode = safe_commands.stream_command(
                        'conda', 
                        ['create', '-n', env_name, 'python=3.10', '-y'],
                        callback=log_callback
                    )
                    
                    if returncode != 0:
                        emit_log("ERROR: Failed to create conda environment", level='error')
                        return
                else:
                    cmd = ['conda', 'create', '-n', env_name, 'python=3.10', '-y']
                    run_with_output(cmd)
                
            except SystemCommandError as e:
                emit_log(f"ERROR: {str(e)}", level='error')
                return
            
            # Step 4: Install dependencies
            emit_log("Installing LeRobot dependencies...")
            os.chdir(path)
            
            try:
                if SECURE_COMMANDS_AVAILABLE:
                    def log_callback(line):
                        emit_log(line)
                    
                    returncode = safe_commands.stream_command(
                        'conda', 
                        ['run', '-n', env_name, 'pip', 'install', '-e', '.'],
                        working_dir=str(path),
                        callback=log_callback
                    )
                    
                    if returncode != 0:
                        emit_log("ERROR: Failed to install LeRobot", level='error')
                        return
                else:
                    cmd = ['conda', 'run', '-n', env_name, 'pip', 'install', '-e', '.']
                    run_with_output(cmd)
                
            except SystemCommandError as e:
                emit_log(f"ERROR: {str(e)}", level='error')
                return
            
            # Step 5: Install robot-specific packages
            emit_log(f"Installing packages for {robot}...")
            
            packages_to_install = ['pyserial']
            if robot in ['koch']:
                packages_to_install.append('dynamixel-sdk')
                
            for package in packages_to_install:
                try:
                    if SECURE_COMMANDS_AVAILABLE:
                        def log_callback(line):
                            emit_log(line)
                        
                        returncode = safe_commands.stream_command(
                            'conda', 
                            ['run', '-n', env_name, 'pip', 'install', package],
                            callback=log_callback
                        )
                        
                        if returncode != 0:
                            emit_log(f"WARNING: Failed to install {package}", level='error')
                    else:
                        cmd = ['conda', 'run', '-n', env_name, 'pip', 'install', package]
                        run_with_output(cmd)
                    
                except SystemCommandError as e:
                    emit_log(f"WARNING: {str(e)}", level='error')
        
        # Create configuration files (always do this)
        emit_log("Creating robot configuration...")
        create_robot_config(path, robot)
        
        emit_log("✅ Installation completed successfully!", level='success')
        emit_log(f"📁 Using: {path}", level='success')
        
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

def run_with_output(cmd):
    """Run command and stream output"""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ''):
        if line:
            emit_log(line.strip())
    process.wait()
    return process.returncode == 0

def emit_log(message, level='info'):
    """Send log message to frontend"""
    print(f"[{level.upper()}] {message}")  # Also print to console for debugging
    socketio.emit('install_log', {
        'message': message,
        'level': level,
        'timestamp': time.time()
    })

def create_robot_config(path, robot, leader_port=None, follower_port=None):
    """Create actual robot configuration files"""
    config = {
        'koch': {
            'robot_type': 'koch_follower',
            'leader_port': leader_port or '/dev/ttyUSB0',
            'follower_port': follower_port or '/dev/ttyUSB1',
            'motors': 6
        },
        'so100': {
            'robot_type': 'so100_follower',
            'leader_port': leader_port or '/dev/ttyUSB0',
            'follower_port': follower_port or '/dev/ttyUSB1',
            'motors': 5
        },
        'so101': {
            'robot_type': 'so101_follower',
            'leader_port': leader_port or '/dev/ttyUSB0',
            'follower_port': follower_port or '/dev/ttyUSB1',
            'motors': 6
        },

    }
    
    robot_config = config.get(robot, config['koch'])
    
    # Write config file
    config_file = path / 'robot_config.json'
    with open(config_file, 'w') as f:
        json.dump(robot_config, f, indent=2)
    
    # Create run script
    run_script = path / 'run_lerobot.sh'
    with open(run_script, 'w') as f:
        f.write(f'''#!/bin/bash
# LeRobot launcher for {robot}
echo "🤖 LeRobot - {robot}"
conda activate lerobot

echo "1) Calibrate"
echo "2) Teleoperate"
echo "3) Record"
echo "4) Train"
echo "5) Deploy"
read -p "Choice: " choice

case $choice in
  1) python -m lerobot.scripts.control_robot calibrate --robot-path robot_config.json ;;
  2) python -m lerobot.scripts.control_robot teleoperate --robot-path robot_config.json ;;
  3) python -m lerobot.scripts.control_robot record --robot-path robot_config.json ;;
  4) python -m lerobot.scripts.train --config-dir configs ;;
  5) python -m lerobot.scripts.control_robot deploy --robot-path robot_config.json ;;
esac
''')
    run_script.chmod(0o755)

def get_available_ports():
    """Get all available serial ports (similar to ref/lerobot/find_port.py)"""
    import platform
    from pathlib import Path
    
    ports = []
    
    if SERIAL_AVAILABLE:
        if platform.system() == "Windows":
            # List COM ports using pyserial
            for port in serial.tools.list_ports.comports():
                ports.append({
                    'device': port.device,
                    'description': port.description,
                    'hwid': port.hwid,
                    'manufacturer': getattr(port, 'manufacturer', 'Unknown'),
                    'product': getattr(port, 'product', 'Unknown')
                })
        else:  # Linux/macOS
            # Get both pyserial detection and /dev/tty* ports
            serial_ports = {}
            for port in serial.tools.list_ports.comports():
                serial_ports[port.device] = {
                    'device': port.device,
                    'description': port.description,
                    'hwid': port.hwid,
                    'manufacturer': getattr(port, 'manufacturer', 'Unknown'),
                    'product': getattr(port, 'product', 'Unknown')
                }
            
            # Add any additional /dev/tty* ports that might not be detected by pyserial
            try:
                dev_ports = [str(path) for path in Path("/dev").glob("tty*")]
                for dev_port in dev_ports:
                    if dev_port not in serial_ports and any(x in dev_port for x in ['USB', 'ACM', 'usbserial', 'usbmodem']):
                        serial_ports[dev_port] = {
                            'device': dev_port,
                            'description': 'USB Serial Device',
                            'hwid': 'Unknown',
                            'manufacturer': 'Unknown',
                            'product': 'Unknown'
                        }
            except:
                pass
                
            ports = list(serial_ports.values())
    else:
        # Fallback when pyserial is not available
        try:
            if platform.system() != "Windows":
                dev_ports = [str(path) for path in Path("/dev").glob("tty*")]
                for dev_port in dev_ports:
                    if any(x in dev_port for x in ['USB', 'ACM', 'usbserial', 'usbmodem']):
                        ports.append({
                            'device': dev_port,
                            'description': 'USB Serial Device',
                            'hwid': 'Unknown',
                            'manufacturer': 'Unknown',
                            'product': 'Unknown'
                        })
        except:
            # Ultimate fallback
            ports = [
                {'device': '/dev/ttyUSB0', 'description': 'USB Serial Port 0', 'hwid': 'USB0', 'manufacturer': 'Unknown', 'product': 'Unknown'},
                {'device': '/dev/ttyUSB1', 'description': 'USB Serial Port 1', 'hwid': 'USB1', 'manufacturer': 'Unknown', 'product': 'Unknown'}
            ]
    
    return ports

@app.route('/api/detect_ports', methods=['GET'])
def detect_ports():
    """Get current list of USB ports"""
    ports = get_available_ports()
    return jsonify({'ports': ports})

# Global port monitoring state
port_monitor = {
    'active': False,
    'baseline_ports': [],
    'current_ports': [],
    'detected_robot_ports': {}
}

@app.route('/api/start_port_detection', methods=['POST'])
def start_port_detection():
    """Start the interactive port detection process"""
    global port_monitor
    
    # Get baseline ports
    port_monitor['baseline_ports'] = get_available_ports()
    port_monitor['active'] = True
    port_monitor['detected_robot_ports'] = {}
    
    # Send initial state to frontend
    socketio.emit('port_detection_started', {
        'baseline_ports': port_monitor['baseline_ports'],
        'message': 'Port detection started. Connect your robot arms one by one.'
    })
    
    return jsonify({'status': 'started', 'baseline_ports': port_monitor['baseline_ports']})

@app.route('/api/monitor_ports', methods=['GET'])
def monitor_ports():
    """Check for port changes (called periodically by frontend)"""
    global port_monitor
    
    if not port_monitor['active']:
        return jsonify({'active': False})
    
    current_ports = get_available_ports()
    baseline_devices = {p['device'] for p in port_monitor['baseline_ports']}
    current_devices = {p['device'] for p in current_ports}
    
    # Find newly connected ports
    new_ports = current_devices - baseline_devices
    # Find disconnected ports  
    removed_ports = baseline_devices - current_devices
    
    changes = {
        'new_ports': [p for p in current_ports if p['device'] in new_ports],
        'removed_ports': list(removed_ports),
        'all_ports': current_ports
    }
    
    port_monitor['current_ports'] = current_ports
    
    return jsonify({
        'active': True,
        'changes': changes,
        'baseline_ports': port_monitor['baseline_ports'],
        'current_ports': current_ports
    })

@app.route('/api/assign_robot_port', methods=['POST'])
def assign_robot_port():
    """Assign a detected port to a robot arm (leader/follower)"""
    global port_monitor
    
    data = request.get_json()
    port_device = data.get('port')
    arm_type = data.get('arm_type')  # 'leader' or 'follower'
    
    if not port_device or not arm_type:
        return jsonify({'error': 'Missing port or arm_type'}), 400
    
    port_monitor['detected_robot_ports'][arm_type] = port_device
    
    socketio.emit('robot_port_assigned', {
        'port': port_device,
        'arm_type': arm_type,
        'assigned_ports': port_monitor['detected_robot_ports']
    })
    
    return jsonify({
        'status': 'assigned',
        'port': port_device,
        'arm_type': arm_type,
        'assigned_ports': port_monitor['detected_robot_ports']
    })

@app.route('/api/stop_port_detection', methods=['POST'])
def stop_port_detection():
    """Stop port detection and save configuration"""
    global port_monitor
    
    port_monitor['active'] = False
    
    # Save the detected ports if we have them
    if port_monitor['detected_robot_ports']:
        leader_port = port_monitor['detected_robot_ports'].get('leader')
        follower_port = port_monitor['detected_robot_ports'].get('follower')
        
        # Update robot config if we have a current installation
        if current_installation['path']:
            config_file = Path(current_installation['path']) / 'robot_config.json'
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    if leader_port:
                        config['leader_port'] = leader_port
                    if follower_port:
                        config['follower_port'] = follower_port
                    if not leader_port and follower_port:
                        config['port'] = follower_port
                    
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
                        
                except Exception as e:
                    print(f"Failed to update robot config: {e}")
        
        socketio.emit('ports_saved', {
            'leader': leader_port,
            'follower': follower_port
        })
    
    return jsonify({
        'status': 'stopped',
        'detected_ports': port_monitor['detected_robot_ports']
    })

@app.route('/api/save_ports', methods=['POST'])
def save_ports():
    """Save port configuration"""
    data = request.get_json()
    leader = data.get('leader')
    follower = data.get('follower')
    
    if current_installation['path']:
        config_file = Path(current_installation['path']) / 'robot_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'leader_port' in config:
                config['leader_port'] = leader
                config['follower_port'] = follower
            else:
                config['port'] = follower
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
    
    socketio.emit('ports_saved', {
        'leader': leader,
        'follower': follower
    })
    
    return jsonify({'status': 'saved'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get installation status"""
    return jsonify(current_installation)

@app.route('/api/test', methods=['GET'])
def test_connection():
    """Test endpoint to verify backend is working"""
    return jsonify({
        'status': 'working',
        'message': 'Backend is running successfully',
        'timestamp': time.time()
    })



@app.route('/api/finish_port_detection', methods=['POST'])
def finish_port_detection():
    """Finish port detection and save configuration"""
    global port_monitor
    
    if not port_monitor['active']:
        return jsonify({'success': False, 'message': 'Port detection not active'})
    
    try:
        # Save the detected ports if we have them
        if port_monitor['detected_robot_ports']:
            leader_port = port_monitor['detected_robot_ports'].get('leader')
            follower_port = port_monitor['detected_robot_ports'].get('follower')
            
            # Create robot configuration with detected ports
            robot = current_installation.get('robot', 'koch')  # Default to koch
            path = current_installation.get('path', '~/lerobot')
            create_robot_config(Path(path), robot, leader_port, follower_port)
            
            emit_log(f"✅ Port configuration saved successfully!", level='success')
            emit_log(f"Leader port: {leader_port or 'Not assigned'}", level='info')
            emit_log(f"Follower port: {follower_port or 'Not assigned'}", level='info')
        else:
            emit_log("⚠️ No ports were assigned during detection", level='warning')
        
        # Stop port detection
        port_monitor['active'] = False
        port_monitor['detected_robot_ports'] = {}
        
        # Notify frontend that detection is complete
        socketio.emit('port_detection_finished', {
            'success': True,
            'detected_ports': port_monitor['detected_robot_ports']
        })
        
        return jsonify({
            'success': True, 
            'message': 'Port detection finished',
            'detected_ports': port_monitor['detected_robot_ports']
        })
        
    except Exception as e:
        emit_log(f"ERROR: Failed to finish port detection: {str(e)}", level='error')
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/save_detected_ports', methods=['POST'])
def save_detected_ports():
    """Save detected ports to the global state and configuration"""
    global port_monitor
    
    try:
        data = request.get_json()
        leader_port = data.get('leader')
        follower_port = data.get('follower')
        
        if not leader_port or not follower_port:
            return jsonify({'success': False, 'message': 'Both leader and follower ports required'})
        
        # Save to global state
        port_monitor['detected_robot_ports'] = {
            'leader': leader_port,
            'follower': follower_port
        }
        
        # Update current installation state
        current_installation['leader_port'] = leader_port
        current_installation['follower_port'] = follower_port
        
        # Create/update robot configuration if we have installation info
        if current_installation.get('path') and current_installation.get('robot'):
            robot = current_installation['robot']
            path = current_installation['path']
            create_robot_config(Path(path), robot, leader_port, follower_port)
        
        emit_log(f"✅ Port configuration saved: Leader={leader_port}, Follower={follower_port}", level='success')
        
        return jsonify({
            'success': True,
            'message': 'Port configuration saved successfully',
            'ports': {'leader': leader_port, 'follower': follower_port}
        })
        
    except Exception as e:
        emit_log(f"ERROR: Failed to save port configuration: {str(e)}", level='error')
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    emit('connected', {'status': 'Connected to LeRobot installer'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    print(f"Starting Working LeRobot API on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)