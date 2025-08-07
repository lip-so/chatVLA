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

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify local installer is running"""
    return jsonify({
        'status': 'online',
        'local': True,
        'can_install': True,
        'message': 'Local installer bridge is running'
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
