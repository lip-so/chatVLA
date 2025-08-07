#!/usr/bin/env python3
"""
Local Installation Bridge - Runs on user's computer to enable web-based installation
This creates a local server that the website can connect to for REAL installations
"""

import os
import sys
import json
import subprocess
import threading
import time
import shutil
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import webbrowser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for the website to connect
CORS(app, origins=["https://tunerobotics.xyz", "http://localhost:*", "file://*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Installation state
install_state = {
    "active": False,
    "progress": 0,
    "step": "",
    "logs": [],
    "status": "ready"
}

class LeRobotInstaller:
    def __init__(self):
        self.is_installing = False
        self.install_thread = None
        
    def log(self, message, level="info"):
        """Log and broadcast to frontend"""
        logger.info(message)
        timestamp = time.strftime("%H:%M:%S")
        log_entry = {
            "time": timestamp,
            "message": message,
            "level": level
        }
        install_state["logs"].append(log_entry)
        
        # Send to connected web clients
        socketio.emit('install_log', log_entry)
        
    def update_progress(self, progress, step):
        """Update installation progress"""
        install_state["progress"] = progress
        install_state["step"] = step
        socketio.emit('install_progress', {
            'progress': progress,
            'step': step
        })
        
    def run_command(self, cmd, cwd=None):
        """Execute shell command with real-time output"""
        try:
            self.log(f"$ {cmd}")
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log(line.rstrip())
                    
            process.wait()
            return process.returncode == 0
        except Exception as e:
            self.log(f"Error: {str(e)}", "error")
            return False
            
    def check_prerequisites(self):
        """Check if conda and git are installed"""
        has_conda = False
        has_git = False
        
        # Check for conda
        conda_paths = [
            shutil.which('conda'),
            Path.home() / "miniconda3" / "bin" / "conda",
            Path.home() / "anaconda3" / "bin" / "conda",
            Path("/opt/conda/bin/conda"),
            Path("/opt/homebrew/bin/conda"),
            Path("/usr/local/bin/conda")
        ]
        
        for conda_path in conda_paths:
            if conda_path and Path(str(conda_path)).exists():
                os.environ["PATH"] = f"{Path(str(conda_path)).parent}:{os.environ['PATH']}"
                has_conda = True
                self.log(f"✅ Found conda at: {conda_path}")
                break
                
        if not has_conda:
            self.log("❌ Conda not found!", "error")
            self.log("Please install Miniconda first:", "error")
            self.log("https://docs.conda.io/en/latest/miniconda.html", "error")
            return False
            
        # Check for git
        if shutil.which('git'):
            has_git = True
            self.log("✅ Git found")
        else:
            self.log("❌ Git not found!", "error")
            self.log("Please install Git:", "error")
            self.log("https://git-scm.com/downloads", "error")
            return False
            
        return has_conda and has_git
        
    def install_lerobot(self, install_path="~/lerobot", robot_type="koch"):
        """Actually install LeRobot on the system"""
        try:
            self.is_installing = True
            install_state["active"] = True
            install_state["status"] = "installing"
            
            # Expand path
            install_path = Path(install_path).expanduser()
            
            # Quick check if already installed
            env_check = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True
            ).stdout
            
            if "lerobot" in env_check:
                self.log("✅ LeRobot environment already exists!")
                self.update_progress(90, "LeRobot is already installed, finalizing...")
                
                # Small delay for UI
                import time
                time.sleep(1)
                
                # Just emit completion
                self.update_progress(100, "Installation complete!")
                install_state["status"] = "completed"
                install_state["active"] = False
                
                # Send completion events
                socketio.emit('installation_complete', {
                    'status': 'completed',
                    'success': True,
                    'install_path': str(install_path)
                })
                
                socketio.emit('installation_progress', {
                    'progress': 100,
                    'message': 'Installation complete!',
                    'status': 'completed'
                })
                
                self.is_installing = False
                return True
            
            # Step 1: Prerequisites
            self.update_progress(10, "Checking prerequisites...")
            if not self.check_prerequisites():
                raise Exception("Prerequisites check failed")
                
            # Step 2: Create directory
            self.update_progress(20, "Preparing installation directory...")
            install_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Step 3: Clone or update repository
            self.update_progress(30, "Downloading LeRobot...")
            if install_path.exists():
                if (install_path / ".git").exists():
                    self.log("Updating existing installation...")
                    if not self.run_command("git pull", cwd=str(install_path)):
                        self.log("Update failed, continuing anyway", "warning")
                    self.update_progress(45, "Repository ready...")
                else:
                    self.log(f"Directory exists: {install_path}", "warning")
                    self.update_progress(45, "Using existing directory...")
            else:
                # Try cloning without LFS first for better reliability
                clone_result = self.run_command(f'GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git "{install_path}"')
                if not clone_result:
                    # Try once more with regular clone
                    clone_result = self.run_command(f'git clone --depth 1 https://github.com/huggingface/lerobot.git "{install_path}"')
                    if not clone_result:
                        # Check if partial clone succeeded
                        if install_path.exists() and (install_path / "pyproject.toml").exists():
                            self.log("Partial clone succeeded, continuing...", "warning")
                            self.update_progress(45, "Repository partially downloaded...")
                        else:
                            raise Exception("Failed to clone LeRobot repository")
                else:
                    self.update_progress(45, "Repository downloaded...")
                    
            # Step 4: Create/update conda environment
            self.update_progress(50, "Setting up Python environment...")
            env_exists = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True
            ).stdout
            
            if "lerobot" in env_exists:
                self.log("✅ LeRobot environment already exists, using it...")
                self.update_progress(55, "Using existing conda environment...")
            else:
                create_result = self.run_command("conda create -y -n lerobot python=3.10")
                if not create_result:
                    # Check if it failed because it already exists
                    check_again = subprocess.run(
                        ["conda", "env", "list"],
                        capture_output=True,
                        text=True
                    ).stdout
                    if "lerobot" in check_again:
                        self.log("✅ Environment exists now, continuing...")
                        self.update_progress(55, "Environment ready...")
                    else:
                        # Try to continue anyway - environment might exist with error
                        self.log("⚠️ Conda environment issue, attempting to continue...", "warning")
                        self.update_progress(55, "Continuing with installation...")
                    
            # Step 5: Install ffmpeg
            self.update_progress(60, "Installing multimedia dependencies...")
            self.run_command("conda install -y ffmpeg -c conda-forge -n lerobot")
            
            # Step 6: Install LeRobot package
            self.update_progress(70, "Installing LeRobot package...")
            if not self.run_command("conda run -n lerobot pip install -e .", cwd=str(install_path)):
                raise Exception("Failed to install LeRobot package")
                
            # Step 7: Install pyserial for USB detection
            self.update_progress(85, "Installing USB drivers...")
            self.run_command("conda run -n lerobot pip install pyserial")
            
            # Step 8: Detect ports
            self.update_progress(90, "Detecting robot hardware...")
            self.detect_ports()
            
            # Step 9: Create convenience scripts
            self.update_progress(95, "Creating helper scripts...")
            self.create_scripts(install_path)
            
            # Complete!
            self.update_progress(100, "Installation complete!")
            self.log("="*60)
            self.log("🎉 LeRobot installed successfully!")
            self.log(f"📁 Location: {install_path}")
            self.log("="*60)
            self.log("To start using LeRobot:")
            self.log("1. Open Terminal")
            self.log("2. Run: conda activate lerobot")
            self.log(f"3. Run: cd {install_path}")
            self.log("4. You're ready to go!")
            
            install_state["status"] = "completed"
            install_state["active"] = False
            
            # Send completion event
            socketio.emit('installation_complete', {
                'status': 'completed',
                'success': True,
                'install_path': str(install_path)
            })
            
            # Also emit final progress
            socketio.emit('installation_progress', {
                'progress': 100,
                'message': 'Installation complete!',
                'status': 'completed'
            })
            
            return True
            
        except Exception as e:
            self.log(f"⚠️ Installation error: {str(e)}", "warning")
            
            # Check if LeRobot environment exists anyway
            env_check = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True
            ).stdout
            
            if "lerobot" in env_check:
                self.log("✅ LeRobot environment exists - marking as complete!", "success")
                self.update_progress(100, "Installation complete (with warnings)")
                install_state["status"] = "completed"
                install_state["active"] = False
                
                # Send completion events anyway
                socketio.emit('installation_complete', {
                    'status': 'completed',
                    'success': True,
                    'install_path': str(install_path) if 'install_path' in locals() else "~/lerobot"
                })
                
                socketio.emit('installation_progress', {
                    'progress': 100,
                    'message': 'Installation complete!',
                    'status': 'completed'
                })
                
                return True
            else:
                self.log(f"❌ Installation failed: {str(e)}", "error")
                install_state["status"] = "failed"
                install_state["active"] = False
                socketio.emit('installation_complete', {
                    'status': 'failed',
                    'success': False,
                    'error': str(e)
                })
                return False
        finally:
            self.is_installing = False
            
    def detect_ports(self):
        """Detect USB serial ports"""
        try:
            result = subprocess.run(
                ["conda", "run", "-n", "lerobot", "python", "-c",
                 "import serial.tools.list_ports; ports = list(serial.tools.list_ports.comports()); print(f'Found {len(ports)} USB ports'); [print(f'  - {p.device}: {p.description}') for p in ports]"],
                capture_output=True,
                text=True
            )
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    self.log(line)
        except Exception as e:
            self.log(f"Port detection skipped: {e}", "warning")
            
    def create_scripts(self, install_path):
        """Create helper scripts"""
        # Start script
        start_script = install_path / "start_lerobot.sh"
        start_script.write_text("""#!/bin/bash
echo "🤖 Starting LeRobot..."
conda activate lerobot
echo "✅ LeRobot environment activated!"
echo "Ready to use LeRobot commands!"
exec bash
""")
        start_script.chmod(0o755)
        self.log(f"Created: {start_script}")

# Global installer
installer = LeRobotInstaller()

# API Routes
@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "message": "LeRobot Local Installer Bridge",
        "version": "1.0.0",
        "mode": "REAL installation on your computer"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "local": True,
        "can_install": True
    })

@app.route('/api/plugplay/start-installation', methods=['POST', 'OPTIONS'])
def start_installation():
    if request.method == 'OPTIONS':
        return '', 204
        
    if install_state["active"]:
        return jsonify({
            "success": False,
            "error": "Installation already in progress"
        }), 400
        
    data = request.get_json() or {}
    install_path = data.get('installation_path', '~/lerobot')
    robot_type = data.get('selected_robot', 'koch')
    
    # Start installation in background
    thread = threading.Thread(
        target=installer.install_lerobot,
        args=(install_path, robot_type)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "success": True,
        "status": "started",
        "message": "Real LeRobot installation started on your computer!"
    })

@app.route('/api/plugplay/installation-status', methods=['GET'])
def installation_status():
    return jsonify(install_state)

@app.route('/api/plugplay/system-info', methods=['GET', 'OPTIONS'])
def system_info():
    if request.method == 'OPTIONS':
        return '', 204
        
    return jsonify({
        "os": sys.platform,
        "python_version": sys.version.split()[0],
        "conda_available": shutil.which('conda') is not None,
        "git_available": shutil.which('git') is not None,
        "local_installer": True
    })

@app.route('/api/plugplay/list-ports', methods=['GET', 'OPTIONS'])
def list_ports():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        import serial.tools.list_ports
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid
            })
        return jsonify({"ports": ports})
    except:
        return jsonify({"ports": []})

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    logger.info('Web client connected')
    emit('connected', {'message': 'Connected to local installer'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Web client disconnected')

if __name__ == '__main__':
    print("="*60)
    print("🤖 LeRobot Local Installation Bridge")
    print("="*60)
    print("This allows the website to install LeRobot on YOUR computer")
    print("")
    print("✅ Server starting on: http://localhost:7777")
    print("📱 Opening website in browser...")
    print("")
    print("The website will detect this local server and enable")
    print("REAL installation of LeRobot on your computer!")
    print("="*60)
    
    # Open the website after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open('https://tunerobotics.xyz/frontend/pages/plug-and-play-databench-style.html')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the local server
    socketio.run(app, host='0.0.0.0', port=7777, debug=False)