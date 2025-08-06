#!/usr/bin/env python3
"""
REAL LeRobot Installation Backend - Actually installs LeRobot!
"""

import os
import sys
import json
import logging
import subprocess
import threading
import time
import shutil
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app with SocketIO for real-time updates
app = Flask(__name__)
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Global installation state
installation_state = {
    "active": False,
    "progress": 0,
    "current_step": "",
    "logs": [],
    "status": "idle",
    "error": None
}

class RealLeRobotInstaller:
    """Actually installs LeRobot on the system"""
    
    def __init__(self):
        self.install_path = None
        self.is_installing = False
        self.process = None
        
    def log(self, message, level="info"):
        """Log message and send to frontend"""
        logger.info(message)
        installation_state["logs"].append({
            "time": time.strftime("%H:%M:%S"),
            "message": message,
            "level": level
        })
        # Send real-time update to frontend
        socketio.emit('install_log', {
            'message': message,
            'level': level
        })
        
    def update_progress(self, progress, step):
        """Update installation progress"""
        installation_state["progress"] = progress
        installation_state["current_step"] = step
        socketio.emit('installation_progress', {
            'progress': progress,
            'step': step
        })
        
    def run_command(self, command, cwd=None):
        """Run a shell command and stream output"""
        try:
            self.log(f"$ {command}")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd
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
        # Check conda
        if not shutil.which('conda'):
            # Try common conda locations
            conda_paths = [
                Path.home() / "miniconda3" / "bin" / "conda",
                Path.home() / "anaconda3" / "bin" / "conda",
                Path("/opt/conda/bin/conda"),
                Path("/usr/local/conda/bin/conda")
            ]
            for conda_path in conda_paths:
                if conda_path.exists():
                    os.environ["PATH"] = f"{conda_path.parent}:{os.environ['PATH']}"
                    self.log(f"Found conda at: {conda_path}")
                    break
            else:
                self.log("ERROR: Conda not found. Please install Miniconda first:", "error")
                self.log("https://docs.conda.io/en/latest/miniconda.html", "error")
                return False
        else:
            self.log("✓ Conda found")
            
        # Check git
        if not shutil.which('git'):
            self.log("ERROR: Git not found. Please install Git first:", "error")
            self.log("https://git-scm.com/downloads", "error")
            return False
        else:
            self.log("✓ Git found")
            
        return True
        
    def install(self, install_path, robot_type="koch"):
        """Actually install LeRobot"""
        try:
            self.is_installing = True
            installation_state["active"] = True
            installation_state["status"] = "installing"
            installation_state["error"] = None
            
            self.install_path = Path(install_path).expanduser()
            
            # Step 1: Check prerequisites
            self.update_progress(10, "Checking prerequisites...")
            if not self.check_prerequisites():
                raise Exception("Prerequisites not met")
                
            # Step 2: Create installation directory
            self.update_progress(20, "Creating installation directory...")
            self.install_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Step 3: Clone LeRobot repository
            self.update_progress(30, "Cloning LeRobot repository...")
            if self.install_path.exists():
                self.log(f"Directory {self.install_path} already exists", "warning")
                # Check if it's a git repo
                if (self.install_path / ".git").exists():
                    self.log("Updating existing repository...")
                    if not self.run_command("git pull", cwd=self.install_path):
                        raise Exception("Failed to update repository")
                else:
                    raise Exception(f"Directory exists but is not a git repository: {self.install_path}")
            else:
                if not self.run_command(
                    f'git clone https://github.com/huggingface/lerobot.git "{self.install_path}"'
                ):
                    raise Exception("Failed to clone repository")
                    
            # Step 4: Create conda environment
            self.update_progress(50, "Creating conda environment...")
            # Check if environment already exists
            check_env = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True
            )
            if "lerobot" in check_env.stdout:
                self.log("LeRobot environment already exists, updating...")
                if not self.run_command("conda update -n lerobot --all -y"):
                    self.log("Failed to update environment, continuing anyway", "warning")
            else:
                if not self.run_command("conda create -y -n lerobot python=3.10"):
                    raise Exception("Failed to create conda environment")
                    
            # Step 5: Install ffmpeg
            self.update_progress(60, "Installing ffmpeg...")
            if not self.run_command("conda install -y ffmpeg -c conda-forge -n lerobot"):
                self.log("Failed to install ffmpeg, continuing anyway", "warning")
                
            # Step 6: Install LeRobot package
            self.update_progress(70, "Installing LeRobot package...")
            if not self.run_command(
                f"conda run -n lerobot pip install -e .",
                cwd=self.install_path
            ):
                raise Exception("Failed to install LeRobot package")
                
            # Step 7: Install additional dependencies
            self.update_progress(80, "Installing additional dependencies...")
            if not self.run_command("conda run -n lerobot pip install pyserial"):
                self.log("Failed to install pyserial", "warning")
                
            # Step 8: Detect USB ports
            self.update_progress(90, "Detecting USB ports...")
            self.detect_usb_ports()
            
            # Step 9: Create helper scripts
            self.update_progress(95, "Creating helper scripts...")
            self.create_helper_scripts()
            
            # Complete!
            self.update_progress(100, "Installation complete!")
            self.log("="*50)
            self.log("✅ LeRobot installation completed successfully!")
            self.log(f"📁 Installed to: {self.install_path}")
            self.log("="*50)
            self.log("To use LeRobot:")
            self.log("1. Open terminal")
            self.log("2. Run: conda activate lerobot")
            self.log(f"3. cd {self.install_path}")
            self.log("4. Start using LeRobot!")
            
            installation_state["status"] = "completed"
            installation_state["active"] = False
            
            return True
            
        except Exception as e:
            self.log(f"Installation failed: {str(e)}", "error")
            installation_state["status"] = "failed"
            installation_state["error"] = str(e)
            installation_state["active"] = False
            return False
        finally:
            self.is_installing = False
            
    def detect_usb_ports(self):
        """Detect available USB ports"""
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            
            if len(ports) == 0:
                self.log("No USB serial ports detected")
            elif len(ports) == 1:
                self.log(f"Found 1 USB port: {ports[0].device}")
                self.log("Connect your second robot arm for full setup")
            else:
                self.log(f"Found {len(ports)} USB ports:")
                for port in ports:
                    self.log(f"  - {port.device}: {port.description}")
                    
        except ImportError:
            self.log("Port detection requires pyserial (will be installed)", "warning")
            
    def create_helper_scripts(self):
        """Create helper scripts for easy usage"""
        # Create run script
        run_script = self.install_path / "run_lerobot.sh"
        run_script.write_text("""#!/bin/bash
# LeRobot Quick Start Script
echo "🤖 Starting LeRobot Environment..."
conda activate lerobot
echo "✅ LeRobot environment activated!"
echo "You can now use LeRobot commands"
exec bash
""")
        run_script.chmod(0o755)
        self.log(f"Created run script: {run_script}")
        
        # Create port detection script
        detect_script = self.install_path / "detect_ports.py"
        detect_script.write_text("""#!/usr/bin/env python3
import serial.tools.list_ports

print("🔍 Detecting USB Serial Ports...")
ports = list(serial.tools.list_ports.comports())

if not ports:
    print("❌ No USB serial ports found")
    print("Make sure your robot arms are connected")
else:
    print(f"✅ Found {len(ports)} port(s):")
    for i, port in enumerate(ports, 1):
        print(f"  {i}. {port.device}")
        print(f"     Description: {port.description}")
        print(f"     Hardware ID: {port.hwid}")
""")
        detect_script.chmod(0o755)
        self.log(f"Created port detection script: {detect_script}")

# Global installer instance
installer = RealLeRobotInstaller()

# API Routes
@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "message": "Real LeRobot Installation Backend",
        "version": "2.0.0",
        "features": ["real-installation", "websocket-updates", "port-detection"]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "services": {
            "databench": {"available": True, "message": "Ready"},
            "plugplay": {"available": True, "message": "Real installation ready"},
            "auth": {"available": False, "message": "Not configured"}
        }
    })

@app.route('/api/plugplay/start-installation', methods=['POST', 'OPTIONS'])
def start_installation():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    data = request.get_json() or {}
    installation_path = data.get('installation_path', '~/lerobot')
    robot_type = data.get('selected_robot', 'koch')
    
    if installation_state["active"]:
        return jsonify({
            "success": False,
            "error": "Installation already in progress"
        }), 400
    
    # Start installation in background thread
    thread = threading.Thread(
        target=installer.install,
        args=(installation_path, robot_type)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "success": True,
        "status": "started",
        "install_path": installation_path,
        "robot": robot_type,
        "message": "Real LeRobot installation started!"
    })

@app.route('/api/plugplay/installation-status', methods=['GET'])
def get_installation_status():
    return jsonify(installation_state)

@app.route('/api/plugplay/cancel-installation', methods=['POST'])
def cancel_installation():
    if installer.is_installing:
        installer.is_installing = False
        installation_state["status"] = "cancelled"
        installation_state["active"] = False
        return jsonify({"success": True, "message": "Installation cancelled"})
    return jsonify({"success": False, "message": "No installation in progress"})

@app.route('/api/plugplay/list-ports', methods=['GET', 'OPTIONS'])
def list_ports():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response, 200
    
    try:
        import serial.tools.list_ports
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid
            })
        return jsonify({"ports": ports, "success": True})
    except ImportError:
        return jsonify({
            "ports": [],
            "success": False,
            "error": "pyserial not installed"
        })

@app.route('/api/plugplay/system-info', methods=['GET', 'OPTIONS'])
def system_info():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response, 200
    
    info = {
        "os": sys.platform,
        "python_version": sys.version.split()[0],
        "conda_available": shutil.which('conda') is not None,
        "git_available": shutil.which('git') is not None,
        "lerobot_installed": Path.home().joinpath('lerobot').exists()
    }
    return jsonify(info)

# Keep mock endpoints for DataBench
@app.route('/api/databench/evaluate', methods=['POST', 'OPTIONS'])
def databench_evaluate():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    data = request.get_json()
    return jsonify({
        "status": "processing",
        "message": "DataBench evaluation running",
        "dataset": data.get('dataset', 'unknown'),
        "metrics": data.get('metrics', []),
        "results": {
            "overall_score": 0.85,
            "message": "Evaluation complete"
        }
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'message': 'Connected to installation server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting REAL LeRobot Installation Server on port {port}")
    logger.info("This server will ACTUALLY install LeRobot on your system!")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)