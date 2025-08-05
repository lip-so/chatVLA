#!/usr/bin/env python3
"""
Cloud-deployable Plug & Play API for Tune Robotics website.
Provides LeRobot installation guidance and USB port information via web interface.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Try to import pyserial for USB port detection
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# Check if databench is available
try:
    from pathlib import Path
    databench_path = Path(__file__).parent.parent / "databench"
    DATABENCH_AVAILABLE = databench_path.exists()
except:
    DATABENCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tune-robotics-plug-and-play'
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/', methods=['GET'])
def index():
    """Serve the main plug & play page"""
    return """
    <h1>Tune Robotics - Plug & Play API</h1>
    <p>Cloud-deployed backend for automated LeRobot installation assistance.</p>
    <ul>
        <li><a href="/health">Health Check</a></li>
        <li><a href="/api/system_info">System Information</a></li>
        <li><a href="/api/installation_guide">Installation Guide</a></li>
    </ul>
    """

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Tune Robotics Plug & Play API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "capabilities": [
            "Installation guidance",
            "System requirements checking",
            "USB port information",
            "LeRobot setup instructions"
        ]
    })

@app.route('/api/system_info', methods=['GET'])
def system_info():
    """Provide system information and requirements"""
    return jsonify({
        "git_available": True,  # Cloud service assumption
        "git_version": "2.39+",
        "conda_available": True,
        "python_version": "3.10+",
        "requirements": {
            "git": {
                "required": True,
                "install_url": "https://git-scm.com/downloads",
                "description": "Git is required to clone the LeRobot repository"
            },
            "conda": {
                "required": True,
                "install_url": "https://docs.conda.io/en/latest/miniconda.html",
                "description": "Conda/Miniconda is required for environment management"
            },
            "python": {
                "required": True,
                "version": "3.8+",
                "description": "Python 3.8 or higher is required"
            }
        }
    })

@app.route('/api/installation_guide', methods=['GET'])
def installation_guide():
    """Provide step-by-step installation guide"""
    return jsonify({
        "title": "LeRobot Installation Guide",
        "description": "Complete setup instructions for LeRobot with robotic arms",
        "steps": [
            {
                "id": "prerequisites",
                "title": "Check Prerequisites",
                "description": "Ensure Git and Conda are installed",
                "commands": [
                    "git --version",
                    "conda --version"
                ],
                "help_links": {
                    "git": "https://git-scm.com/downloads",
                    "conda": "https://docs.conda.io/en/latest/miniconda.html"
                }
            },
            {
                "id": "clone_repository",
                "title": "Clone LeRobot Repository",
                "description": "Download the latest LeRobot code",
                "commands": [
                    "git clone https://github.com/huggingface/lerobot.git",
                    "cd lerobot"
                ]
            },
            {
                "id": "create_environment",
                "title": "Create Conda Environment",
                "description": "Set up isolated Python environment",
                "commands": [
                    "conda create -n lerobot python=3.10",
                    "conda activate lerobot"
                ]
            },
            {
                "id": "install_dependencies",
                "title": "Install Dependencies",
                "description": "Install LeRobot and required packages",
                "commands": [
                    "pip install -e .",
                    "pip install pyserial",  # For USB communication
                ]
            },
            {
                "id": "usb_setup",
                "title": "USB Port Setup",
                "description": "Configure USB ports for robotic arms",
                "instructions": [
                    "Connect your robotic arms via USB",
                    "Check available ports with the USB detection tool",
                    "Note the port names for configuration"
                ]
            },
            {
                "id": "verification",
                "title": "Verify Installation",
                "description": "Test that everything works",
                "commands": [
                    "python -c \"import lerobot; print('LeRobot installed successfully!')\"",
                    "python -c \"import serial; print('USB communication ready!')\""
                ]
            }
        ]
    })

@app.route('/api/start_installation', methods=['POST'])
def start_installation():
    """Provide installation instructions based on user's system"""
    data = request.get_json()
    installation_path = data.get('installation_path', '~/lerobot')
    
    # Generate customized installation script
    script_content = f"""#!/bin/bash
# Auto-generated script by Tune Robotics Plug & Play
echo "Starting LeRobot installation..."
echo "Installation path: {installation_path}"

cd {installation_path}
git clone https://github.com/huggingface/lerobot.git
cd lerobot

echo "Cloning LeRobot repository..."
git clone https://github.com/huggingface/lerobot.git

echo "Creating conda environment..."
conda create -n lerobot python=3.10 -y

echo "Installing dependencies..."
conda run -n lerobot pip install -e .

echo "Installation complete!"
echo "Connect your robotic arms and run USB detection"
echo "Activate environment with: conda activate lerobot"
"""
    
    return jsonify({
        "success": True,
        "message": "Installation script generated",
        "installation_path": installation_path,
        "script": script_content,
        "next_steps": [
            "Save the script to install_lerobot.sh",
            "Run: chmod +x install_lerobot.sh",
            "Execute: ./install_lerobot.sh",
            "Connect USB devices and scan ports"
        ]
    })

@app.route('/api/scan_usb_ports', methods=['GET'])
def scan_usb_ports():
    """Provide USB port detection guidance"""
    return jsonify({
        "message": "USB port detection guidance",
        "instructions": [
            "Connect your robotic arms via USB cables",
            "Use the following Python code to detect ports:",
        ],
        "detection_code": """
import serial.tools.list_ports

def scan_usb_ports():
    ports = []
    for port in serial.tools.list_ports.comports():
        ports.append({
            'device': port.device,
            'description': port.description,
            'hwid': port.hwid
        })
    return ports

# Run detection
detected_ports = scan_usb_ports()
for port in detected_ports:
    print(f"Found: {port['device']} - {port['description']}")
""",
        "common_ports": {
            "macOS": ["/dev/cu.usbmodem*", "/dev/tty.usbserial*"],
            "Linux": ["/dev/ttyACM*", "/dev/ttyUSB*"],
            "Windows": ["COM1", "COM2", "COM3", "..."]
        },
        "troubleshooting": [
            "Ensure USB drivers are installed",
            "Try different USB cables",
            "Check if devices appear in system device manager",
            "Verify robotic arms are powered on"
        ]
    })

@app.route('/api/cancel_installation', methods=['POST'])
def cancel_installation():
    """Handle installation cancellation"""
    return jsonify({
        "success": True,
        "message": "Installation guidance session ended",
        "note": "No active installation to cancel - this service provides guidance only"
    })

@app.route('/api/status', methods=['GET'])
def status():
    """Service status endpoint"""
    return jsonify({
        "status": "online",
        "service": "Tune Robotics Plug & Play",
        "mode": "guidance",
        "description": "Provides installation guidance and setup instructions"
    })

# WebSocket events for real-time guidance
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status', {
        'message': 'Connected to Tune Robotics Plug & Play service',
        'type': 'info'
    })
    logger.info('Client connected to guidance service')

@socketio.on('request_guidance')
def handle_guidance_request(data):
    """Provide real-time installation guidance"""
    step = data.get('step', 'start')
    
    guidance_steps = {
        'start': 'Welcome! Let\'s set up LeRobot on your system.',
        'prerequisites': 'First, ensure Git and Conda are installed on your system.',
        'clone': 'Clone the LeRobot repository to your chosen directory.',
        'environment': 'Create a dedicated conda environment for LeRobot.',
        'install': 'Install LeRobot and its dependencies.',
        'usb': 'Connect and configure your robotic arms.',
        'complete': 'Installation complete! Your robotic arms are ready to use.'
    }
    
    emit('guidance_update', {
        'step': step,
        'message': guidance_steps.get(step, 'Unknown step'),
        'type': 'info'
    })

class PlugPlayInstallationManager:
    """Manages LeRobot installation process"""
    
    def __init__(self):
        self.status = "idle"
        self.progress = 0
        self.message = "Ready to start installation"
        self.is_running = False
        self.install_path = None
        self.socketio = None  # Will be set by main.py after initialization
        
    def start_installation(self, install_path="./lerobot", selected_port=None):
        """Start the REAL installation process"""
        if self.is_running:
            return {
                "success": False,
                "error": "Installation already running",
                "status": "running"
            }
        
        import threading
        from pathlib import Path
        
        # Expand path
        install_path = str(Path(install_path).expanduser().resolve())
        self.install_path = install_path
        
        # Start real installation in background thread
        self.is_running = True
        self.status = "running"
        self.progress = 0
        self.message = "Starting real LeRobot installation..."
        
        thread = threading.Thread(target=self._run_real_installation, args=(install_path,))
        thread.daemon = True
        thread.start()
        
        return {
            "success": True,
            "status": "started",
            "message": "Real LeRobot installation started",
            "install_path": install_path,
            "selected_port": selected_port
        }
    
    def _run_real_installation(self, install_path):
        """Actually install LeRobot"""
        import subprocess
        import os
        from pathlib import Path
        
        print(f"🚀 STARTING REAL INSTALLATION to {install_path}")
        
        try:
            install_dir = Path(install_path)
            
            # Step 1: Check prerequisites
            self.progress = 10
            self.message = "Checking prerequisites..."
            self._emit_progress()
            
            # Check git
            try:
                result = subprocess.run(['git', '--version'], check=True, capture_output=True, text=True, timeout=30)
                self.message = f"✓ Git found: {result.stdout.strip()}"
                self._emit_progress()
                print(f"✓ Git check passed: {result.stdout.strip()}")
            except subprocess.TimeoutExpired:
                self._set_error("Git command timed out")
                return
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                self._set_error(f"Git not found. Error: {str(e)}")
                return
            
            # Check conda
            try:
                result = subprocess.run(['conda', '--version'], check=True, capture_output=True, text=True, timeout=30)
                self.message = f"✓ Conda found: {result.stdout.strip()}"
                self._emit_progress()
                print(f"✓ Conda check passed: {result.stdout.strip()}")
            except subprocess.TimeoutExpired:
                self._set_error("Conda command timed out")
                return
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                self._set_error(f"Conda not found. Error: {str(e)}")
                return
            
            # Step 2: Clone repository
            self.progress = 30
            self.message = "Cloning LeRobot repository from GitHub..."
            self._emit_progress()
            print(f"📥 Cloning repository to {install_dir}")
            
            # Remove existing directory if it exists
            if install_dir.exists():
                import shutil
                print(f"🗑️ Removing existing directory: {install_dir}")
                shutil.rmtree(install_dir)
            
            # Clone the repository
            print("🔄 Running git clone command...")
            result = subprocess.run([
                'git', 'clone', 
                'https://github.com/huggingface/lerobot.git', 
                str(install_dir)
            ], capture_output=True, text=True, timeout=300)
            
            print(f"Git clone exit code: {result.returncode}")
            if result.stdout:
                print(f"Git clone stdout: {result.stdout}")
            if result.stderr:
                print(f"Git clone stderr: {result.stderr}")
            
            if result.returncode != 0:
                self._set_error(f"Failed to clone repository: {result.stderr}")
                return
            
            self.message = f"✓ Repository cloned to {install_path}"
            self._emit_progress()
            print(f"✅ Repository successfully cloned to {install_path}")
            
            # Step 3: Create conda environment
            self.progress = 50
            self.message = "Creating conda environment 'lerobot' with Python 3.10..."
            self._emit_progress()
            
            result = subprocess.run([
                'conda', 'create', '-y', '-n', 'lerobot', 'python=3.10'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                self._set_error(f"Failed to create conda environment: {result.stderr}")
                return
            
            self.message = "✓ Conda environment 'lerobot' created"
            self._emit_progress()
            
            # Step 4: Install FFmpeg
            self.progress = 70
            self.message = "Installing FFmpeg from conda-forge (conda install ffmpeg -c conda-forge)..."
            self._emit_progress()
            print("📹 Running: conda install ffmpeg -c conda-forge -n lerobot...")
            
            result = subprocess.run([
                'conda', 'install', '-y', 'ffmpeg', '-c', 'conda-forge', '-n', 'lerobot'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                self._set_error(f"Failed to install FFmpeg: {result.stderr}")
                return
            
            self.message = "✓ FFmpeg installed"
            self._emit_progress()
            
            # Step 5: Install LeRobot
            self.progress = 90
            self.message = "Installing LeRobot package with pip install -e . (this may take several minutes)..."
            self._emit_progress()
            print("📦 Running: pip install -e . in the lerobot directory...")
            
            result = subprocess.run([
                'conda', 'run', '-n', 'lerobot', 'pip', 'install', '-e', '.'
            ], cwd=install_dir, capture_output=True, text=True, timeout=1200)
            
            print(f"pip install exit code: {result.returncode}")
            if result.stdout:
                print(f"pip install stdout (last 500 chars): {result.stdout[-500:]}")
            if result.stderr:
                print(f"pip install stderr: {result.stderr}")
            
            if result.returncode != 0:
                self._set_error(f"Failed to install LeRobot: {result.stderr}")
                return
            
            self.message = "✓ LeRobot package installed successfully"
            self._emit_progress()
            print("✅ LeRobot package installed!")
            
            # Step 6: Install pyserial for USB detection
            self.progress = 95
            self.message = "Installing pyserial for USB port detection..."
            self._emit_progress()
            print("🔌 Installing pyserial...")
            
            result = subprocess.run([
                'conda', 'run', '-n', 'lerobot', 'pip', 'install', 'pyserial'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Warning: pyserial installation failed: {result.stderr}")
            else:
                print("✅ pyserial installed!")
            
            # Installation complete
            self.progress = 100
            self.message = f"🎉 LeRobot successfully installed to {install_path}!"
            self.status = "completed"
            self.is_running = False
            self._emit_progress()
            
            # Print summary of what was done
            print("\n" + "="*60)
            print("✅ INSTALLATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Commands executed:")
            print(f"1. git clone https://github.com/huggingface/lerobot.git {install_path}")
            print("2. conda create -y -n lerobot python=3.10")
            print("3. conda install -y ffmpeg -c conda-forge -n lerobot")
            print(f"4. cd {install_path} && pip install -e .")
            print("5. pip install pyserial")
            print("\n📍 LeRobot is now installed and ready to use!")
            print(f"📁 Installation location: {install_path}")
            print("🐍 Conda environment: lerobot")
            print("\nTo use LeRobot:")
            print("  conda activate lerobot")
            print(f"  cd {install_path}")
            print("="*60 + "\n")
            
        except subprocess.TimeoutExpired:
            self._set_error("Installation timed out. Please try again.")
        except Exception as e:
            self._set_error(f"Installation failed: {str(e)}")
    
    def _set_error(self, error_message):
        """Set error state"""
        self.status = "failed"
        self.message = error_message
        self.is_running = False
        self._emit_progress()
        print(f"Installation Error: {error_message}")
    
    def _emit_progress(self):
        """Emit progress update via WebSocket and console"""
        status = self.get_status()
        print(f"Progress: {status['progress']}% - {status['message']}")
        
        # Emit WebSocket event if socketio is available
        if self.socketio:
            self.socketio.emit('installation_progress', {
                'progress': status['progress'],
                'message': status['message'],
                'status': status['status'],
                'install_path': status['install_path']
            })
    
    def get_status(self):
        """Get current installation status"""
        return {
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "is_running": self.is_running,
            "install_path": self.install_path
        }

class USBPortDetector:
    """Detects available USB ports"""
    
    def __init__(self):
        self.detected_ports = []
        
    def scan_ports(self):
        """Scan for available USB ports"""
        ports = []
        if SERIAL_AVAILABLE:
            try:
                for port in serial.tools.list_ports.comports():
                    port_info = {
                        'device': port.device,
                        'description': port.description,
                        'hwid': port.hwid,
                        'manufacturer': getattr(port, 'manufacturer', 'Unknown'),
                        'product': getattr(port, 'product', 'Unknown'),
                        'serial_number': getattr(port, 'serial_number', 'Unknown')
                    }
                    ports.append(port_info)
            except Exception as e:
                logger.error(f"Error scanning ports: {e}")
        
        self.detected_ports = ports
        return ports

if __name__ == '__main__':
    # Get port from environment (for deployment platforms)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("Starting Tune Robotics Plug & Play API")
    logger.info(f"Running on port {port}")
    
    # Run with host 0.0.0.0 for external access
    socketio.run(app, host='0.0.0.0', port=port, debug=False) 