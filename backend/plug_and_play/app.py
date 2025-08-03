"""
Flask backend API for the LeRobot Installation Assistant.
Provides REST API endpoints and WebSocket communication for real-time updates.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import os
import json
from pathlib import Path
import subprocess
import shutil
from typing import Dict, Any, List
import sys

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up static directories with absolute paths
FRONTEND_DIR = PROJECT_ROOT / 'frontend'
STATIC_DIR = FRONTEND_DIR
PAGES_DIR = FRONTEND_DIR / 'pages'
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

app = Flask(__name__, 
            static_folder=str(FRONTEND_DIR),
            static_url_path='/')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'lerobot-installer-secret-key')
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state for installation process
installation_state = {
    'is_running': False,
    'current_step': None,
    'installation_path': None,
    'progress': 0,
    'status': 'ready'
}

class InstallationManager:
    """Manages the LeRobot installation process."""
    
    def __init__(self):
        self.is_running = False
        self.installation_path = None
        self.current_step = None
        
    def start_installation(self, installation_path: str):
        """Start the installation process in a separate thread."""
        if self.is_running:
            return False
            
        self.is_running = True
        self.installation_path = Path(installation_path)
        
        # Update global state
        installation_state['is_running'] = True
        installation_state['installation_path'] = installation_path
        installation_state['status'] = 'running'
        installation_state['progress'] = 0
        
        # Start installation in background thread
        thread = threading.Thread(target=self._run_installation)
        thread.daemon = True
        thread.start()
        
        return True
        
    def cancel_installation(self):
        """Cancel the running installation."""
        self.is_running = False
        installation_state['is_running'] = False
        installation_state['status'] = 'cancelled'
        self._emit_log("Installation cancelled by user")
        
    def _run_installation(self):
        """Execute all installation steps."""
        try:
            steps = [
                ('checking_prerequisites', 'Checking system requirements', self._check_prerequisites),
                ('cloning_repository', 'Cloning LeRobot repository', self._clone_repository),
                ('creating_environment', 'Creating conda environment', self._create_environment),
                ('installing_ffmpeg', 'Installing FFmpeg', self._install_ffmpeg),
                ('installing_lerobot', 'Installing LeRobot package', self._install_lerobot),
                ('detecting_ports', 'Setting up robotic arm connections', self._setup_arm_connections)
            ]
            
            total_steps = len(steps)
            
            for i, (step_id, step_name, step_function) in enumerate(steps):
                if not self.is_running:
                    return
                    
                self.current_step = step_id
                installation_state['current_step'] = step_id
                installation_state['progress'] = int((i / total_steps) * 100)
                
                self._emit_progress(step_id, step_name, installation_state['progress'])
                
                if not step_function():
                    self._emit_error(f"Installation failed at step: {step_name}")
                    installation_state['status'] = 'failed'
                    return
                    
            # Installation completed successfully
            installation_state['progress'] = 100
            installation_state['status'] = 'completed'
            self._emit_completion("Installation completed successfully!")
            
        except Exception as e:
            self._emit_error(f"Unexpected error: {str(e)}")
            installation_state['status'] = 'failed'
        finally:
            self.is_running = False
            installation_state['is_running'] = False
            
    def _check_prerequisites(self) -> bool:
        """Check if required tools are available."""
        self._emit_log("Checking system prerequisites...")
        
        # Check Conda
        if not self._check_command('conda'):
            self._emit_error("Conda is not installed or not in PATH. Please install Miniconda or Anaconda first.")
            return False
        self._emit_log("Conda found - OK")
        
        # Check Git
        if not self._check_command('git'):
            self._emit_error("Git is not installed or not in PATH. Please install Git first.")
            return False
        self._emit_log("Git found - OK")
        
        # Check installation directory
        try:
            self.installation_path.parent.mkdir(parents=True, exist_ok=True)
            self._emit_log(f"Installation directory prepared: {self.installation_path}")
        except Exception as e:
            self._emit_error(f"Cannot create installation directory: {str(e)}")
            return False
            
        return True
        
    def _clone_repository(self) -> bool:
        """Clone the LeRobot repository."""
        self._emit_log("Cloning LeRobot repository from GitHub...")
        
        if self.installation_path.exists():
            self._emit_log(f"Directory {self.installation_path} already exists, removing...")
            try:
                shutil.rmtree(self.installation_path)
            except Exception as e:
                self._emit_error(f"Could not remove existing directory: {str(e)}")
                return False
                
        command = f'git clone https://github.com/huggingface/lerobot.git "{self.installation_path}"'
        return self._run_command(command)
        
    def _create_environment(self) -> bool:
        """Create conda environment."""
        self._emit_log("Creating conda environment 'lerobot' with Python 3.10...")
        command = 'conda create -y -n lerobot python=3.10'
        return self._run_command(command)
        
    def _install_ffmpeg(self) -> bool:
        """Install FFmpeg."""
        self._emit_log("Installing FFmpeg from conda-forge...")
        command = 'conda install -y ffmpeg -c conda-forge -n lerobot'
        return self._run_command(command)
        
    def _install_lerobot(self) -> bool:
        """Install LeRobot package."""
        self._emit_log("Installing LeRobot package...")
        command = 'conda run -n lerobot pip install -e .'
        return self._run_command(command, cwd=self.installation_path)
    
    def _setup_arm_connections(self) -> bool:
        """Set up robotic arm USB connections through guided detection."""
        self._emit_log("Setting up robotic arm connections...")
        self._emit_log("This step will help you identify your robotic arm USB ports.")
        
        if not SERIAL_AVAILABLE:
            self._emit_log("Installing pyserial for USB port detection...")
            if not self._run_command('conda run -n lerobot pip install pyserial'):
                self._emit_log("WARNING: Could not install pyserial. Port detection will be skipped.")
                return True
        
        # Emit special event to trigger port detection UI
        socketio.emit('start_port_detection', {
            'message': 'Please follow the on-screen instructions to identify your robotic arm ports',
            'step': 'detecting_ports'
        })
        
        self._emit_log("Please use the interactive interface to identify your arm ports.")
        self._emit_log("The system will guide you through connecting and disconnecting each arm.")
        
        # Wait for port detection completion or user skip
        # In a real implementation, this would wait for a completion signal
        import time
        time.sleep(5)  # Give user time to see the instruction
        
        # Copy port detection tools to installation directory
        self._copy_port_detection_tools()
        
        self._emit_log("Port detection step ready. You can configure ports now or later.")
        return True
        
    def _run_command(self, command: str, cwd=None) -> bool:
        """Execute a shell command."""
        if not self.is_running:
            return False
            
        try:
            self._emit_log(f"Executing: {command}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                shell=True
            )
            
            while True:
                if not self.is_running:
                    process.terminate()
                    return False
                    
                line = process.stdout.readline()
                if not line:
                    break
                    
                self._emit_log(line.rstrip())
                
            process.wait()
            
            if process.returncode != 0:
                self._emit_error(f"Command failed with exit code: {process.returncode}")
                return False
                
            return True
            
        except Exception as e:
            self._emit_error(f"Error executing command: {str(e)}")
            return False
            
    def _check_command(self, command: str) -> bool:
        """Check if a command is available."""
        try:
            result = subprocess.run(
                [command, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
            
    def _emit_log(self, message: str):
        """Emit log message to frontend."""
        socketio.emit('log_message', {'message': message})
        
    def _emit_progress(self, step: str, description: str, progress: int):
        """Emit progress update to frontend."""
        socketio.emit('progress_update', {
            'step': step,
            'description': description,
            'progress': progress
        })
        
    def _emit_error(self, message: str):
        """Emit error message to frontend."""
        socketio.emit('error_message', {'message': message})
        
    def _emit_completion(self, message: str):
        """Emit completion message to frontend."""
        socketio.emit('installation_complete', {'message': message})
        
    def _copy_port_detection_tools(self):
        """Copy port detection utilities to installation directory."""
        try:
            utils_dir = PROJECT_ROOT / "backend" / "plug_and_play" / "utils"
            source_file = utils_dir / "lerobot" / "find_port.py"
            
            if source_file.exists():
                target_dir = self.installation_path / "lerobot"
                target_dir.mkdir(exist_ok=True)
                shutil.copy2(source_file, target_dir / "find_port.py")
                self._emit_log("✅ Port detection tool copied to installation directory")
                self._emit_log("   Run interactive detection with: python lerobot/find_port.py")
            else:
                self._emit_log("⚠️  Port detection tool not found in utils directory")
                
        except Exception as e:
            self._emit_log(f"Failed to copy port detection tools: {e}")

# Global installation manager
installer = InstallationManager()

# Port detection utilities

class PortDetectionManager:
    """Manages USB port detection for robotic arms."""
    
    def __init__(self):
        self.initial_ports = []
        self.current_ports = []
        self.leader_port = None
        self.follower_port = None
    
    def get_available_ports(self) -> List[Dict[str, Any]]:
        """Get list of available serial ports."""
        if not SERIAL_AVAILABLE:
            return []
        
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid,
                'vid': getattr(port, 'vid', None),
                'pid': getattr(port, 'pid', None),
                'serial_number': getattr(port, 'serial_number', None)
            })
        return ports
    
    def detect_port_changes(self, previous_ports: List[str]) -> Dict[str, List[str]]:
        """Detect changes in available ports."""
        current_ports = [port['device'] for port in self.get_available_ports()]
        previous_port_names = set(previous_ports)
        current_port_names = set(current_ports)
        
        return {
            'added': list(current_port_names - previous_port_names),
            'removed': list(previous_port_names - current_port_names),
            'current': current_ports
        }
    
    def save_port_configuration(self, leader_port: str, follower_port: str) -> bool:
        """Save port configuration to file."""
        try:
            config_content = f'''# LeRobot Port Configuration
# Generated by USB Port Detection Tool

LEADER_ARM_PORT = "{leader_port}"
FOLLOWER_ARM_PORT = "{follower_port}"

# Port Details:
# Leader Arm:  {leader_port}
# Follower Arm: {follower_port}
'''
            config_file = PROJECT_ROOT / 'backend' / 'lerobot_ports.py'
            
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            self.leader_port = leader_port
            self.follower_port = follower_port
            
            return True
            
        except Exception as e:
            print(f"Failed to save port configuration: {e}")
            return False

# Global port detection manager
port_detector = PortDetectionManager()

# REST API endpoints

@app.route('/')
def serve_frontend():
    """Serve the main HTML page."""
    return send_from_directory(str(FRONTEND_DIR), 'index.html')

@app.route('/pages/<path:filename>')
def serve_pages(filename):
    """Serve page files."""
    return send_from_directory(str(PAGES_DIR), filename)

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc)."""
    # Security check: don't allow path traversal
    if '..' in path:
        return 'Forbidden', 403
    return send_from_directory(str(FRONTEND_DIR), path)

@app.route('/api/browse-directory', methods=['POST'])
def browse_directory():
    """List contents of a directory for browsing."""
    try:
        data = request.get_json()
        current_path = data.get('path', str(Path.home()))
        
        # Normalize and validate the path
        try:
            path_obj = Path(current_path).resolve()
        except (OSError, ValueError):
            path_obj = Path.home()
        
        # Ensure path exists and is a directory
        if not path_obj.exists() or not path_obj.is_dir():
            path_obj = Path.home()
        
        try:
            # Get directory contents
            items = []
            
            # Add parent directory entry if not at root
            if path_obj.parent != path_obj:
                items.append({
                    'name': '..',
                    'path': str(path_obj.parent),
                    'type': 'parent',
                    'is_dir': True
                })
            
            # List directory contents
            for item in sorted(path_obj.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                try:
                    # Skip hidden files/folders on non-Windows systems
                    if item.name.startswith('.') and os.name != 'nt':
                        continue
                    
                    # Skip system/protected directories
                    if item.name in ['System Volume Information', '$Recycle.Bin', 'Recovery']:
                        continue
                        
                    items.append({
                        'name': item.name,
                        'path': str(item),
                        'type': 'directory' if item.is_dir() else 'file',
                        'is_dir': item.is_dir(),
                        'size': item.stat().st_size if item.is_file() else None,
                        'modified': item.stat().st_mtime if item.exists() else None
                    })
                except (PermissionError, OSError):
                    # Skip items we can't access
                    continue
                    
            return jsonify({
                'current_path': str(path_obj),
                'items': items,
                'can_create': True  # Assume we can create directories here
            })
            
        except PermissionError:
            return jsonify({
                'error': 'Permission denied',
                'current_path': str(Path.home()),
                'items': [],
                'can_create': False
            }), 403
            
    except Exception as e:
        return jsonify({
            'error': f'Failed to browse directory: {str(e)}',
            'current_path': str(Path.home()),
            'items': [],
            'can_create': False
        }), 500

@app.route('/api/create-directory', methods=['POST'])
def create_directory():
    """Create a new directory."""
    try:
        data = request.get_json()
        parent_path = data.get('parent_path')
        directory_name = data.get('directory_name')
        
        if not parent_path or not directory_name:
            return jsonify({'error': 'Parent path and directory name are required'}), 400
        
        # Validate directory name
        if not directory_name.strip() or '/' in directory_name or '\\' in directory_name:
            return jsonify({'error': 'Invalid directory name'}), 400
        
        parent_dir = Path(parent_path).resolve()
        new_dir = parent_dir / directory_name.strip()
        
        # Check if directory already exists
        if new_dir.exists():
            return jsonify({'error': 'Directory already exists'}), 400
        
        # Create directory
        new_dir.mkdir(parents=True, exist_ok=False)
        
        return jsonify({
            'message': 'Directory created successfully',
            'path': str(new_dir)
        })
        
    except PermissionError:
        return jsonify({'error': 'Permission denied - cannot create directory here'}), 403
    except Exception as e:
        return jsonify({'error': f'Failed to create directory: {str(e)}'}), 500

@app.route('/api/get-home-directory', methods=['GET'])
def get_home_directory():
    """Get the user's home directory."""
    try:
        home_path = str(Path.home())
        return jsonify({
            'home_path': home_path,
            'suggested_paths': [
                str(Path.home()),
                str(Path.home() / 'Documents'),
                str(Path.home() / 'Desktop'),
                str(Path.home() / 'lerobot')
            ]
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get home directory: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current installation status."""
    return jsonify(installation_state)

@app.route('/api/start-installation', methods=['POST'])
def start_installation():
    """Start the installation process."""
    data = request.get_json()
    installation_path = data.get('installation_path')
    
    if not installation_path:
        return jsonify({'error': 'Installation path is required'}), 400
        
    if installer.start_installation(installation_path):
        return jsonify({'message': 'Installation started successfully'})
    else:
        return jsonify({'error': 'Installation is already running'}), 400

@app.route('/api/cancel-installation', methods=['POST'])
def cancel_installation():
    """Cancel the running installation."""
    installer.cancel_installation()
    return jsonify({'message': 'Installation cancelled'})

@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """Get system information and prerequisites status."""
    info = {
        'conda_available': installer._check_command('conda'),
        'git_available': installer._check_command('git'),
        'default_path': str(Path.home() / 'lerobot'),
        'serial_available': SERIAL_AVAILABLE
    }
    return jsonify(info)

@app.route('/api/list-ports', methods=['GET'])
def list_ports():
    """Get list of available serial ports."""
    if not SERIAL_AVAILABLE:
        return jsonify({
            'error': 'pyserial not installed',
            'ports': [],
            'message': 'Please install pyserial: pip install pyserial'
        }), 400
    
    try:
        ports = port_detector.get_available_ports()
        return jsonify({
            'ports': ports,
            'count': len(ports)
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to list ports: {str(e)}',
            'ports': []
        }), 500

@app.route('/api/detect-port-changes', methods=['POST'])
def detect_port_changes():
    """Detect changes in available ports."""
    if not SERIAL_AVAILABLE:
        return jsonify({'error': 'pyserial not installed'}), 400
    
    try:
        data = request.get_json()
        previous_ports = data.get('previous_ports', [])
        
        changes = port_detector.detect_port_changes(previous_ports)
        return jsonify(changes)
        
    except Exception as e:
        return jsonify({'error': f'Failed to detect port changes: {str(e)}'}), 500

@app.route('/api/save-port-config', methods=['POST'])
def save_port_config():
    """Save detected port configuration."""
    try:
        data = request.get_json()
        leader_port = data.get('leader_port')
        follower_port = data.get('follower_port')
        
        if not leader_port or not follower_port:
            return jsonify({'error': 'Both leader and follower ports are required'}), 400
        
        if leader_port == follower_port:
            return jsonify({'error': 'Leader and follower ports cannot be the same'}), 400
        
        success = port_detector.save_port_configuration(leader_port, follower_port)
        
        if success:
            # Emit port configuration update to all clients
            socketio.emit('port_config_saved', {
                'leader_port': leader_port,
                'follower_port': follower_port
            })
            
            return jsonify({
                'message': 'Port configuration saved successfully',
                'leader_port': leader_port,
                'follower_port': follower_port
            })
        else:
            return jsonify({'error': 'Failed to save port configuration'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to save configuration: {str(e)}'}), 500

@app.route('/port-detection')
def serve_port_detection():
    """Serve the port detection page."""
    return send_from_directory(str(PAGES_DIR), 'port-detection.html')

# WebSocket events

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('status_update', installation_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    pass

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Use 0.0.0.0 for deployment, 127.0.0.1 for local development
    host = '0.0.0.0' if not debug else '127.0.0.1'
    
    socketio.run(app, host=host, port=port, debug=debug)