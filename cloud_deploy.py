#!/usr/bin/env python3
"""
Comprehensive cloud deployment with full DataBench and Plug & Play functionality
Designed to work on Railway cloud deployment
"""

import os
import sys
import json
import threading
import time
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from flask import Flask, Blueprint, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')

# Add backend to path
PROJECT_ROOT = Path(__file__).parent.absolute()
backend_path = PROJECT_ROOT / 'backend'
sys.path.insert(0, str(backend_path))

# Check for dependencies
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("Serial/pyserial not available - USB port detection will be limited")

# Try to import auth modules
try:
    from auth.firestore_service import get_firestore_service
    from auth.firebase_auth import requires_firebase_auth
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    logger.warning("Firebase auth not available - running without authentication")

# ================================
# DATABENCH FUNCTIONALITY
# ================================

# DataBench configuration
DATABENCH_PATH = backend_path / "databench"
RESULTS_DIR = DATABENCH_PATH / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Metric mapping for DataBench
METRIC_CODES = {
    'a': 'action_consistency',
    'v': 'visual_diversity', 
    'h': 'hfv_overall_score',
    't': 'trajectory_quality',
    'c': 'dataset_coverage',
    'r': 'robot_action_quality'
}

METRIC_NAMES = {
    'a': 'Action Consistency',
    'v': 'Visual Diversity',
    'h': 'High-Fidelity Vision', 
    't': 'Trajectory Quality',
    'c': 'Dataset Coverage',
    'r': 'Robot Action Quality'
}

class CloudDataBenchEvaluator:
    """Cloud-ready DataBench evaluator that works on Railway"""
    
    def __init__(self):
        self.databench_script = DATABENCH_PATH / "scripts" / "evaluate.py"
        
    def validate_request(self, data):
        """Validate the evaluation request"""
        errors = []
        
        if not data.get('dataset'):
            errors.append("Dataset path is required")
            
        if not data.get('metrics'):
            errors.append("At least one metric must be selected")
        else:
            # Validate metric codes
            invalid_metrics = [m for m in data['metrics'].split(',') if m not in METRIC_CODES]
            if invalid_metrics:
                errors.append(f"Invalid metrics: {', '.join(invalid_metrics)}")
                
        if data.get('subset'):
            try:
                subset = int(data['subset'])
                if subset <= 0 or subset > 10000:
                    errors.append("Subset size must be between 1 and 10000")
            except ValueError:
                errors.append("Subset size must be a valid number")
                
        return errors
        
    def build_command(self, data):
        """Build the databench evaluation command"""
        cmd = [
            "python", str(self.databench_script),
            "--data", data['dataset'],
            "--metrics", data['metrics']
        ]
        
        if data.get('subset'):
            cmd.extend(["--subset", str(data['subset'])])
            
        # Generate output filename
        dataset_name = data['dataset'].replace('/', '_').replace(':', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"results_{dataset_name}_{timestamp}.json"
        
        cmd.extend(["--output", str(output_file)])
        
        return cmd, output_file
        
    def run_evaluation(self, data):
        """Run the databench evaluation on cloud"""
        try:
            # Validate request
            errors = self.validate_request(data)
            if errors:
                return {"error": "; ".join(errors)}, 400
                
            # Build command
            cmd, output_file = self.build_command(data)
            
            logger.info(f"Running DataBench command: {' '.join(cmd)}")
            
            # Set environment variables
            env = os.environ.copy()
            if data.get('token'):
                env['HF_TOKEN'] = data['token']
            env['PYTHONPATH'] = str(DATABENCH_PATH)
            
            # Run evaluation with timeout
            result = subprocess.run(
                cmd,
                cwd=DATABENCH_PATH,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for cloud deployment
            )
            
            if result.returncode != 0:
                logger.error(f"DataBench evaluation failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout}")
                return {
                    "error": f"Evaluation failed: {result.stderr}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }, 500
                
            # Read results
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                logger.info(f"DataBench evaluation completed successfully")
                return {"results": results, "output_file": str(output_file)}, 200
            else:
                return {"error": "Results file not found", "stdout": result.stdout}, 500
                
        except subprocess.TimeoutExpired:
            logger.error("DataBench evaluation timed out")
            return {"error": "Evaluation timed out (max 1 hour)"}, 408
        except Exception as e:
            logger.exception("DataBench evaluation failed")
            return {"error": str(e)}, 500

# ================================
# PLUG & PLAY FUNCTIONALITY
# ================================

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

class CloudPlugPlayManager:
    """Cloud-ready Plug & Play manager for Railway deployment"""
    
    def __init__(self, socketio=None):
        self.socketio = socketio
        self.is_running = False
        
    def emit_log(self, message, level='info'):
        """Send log message to frontend via SocketIO"""
        if self.socketio:
            self.socketio.emit('install_log', {
                'message': message,
                'level': level,
                'timestamp': time.time()
            })
        logger.info(f"[{level.upper()}] {message}")
        
    def start_real_installation(self, install_path="./lerobot", robot="koch", use_existing=False):
        """Start real LeRobot installation on cloud"""
        global current_installation
        
        if self.is_running:
            return {
                "success": False,
                "error": "Installation already running",
                "status": "running"
            }
        
        current_installation['running'] = True
        current_installation['path'] = str(install_path)
        current_installation['robot'] = robot
        self.is_running = True
        
        # Start installation in background thread
        thread = threading.Thread(
            target=self._run_cloud_installation, 
            args=(install_path, robot, use_existing)
        )
        thread.daemon = True
        thread.start()
        
        return {
            "success": True,
            "status": "started",
            "message": "Real LeRobot installation started on cloud",
            "install_path": install_path,
            "robot": robot
        }
    
    def _run_cloud_installation(self, install_path, robot, use_existing=False):
        """Actually install LeRobot on the cloud server"""
        global current_installation
        
        try:
            current_installation['progress'] = 10
            self.emit_log(f"Starting cloud installation for {robot} robot...")
            self.emit_log(f"Installation path: {install_path}")
            
            # Step 1: Check prerequisites
            current_installation['step'] = 'checking_prerequisites'
            current_installation['progress'] = 20
            self.emit_log("Checking system prerequisites...")
            
            if not self._check_cloud_prerequisites():
                self.emit_log("Prerequisites check failed", level='error')
                return
            
            # Step 2: Clone repository if needed
            if not use_existing:
                current_installation['step'] = 'cloning_repository'
                current_installation['progress'] = 30
                self.emit_log("Cloning LeRobot repository...")
                
                if not self._clone_lerobot_repo(install_path):
                    self.emit_log("Repository cloning failed", level='error')
                    return
            
            # Step 3: Set up environment
            current_installation['step'] = 'creating_environment'
            current_installation['progress'] = 50
            self.emit_log("Setting up Python environment...")
            
            if not self._setup_cloud_environment():
                self.emit_log("Environment setup failed", level='error')
                return
            
            # Step 4: Install dependencies
            current_installation['step'] = 'installing_dependencies'
            current_installation['progress'] = 70
            self.emit_log("Installing LeRobot dependencies...")
            
            if not self._install_cloud_dependencies():
                self.emit_log("Dependency installation failed", level='error')
                return
            
            # Step 5: Configure robot
            current_installation['step'] = 'configuring_robot'
            current_installation['progress'] = 90
            self.emit_log(f"Configuring {robot} robot...")
            
            if not self._configure_robot(robot):
                self.emit_log("Robot configuration failed", level='error')
                return
            
            # Installation complete
            current_installation['progress'] = 100
            current_installation['step'] = 'completed'
            self.emit_log("LeRobot installation completed successfully on cloud!", level='success')
            self.emit_log(f"Robot type: {robot}", level='success')
            self.emit_log(f"Installation path: {install_path}", level='success')
            
            # Emit completion event
            if self.socketio:
                self.socketio.emit('installation_complete', {
                    'path': str(install_path),
                    'robot': robot,
                    'next_step': 'ready_for_use'
                })
            
        except Exception as e:
            self.emit_log(f"Installation failed: {str(e)}", level='error')
        finally:
            current_installation['running'] = False
            self.is_running = False
    
    def _check_cloud_prerequisites(self):
        """Check if cloud environment has required tools"""
        try:
            # Check Python
            result = subprocess.run(['python', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.emit_log("Python not available", level='error')
                return False
            
            python_version = result.stdout.strip()
            self.emit_log(f"Found {python_version}")
            
            # Check git
            result = subprocess.run(['git', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.emit_log("Git not available", level='error')
                return False
            
            git_version = result.stdout.strip()
            self.emit_log(f"Found {git_version}")
            
            return True
            
        except Exception as e:
            self.emit_log(f"Prerequisites check error: {str(e)}", level='error')
            return False
    
    def _clone_lerobot_repo(self, install_path):
        """Clone LeRobot repository"""
        try:
            # Create installation directory
            install_dir = Path(install_path)
            install_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone the repository
            clone_cmd = [
                'git', 'clone', 
                'https://github.com/huggingface/lerobot.git',
                str(install_dir / 'lerobot')
            ]
            
            result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.emit_log(f"Git clone failed: {result.stderr}", level='error')
                return False
            
            self.emit_log("Repository cloned successfully")
            return True
            
        except Exception as e:
            self.emit_log(f"Repository cloning error: {str(e)}", level='error')
            return False
    
    def _setup_cloud_environment(self):
        """Set up Python environment for cloud deployment"""
        try:
            # For cloud deployment, we'll work with the existing environment
            self.emit_log("Using existing cloud environment")
            
            # Verify pip is available
            result = subprocess.run(['pip', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.emit_log("pip not available", level='error')
                return False
            
            pip_version = result.stdout.strip()
            self.emit_log(f"Found {pip_version}")
            
            return True
            
        except Exception as e:
            self.emit_log(f"Environment setup error: {str(e)}", level='error')
            return False
    
    def _install_cloud_dependencies(self):
        """Install LeRobot dependencies in cloud environment"""
        try:
            # Install essential dependencies for cloud operation
            dependencies = [
                'torch',
                'torchvision', 
                'transformers',
                'datasets',
                'opencv-python',
                'numpy',
                'scipy',
                'matplotlib',
                'pyyaml'
            ]
            
            for dep in dependencies:
                self.emit_log(f"Installing {dep}...")
                result = subprocess.run(
                    ['pip', 'install', dep], 
                    capture_output=True, text=True, timeout=180
                )
                
                if result.returncode != 0:
                    self.emit_log(f"Failed to install {dep}: {result.stderr}", level='warning')
                    # Continue with other dependencies
                else:
                    self.emit_log(f"Successfully installed {dep}")
            
            return True
            
        except Exception as e:
            self.emit_log(f"Dependency installation error: {str(e)}", level='error')
            return False
    
    def _configure_robot(self, robot):
        """Configure robot settings for cloud operation"""
        try:
            self.emit_log(f"Configuring {robot} robot for cloud operation...")
            
            # Create robot configuration based on robot type
            robot_configs = {
                'koch': {
                    'dof': 6,
                    'type': 'leader_follower',
                    'description': 'Koch Follower (6-DOF leader-follower arm)'
                },
                'so100': {
                    'dof': 5,
                    'type': 'desktop_arm',
                    'description': 'SO-100 Follower (5-DOF desktop arm)'
                },
                'so101': {
                    'dof': 6,
                    'type': 'precision_arm',
                    'description': 'SO-101 Follower (6-DOF precision arm)'
                }
            }
            
            config = robot_configs.get(robot, robot_configs['koch'])
            
            self.emit_log(f"Robot configured: {config['description']}")
            current_installation['robot_config'] = config
            
            return True
            
        except Exception as e:
            self.emit_log(f"Robot configuration error: {str(e)}", level='error')
            return False
    
    def get_usb_ports(self):
        """Get available USB ports on cloud server"""
        ports = []
        
        if SERIAL_AVAILABLE:
            try:
                available_ports = serial.tools.list_ports.comports()
                ports = [
                    {
                        'device': port.device,
                        'description': port.description,
                        'hwid': port.hwid
                    }
                    for port in available_ports
                ]
            except Exception as e:
                logger.warning(f"USB port detection failed: {e}")
        
        return ports

# ================================
# FLASK APPLICATION SETUP
# ================================

# Initialize Flask app
app = Flask(__name__, 
            static_folder='frontend',
            static_url_path='')

# App configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cloud-deploy-secret-key')

# Initialize extensions
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize managers
databench_evaluator = CloudDataBenchEvaluator()
plugplay_manager = CloudPlugPlayManager(socketio)

# ================================
# DATABENCH API ROUTES
# ================================

databench_bp = Blueprint('databench', __name__, url_prefix='/api/databench')

@databench_bp.route('/metrics', methods=['GET'])
def get_databench_metrics():
    """Get available DataBench metrics"""
    metrics = [
        {
            'code': code,
            'name': METRIC_NAMES[code],
            'description': f"Evaluate {METRIC_NAMES[code].lower()}"
        }
        for code in METRIC_CODES.keys()
    ]
    return jsonify({"metrics": metrics})

@databench_bp.route('/evaluate', methods=['POST'])
def evaluate_dataset():
    """Evaluate dataset using DataBench"""
    try:
        data = request.get_json()
        logger.info(f"DataBench evaluation request: {data}")
        
        result, status_code = databench_evaluator.run_evaluation(data)
        return jsonify(result), status_code
        
    except Exception as e:
        logger.exception("DataBench evaluation error")
        return jsonify({"error": str(e)}), 500

# ================================
# PLUG & PLAY API ROUTES
# ================================

plugplay_bp = Blueprint('plugplay', __name__, url_prefix='/api/plugplay')

@plugplay_bp.route('/start-installation', methods=['POST'])
def start_plugplay_installation():
    """Start LeRobot installation"""
    try:
        data = request.get_json()
        logger.info(f"Plug & Play installation request: {data}")
        
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Handle different parameter formats
        path = data.get('installation_path') or data.get('path', './lerobot')
        robot = data.get('selected_robot') or data.get('robot', 'koch')
        use_existing = data.get('use_existing', False)
        
        result = plugplay_manager.start_real_installation(path, robot, use_existing)
        return jsonify(result)
        
    except Exception as e:
        logger.exception("Plug & Play installation error")
        return jsonify({"success": False, "error": str(e)}), 500

@plugplay_bp.route('/installation-status', methods=['GET'])
def get_installation_status():
    """Get installation status"""
    return jsonify(current_installation)

@plugplay_bp.route('/cancel-installation', methods=['POST'])
def cancel_plugplay_installation():
    """Cancel running installation"""
    global current_installation
    current_installation['running'] = False
    plugplay_manager.is_running = False
    plugplay_manager.emit_log("Installation cancelled by user", level='warning')
    return jsonify({'success': True, 'message': 'Installation cancelled'})

@plugplay_bp.route('/system-info', methods=['GET'])
def get_system_info():
    """Get system information"""
    return jsonify({
        "os": sys.platform,
        "python_version": sys.version,
        "capabilities": {
            "usb_detection": SERIAL_AVAILABLE,
            "installation": True,
            "databench": True,
            "auth": AUTH_AVAILABLE
        },
        "cloud_deployment": True
    })

@plugplay_bp.route('/list-ports', methods=['GET'])
def list_usb_ports():
    """List available USB ports"""
    ports = plugplay_manager.get_usb_ports()
    return jsonify({"ports": ports})

@plugplay_bp.route('/save-port-config', methods=['POST'])
def save_port_configuration():
    """Save port configuration"""
    try:
        data = request.get_json()
        # In cloud deployment, save configuration to global state
        current_installation['leader_port'] = data.get('leader_port')
        current_installation['follower_port'] = data.get('follower_port')
        
        return jsonify({'success': True, 'message': 'Port configuration saved'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================================
# STATIC FILE ROUTES
# ================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/pages/<path:filename>')
def serve_page(filename):
    return send_from_directory('pages', filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('js', filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'mode': 'cloud_deploy',
        'features': {
            'databench': True,
            'plugplay': True,
            'usb_detection': SERIAL_AVAILABLE,
            'auth': AUTH_AVAILABLE
        }
    })

# ================================
# ERROR HANDLERS
# ================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found', 'path': request.path}), 404
    try:
        return send_from_directory('.', '404.html'), 404
    except:
        return 'Page not found', 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.exception("Internal server error")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
    return 'Internal server error', 500

# ================================
# SOCKETIO EVENT HANDLERS
# ================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status', {'message': 'Connected to cloud server', 'features': {
        'databench': True,
        'plugplay': True,
        'cloud_deployment': True
    }})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass

# Register blueprints
app.register_blueprint(databench_bp)
app.register_blueprint(plugplay_bp)

# This allows gunicorn to import the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting comprehensive cloud deployment on port {port}")
    print(f"Features: DataBench={True}, PlugPlay={True}, USB={SERIAL_AVAILABLE}, Auth={AUTH_AVAILABLE}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)