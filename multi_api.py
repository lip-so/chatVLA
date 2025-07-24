#!/usr/bin/env python3
"""
Tune Robotics Unified API - Real Implementation
Integrates actual DataBench evaluation and Plug & Play installation functionality
Version: 2.0.1 - Fixed DataBench method calls and result formatting
"""

import os
import sys
import logging
import threading
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add databench and plug-and-play to Python path
project_root = Path(__file__).parent
databench_path = project_root / 'databench'
plugplay_path = project_root / 'Plug-and-play'

sys.path.append(str(databench_path))
sys.path.append(str(plugplay_path / 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("🔧 Initializing Tune Robotics Unified API...")
print(f"🐍 Python path: {os.getcwd()}")
print(f"📁 DataBench path: {databench_path}")
print(f"📁 Plug & Play path: {plugplay_path}")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tune-robotics-unified-api'
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")

print("✅ Flask app and SocketIO initialized")

# ============================================================================
# IMPORT REAL FUNCTIONALITY
# ============================================================================

# Import real DataBench functionality
try:
    # Ensure proper path setup for Railway deployment
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(databench_path) not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{databench_path}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = str(databench_path)
    
    # Verify databench directory structure
    if not databench_path.exists():
        raise ImportError(f"DataBench directory not found: {databench_path}")
    
    scripts_path = databench_path / 'scripts'
    if not scripts_path.exists():
        raise ImportError(f"DataBench scripts directory not found: {scripts_path}")
    
    # Import with proper error handling
    from scripts.evaluate import RoboticsDatasetBenchmark, METRIC_MAPPING
    from scripts.config_loader import get_config_loader, get_config
    DATABENCH_AVAILABLE = True
    print("✅ DataBench evaluation system loaded")
    logger.info(f"DataBench path: {databench_path}")
    logger.info(f"Python path: {os.environ.get('PYTHONPATH')}")
except ImportError as e:
    logger.error(f"Failed to import DataBench: {e}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"DataBench path exists: {databench_path.exists()}")
    logger.error(f"Python path: {sys.path}")
    DATABENCH_AVAILABLE = False

# Import real Plug & Play functionality  
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
    print("✅ Serial port detection available")
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("pyserial not available - USB detection limited")

# ============================================================================
# REAL DATABENCH IMPLEMENTATION
# ============================================================================

class DataBenchAPI:
    """Real DataBench evaluation API using the actual benchmark system"""
    
    def __init__(self):
        self.current_evaluation = None
        self.evaluation_thread = None
        
    def evaluate_dataset(self, dataset_name: str, metrics: List[str], subset: Optional[int] = None, 
                        token: Optional[str] = None) -> Dict[str, Any]:
        """Perform real dataset evaluation using DataBench"""
        
        if not DATABENCH_AVAILABLE:
            raise Exception("DataBench system not available - missing dependencies")
            
        try:
            logger.info(f"Starting real DataBench evaluation: {dataset_name}")
            
            # Set HuggingFace token if provided
            if token:
                os.environ['HF_TOKEN'] = token
                
            # Initialize the real benchmark system with Railway-optimized settings
            config_overrides = {}
            if subset:
                config_overrides['general'] = {'subset_size': subset}
            
            # Add Railway-specific optimizations
            config_overrides['computation'] = {
                'batch_size': 8,  # Smaller batch size for Railway memory limits
                'num_workers': 2,  # Limit workers for Railway
                'device': 'cpu'   # Force CPU for Railway compatibility
            }
                
            benchmark = RoboticsDatasetBenchmark(
                data_path=dataset_name,
                subset=subset,
                config_overrides=config_overrides
            )
            
            # Perform the actual evaluation
            logger.info(f"Evaluating metrics: {metrics}")
            start_time = datetime.now()
            
            # Run evaluation with timeout protection
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Evaluation timed out")
            
            # Set 15 minute timeout for Railway
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(900)  # 15 minutes
            
            try:
                results = benchmark.run_evaluation(metrics)
            finally:
                signal.alarm(0)  # Disable timeout
            
            end_time = datetime.now()
            evaluation_duration = (end_time - start_time).total_seconds()
            
            # Format results for API response
            formatted_results = {}
            overall_scores = []
            
            for metric_code in metrics:
                if metric_code in METRIC_MAPPING:
                    metric_name = METRIC_MAPPING[metric_code]
                    if metric_name in results:
                        # Handle different result formats
                        score = results[metric_name]
                        if isinstance(score, dict):
                            score = score.get('overall_score', score.get('score', 0.0))
                        elif isinstance(score, (int, float)):
                            score = float(score)
                        else:
                            score = 0.0
                        
                        formatted_results[metric_code] = round(score, 3)
                        overall_scores.append(score)
            
            # Calculate overall score
            if overall_scores:
                overall_score = sum(overall_scores) / len(overall_scores)
                formatted_results['overall_score'] = round(float(overall_score), 3)
            
            return {
                "success": True,
                "dataset": dataset_name,
                "results": formatted_results,
                "detailed_results": results,
                "metadata": {
                    "subset_size": subset,
                    "evaluation_time": start_time.isoformat(),
                    "duration_seconds": round(evaluation_duration, 2),
                    "metrics_evaluated": len(formatted_results),
                    "mode": "railway_optimized"
                }
            }
            
        except TimeoutError:
            logger.error(f"DataBench evaluation timed out for dataset: {dataset_name}")
            return {
                "success": False,
                "error": "Evaluation timed out after 15 minutes",
                "dataset": dataset_name
            }
        except Exception as e:
            logger.error(f"DataBench evaluation failed: {e}")
            logger.exception("Full error traceback:")
            return {
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
                "dataset": dataset_name
            }

# ============================================================================
# REAL PLUG & PLAY IMPLEMENTATION  
# ============================================================================

class PlugPlayInstallationManager:
    """Real installation manager for LeRobot setup"""
    
    def __init__(self):
        self.is_running = False
        self.installation_path = None
        self.current_step = None
        self.progress = 0
        
    def start_installation(self, installation_path: str):
        """Start real LeRobot installation"""
        if self.is_running:
            return False
            
        self.is_running = True
        self.installation_path = Path(installation_path)
        self.progress = 0
        
        # Start installation in separate thread
        installation_thread = threading.Thread(
            target=self._run_installation,
            daemon=True
        )
        installation_thread.start()
        return True
        
    def _run_installation(self):
        """Execute the actual installation steps"""
        try:
            steps = [
                ("checking_prerequisites", "Checking Prerequisites", self._check_prerequisites),
                ("cloning_repository", "Cloning Repository", self._clone_repository),
                ("creating_environment", "Creating Environment", self._create_environment),
                ("installing_ffmpeg", "Installing FFmpeg", self._install_ffmpeg),
                ("installing_lerobot", "Installing LeRobot", self._install_lerobot),
                ("detecting_usb_ports", "USB Port Detection", self._setup_usb_detection)
            ]
            
            for i, (step_name, step_text, step_func) in enumerate(steps):
                if not self.is_running:
                    break
                    
                self.current_step = step_name
                self.progress = int((i / len(steps)) * 100)
                
                socketio.emit('installation_progress', {
                    'step': step_name,
                    'progress': self.progress,
                    'status': step_text,
                    'message': f"Executing: {step_text}"
                })
                
                success = step_func()
                if not success:
                    socketio.emit('installation_error', {
                        'step': step_name,
                        'message': f"Failed at step: {step_text}"
                    })
                    return
                    
            # Installation completed
            self.progress = 100
            socketio.emit('installation_complete', {
                'message': 'LeRobot installation completed successfully!',
                'installation_path': str(self.installation_path)
            })
            
        except Exception as e:
            socketio.emit('installation_error', {
                'message': f"Installation failed: {str(e)}"
            })
        finally:
            self.is_running = False
            
    def _check_prerequisites(self) -> bool:
        """Check if required tools are available"""
        logger.info("Checking prerequisites...")
        
        # Check Conda
        if not self._check_command('conda'):
            socketio.emit('installation_log', {
                'message': 'ERROR: Conda not found. Please install Miniconda or Anaconda first.',
                'level': 'error'
            })
            return False
            
        # Check Git
        if not self._check_command('git'):
            socketio.emit('installation_log', {
                'message': 'ERROR: Git not found. Please install Git first.',
                'level': 'error'
            })
            return False
            
        socketio.emit('installation_log', {
            'message': 'Prerequisites check passed - Conda and Git available',
            'level': 'info'
        })
        return True
        
    def _clone_repository(self) -> bool:
        """Clone LeRobot repository"""
        logger.info("Cloning LeRobot repository...")
        
        if self.installation_path.exists():
            shutil.rmtree(self.installation_path)
            
        command = f'git clone https://github.com/huggingface/lerobot.git "{self.installation_path}"'
        return self._run_command(command)
        
    def _create_environment(self) -> bool:
        """Create conda environment"""
        logger.info("Creating conda environment...")
        command = 'conda create -y -n lerobot python=3.10'
        return self._run_command(command)
        
    def _install_ffmpeg(self) -> bool:
        """Install FFmpeg"""
        logger.info("Installing FFmpeg...")
        command = 'conda install -y ffmpeg -c conda-forge -n lerobot'
        return self._run_command(command)
        
    def _install_lerobot(self) -> bool:
        """Install LeRobot package"""
        logger.info("Installing LeRobot package...")
        command = 'conda run -n lerobot pip install -e .'
        return self._run_command(command, cwd=self.installation_path)
        
    def _setup_usb_detection(self) -> bool:
        """Setup USB port detection"""
        logger.info("Setting up USB port detection...")
        
        # Install pyserial in lerobot environment
        command = 'conda run -n lerobot pip install pyserial'
        success = self._run_command(command)
        
        if success:
            socketio.emit('installation_log', {
                'message': 'USB port detection tools installed. Use the scan function to detect robotic arms.',
                'level': 'info'
            })
        
        return success
        
    def _check_command(self, command: str) -> bool:
        """Check if a command is available"""
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, 
                         check=True, 
                         timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
            
    def _run_command(self, command: str, cwd: Optional[Path] = None) -> bool:
        """Run a command and emit output"""
        try:
            socketio.emit('installation_log', {
                'message': f'Running: {command}',
                'level': 'info'
            })
            
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd
            )
            
            for line in process.stdout:
                if line.strip():
                    socketio.emit('installation_log', {
                        'message': line.strip(),
                        'level': 'info'
                    })
                    
            process.wait()
            
            if process.returncode == 0:
                socketio.emit('installation_log', {
                    'message': f'Command completed successfully',
                    'level': 'info'
                })
                return True
            else:
                socketio.emit('installation_log', {
                    'message': f'Command failed with exit code {process.returncode}',
                    'level': 'error'
                })
                return False
                
        except Exception as e:
            socketio.emit('installation_log', {
                'message': f'Command execution failed: {str(e)}',
                'level': 'error'
            })
            return False

class USBPortDetector:
    """Real USB port detection for robotic arms"""
    
    def __init__(self):
        self.detected_ports = []
        
    def scan_ports(self) -> List[Dict[str, Any]]:
        """Scan for available USB ports"""
        if not SERIAL_AVAILABLE:
            return []
            
        ports = []
        try:
            for port in serial.tools.list_ports.comports():
                port_info = {
                    'device': port.device,
                    'description': port.description,
                    'hwid': port.hwid,
                    'vid': getattr(port, 'vid', None),
                    'pid': getattr(port, 'pid', None),
                    'serial_number': getattr(port, 'serial_number', None),
                    'manufacturer': getattr(port, 'manufacturer', None),
                    'product': getattr(port, 'product', None)
                }
                ports.append(port_info)
                
        except Exception as e:
            logger.error(f"Failed to scan ports: {e}")
            
        self.detected_ports = ports
        return ports

# ============================================================================
# INITIALIZE REAL MANAGERS
# ============================================================================

databench_api = DataBenchAPI()
installation_manager = PlugPlayInstallationManager()
usb_detector = USBPortDetector()

print("✅ Real functionality managers initialized")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Root endpoint - API info page"""
    return jsonify({
        "name": "Tune Robotics Unified API - Real Implementation",
        "description": "Actual DataBench evaluation and Plug & Play installation",
        "services": ["DataBench", "Plug & Play"],
        "capabilities": {
            "databench_available": DATABENCH_AVAILABLE,
            "serial_available": SERIAL_AVAILABLE
        },
        "endpoints": {
            "health": "/health",
            "databench_evaluate": "/api/evaluate",
            "plugplay_install": "/api/start_installation",
            "plugplay_usb": "/api/scan_usb_ports"
        },
        "status": "running",
        "version": "2.0.0-real",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with detailed system status"""
    try:
        # Check DataBench availability
        databench_status = "available" if DATABENCH_AVAILABLE else "unavailable"
        databench_details = {}
        
        if DATABENCH_AVAILABLE:
            databench_details = {
                "path_exists": databench_path.exists(),
                "scripts_exists": (databench_path / 'scripts').exists(),
                "config_exists": (databench_path / 'config.yaml').exists(),
                "pythonpath_set": str(databench_path) in os.environ.get('PYTHONPATH', '')
            }
        
        # Check system resources
        import psutil
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        return jsonify({
            "status": "healthy",
            "services": {
                "databench": {
                    "status": databench_status,
                    "details": databench_details
                },
                "plugplay": {
                    "status": "available",
                    "serial_available": SERIAL_AVAILABLE
                }
            },
            "system": {
                "memory_usage_percent": memory_info.percent,
                "memory_available_mb": memory_info.available // (1024 * 1024),
                "disk_usage_percent": disk_info.percent,
                "disk_free_gb": disk_info.free // (1024 * 1024 * 1024)
            },
            "environment": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "port": int(os.environ.get('PORT', 5000)),
                "railway_deployment": os.environ.get('RAILWAY_ENVIRONMENT_NAME') is not None
            },
            "version": "2.0.1-railway",
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

# ============================================================================
# DATABENCH ENDPOINTS - REAL IMPLEMENTATION
# ============================================================================

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get available DataBench metrics"""
    return jsonify({
        "metrics": {
            "a": {
                "name": "Action Consistency",
                "description": "Visual-text alignment, action-observation consistency, temporal coherence"
            },
            "v": {
                "name": "Visual Diversity", 
                "description": "Pairwise distances, clustering analysis, entropy of visual content"
            },
            "h": {
                "name": "High-Fidelity Vision",
                "description": "Multi-view setup, resolution, environment quality, prompt clarity"
            },
            "t": {
                "name": "Trajectory Quality",
                "description": "Synchronization, frequency, data completeness of trajectories"
            },
            "c": {
                "name": "Dataset Coverage",
                "description": "Scale, task diversity, visual variety, failure rates"
            },
            "r": {
                "name": "Robot Action Quality",
                "description": "Action smoothness, joint limits, physical feasibility"
            }
        }
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate_dataset():
    """Real DataBench evaluation using actual benchmark system"""
    try:
        data = request.get_json()
        dataset = data.get('dataset', '')
        metrics = data.get('metrics', 'a,v,t').split(',')
        subset = data.get('subset', 5)
        token = data.get('token', None)
        
        # Clean up metrics
        metrics = [m.strip() for m in metrics if m.strip()]
        
        logger.info(f"Real evaluation request: dataset={dataset}, metrics={metrics}, subset={subset}")
        
        # Perform real evaluation
        result = databench_api.evaluate_dataset(dataset, metrics, subset, token)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Evaluation failed: {str(e)}"
        }), 500

# ============================================================================
# PLUG & PLAY ENDPOINTS - REAL IMPLEMENTATION
# ============================================================================

@app.route('/api/system_info', methods=['GET'])
def get_system_info():
    """Get real system information"""
    return jsonify({
        "conda_available": installation_manager._check_command('conda'),
        "git_available": installation_manager._check_command('git'),
        "serial_available": SERIAL_AVAILABLE,
        "default_path": str(Path.home() / 'lerobot'),
        "platform": sys.platform
    })

@app.route('/api/start_installation', methods=['POST'])
def start_installation():
    """Start real LeRobot installation"""
    try:
        data = request.get_json()
        installation_path = data.get('installation_path')
        
        if not installation_path:
            return jsonify({'error': 'Installation path is required'}), 400
            
        success = installation_manager.start_installation(installation_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Real installation started',
                'installation_path': installation_path
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Installation already running'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scan_usb_ports', methods=['GET'])
def scan_usb_ports():
    """Real USB port scanning"""
    try:
        ports = usb_detector.scan_ports()
        
        return jsonify({
            "success": True,
            "ports": ports,
            "count": len(ports),
            "message": f"Found {len(ports)} USB devices",
            "serial_available": SERIAL_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"USB scan error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "ports": []
        }), 500

@app.route('/api/cancel_installation', methods=['POST'])
def cancel_installation():
    """Cancel running installation"""
    installation_manager.is_running = False
    return jsonify({'message': 'Installation cancelled'})

@app.route('/api/status', methods=['GET'])
def get_installation_status():
    """Get current installation status for Plug & Play frontend"""
    return jsonify({
        'is_running': installation_manager.is_running,
        'progress': installation_manager.progress,
        'current_step': installation_manager.current_step,
        'status': 'running' if installation_manager.is_running else 'ready'
    })

@app.route('/api/browse-directory', methods=['POST'])
def browse_directory():
    """Browse directory endpoint for frontend compatibility"""
    try:
        data = request.get_json()
        current_path = data.get('path', str(Path.home()))
        
        path_obj = Path(current_path).resolve()
        if not path_obj.exists() or not path_obj.is_dir():
            path_obj = Path.home()
        
        items = []
        if path_obj.parent != path_obj:
            items.append({
                'name': '..',
                'path': str(path_obj.parent),
                'type': 'parent',
                'is_dir': True
            })
        
        for item in sorted(path_obj.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                if item.name.startswith('.'):
                    continue
                    
                items.append({
                    'name': item.name,
                    'path': str(item),
                    'type': 'directory' if item.is_dir() else 'file',
                    'is_dir': item.is_dir(),
                    'size': item.stat().st_size if item.is_file() else None
                })
            except (PermissionError, OSError):
                continue
                
        return jsonify({
            'current_path': str(path_obj),
            'items': items,
            'can_create': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to browse directory: {str(e)}',
            'current_path': str(Path.home()),
            'items': [],
            'can_create': False
        }), 500

# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected to real API")
    emit('connected', {'message': 'Connected to real Tune Robotics API'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 Starting Tune Robotics Unified API - REAL IMPLEMENTATION")
    print(f"📋 DataBench Available: {DATABENCH_AVAILABLE}")
    print(f"🔌 USB Detection Available: {SERIAL_AVAILABLE}")
    print(f"🌐 Port: {port}")
    print(f"❤️ Health endpoint: /health")
    
    logger.info("Starting Real Tune Robotics Unified API")
    logger.info(f"DataBench Available: {DATABENCH_AVAILABLE}")
    logger.info(f"Serial Available: {SERIAL_AVAILABLE}")
    logger.info(f"Port: {port}")
    
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        logger.error(f"Failed to start server: {e}")
        raise 