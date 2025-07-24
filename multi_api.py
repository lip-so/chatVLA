#!/usr/bin/env python3
"""
Unified API for Tune Robotics - DataBench and Plug & Play services.
Cloud-deployable backend that provides both dataset evaluation and installation guidance.
"""

import os
import json
import subprocess
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tune-robotics-unified-api'
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
DATABENCH_PATH = Path(__file__).parent / "databench"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    """Serve the main API page"""
    return """
    <h1>Tune Robotics - Unified API</h1>
    <p>Cloud backend providing DataBench evaluation and Plug & Play installation guidance.</p>
    <h2>Services:</h2>
    <ul>
        <li><strong>DataBench</strong>: Dataset quality evaluation</li>
        <li><strong>Plug & Play</strong>: LeRobot installation guidance</li>
    </ul>
    <h2>Endpoints:</h2>
    <ul>
        <li><a href="/health">Health Check</a></li>
        <li><a href="/api/metrics">DataBench Metrics</a></li>
        <li><a href="/api/system_info">System Info</a></li>
        <li><a href="/api/installation_guide">Installation Guide</a></li>
    </ul>
    """

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "services": ["DataBench", "Plug & Play"],
        "databench_exists": DATABENCH_PATH.exists(),
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

# ============================================================================
# DATABENCH ENDPOINTS
# ============================================================================

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get available DataBench metrics"""
    return jsonify({
        "metrics": {
            "a": {
                "name": "Action Consistency",
                "description": "Visual-text alignment, action-observation consistency, and temporal coherence"
            },
            "v": {
                "name": "Visual Diversity", 
                "description": "Pairwise distances, clustering analysis, and entropy of visual content"
            },
            "h": {
                "name": "High-Fidelity Vision",
                "description": "Multi-view setup, resolution, environment quality, and prompt clarity"
            },
            "t": {
                "name": "Trajectory Quality",
                "description": "Synchronization, frequency, and data completeness of trajectories"
            },
            "c": {
                "name": "Dataset Coverage",
                "description": "Scale, task diversity, visual variety, and failure rates"
            },
            "r": {
                "name": "Robot Action Quality",
                "description": "Action smoothness, joint limits, and physical feasibility"
            }
        }
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate_dataset():
    """Evaluate dataset quality - Cloud version provides guidance and demo results"""
    try:
        data = request.get_json()
        dataset = data.get('dataset', '')
        metrics = data.get('metrics', 'a,v,t').split(',')
        subset = data.get('subset', 5)
        
        logger.info(f"Evaluation request: dataset={dataset}, metrics={metrics}, subset={subset}")
        
        # Generate realistic demo results for cloud deployment
        import random
        results = {}
        
        # Base scores with some variance
        base_scores = {
            'a': 0.75 + random.uniform(-0.15, 0.15),  # Action consistency
            'v': 0.82 + random.uniform(-0.12, 0.12),  # Visual diversity  
            'h': 0.68 + random.uniform(-0.18, 0.18),  # High-fidelity vision
            't': 0.79 + random.uniform(-0.14, 0.14),  # Trajectory quality
            'c': 0.71 + random.uniform(-0.16, 0.16),  # Dataset coverage
            'r': 0.77 + random.uniform(-0.13, 0.13)   # Robot action quality
        }
        
        for metric in metrics:
            metric = metric.strip()
            if metric in base_scores:
                # Ensure scores stay within valid range
                score = max(0.0, min(1.0, base_scores[metric]))
                results[metric] = round(score, 3)
        
        # Calculate overall score
        if results:
            overall_score = sum(results.values()) / len(results)
            results['overall_score'] = round(overall_score, 3)
        
        response = {
            "success": True,
            "dataset": dataset,
            "results": results,
            "metadata": {
                "subset_size": subset,
                "evaluation_time": datetime.now().isoformat(),
                "mode": "cloud_demo",
                "note": "These are realistic demo scores. For full evaluation, set up local DataBench."
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "guidance": "For full DataBench functionality, run: python databench_api.py"
        }), 500

# ============================================================================
# PLUG & PLAY ENDPOINTS  
# ============================================================================

@app.route('/api/system_info', methods=['GET'])
def system_info():
    """Provide system information and requirements for installation"""
    return jsonify({
        "git_available": True,
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
        },
        "note": "This service provides installation guidance. Actual installation runs on your system."
    })

@app.route('/api/installation_guide', methods=['GET'])
def installation_guide():
    """Provide complete installation guide"""
    return jsonify({
        "title": "LeRobot Installation Guide",
        "description": "Complete automated setup for LeRobot with robotic arms",
        "steps": [
            {
                "id": "prerequisites",
                "title": "Check Prerequisites", 
                "description": "Ensure Git and Conda are installed",
                "commands": ["git --version", "conda --version"],
                "help_links": {
                    "git": "https://git-scm.com/downloads",
                    "conda": "https://docs.conda.io/en/latest/miniconda.html"
                }
            },
            {
                "id": "clone_repository",
                "title": "Clone LeRobot Repository",
                "description": "Download the latest LeRobot framework",
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
                    "conda create -n lerobot python=3.10 -y",
                    "conda activate lerobot"
                ]
            },
            {
                "id": "install_dependencies",
                "title": "Install Dependencies",
                "description": "Install LeRobot and required packages", 
                "commands": [
                    "pip install -e .",
                    "pip install pyserial"
                ]
            },
            {
                "id": "usb_setup",
                "title": "USB Port Configuration",
                "description": "Configure robotic arm connections",
                "instructions": [
                    "Connect robotic arms via USB",
                    "Run port detection script",
                    "Note port assignments for use"
                ]
            },
            {
                "id": "verification",
                "title": "Verify Installation", 
                "description": "Test the complete setup",
                "commands": [
                    "python -c \"import lerobot; print('✅ LeRobot ready!')\"",
                    "python -c \"import serial; print('✅ USB ready!')\""
                ]
            }
        ]
    })

@app.route('/api/start_installation', methods=['POST'])
def start_installation():
    """Generate installation script and guidance"""
    data = request.get_json()
    installation_path = data.get('installation_path', '~/lerobot')
    
    # Generate platform-specific installation script
    script_content = f"""#!/bin/bash
# Automated LeRobot Installation
# Generated by Tune Robotics

echo "🤖 Starting LeRobot installation..."
echo "📁 Target: {installation_path}"

# Create and navigate to installation directory
mkdir -p "{installation_path}"
cd "{installation_path}"

# Clone LeRobot repository
echo "📥 Cloning LeRobot..."
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Create conda environment
echo "🏗️ Creating environment..."
conda create -n lerobot python=3.10 -y
source activate lerobot || conda activate lerobot

# Install framework and dependencies
echo "📦 Installing LeRobot..."
pip install -e .
pip install pyserial

echo "✅ Installation complete!"
echo "🔌 Connect robotic arms and test USB ports"
echo "🚀 Activate: conda activate lerobot"
"""

    return jsonify({
        "success": True,
        "installation_path": installation_path,
        "script": script_content,
        "instructions": [
            "Copy the script below to install_lerobot.sh",
            "Make executable: chmod +x install_lerobot.sh", 
            "Run: ./install_lerobot.sh",
            "Connect USB devices for port detection"
        ],
        "next_steps": [
            "Save installation script to file",
            "Execute the script in terminal",
            "Connect and configure robotic arms",
            "Test the complete setup"
        ]
    })

@app.route('/api/scan_usb_ports', methods=['GET'])
def scan_usb_ports():
    """Provide USB port detection guidance and code"""
    return jsonify({
        "message": "USB Port Detection Guide",
        "instructions": [
            "Connect robotic arms via USB",
            "Run the detection code below",
            "Note the port names for configuration"
        ],
        "detection_code": """import serial.tools.list_ports

def detect_robotic_arms():
    print("🔍 Scanning for USB devices...")
    ports = []
    
    for port in serial.tools.list_ports.comports():
        ports.append({
            'device': port.device,
            'description': port.description,
            'hwid': port.hwid
        })
        print(f"Found: {port.device} - {port.description}")
    
    return ports

# Run detection
detected_ports = detect_robotic_arms()
print(f"\\n✅ Found {len(detected_ports)} USB devices")""",
        "common_patterns": {
            "macOS": ["/dev/cu.usbmodem*", "/dev/tty.usbserial*"],
            "Linux": ["/dev/ttyACM*", "/dev/ttyUSB*"], 
            "Windows": ["COM1", "COM2", "COM3", "COM4", "..."]
        },
        "tips": [
            "Ensure USB drivers are installed",
            "Try different high-quality USB cables",
            "Check device manager for recognized devices",
            "Verify robotic arms are powered on"
        ]
    })

@app.route('/api/cancel_installation', methods=['POST'])
def cancel_installation():
    """Handle installation cancellation"""
    return jsonify({
        "success": True,
        "message": "Guidance session ended",
        "note": "This service provides guidance only - no active installation to cancel"
    })

@app.route('/api/status', methods=['GET'])
def status():
    """Service status endpoint"""
    return jsonify({
        "status": "online",
        "services": ["DataBench", "Plug & Play"],
        "mode": "cloud_guidance",
        "description": "Provides dataset evaluation and installation guidance"
    })

# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status', {
        'message': 'Connected to Tune Robotics services',
        'services': ['DataBench', 'Plug & Play'],
        'type': 'info'
    })
    logger.info('Client connected to unified API')

@socketio.on('request_guidance')
def handle_guidance_request(data):
    """Provide real-time installation guidance"""
    step = data.get('step', 'start')
    
    guidance = {
        'start': 'Welcome! Ready to set up LeRobot?',
        'prerequisites': 'Checking system requirements...',
        'clone': 'Downloading LeRobot repository...',
        'environment': 'Creating conda environment...',
        'install': 'Installing dependencies...',
        'usb': 'Configuring robotic arms...',
        'complete': 'Setup complete! Ready to use.'
    }
    
    emit('guidance_update', {
        'step': step,
        'message': guidance.get(step, 'Processing...'),
        'type': 'info'
    })

if __name__ == '__main__':
    # Get port from environment for cloud deployment
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("Starting Tune Robotics Unified API")
    logger.info(f"Services: DataBench, Plug & Play")
    logger.info(f"Port: {port}")
    
    # Run with external access for cloud deployment
    socketio.run(app, host='0.0.0.0', port=port, debug=False) 