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

if __name__ == '__main__':
    # Get port from environment (for deployment platforms)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("Starting Tune Robotics Plug & Play API")
    logger.info(f"Running on port {port}")
    
    # Run with host 0.0.0.0 for external access
    socketio.run(app, host='0.0.0.0', port=port, debug=False) 