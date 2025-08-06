#!/usr/bin/env python3
"""
Simple start script for Railway that will definitely work
"""

import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a simple Flask app
app = Flask(__name__)
CORS(app, origins=["*"])

@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "message": "ChatVLA Backend is running",
        "version": "1.0.0"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "services": {
            "databench": {"available": True, "message": "Mock service ready"},
            "plugplay": {"available": True, "message": "Installation service ready"},
            "auth": {"available": False, "message": "Not configured"}
        }
    })

@app.route('/api/databench/evaluate', methods=['POST', 'OPTIONS'])
def databench_evaluate():
    if request.method == 'OPTIONS':
        # Handle CORS preflight
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    # For now, return a mock response
    data = request.get_json()
    return jsonify({
        "status": "processing",
        "message": "DataBench evaluation service is being configured",
        "dataset": data.get('dataset', 'unknown'),
        "metrics": data.get('metrics', []),
        "results": {
            "overall_score": 0.75,
            "message": "Mock results - backend is being deployed"
        }
    })

@app.route('/api/plugplay/install', methods=['POST', 'OPTIONS'])
def plugplay_install():
    if request.method == 'OPTIONS':
        # Handle CORS preflight
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    # Mock LeRobot installation response
    data = request.get_json() or {}
    package = data.get('package', 'lerobot')
    logger.info(f"Received install request for package: {package}")
    
    return jsonify({
        "status": "success",
        "package": package,
        "message": f"Installing {package}...",
        "steps": [
            {"step": 1, "description": "Checking system requirements", "status": "completed"},
            {"step": 2, "description": "Downloading package", "status": "completed"},
            {"step": 3, "description": "Installing dependencies", "status": "in_progress"},
            {"step": 4, "description": "Configuring environment", "status": "pending"},
            {"step": 5, "description": "Running tests", "status": "pending"}
        ],
        "progress": 60,
        "estimated_time": "2 minutes remaining"
    })

@app.route('/api/plugplay/detect', methods=['GET', 'OPTIONS'])
def plugplay_detect():
    if request.method == 'OPTIONS':
        # Handle CORS preflight
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response, 200
    
    # Mock port detection response
    return jsonify({
        "status": "success",
        "devices": [
            {"port": "/dev/ttyUSB0", "type": "SO-100", "status": "ready"},
            {"port": "/dev/ttyUSB1", "type": "Koch v1.1", "status": "ready"}
        ],
        "message": "2 devices detected"
    })

@app.route('/api/plugplay/configure', methods=['POST', 'OPTIONS'])
def plugplay_configure():
    if request.method == 'OPTIONS':
        # Handle CORS preflight
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    # Mock configuration response
    data = request.get_json() or {}
    return jsonify({
        "status": "success",
        "configuration": data,
        "message": "Configuration saved successfully"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)