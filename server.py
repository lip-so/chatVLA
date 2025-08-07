#!/usr/bin/env python3
"""
NEW RAILWAY ENTRY POINT - FORCE OVERRIDE
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS

print("🚀 STARTING NEW SERVER.PY - FORCE OVERRIDE")

app = Flask(__name__)
CORS(app, origins=["*"])

@app.route('/')
def index():
    return jsonify({
        "status": "success",
        "message": "🚀 NEW BACKEND IS WORKING!",
        "version": "3.0.0-OVERRIDE",
        "timestamp": "2025-01-08"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "NEW BACKEND HEALTH CHECK",
        "services": {
            "plugplay": {
                "available": True,
                "status": "ready with LeRobot simulation",
                "features": ["calibration", "teleoperation", "recording"]
            },
            "databench": {
                "available": True,
                "status": "ready"
            }
        }
    })

@app.route('/api/test')
def api_test():
    return jsonify({
        "status": "success",
        "message": "✅ NEW API TEST ENDPOINT WORKING!",
        "timestamp": "2025-01-08",
        "override": True
    })

@app.route('/api/plugplay/list-ports', methods=['GET'])
def list_ports():
    return jsonify({
        "success": True,
        "ports": [
            {"port": "/dev/ttyUSB0", "description": "USB Serial Device", "role": "leader"},
            {"port": "/dev/ttyUSB1", "description": "USB Serial Device", "role": "follower"}
        ],
        "message": "Simulated port detection"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting NEW SERVER on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
