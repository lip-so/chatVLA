#!/usr/bin/env python3
"""
Simple start script for Railway that will definitely work
"""

import os
from flask import Flask, jsonify, request
from flask_cors import CORS

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
            "databench": {"available": False, "message": "Loading..."},
            "plugplay": {"available": False, "message": "Loading..."},
            "auth": {"available": False, "message": "Loading..."}
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)