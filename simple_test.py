#!/usr/bin/env python3
"""
Very simple test to see if Railway can run basic Flask
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])

@app.route('/')
def index():
    return jsonify({
        "status": "success",
        "message": "Simple test app is working!",
        "version": "test-1.0",
        "port": os.environ.get('PORT', 'unknown')
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "test": "simple app working"
    })

@app.route('/api/test')
def api_test():
    return jsonify({
        "status": "success",
        "message": "API test endpoint working!",
        "timestamp": "2025-01-08"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting simple test app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
