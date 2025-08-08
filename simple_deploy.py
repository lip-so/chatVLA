#!/usr/bin/env python3
"""
Simple deployment entry point
Minimal Flask app for basic functionality
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))

from flask import Flask, jsonify, send_from_directory, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Basic routes
@app.route('/')
def index():
    """Serve the main index page"""
    return send_file('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Simple deploy running"})

@app.route('/pages/<path:filename>')
def pages(filename):
    """Serve pages from frontend/pages"""
    return send_from_directory('frontend/pages', filename)

@app.route('/js/<path:filename>')
def js(filename):
    """Serve JavaScript files"""
    return send_from_directory('frontend/js', filename)

@app.route('/css/<path:filename>')
def css(filename):
    """Serve CSS files"""
    return send_from_directory('frontend/css', filename)

@app.route('/assets/<path:filename>')
def assets(filename):
    """Serve asset files"""
    return send_from_directory('frontend/assets', filename)

# Try to import and add API routes if available
try:
    from backend.api.main import app as main_app
    # Copy routes from main app if available
    for rule in main_app.url_map.iter_rules():
        if rule.endpoint.startswith('api'):
            app.add_url_rule(rule.rule, rule.endpoint, 
                           main_app.view_functions[rule.endpoint], 
                           methods=rule.methods)
except Exception as e:
    print(f"Could not import full backend: {e}")
    
    # Add minimal API endpoints
    @app.route('/api/health')
    def api_health():
        return jsonify({"status": "limited", "message": "Simple mode active"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)