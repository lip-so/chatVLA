#!/usr/bin/env python3
"""
Production entry point that combines main app with plug & play features
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

# Import comprehensive backend
from simple_deploy import app as comprehensive_app, socketio

# Import required components
from flask import Flask, Blueprint
from flask_cors import CORS
from flask_socketio import SocketIO

# Create combined app
app = Flask(__name__, 
            static_folder='frontend',
            static_url_path='')

# Configure app
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize CORS
CORS(app, origins=["*"])

# Import all routes from comprehensive backend
for rule in comprehensive_app.url_map.iter_rules():
    # Skip static and root routes as we'll handle them separately
    if not rule.rule.startswith('/static') and rule.rule != '/':
        app.add_url_rule(
            rule.rule,
            endpoint=rule.endpoint,
            view_func=comprehensive_app.view_functions[rule.endpoint],
            methods=rule.methods
        )

# Static file serving
# Use comprehensive backend for static file serving
@app.route('/')
def index():
    return comprehensive_app.view_functions['index']()

@app.route('/pages/<path:filename>')
def serve_page(filename):
    return comprehensive_app.view_functions['serve_page'](filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    return comprehensive_app.view_functions['serve_css'](filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return comprehensive_app.view_functions['serve_js'](filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return comprehensive_app.view_functions['serve_assets'](filename)

# Initialize SocketIO with the combined app
socketio = SocketIO(app, cors_allowed_origins="*")

# Use socketio from comprehensive backend directly
socketio = getattr(comprehensive_app, 'socketio', socketio)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"""
    Tune Robotics Production Server
    ===============================
    Starting on port {port}
    
    Services:
    - Main App: /
    - DataBench API: /api/databench/*
    - Plug & Play API: /api/plugplay/*
    - WebSocket: Socket.IO enabled
    """)
    
    # Use socketio.run for WebSocket support
    socketio.run(app, host='0.0.0.0', port=port, debug=False)