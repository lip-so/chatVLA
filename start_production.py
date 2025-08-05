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

# Import both apps
from api.main import app as main_app
from plug_and_play.working_api import app as working_app, socketio

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

# Import all routes from main_app
for rule in main_app.url_map.iter_rules():
    # Skip static and root routes as we'll handle them separately
    if not rule.rule.startswith('/static') and rule.rule != '/':
        app.add_url_rule(
            rule.rule,
            endpoint=rule.endpoint,
            view_func=main_app.view_functions[rule.endpoint],
            methods=rule.methods
        )

# Import all routes from working_app
for rule in working_app.url_map.iter_rules():
    # Skip routes that conflict with main_app
    if rule.rule.startswith('/api/plugplay') or rule.rule.startswith('/api/install'):
        # Check if endpoint already exists
        if rule.endpoint not in app.view_functions:
            app.add_url_rule(
                rule.rule,
                endpoint=rule.endpoint,
                view_func=working_app.view_functions[rule.endpoint],
                methods=rule.methods
            )

# Static file serving
@app.route('/')
def index():
    return working_app.send_static_file('index.html')

@app.route('/pages/<path:filename>')
def serve_page(filename):
    return working_app.send_static_file(f'pages/{filename}')

@app.route('/css/<path:filename>')
def serve_css(filename):
    return working_app.send_static_file(f'css/{filename}')

@app.route('/js/<path:filename>')
def serve_js(filename):
    return working_app.send_static_file(f'js/{filename}')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return working_app.send_static_file(f'assets/{filename}')

# Initialize SocketIO with the combined app
socketio = SocketIO(app, cors_allowed_origins="*")

# Copy socket event handlers from working_app
working_socketio = getattr(working_app, '_socketio', None)
if working_socketio:
    for event in ['connect', 'disconnect', 'install_log']:
        if hasattr(working_socketio, 'handlers'):
            handlers = working_socketio.handlers.get(event, {})
            for namespace, handler_list in handlers.items():
                for handler in handler_list:
                    socketio.on(event, namespace=namespace)(handler)

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