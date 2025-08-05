#!/usr/bin/env python3
"""
Minimal Flask app for Railway deployment testing
"""
import os
import sys
from pathlib import Path
from flask import Flask, jsonify, send_file

# Set up paths
PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Create minimal Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Serve main page"""
    try:
        return send_file('index.html')
    except:
        return """
<!DOCTYPE html>
<html>
<head><title>Tune Robotics</title></head>
<body>
    <h1>Tune Robotics - Backend Active!</h1>
    <p>✅ Flask backend is running successfully</p>
    <p><a href="/api/test">Test API</a></p>
</body>
</html>
"""

@app.route('/api/test')
def test():
    """Test API endpoint"""
    return jsonify({
        'status': 'working',
        'message': 'Minimal backend is running successfully',
        'host': '0.0.0.0',
        'port': os.environ.get('PORT', 5000)
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Get port from environment (Railway requirement)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting minimal Tune Robotics app on port {port}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    
    # Run with Railway requirements: host=0.0.0.0, port from env
    app.run(
        host='0.0.0.0', 
        port=port,
        debug=False
    )