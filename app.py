#!/usr/bin/env python3
"""
Railway-compatible Flask app
"""
import os
from flask import Flask, jsonify, send_file
from pathlib import Path

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
    <p>✅ Backend is running on Railway!</p>
    <p><a href="/api/test">Test API</a></p>
</body>
</html>
"""

@app.route('/api/test')
def test():
    """Test API endpoint"""
    return jsonify({
        'status': 'working',
        'message': 'Railway Flask backend is running successfully',
        'port': os.environ.get('PORT', '5000')
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

# For Railway compatibility
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)