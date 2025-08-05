#!/usr/bin/env python3
"""
Ultra minimal Flask app - absolutely basic
"""
import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Ultra Minimal Flask App Works!</h1><p>Backend is ACTIVE</p>'

@app.route('/health')
def health():
    return {'status': 'healthy'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Ultra minimal app starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)