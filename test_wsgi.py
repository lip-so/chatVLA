#!/usr/bin/env python3
"""
Test script to verify WSGI setup
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"Python path: {sys.path[:3]}")
print(f"Working directory: {os.getcwd()}")

try:
    print("Testing WSGI import...")
    from backend.plug_and_play.wsgi import application
    print(f"✅ WSGI application imported successfully: {type(application)}")
    
    print("\nTesting Flask app...")
    from backend.plug_and_play.working_api import app, socketio
    print(f"✅ Flask app imported: {type(app)}")
    print(f"✅ SocketIO imported: {type(socketio)}")
    
    print("\nTesting routes...")
    with app.test_client() as client:
        response = client.get('/api/test')
        print(f"✅ /api/test endpoint: {response.status_code} - {response.get_json()}")
        
        response = client.get('/')
        print(f"✅ / endpoint: {response.status_code}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()