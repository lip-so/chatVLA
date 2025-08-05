#!/usr/bin/env python3
"""
Railway debug startup script with extensive logging
"""
import os
import sys
import traceback
from pathlib import Path

def debug_environment():
    """Debug the Railway environment"""
    print("=" * 50)
    print("RAILWAY DEPLOYMENT DEBUG")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent.absolute()}")
    
    print("\nEnvironment variables:")
    for key, value in sorted(os.environ.items()):
        if key.startswith(('PORT', 'RAILWAY', 'PYTHON', 'FLASK')):
            print(f"  {key}={value}")
    
    print(f"\nContents of current directory:")
    try:
        for item in sorted(os.listdir('.')):
            print(f"  {item}")
    except Exception as e:
        print(f"  Error listing directory: {e}")
    
    print(f"\nPython path:")
    for path in sys.path:
        print(f"  {path}")

def test_imports():
    """Test all critical imports"""
    print("\n" + "=" * 30)
    print("TESTING IMPORTS")
    print("=" * 30)
    
    try:
        print("Testing Flask...")
        import flask
        print(f"✅ Flask {flask.__version__}")
    except Exception as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        print("Testing Flask-SocketIO...")
        import flask_socketio
        print(f"✅ Flask-SocketIO {flask_socketio.__version__}")
    except Exception as e:
        print(f"❌ Flask-SocketIO import failed: {e}")
        return False
    
    try:
        print("Testing backend module...")
        import backend
        print("✅ Backend module imported")
    except Exception as e:
        print(f"❌ Backend module failed: {e}")
        return False
    
    try:
        print("Testing working_api...")
        from backend.plug_and_play.working_api import app, socketio
        print("✅ Working API imported successfully")
        print(f"   App: {type(app)}")
        print(f"   SocketIO: {type(socketio)}")
        return True
    except Exception as e:
        print(f"❌ Working API import failed: {e}")
        traceback.print_exc()
        return False

def start_app():
    """Start the application if imports work"""
    print("\n" + "=" * 30)
    print("STARTING APPLICATION")
    print("=" * 30)
    
    try:
        from backend.plug_and_play.working_api import app, socketio
        
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting server on port {port}")
        
        # Test a simple route first
        with app.test_client() as client:
            response = client.get('/api/test')
            print(f"Test endpoint response: {response.status_code}")
            if response.status_code == 200:
                print(f"Response data: {response.get_json()}")
        
        print(f"Starting SocketIO server...")
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=port,
            debug=False,
            use_reloader=False,
            log_output=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    # Set up paths first
    PROJECT_ROOT = Path(__file__).parent.absolute()
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Set environment
    os.environ['PYTHONPATH'] = str(PROJECT_ROOT)
    os.environ['FLASK_ENV'] = 'production'
    
    # Debug environment
    debug_environment()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed - exiting")
        sys.exit(1)
    
    # Start the app
    start_app()