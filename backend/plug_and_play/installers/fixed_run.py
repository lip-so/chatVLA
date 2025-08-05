#!/usr/bin/env python3
"""
Fixed launcher for LeRobot Installation Assistant
Uses compatible Flask version that works with LeRobot
"""

import sys
import os
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def check_and_install_dependencies():
    """Check and install compatible dependencies."""
    print("🔧 Checking dependencies...")
    
    required_packages = {
        'flask': 'flask>=3.0.3',
        'flask_socketio': 'flask-socketio>=5.0.0', 
        'flask_cors': 'flask-cors>=4.0.0'
    }
    
    missing = []
    for package, install_spec in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(install_spec)
    
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install'
            ] + missing, check=True)
            print("Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install dependencies: {e}")
            return False
    else:
        print("All dependencies are available")
        return True

def start_backend():
    """Start the Flask backend server."""
    backend_dir = Path(__file__).parent.parent / 'backend'
    app_path = backend_dir / 'app.py'
    
    if not app_path.exists():
        print(f"ERROR: Backend not found at {app_path}")
        return None
    
    print("🚀 Starting Flask backend...")
    
    try:
        # Change to backend directory for proper file serving
        original_cwd = os.getcwd()
        os.chdir(backend_dir)
        
        # Start Flask server
        process = subprocess.Popen([
            sys.executable, str(app_path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Restore original working directory
        os.chdir(original_cwd)
        
        return process
        
    except Exception as e:
        print(f"ERROR: Failed to start backend: {e}")
        return None

def check_server_ready(max_attempts=30):
    """Check if server is ready to accept connections."""
    import urllib.request
    import urllib.error
    
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen('http://127.0.0.1:5000', timeout=1):
                return True
        except:
            time.sleep(0.5)
    return False

def open_browser():
    """Open the web interface in browser."""
    url = 'http://127.0.0.1:5000'
    
    if check_server_ready():
                    print(f"Server ready! Opening {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
                            print(f"WARNING: Could not open browser automatically: {e}")
            print(f"Please manually open: {url}")
    else:
                    print(f"WARNING: Server not ready. Please try opening: {url}")

def main():
    """Main entry point."""
    print("🤖 LeRobot Installation Assistant")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7+ required")
        return 1
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        print("ERROR: Failed to set up dependencies")
        print("Please install manually:")
        print("pip install 'flask>=3.0.3' 'flask-socketio>=5.0.0' 'flask-cors>=4.0.0'")
        return 1
    
    # Start backend server
    server_process = start_backend()
    if not server_process:
        print("ERROR: Failed to start server")
        return 1
    
    # Open browser in separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
            print(f"Server started successfully!")
            print(f"Web interface: http://127.0.0.1:5000")
            print(f"For port detection: http://127.0.0.1:5000/port-detection.html")
    print()
    print("Press Ctrl+C to stop the server")
    
    try:
        # Wait for server process
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
    
    print("👋 Server stopped")
    return 0

if __name__ == '__main__':
    sys.exit(main())