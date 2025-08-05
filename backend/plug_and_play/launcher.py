#!/usr/bin/env python3
"""
LeRobot Plug & Play Launcher
Ensures the backend starts properly and handles system commands safely.
"""

import os
import sys
import time
import subprocess
import signal
import atexit
from pathlib import Path

def check_port(port):
    """Check if port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def kill_port_process(port):
    """Kill any process using the specified port"""
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(1)
                        os.kill(int(pid), signal.SIGKILL)
                    except:
                        pass
    except:
        pass

def start_backend(port=5003):
    """Start the backend server properly"""
    print(f"🚀 Starting LeRobot Plug & Play Backend on port {port}")
    
    # Kill any existing processes on this port
    if not check_port(port):
        print(f"WARNING: Port {port} is in use, cleaning up...")
        kill_port_process(port)
        time.sleep(2)
    
    # Set environment variables for proper startup
    env = os.environ.copy()
    env['FLASK_ENV'] = 'development'
    env['PYTHONUNBUFFERED'] = '1'  # Ensure output is not buffered
    
    # Start the backend process
    backend_script = Path(__file__).parent / 'working_api.py'
    
    try:
        process = subprocess.Popen([
            sys.executable, str(backend_script)
        ], 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
        )
        
        # Register cleanup on exit
        def cleanup():
            if process.poll() is None:
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
        
        atexit.register(cleanup)
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        for i in range(10):
            if not check_port(port):
                print("Backend started successfully!")
                return process
            time.sleep(1)
        
        print("ERROR: Backend failed to start within 10 seconds")
        cleanup()
        return None
        
    except Exception as e:
        print(f"ERROR: Failed to start backend: {e}")
        return None

def open_frontend():
    """Open the frontend in the default browser"""
    frontend_url = 'http://localhost:5003/pages/plug-and-play-databench-style.html'
    
    print("🌐 Opening Plug & Play frontend...")
    try:
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['open', frontend_url])
        elif sys.platform.startswith('linux'):  # Linux
            subprocess.run(['xdg-open', frontend_url])
        elif sys.platform == 'win32':  # Windows
            subprocess.run(['start', frontend_url], shell=True)
        else:
            print(f"Please open: {frontend_url}")
    except Exception as e:
        print(f"Please manually open: {frontend_url}")

def main():
    """Main launcher function"""
    print("🤖 LeRobot Plug & Play System")
    print("=" * 50)
    
    # Start backend
    process = start_backend()
    if not process:
        sys.exit(1)
    
    # Open frontend
    time.sleep(2)
    open_frontend()
    
    print("\n📋 System Status:")
    print("Backend: Running on http://localhost:5003")
    print("Main page: Available at http://localhost:5003")
    print("Plug & Play: Available at http://localhost:5003/pages/plug-and-play-databench-style.html")
    print("\n🎯 Instructions:")
    print("1. Select your robot in the browser")
    print("2. Choose installation type (fresh or existing)")
    print("3. Follow the 3-step setup process")
    print("\nPress Ctrl+C to stop the system")
    
    try:
        # Keep the process alive and show output
        while True:
            output = process.stdout.readline()
            if output:
                print(f"[BACKEND] {output.strip()}")
            if process.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if process.poll() is None:
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()
        print("System stopped")

if __name__ == '__main__':
    main()