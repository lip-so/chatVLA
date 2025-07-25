#!/usr/bin/env python3
"""
Launcher script for the Tune Robotics Plug & Play Installation Assistant.
This script starts the backend server and optionally opens the web interface.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if Python and required packages are available."""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"Python {sys.version.split()[0]} detected")
    
    # Check if we're in the correct directory
    if not Path("Plug-and-play").exists():
        print("Error: Plug-and-play directory not found.")
        print("   Please run this script from the root directory of the project.")
        return False
    
    print("Plug-and-play directory found")
    return True

def install_backend_dependencies():
    """Install required Python packages for the backend."""
    print("\nInstalling backend dependencies...")
    
    requirements_file = Path("Plug-and-play/backend/requirements.txt")
    if not requirements_file.exists():
        print("Error: requirements.txt not found in Plug-and-play/backend/")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", 
            str(requirements_file)
        ])
        print("Backend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def start_backend():
    """Start the Flask backend server."""
    print("\nStarting Plug & Play backend server...")
    
    backend_dir = Path("Plug-and-play/backend")
    app_file = backend_dir / "app.py"
    
    if not app_file.exists():
        print("Error: app.py not found in Plug-and-play/backend/")
        return None
    
    try:
        # Change to backend directory and start the server
        process = subprocess.Popen([
            sys.executable, "app.py"
        ], cwd=backend_dir)
        
        print("Backend server starting...")
        print("   Server will be available at: http://localhost:5002")
        return process
    except Exception as e:
        print(f"Error starting backend: {e}")
        return None

def open_web_interface():
    """Open the web interface in the default browser."""
    print("\nOpening web interface...")
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    try:
        # Check if plug-and-play.html exists
        if Path("plug-and-play.html").exists():
            # Use file:// URL to open the local HTML file
            file_url = f"file://{Path.cwd().absolute()}/plug-and-play.html"
            webbrowser.open(file_url)
            print("Web interface opened in your default browser")
        else:
            print("Error: plug-and-play.html not found")
            print("   You can manually navigate to: http://localhost:5002")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("   Please manually open: http://localhost:5002")

def main():
    """Main launcher function."""
    print("Tune Robotics - Plug & Play Installation Assistant")
    print("=" * 55)
    
    # Check system requirements
    if not check_requirements():
        sys.exit(1)
    
    # Install dependencies
    if not install_backend_dependencies():
        print("\nWarning: Could not install all dependencies.")
        print("   You may need to install them manually:")
        print("   pip install -r Plug-and-play/backend/requirements.txt")
    
    # Start backend server
    backend_process = start_backend()
    if not backend_process:
        sys.exit(1)
    
    # Open web interface
    open_web_interface()
    
    print("\n" + "=" * 55)
    print("Plug & Play system is now running!")
    print("\nTo use the installation assistant:")
    print("1. Open your browser to the page that just opened")
    print("2. Configure your installation directory")
    print("3. Click 'Start Installation' to begin")
    print("\nPress Ctrl+C to stop the server when you're done.")
    print("=" * 55)
    
    try:
        # Keep the script running while the backend runs
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down Plug & Play system...")
        backend_process.terminate()
        print("System stopped successfully")

if __name__ == "__main__":
    main() 