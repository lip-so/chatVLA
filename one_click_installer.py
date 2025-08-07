#!/usr/bin/env python3
"""
ONE-CLICK INSTALLER - Downloads and runs everything automatically
"""

import os
import sys
import subprocess
import tempfile
import urllib.request
import webbrowser
import time
import platform
from pathlib import Path

def download_and_install():
    """Download chatVLA and start the installer bridge automatically"""
    
    print("🤖 Tune Robotics - One-Click LeRobot Installer")
    print("=" * 50)
    
    # Determine installation directory
    install_dir = Path.home() / "chatVLA"
    
    # Check if already installed
    if not install_dir.exists():
        print("📦 Downloading ChatVLA...")
        
        # Clone the repository
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/lip-so/chatVLA.git",
                str(install_dir)
            ], check=True)
            print("✅ Downloaded successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to download. Please check your internet connection.")
            return False
    else:
        print("✅ ChatVLA already downloaded")
        
    # Install Python dependencies
    print("📚 Installing dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "flask", "flask-cors", "flask-socketio", "pyserial", "eventlet",
        "--quiet"
    ])
    
    # Start the local installer bridge
    print("🚀 Starting installation bridge...")
    bridge_script = install_dir / "local_installer_bridge.py"
    
    # Start bridge in background
    if platform.system() == "Windows":
        subprocess.Popen([sys.executable, str(bridge_script)], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen([sys.executable, str(bridge_script)])
    
    # Wait for bridge to start
    time.sleep(3)
    
    # Open the website
    print("🌐 Opening website...")
    webbrowser.open("https://tunerobotics.xyz/frontend/pages/plug-and-play-databench-style.html")
    
    print("\n" + "="*50)
    print("✅ READY! The website will now install LeRobot automatically!")
    print("Just click 'Install LeRobot' on the website!")
    print("="*50)
    
    # Keep running to maintain the bridge
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Installation bridge stopped")

if __name__ == "__main__":
    download_and_install()