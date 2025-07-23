#!/usr/bin/env python3
"""
🤖 LEROBOT INSTALLATION ASSISTANT 🤖
====================================

Easy one-click installer for LeRobot with automatic USB port detection!

This installer is designed for users without technical background.
Just run this script and follow the instructions!

Features:
- Automatic LeRobot installation
- USB port detection for robotic arms  
- User-friendly web interface
- No complex setup required
"""

import sys
import os
from pathlib import Path

def show_welcome():
    """Show welcome message."""
    print("🤖" + "=" * 68 + "🤖")
    print("                 LEROBOT INSTALLATION ASSISTANT")  
    print("🤖" + "=" * 68 + "🤖")
    print()
    print("✨ Features:")
    print("  • Automatic LeRobot installation with all dependencies")
    print("  • Smart USB port detection for your robotic arms")
    print("  • User-friendly web interface - no command line needed!")
    print("  • Works on Windows, macOS, and Linux")
    print()
    print("🎯 Perfect for users without technical background!")
    print()

def check_python():
    """Check Python version."""
    if sys.version_info < (3, 7):
        print("❌ PYTHON VERSION ERROR")
        print(f"   Your Python version: {sys.version}")
        print("   Required: Python 3.7 or higher")
        print()
        print("💡 Please install a newer Python version:")
        print("   https://www.python.org/downloads/")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - OK")
    return True

def check_prerequisites():
    """Check system prerequisites."""
    print("🔍 Checking your system...")
    
    if not check_python():
        return False
    
    # Check for conda
    try:
        import subprocess
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Conda - OK")
        else:
            print("⚠️  Conda not found")
            print("   Don't worry! The installer will guide you through setup")
    except:
        print("⚠️  Conda not found") 
        print("   Don't worry! The installer will guide you through setup")
    
    # Check for git
    try:
        import subprocess
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Git - OK")
        else:
            print("⚠️  Git not found")
            print("   Don't worry! The installer will guide you through setup")
    except:
        print("⚠️  Git not found")
        print("   Don't worry! The installer will guide you through setup")
    
    return True

def launch_installer():
    """Launch the best available installer."""
    print()
    print("🚀 Starting Installation Assistant...")
    print()
    
    # Try web interface first (modern, user-friendly)
    web_launcher = Path(__file__).parent / "installers" / "fixed_run.py"
    if web_launcher.exists():
        print("🌐 Launching web interface...")
        try:
            os.system(f'python "{web_launcher}"')
            return
        except KeyboardInterrupt:
            print("\n👋 Installation assistant stopped by user")
            return
        except Exception as e:
            print(f"Web interface failed: {e}")
    
    # Try GUI installer as fallback
    gui_launcher = Path(__file__).parent / "installers" / "lerobot_installer.py"
    if gui_launcher.exists():
        print("🖥️  Launching desktop application...")
        try:
            os.system(f'python "{gui_launcher}"')
            return
        except Exception as e:
            print(f"Desktop app failed: {e}")
    
    # Manual installation instructions
    print("📋 MANUAL INSTALLATION INSTRUCTIONS")
    print("=" * 50)
    print("If automatic installers don't work, follow these steps:")
    print()
    print("1. Install prerequisites:")
    print("   • Git: https://git-scm.com/downloads")
    print("   • Miniconda: https://docs.conda.io/en/latest/miniconda.html")
    print()
    print("2. Run these commands in your terminal:")
    print("   conda create -y -n lerobot python=3.10")
    print("   conda activate lerobot")
    print("   git clone https://github.com/huggingface/lerobot.git")
    print("   cd lerobot")
    print("   conda install -y ffmpeg -c conda-forge")
    print("   pip install -e .")
    print("   pip install pyserial")
    print()
    print("3. For USB port detection:")
    print("   python lerobot/find_port.py")

def main():
    """Main entry point."""
    show_welcome()
    
    if not check_prerequisites():
        print()
        print("❌ System check failed. Please fix the issues above and try again.")
        return 1
    
    print()
    print("🎉 System check passed! Your computer is ready for LeRobot!")
    
    try:
        launch_installer()
    except KeyboardInterrupt:
        print("\n👋 Installation cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())