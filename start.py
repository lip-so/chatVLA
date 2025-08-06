#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    
    # Change to project root
    os.chdir(script_dir)
    
    # Start the server
    print("Starting Tune Robotics Server...")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable,
            "force_railway_fix.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
