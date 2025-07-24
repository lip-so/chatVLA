#!/usr/bin/env python3
"""
Fix Port Conflict for DataBench
Kills any processes using port 5002 and starts DataBench server
"""

import subprocess
import sys
import time
import requests

def kill_port_5002():
    """Kill any processes using port 5002"""
    print("🔍 Checking for processes on port 5002...")
    
    try:
        # Find processes using port 5002
        result = subprocess.run(['lsof', '-ti:5002'], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"Found {len(pids)} process(es) using port 5002")
            
            # Kill the processes
            for pid in pids:
                print(f"Killing process {pid}...")
                subprocess.run(['kill', '-9', pid], capture_output=True)
            
            print("✅ Port 5002 is now free")
            time.sleep(1)
        else:
            print("✅ Port 5002 is already free")
            
    except FileNotFoundError:
        print("⚠️  lsof command not found. Trying alternative method...")
        # Alternative for systems without lsof
        try:
            subprocess.run(['pkill', '-f', 'databench_api.py'], capture_output=True)
            print("✅ Killed any existing databench processes")
        except:
            print("⚠️  Could not kill existing processes automatically")

def start_databench():
    """Start the DataBench server"""
    print("\n🚀 Starting DataBench server...")
    
    try:
        # Start the server in background
        process = subprocess.Popen([
            sys.executable, 'databench_api.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test if server is running
        try:
            response = requests.get('http://localhost:5002/health', timeout=5)
            if response.status_code == 200:
                print("✅ DataBench server started successfully!")
                print("🌐 Server running at: http://localhost:5002/")
                print("📊 Health check: http://localhost:5002/health")
                return True
        except:
            pass
            
        print("❌ Failed to start DataBench server")
        print("💡 Try running manually: python databench_api.py")
        return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    """Main function"""
    print("🔧 DataBench Port Conflict Fixer")
    print("="*40)
    
    # Kill existing processes
    kill_port_5002()
    
    # Start DataBench
    success = start_databench()
    
    if success:
        print("\n🎉 DataBench is ready!")
        print("\n💡 Next steps:")
        print("   1. Open your browser to: http://localhost:5002/")
        print("   2. Try evaluating a dataset like 'gribok201/150episodes6'")
        print("   3. Select some metrics and run evaluation")
        print("\n⏹️  To stop the server: Press Ctrl+C in this terminal")
        
        # Keep the script running so user can see server output
        try:
            print("\n" + "="*40)
            print("Server is running... Press Ctrl+C to stop")
            print("="*40)
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping DataBench server...")
            kill_port_5002()
            print("✅ Server stopped")
    else:
        print("\n💡 Alternative solutions:")
        print("   1. Deploy to Railway (see DEPLOY_DATABENCH.md)")
        print("   2. Use a different port: python databench_api.py --port 5003")
        print("   3. Restart your computer to clear port conflicts")
    
    return 0 if success else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")
        sys.exit(0) 