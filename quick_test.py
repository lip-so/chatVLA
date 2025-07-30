#!/usr/bin/env python3
"""Quick test to verify all pages are working"""

import subprocess
import time
import requests
import sys

def test_server():
    print("🧪 Testing reorganized application...")
    
    # Start server
    env = {'PORT': '5004'}
    server = subprocess.Popen([sys.executable, 'backend/api/main.py'], env=env)
    time.sleep(3)
    
    base_url = "http://localhost:5004"
    tests = [
        ("/", "Main page"),
        ("/pages/vision.html", "Vision page"),
        ("/pages/databench.html", "DataBench page"), 
        ("/pages/plug-and-play.html", "Plug & Play page"),
        ("/pages/port-detection.html", "Port Detection page"),
        ("/assets/logo.png", "Logo file"),
        ("/css/styles.css", "CSS file"),
        ("/js/app.js", "JavaScript file"),
        ("/health", "Health endpoint"),
        ("/api/databench/metrics", "DataBench API"),
        ("/api/plugplay/system-info", "Plug & Play API")
    ]
    
    passed = 0
    failed = 0
    
    for endpoint, name in tests:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}")
                passed += 1
            else:
                print(f"❌ {name} - Status: {response.status_code}")
                failed += 1
        except Exception as e:
            print(f"❌ {name} - Error: {e}")
            failed += 1
    
    # Stop server
    server.terminate()
    server.wait()
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = test_server()
    if success:
        print("🎉 All tests passed! Your reorganized application is working perfectly.")
    else:
        print("⚠️  Some tests failed.")
    sys.exit(0 if success else 1)