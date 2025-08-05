#!/usr/bin/env python3
"""
Health check script to verify the backend is working properly
"""
import requests
import sys
import time

def check_backend_health():
    """Check if the backend is responding"""
    try:
        # Check local backend if running
        print("Checking local backend...")
        response = requests.get('http://localhost:5000/api/test', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Local backend is working: {data['message']}")
            return True
    except Exception as e:
        print(f"❌ Local backend not accessible: {e}")
    
    try:
        # Check Railway production backend
        print("\nChecking Railway production backend...")
        response = requests.get('https://chatvla-production.up.railway.app/api/test', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Production backend is working: {data['message']}")
            print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
            return True
        else:
            print(f"❌ Production backend returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Production backend not accessible: {e}")
    
    return False

def check_frontend():
    """Check if the frontend is being served"""
    try:
        print("\nChecking frontend...")
        response = requests.get('https://chatvla-production.up.railway.app/', timeout=10)
        if response.status_code == 200:
            if 'Tune Robotics' in response.text:
                print("✅ Frontend is being served correctly")
                return True
            else:
                print("⚠️ Frontend is served but content may be incorrect")
        else:
            print(f"❌ Frontend returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend not accessible: {e}")
    
    return False

if __name__ == "__main__":
    print("🔍 Checking Tune Robotics deployment health...")
    print("=" * 50)
    
    backend_ok = check_backend_health()
    frontend_ok = check_frontend()
    
    print("\n" + "=" * 50)
    if backend_ok and frontend_ok:
        print("✅ All systems are working correctly!")
        sys.exit(0)
    elif backend_ok:
        print("⚠️ Backend is working but frontend has issues")
        sys.exit(1)
    else:
        print("❌ Backend is not responding - deployment needs attention")
        sys.exit(1)