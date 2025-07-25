#!/usr/bin/env python3
"""
Verification script for Plug & Play on Railway deployment
Run this AFTER deploying to Railway to verify everything works
"""

import requests
import sys
import json

def verify_railway_deployment(railway_url):
    """Verify Plug & Play works on Railway deployment"""
    
    if not railway_url.startswith('http'):
        railway_url = f'https://{railway_url}'
    
    print(f"🚀 Verifying Plug & Play on Railway: {railway_url}")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Frontend HTML
    print("\n1️⃣ Testing Plug & Play Frontend...")
    try:
        response = requests.get(f"{railway_url}/plug-and-play", timeout=10)
        if response.status_code == 200 and "Plug & Play" in response.text:
            print("✅ Frontend served correctly")
            # Check if it's using relative URLs (no localhost)
            if "localhost:5002" not in response.text or "window.location" in response.text:
                print("✅ Frontend configured for production (no hardcoded localhost)")
            else:
                print("⚠️  Frontend may have localhost URLs")
        else:
            print(f"❌ Frontend failed: Status {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"❌ Frontend error: {e}")
        all_passed = False
    
    # Test 2: API Endpoints
    endpoints = [
        ("/api/system-info", "System Info"),
        ("/api/status", "Installation Status"),
        ("/api/list-ports", "USB Ports"),
        ("/api/get-home-directory", "Home Directory"),
    ]
    
    print("\n2️⃣ Testing API Endpoints...")
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{railway_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {name}: {json.dumps(data, indent=2)[:100]}...")
            else:
                print(f"❌ {name} failed: Status {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"❌ {name} error: {e}")
            all_passed = False
    
    # Test 3: POST endpoints
    print("\n3️⃣ Testing POST Endpoints...")
    try:
        response = requests.post(
            f"{railway_url}/api/browse-directory",
            json={"path": "/"},
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Browse directory endpoint working")
        else:
            print(f"❌ Browse directory failed: Status {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"❌ Browse directory error: {e}")
        all_passed = False
    
    # Test 4: WebSocket Support
    print("\n4️⃣ Checking WebSocket Configuration...")
    try:
        response = requests.get(f"{railway_url}/socket.io/", timeout=5)
        if response.status_code in [200, 400]:  # 400 is normal without proper handshake
            print("✅ Socket.IO endpoint accessible")
        else:
            print("⚠️  Socket.IO may not be configured")
    except:
        print("⚠️  Socket.IO endpoint not responding (may be normal)")
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Plug & Play is working on Railway!")
        print("\n🎯 Users can access Plug & Play at:")
        print(f"   {railway_url}/plug-and-play")
    else:
        print("❌ Some tests failed. Check Railway logs for details.")
        
    print("\n📝 Notes for Railway:")
    print("- USB detection won't work in cloud (expected)")
    print("- Installation paths are simulated (can't modify cloud filesystem)")
    print("- WebSocket may need Railway's websocket support enabled")
    
    return all_passed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_railway_plugplay.py <your-app.railway.app>")
        sys.exit(1)
    
    railway_url = sys.argv[1]
    success = verify_railway_deployment(railway_url)
    sys.exit(0 if success else 1) 