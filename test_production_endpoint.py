#!/usr/bin/env python3
"""
Test script to verify the exact production endpoint issue
"""

import requests
import json
import sys

def test_endpoint(base_url, endpoint, data):
    """Test an API endpoint and show exactly what's returned"""
    url = f"{base_url}{endpoint}"
    
    print(f"\n🧪 Testing: {url}")
    print(f"📤 Sending: {json.dumps(data)}")
    
    try:
        response = requests.post(
            url, 
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"📊 Status: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        # Try to parse as JSON
        try:
            json_data = response.json()
            print(f"✅ JSON Response: {json.dumps(json_data, indent=2)}")
            return True
        except:
            # If not JSON, show raw content
            content = response.text[:200]
            print(f"❌ HTML/Text Response: {content}")
            
            # Check if it's HTML
            if content.strip().startswith('<'):
                print("🚨 This is HTML, not JSON - that's the problem!")
                print("The frontend is calling an endpoint that returns HTML instead of JSON")
            
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Test the production endpoint issue"""
    
    # Test data
    test_data = {
        "installation_path": "./test_install",
        "selected_robot": "koch"
    }
    
    print("🔍 PRODUCTION ENDPOINT DEBUGGING")
    print("=" * 50)
    
    # Test different possible base URLs
    test_urls = [
        "http://localhost:5000",  # Might be wrong backend
        "https://your-app.railway.app",  # Production URL (replace with actual)
    ]
    
    for base_url in test_urls:
        print(f"\n🌐 Testing base URL: {base_url}")
        
        # Test the problematic endpoint
        success = test_endpoint(base_url, "/api/plugplay/start-installation", test_data)
        
        if not success:
            # Test if a simple endpoint exists
            try:
                response = requests.get(f"{base_url}/health", timeout=5)
                print(f"Health check: {response.status_code}")
                if response.status_code == 200:
                    print("✅ Server is running, but wrong endpoints")
                else:
                    print("❌ Server issues")
            except:
                print("❌ Server not reachable")

if __name__ == "__main__":
    main()