#!/usr/bin/env python3
"""
Test script for DataBench API
Verifies that the API server can start and respond to requests
"""

import requests
import json
import time
import subprocess
import sys
import os
from pathlib import Path

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get('http://localhost:5002/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            print(f"   DataBench path: {data.get('databench_path')}")
            print(f"   DataBench exists: {data.get('databench_exists')}")
            print(f"   Script exists: {data.get('script_exists')}")
            print(f"   Python path: {data.get('python_path')}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server at http://localhost:5002")
        print("   Please start the server with: python databench_api.py")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_metrics_endpoint():
    """Test the metrics endpoint"""
    try:
        response = requests.get('http://localhost:5002/api/metrics', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Metrics endpoint working")
            print(f"   Available metrics: {len(data.get('metrics', []))}")
            for metric in data.get('metrics', []):
                print(f"   - {metric['code']}: {metric['name']}")
            return True
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics endpoint failed: {e}")
        return False

def test_databench_command():
    """Test that databench command works"""
    databench_path = Path(__file__).parent / "databench"
    script_path = databench_path / "scripts" / "evaluate.py"
    
    if not script_path.exists():
        print(f"❌ DataBench script not found: {script_path}")
        return False
    
    try:
        # Set up environment with proper PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = str(databench_path)
        
        # Test with --help to verify the script runs
        result = subprocess.run([
            sys.executable, str(script_path), "--help"
        ], capture_output=True, text=True, timeout=10, env=env)
        
        if result.returncode == 0:
            print("✅ DataBench command line script working")
            return True
        else:
            print(f"❌ DataBench script failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ DataBench command test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing DataBench API Integration")
    print("=" * 40)
    
    # Test databench command first
    print("\n1. Testing DataBench Command Line:")
    databench_ok = test_databench_command()
    
    # Test API endpoints
    print("\n2. Testing API Server:")
    health_ok = test_health_endpoint()
    
    if health_ok:
        print("\n3. Testing API Endpoints:")
        metrics_ok = test_metrics_endpoint()
    else:
        metrics_ok = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    print(f"   DataBench CLI: {'✅ PASS' if databench_ok else '❌ FAIL'}")
    print(f"   API Health: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   API Metrics: {'✅ PASS' if metrics_ok else '❌ FAIL'}")
    
    all_pass = databench_ok and health_ok and metrics_ok
    if all_pass:
        print("\n🎉 All tests passed! DataBench API is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Open databench.html in your browser")
        print("   2. Or visit http://localhost:5002/")
        print("   3. Try evaluating a dataset like 'gribok201/150episodes6'")
    else:
        print("\n⚠️  Some tests failed. Please check the setup:")
        if not databench_ok:
            print("   - Ensure databench folder is present with scripts/evaluate.py")
            print("   - Install dependencies: pip install -r databench/requirements.txt")
        if not health_ok:
            print("   - Start the API server: python databench_api.py")
        if not metrics_ok:
            print("   - Check API server logs for errors")
    
    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main()) 