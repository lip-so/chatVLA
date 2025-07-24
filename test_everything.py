#!/usr/bin/env python3
"""
Complete Test Suite for Tune Robotics System
Tests DataBench, Plug & Play, and all integrations
"""

import requests
import subprocess
import sys
import time
from pathlib import Path

def test_databench_server():
    """Test if DataBench server is running and responding"""
    print("🧪 Testing DataBench Server...")
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:5002/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ DataBench server is running!")
            print(f"   - DataBench path: {data.get('databench_path')}")
            print(f"   - Script exists: {data.get('script_exists')}")
            print(f"   - Python path set: {data.get('python_path', 'Not set') != 'Not set'}")
            return True
        else:
            print(f"❌ DataBench health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to DataBench server at http://localhost:5002")
        print("   💡 Try running: python databench_api.py")
        return False
    except Exception as e:
        print(f"❌ DataBench test failed: {e}")
        return False

def test_databench_metrics():
    """Test DataBench metrics endpoint"""
    print("\n🧪 Testing DataBench Metrics...")
    
    try:
        response = requests.get('http://localhost:5002/api/metrics', timeout=5)
        if response.status_code == 200:
            data = response.json()
            metrics = data.get('metrics', [])
            print(f"✅ Metrics endpoint working! Found {len(metrics)} metrics:")
            for metric in metrics[:3]:  # Show first 3
                print(f"   - {metric['code']}: {metric['name']}")
            return True
        else:
            print(f"❌ Metrics endpoint failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def test_plug_and_play():
    """Test Plug & Play functionality"""
    print("\n🧪 Testing Plug & Play...")
    
    # Check if file exists and is accessible
    pnp_file = Path("plug-and-play.html")
    if pnp_file.exists():
        print("✅ Plug & Play HTML file exists")
        
        # Basic file size check
        size = pnp_file.stat().st_size
        if size > 10000:  # Should be at least 10KB
            print(f"✅ File size looks good: {size:,} bytes")
            return True
        else:
            print(f"⚠️  File seems small: {size} bytes")
            return False
    else:
        print("❌ Plug & Play HTML file not found")
        return False

def test_website_files():
    """Test if all website files exist"""
    print("\n🧪 Testing Website Files...")
    
    required_files = [
        "landing.html",
        "Vision.html", 
        "databench.html",
        "plug-and-play.html",
        "styles.css"
    ]
    
    all_good = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_good = False
    
    return all_good

def start_databench_if_needed():
    """Start DataBench server if it's not running"""
    print("\n🚀 Checking DataBench Server Status...")
    
    try:
        # Quick check if server is already running
        response = requests.get('http://localhost:5002/health', timeout=2)
        if response.status_code == 200:
            print("✅ DataBench server is already running!")
            return True
    except:
        pass
    
    print("🔧 DataBench server not responding. Trying to start it...")
    
    # Try to start the server
    try:
        process = subprocess.Popen([
            sys.executable, "databench_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test again
        try:
            response = requests.get('http://localhost:5002/health', timeout=5)
            if response.status_code == 200:
                print("✅ DataBench server started successfully!")
                return True
        except:
            pass
            
        print("❌ Failed to start DataBench server")
        return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def show_usage_instructions():
    """Show instructions for using the system"""
    print("\n" + "="*60)
    print("📖 USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\n🌐 DataBench (Dataset Evaluation):")
    print("   1. Open your browser to: http://localhost:5002/")
    print("   2. Or open databench.html directly")
    print("   3. Enter a HuggingFace dataset (e.g., 'gribok201/150episodes6')")
    print("   4. Select metrics to evaluate")
    print("   5. Start evaluation and wait for results")
    
    print("\n🔧 Plug & Play (LeRobot Installation):")
    print("   1. Open plug-and-play.html in your browser")
    print("   2. Choose installation directory")
    print("   3. Click 'Start Installation'")
    print("   4. Monitor progress in the terminal")
    
    print("\n🎯 Quick Test:")
    print("   1. Visit DataBench: http://localhost:5002/")
    print("   2. Try dataset: 'gribok201/150episodes6'")
    print("   3. Select metric: 'a' (Action Consistency)")
    print("   4. Set subset: 10")
    print("   5. Run evaluation")

def main():
    """Run all tests"""
    print("🤖 Tune Robotics System Test Suite")
    print("="*50)
    
    # Track results
    results = {}
    
    # Test website files
    results['files'] = test_website_files()
    
    # Start DataBench server if needed
    start_databench_if_needed()
    
    # Test DataBench
    results['databench_server'] = test_databench_server()
    if results['databench_server']:
        results['databench_metrics'] = test_databench_metrics()
    else:
        results['databench_metrics'] = False
    
    # Test Plug & Play
    results['plug_and_play'] = test_plug_and_play()
    
    # Show summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📈 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your system is ready to use.")
        show_usage_instructions()
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
        
        if not results['databench_server']:
            print("\n💡 DataBench Fix:")
            print("   Run: python databench_api.py")
            print("   Then try: http://localhost:5002/health")
    
    return 0 if passed_tests == total_tests else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n🛑 Testing cancelled by user")
        sys.exit(0) 