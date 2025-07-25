#!/usr/bin/env python3
"""
Run all tests for the Tune Robotics system and provide a comprehensive summary
"""

import subprocess
import sys
import time
import os
from pathlib import Path

class TestRunner:
    def __init__(self):
        self.results = {}
        self.servers_started = []
        
    def run_test(self, name, command):
        """Run a test and capture results"""
        print(f"\n🧪 Running {name}...")
        print(f"   Command: {command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=120
            )
            
            success = result.returncode == 0
            self.results[name] = {
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            if success:
                print(f"   ✅ {name} PASSED")
            else:
                print(f"   ❌ {name} FAILED (return code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                    
            return success
            
        except subprocess.TimeoutExpired:
            self.results[name] = {
                'success': False,
                'error': 'Test timed out after 120 seconds'
            }
            print(f"   ❌ {name} TIMED OUT")
            return False
            
        except Exception as e:
            self.results[name] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ {name} ERROR: {e}")
            return False
            
    def start_servers(self):
        """Start necessary servers for testing"""
        print("\n🚀 Starting test servers...")
        
        # Kill any existing servers
        subprocess.run("pkill -f 'python multi_api.py' || true", shell=True)
        subprocess.run("pkill -f 'python databench_api.py' || true", shell=True)
        time.sleep(2)
        
        # Start multi_api server on port 5003
        env = os.environ.copy()
        env['PORT'] = '5003'
        env['PYTHONPATH'] = str(Path(__file__).parent / 'databench')
        
        multi_api_proc = subprocess.Popen(
            [sys.executable, 'multi_api.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.servers_started.append(multi_api_proc)
        
        # Start databench_api server on port 5002
        databench_proc = subprocess.Popen(
            [sys.executable, 'databench_api.py'],
            env={'PYTHONPATH': str(Path(__file__).parent / 'databench')},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.servers_started.append(databench_proc)
        
        # Wait for servers to start
        print("   Waiting for servers to start...")
        time.sleep(8)  # Increased wait time for slower startup
        print("   ✅ Servers started")
        
    def stop_servers(self):
        """Stop all started servers"""
        print("\n🛑 Stopping test servers...")
        for proc in self.servers_started:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except:
                proc.kill()
                
        subprocess.run("pkill -f 'python multi_api.py' || true", shell=True)
        subprocess.run("pkill -f 'python databench_api.py' || true", shell=True)
        print("   ✅ Servers stopped")
        
    def run_all_tests(self):
        """Run all tests in the system"""
        print("🤖 Tune Robotics Comprehensive Test Suite")
        print("=" * 60)
        
        # Start servers
        self.start_servers()
        
        try:
            # Test 1: Basic system test
            self.run_test(
                "Basic System Test",
                "python test_everything.py"
            )
            
            # Test 2: DataBench API test
            self.run_test(
                "DataBench API Test", 
                "python test_databench_api.py"
            )
            
            # Test 3: Integration test
            self.run_test(
                "Integration Test",
                "PORT=5003 python test_integration.py"
            )
            
            # Test 4: Plug & Play integration test
            self.run_test(
                "Plug & Play Integration Test",
                "PORT=5003 python test_plugplay_integration.py"
            )
            
            # Test 5: DataBench video loading dependencies
            self.run_test(
                "Video Loading Dependencies",
                "cd databench && PYTHONPATH=/Users/sofiia/chatVLA/databench python scripts/test_video_loading.py --check-deps"
            )
            
            # Test 6: Railway deployment verification
            self.run_test(
                "Railway Deployment Verification",
                "python verify_databench_railway.py"
            )
            
        finally:
            # Always stop servers
            self.stop_servers()
        
        # Print summary
        self.print_summary()
        
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"{test_name:.<40} {status}")
            if result.get('success', False):
                passed += 1
            else:
                failed += 1
                
        print("=" * 60)
        print(f"Total: {len(self.results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed == 0:
            print("\n🎉 All tests passed! The system is working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
            
        # Return exit code based on results
        return 0 if failed == 0 else 1

if __name__ == "__main__":
    runner = TestRunner()
    exit_code = runner.run_all_tests()
    sys.exit(exit_code) 