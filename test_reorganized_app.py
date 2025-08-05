#!/usr/bin/env python3
"""
Test script for the reorganized Tune Robotics application
Tests all functionality and file references
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path

class AppTester:
    def __init__(self, port=5002):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.server_process = None
        self.tests_passed = 0
        self.tests_failed = 0
        
    def log_success(self, message):
        print(f"[PASS] {message}")
        self.tests_passed += 1
        
    def log_error(self, message):
        print(f"[FAIL] {message}")
        self.tests_failed += 1
        
    def log_info(self, message):
        print(f"[INFO] {message}")
        
    def start_server(self):
        """Start the server for testing"""
        self.log_info(f"Starting server on port {self.port}...")
        
        env = os.environ.copy()
        env['PORT'] = str(self.port)
        
        try:
            self.server_process = subprocess.Popen(
                [sys.executable, "backend/api/main.py"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Check if server is running by testing health endpoint
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    self.log_success("Server started successfully")
                    return True
                else:
                    self.log_error(f"Server health check failed: {response.status_code}")
                    return False
            except requests.RequestException as e:
                self.log_error(f"Could not connect to server: {e}")
                return False
                
        except Exception as e:
            self.log_error(f"Failed to start server: {e}")
            return False
            
    def stop_server(self):
        """Stop the test server"""
        if self.server_process:
            self.log_info("Stopping server...")
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            
    def test_directory_structure(self):
        """Test that the directory structure is correct"""
        self.log_info("Testing directory structure...")
        
        required_dirs = [
            "frontend",
            "frontend/css",
            "frontend/js", 
            "frontend/pages",
            "frontend/assets",
            "backend",
            "backend/api",
            "backend/databench",
            "backend/plug_and_play",
            "tests",
            "deployment",
            "docs"
        ]
        
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                self.log_success(f"Directory exists: {dir_path}")
            else:
                self.log_error(f"Directory missing: {dir_path}")
                
    def test_essential_files(self):
        """Test that essential files exist"""
        self.log_info("Testing essential files...")
        
        essential_files = [
            "frontend/index.html",
            "frontend/css/styles.css", 
            "frontend/js/app.js",
            "frontend/assets/logo.png",
            "backend/api/main.py",
            "backend/databench/api.py",
            "backend/plug_and_play/api.py",
            "README.md",
            ".gitignore"
        ]
        
        for file_path in essential_files:
            if Path(file_path).exists():
                self.log_success(f"File exists: {file_path}")
            else:
                self.log_error(f"File missing: {file_path}")
                
    def test_health_endpoint(self):
        """Test the health endpoint"""
        self.log_info("Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.log_success("Health endpoint working")
                else:
                    self.log_error(f"Health endpoint unhealthy: {data}")
            else:
                self.log_error(f"Health endpoint failed: {response.status_code}")
        except Exception as e:
            self.log_error(f"Health endpoint error: {e}")
            
    def test_frontend_files(self):
        """Test that frontend files are served correctly"""
        self.log_info("Testing frontend file serving...")
        
        frontend_tests = [
            ("/", "index.html"),
            ("/css/styles.css", "CSS file"),
            ("/js/app.js", "JavaScript file"),
            ("/assets/logo.png", "Logo image"),
            ("/pages/databench.html", "DataBench page"),
            ("/pages/plug-and-play.html", "Plug & Play page")
        ]
        
        for endpoint, description in frontend_tests:
            try:
                response = requests.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    self.log_success(f"{description} served correctly")
                else:
                    self.log_error(f"{description} failed: {response.status_code}")
            except Exception as e:
                self.log_error(f"{description} error: {e}")
                
    def test_api_endpoints(self):
        """Test API endpoints"""
        self.log_info("Testing API endpoints...")
        
        api_tests = [
            ("/api/databench/metrics", "DataBench metrics"),
            ("/api/plugplay/system-info", "Plug & Play system info"),
            ("/api/plugplay/list-ports", "USB port detection")
        ]
        
        for endpoint, description in api_tests:
            try:
                response = requests.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    data = response.json()
                    self.log_success(f"{description} API working")
                else:
                    self.log_error(f"{description} API failed: {response.status_code}")
            except Exception as e:
                self.log_error(f"{description} API error: {e}")
                
    def test_file_references(self):
        """Test that file references in HTML are correct"""
        self.log_info("Testing file references...")
        
        try:
            # Test main page
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                content = response.text
                
                # Check for correct CSS reference
                if 'href="/css/styles.css"' in content:
                    self.log_success("CSS reference is correct")
                else:
                    self.log_error("CSS reference is incorrect")
                    
                # Check for correct JS references
                if 'src="/js/app.js"' in content:
                    self.log_success("JavaScript references are correct")
                else:
                    self.log_error("JavaScript references are incorrect")
                    
            else:
                self.log_error("Could not fetch main page for reference testing")
                
        except Exception as e:
            self.log_error(f"File reference testing error: {e}")
            
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Testing Reorganized Tune Robotics Application")
        print("=" * 60)
        
        # Test structure without server
        self.test_directory_structure()
        self.test_essential_files()
        
        # Start server for API tests
        if self.start_server():
            self.test_health_endpoint()
            self.test_frontend_files()
            self.test_api_endpoints()
            self.test_file_references()
            self.stop_server()
        else:
            self.log_error("Could not start server, skipping API tests")
            
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = self.tests_passed + self.tests_failed
        print(f"Total tests: {total_tests}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.tests_failed == 0:
            print("\nAll tests passed! The reorganized application is working correctly.")
            print("\nTo start the application:")
            print("  python start.py")
            print("  # or")
            print("  python backend/api/main.py")
            print("\nThen visit http://localhost:5000")
        else:
            print(f"\nWARNING: {self.tests_failed} tests failed. Please fix the issues.")
            
        return self.tests_failed == 0

if __name__ == "__main__":
    tester = AppTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)