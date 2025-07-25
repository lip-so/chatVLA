#!/usr/bin/env python3
"""
Test script to verify Plug & Play integration with unified API
"""

import json
import requests
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_plugplay_endpoints(base_url):
    """Test all Plug & Play related endpoints"""
    
    endpoints_to_test = [
        ("/health", "GET", None, "Health check"),
        ("/api/system_info", "GET", None, "System information"),
        ("/api/status", "GET", None, "Installation status"),
        ("/api/scan_usb_ports", "GET", None, "USB port scanning"),
        ("/api/browse-directory", "POST", {"path": "/"}, "Directory browsing")
    ]
    
    results = []
    
    for endpoint, method, data, description in endpoints_to_test:
        try:
            logger.info(f"Testing {description}: {method} {endpoint}")
            
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=30)
            elif method == "POST":
                response = requests.post(
                    f"{base_url}{endpoint}", 
                    json=data, 
                    timeout=30
                )
            
            if response.status_code == 200:
                result_data = response.json()
                logger.info(f"✅ {description} - OK")
                results.append((endpoint, True, result_data))
            else:
                logger.error(f"❌ {description} - HTTP {response.status_code}")
                results.append((endpoint, False, response.text))
                
        except Exception as e:
            logger.error(f"❌ {description} - Error: {e}")
            results.append((endpoint, False, str(e)))
    
    return results

def test_installation_workflow(base_url):
    """Test the installation workflow"""
    logger.info("🧪 Testing installation workflow")
    
    try:
        # Start installation
        install_data = {
            'installation_path': '/tmp/test_lerobot_install'
        }
        
        response = requests.post(
            f"{base_url}/api/start_installation",
            json=install_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                logger.info("✅ Installation started successfully")
                
                # Wait a moment, then cancel
                time.sleep(2)
                
                cancel_response = requests.post(
                    f"{base_url}/api/cancel_installation",
                    timeout=30
                )
                
                if cancel_response.status_code == 200:
                    logger.info("✅ Installation cancelled successfully")
                    return True
                else:
                    logger.error("❌ Failed to cancel installation")
                    return False
            else:
                logger.error(f"❌ Installation failed to start: {data.get('error')}")
                return False
        else:
            logger.error(f"❌ Installation start failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Installation workflow test failed: {e}")
        return False

def main():
    """Run all Plug & Play integration tests"""
    # Use port from environment or default to 5003 to avoid macOS AirPlay conflict
    port = os.environ.get('PORT', '5003')
    base_url = f"http://localhost:{port}"
    
    logger.info("🔧 Testing Plug & Play integration with unified API")
    
    # Test basic endpoints
    logger.info("📋 Testing API endpoints...")
    endpoint_results = test_plugplay_endpoints(base_url)
    
    # Count successes
    successful_endpoints = sum(1 for _, success, _ in endpoint_results if success)
    total_endpoints = len(endpoint_results)
    
    logger.info(f"📊 Endpoint results: {successful_endpoints}/{total_endpoints} successful")
    
    # Test installation workflow
    logger.info("⚙️ Testing installation workflow...")
    workflow_success = test_installation_workflow(base_url)
    
    # Final summary
    logger.info("📋 Test Summary:")
    logger.info(f"   Endpoints: {successful_endpoints}/{total_endpoints}")
    logger.info(f"   Workflow: {'✅ Pass' if workflow_success else '❌ Fail'}")
    
    if successful_endpoints == total_endpoints and workflow_success:
        logger.info("🎉 All Plug & Play integration tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)