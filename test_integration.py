#!/usr/bin/env python3
"""
Test script to verify DataBench integration works properly
"""

import json
import requests
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Health check passed")
            logger.info(f"DataBench status: {data.get('services', {}).get('databench', {}).get('status')}")
            return True
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return False

def test_metrics_endpoint(base_url):
    """Test the metrics endpoint"""
    try:
        response = requests.get(f"{base_url}/api/metrics", timeout=30)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Metrics endpoint working")
            logger.info(f"Available metrics: {list(data.get('metrics', {}).keys())}")
            return True
        else:
            logger.error(f"❌ Metrics endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Metrics endpoint error: {e}")
        return False

def test_evaluation_endpoint(base_url, test_dataset="lerobot/aloha_sim_insertion_human"):
    """Test the evaluation endpoint with a small dataset"""
    try:
        test_data = {
            'dataset': test_dataset,
            'metrics': 'a,v',  # Just action consistency and visual diversity
            'subset': 2  # Very small subset for testing
        }
        
        logger.info(f"Testing evaluation with dataset: {test_dataset}")
        response = requests.post(
            f"{base_url}/api/evaluate", 
            json=test_data, 
            timeout=600  # 10 minute timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                logger.info("✅ Evaluation test passed")
                logger.info(f"Results: {data.get('results', {})}")
                return True
            else:
                logger.error(f"❌ Evaluation failed: {data.get('error')}")
                return False
        else:
            logger.error(f"❌ Evaluation endpoint failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Evaluation test error: {e}")
        return False

def main():
    """Run integration tests"""
    # Test locally first
    # Use port from environment or default to 5003 to avoid macOS AirPlay conflict
    port = os.environ.get('PORT', '5003')
    base_url = f"http://localhost:{port}"
    
    logger.info("🧪 Running DataBench integration tests")
    
    # Test health
    if not test_health_endpoint(base_url):
        logger.error("❌ Health test failed - aborting")
        return
    
    # Test metrics
    if not test_metrics_endpoint(base_url):
        logger.error("❌ Metrics test failed - aborting")
        return
    
    # Test evaluation (this is the main test)
    logger.info("🔍 Running evaluation test (this may take several minutes)...")
    if test_evaluation_endpoint(base_url):
        logger.info("🎉 All tests passed! DataBench integration is working.")
    else:
        logger.error("❌ Evaluation test failed")

if __name__ == '__main__':
    main()