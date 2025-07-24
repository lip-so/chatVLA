#!/usr/bin/env python3
"""
Railway Startup Script for DataBench
Ensures proper initialization and environment setup for Railway deployment
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_railway_environment():
    """Setup Railway-specific environment variables and paths"""
    
    # Set up paths
    app_root = Path(__file__).parent
    databench_path = app_root / 'databench'
    
    # Ensure PYTHONPATH includes databench
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    paths_to_add = [str(databench_path), str(app_root)]
    
    for path in paths_to_add:
        if path not in current_pythonpath:
            if current_pythonpath:
                current_pythonpath = f"{path}:{current_pythonpath}"
            else:
                current_pythonpath = path
    
    os.environ['PYTHONPATH'] = current_pythonpath
    
    # Setup HuggingFace cache directories
    hf_home = os.environ.get('HF_HOME', '/tmp/huggingface')
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(f"{hf_home}/transformers", exist_ok=True)
    os.makedirs(f"{hf_home}/datasets", exist_ok=True)
    
    # Railway-specific optimizations
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
    
    logger.info("✅ Railway environment setup complete")
    logger.info(f"📁 App root: {app_root}")
    logger.info(f"📁 DataBench path: {databench_path}")
    logger.info(f"🐍 Python path: {os.environ['PYTHONPATH']}")
    logger.info(f"💾 HF cache: {hf_home}")

def verify_dependencies():
    """Verify that critical dependencies are available"""
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        logger.error("❌ PyTorch not available")
        return False
        
    try:
        import transformers
        logger.info(f"✅ Transformers {transformers.__version__} available")
    except ImportError:
        logger.error("❌ Transformers not available")
        return False
        
    try:
        from databench.scripts.evaluate import RoboticsDatasetBenchmark
        logger.info("✅ DataBench evaluation system available")
    except ImportError as e:
        logger.error(f"❌ DataBench not available: {e}")
        return False
    
    # Test Plug & Play functionality
    try:
        import serial
        logger.info("✅ Plug & Play USB detection available")
    except ImportError:
        logger.warning("⚠️ pyserial not available - USB detection limited")
        
    return True

def main():
    """Main startup sequence"""
    logger.info("🚀 Starting Railway deployment for DataBench")
    
    # Setup environment
    setup_railway_environment()
    
    # Verify dependencies
    if not verify_dependencies():
        logger.error("❌ Dependency verification failed")
        sys.exit(1)
    
    # Import and run the main application
    try:
        from multi_api import app, socketio
        port = int(os.environ.get('PORT', 10000))
        
        logger.info(f"🌐 Starting server on port {port}")
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=port, 
            debug=False, 
            allow_unsafe_werkzeug=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()