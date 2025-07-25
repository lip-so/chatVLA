#!/usr/bin/env python3
"""
Verification script for DataBench Railway deployment
Ensures all components are properly configured and working
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

def print_status(message, status):
    """Print colored status message"""
    if status == "OK":
        print(f"✅ {message}")
    elif status == "WARNING":
        print(f"⚠️  {message}")
    else:
        print(f"❌ {message}")

def check_python_path():
    """Verify Python path includes databench"""
    databench_path = Path(__file__).parent / "databench"
    pythonpath = os.environ.get('PYTHONPATH', '')
    
    if str(databench_path) in pythonpath:
        print_status("DataBench in PYTHONPATH", "OK")
        return True
    else:
        print_status(f"DataBench not in PYTHONPATH. Current: {pythonpath}", "ERROR")
        return False

def check_dependencies():
    """Check all required dependencies"""
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'cv2': 'OpenCV',
        'av': 'PyAV (video processing)',
        'imageio': 'ImageIO',
        'sklearn': 'Scikit-learn',
        'flask': 'Flask',
        'flask_socketio': 'Flask-SocketIO',
        'flask_cors': 'Flask-CORS',
        'serial': 'PySerial',
        'ultralytics': 'Ultralytics (YOLO)',
        'spacy': 'spaCy',
        'dtw': 'DTW-Python'
    }
    
    all_ok = True
    print("\n📦 Checking dependencies:")
    
    for module, name in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
            elif module == 'sklearn':
                import sklearn
            else:
                importlib.import_module(module)
            print_status(f"{name} ({module})", "OK")
        except ImportError:
            print_status(f"{name} ({module}) - NOT INSTALLED", "ERROR")
            all_ok = False
    
    return all_ok

def check_databench_structure():
    """Verify DataBench directory structure"""
    databench_path = Path(__file__).parent / "databench"
    
    required_files = [
        "scripts/evaluate.py",
        "scripts/embed_utils.py",
        "scripts/config_loader.py",
        "scripts/dataset_analyzer.py",
        "metrics/__init__.py",
        "metrics/action_consistency.py",
        "metrics/visual_diversity.py",
        "metrics/high_fidelity_vision.py",
        "metrics/trajectory_quality.py",
        "metrics/dataset_coverage.py",
        "metrics/robot_action_quality.py",
        "config.yaml"
    ]
    
    all_ok = True
    print("\n📁 Checking DataBench structure:")
    
    for file_path in required_files:
        full_path = databench_path / file_path
        if full_path.exists():
            print_status(f"{file_path}", "OK")
        else:
            print_status(f"{file_path} - MISSING", "ERROR")
            all_ok = False
    
    return all_ok

def check_databench_imports():
    """Test importing DataBench components"""
    print("\n🔧 Testing DataBench imports:")
    
    try:
        # Add databench to path
        databench_path = Path(__file__).parent / "databench"
        sys.path.insert(0, str(databench_path))
        
        from scripts.evaluate import RoboticsDatasetBenchmark, METRIC_MAPPING
        print_status("RoboticsDatasetBenchmark import", "OK")
        
        from scripts.embed_utils import EmbeddingManager
        print_status("EmbeddingManager import", "OK")
        
        from scripts.config_loader import get_config_loader
        print_status("Config loader import", "OK")
        
        # Check if embed_utils has all required imports
        from scripts import embed_utils
        required_attrs = ['av', 'subprocess', 'imageio', 'cosine_similarity']
        
        for attr in required_attrs:
            if hasattr(embed_utils, attr):
                print_status(f"embed_utils.{attr} available", "OK")
            else:
                # Check if it's imported within the module
                import inspect
                source = inspect.getsource(embed_utils)
                if f'import {attr}' in source or f'from' in source and attr in source:
                    print_status(f"embed_utils imports {attr}", "OK")
                else:
                    print_status(f"embed_utils missing {attr}", "ERROR")
                    return False
        
        return True
        
    except Exception as e:
        print_status(f"Import failed: {e}", "ERROR")
        return False

def check_multi_api():
    """Check if multi_api.py properly integrates DataBench"""
    print("\n🌐 Checking multi_api.py integration:")
    
    try:
        from multi_api import app, DataBenchAPI, DATABENCH_AVAILABLE
        
        if DATABENCH_AVAILABLE:
            print_status("DataBench available in multi_api", "OK")
        else:
            print_status("DataBench NOT available in multi_api", "ERROR")
            return False
        
        # Check if DataBenchAPI is properly configured
        api = DataBenchAPI()
        print_status("DataBenchAPI instantiated", "OK")
        
        return True
        
    except Exception as e:
        print_status(f"multi_api check failed: {e}", "ERROR")
        return False

def check_railway_config():
    """Check Railway configuration"""
    print("\n🚂 Checking Railway configuration:")
    
    railway_toml = Path(__file__).parent / "railway.toml"
    if railway_toml.exists():
        print_status("railway.toml exists", "OK")
        
        # Check content
        content = railway_toml.read_text()
        if "databench" in content.lower():
            print_status("railway.toml references databench", "OK")
        else:
            print_status("railway.toml doesn't reference databench", "WARNING")
    else:
        print_status("railway.toml missing", "ERROR")
        return False
    
    # Check start script
    start_script = Path(__file__).parent / "start_railway.py"
    if start_script.exists():
        print_status("start_railway.py exists", "OK")
    else:
        print_status("start_railway.py missing", "ERROR")
        return False
    
    return True

def main():
    """Run all verification checks"""
    print("🔍 DataBench Railway Deployment Verification")
    print("=" * 50)
    
    checks = [
        ("Python Path", check_python_path),
        ("Dependencies", check_dependencies),
        ("DataBench Structure", check_databench_structure),
        ("DataBench Imports", check_databench_imports),
        ("Multi-API Integration", check_multi_api),
        ("Railway Config", check_railway_config)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_status(f"{name} check failed with error: {e}", "ERROR")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY:")
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"   {check_name}: {status}")
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("✅ All checks passed! DataBench is ready for Railway deployment.")
        print("\n💡 Next steps:")
        print("   1. Commit and push changes")
        print("   2. Railway will automatically deploy")
        print("   3. Check Railway logs for any runtime issues")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   1. Install missing dependencies: pip install -r requirements-full.txt")
        print("   2. Ensure databench folder is present")
        print("   3. Set PYTHONPATH to include databench")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 