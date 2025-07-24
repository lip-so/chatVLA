#!/usr/bin/env python3
"""
DataBench Startup Script
Automatically sets up and runs the DataBench web interface
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check if databench directory exists
    databench_path = Path(__file__).parent / "databench"
    if not databench_path.exists():
        print("❌ DataBench directory not found!")
        print("   Please ensure the databench folder is in the same directory as this script.")
        print(f"   Expected path: {databench_path}")
        return False
    
    # Check if evaluation script exists
    eval_script = databench_path / "scripts" / "evaluate.py"
    if not eval_script.exists():
        print("❌ DataBench evaluation script not found!")
        print(f"   Expected: {eval_script}")
        return False
    
    # Check Python packages for Flask
    try:
        import flask
        import flask_cors
        print("✅ Flask dependencies found")
    except ImportError as e:
        print("❌ Flask dependencies missing!")
        print(f"   Missing: {e}")
        print("   Run: pip install -r databench_requirements.txt")
        return False
    
    # Check for databench dependencies
    try:
        import torch
        import transformers
        print("✅ Core ML dependencies found")
    except ImportError:
        print("⚠️  Some DataBench dependencies may be missing")
        print("   Run: pip install -r databench/requirements.txt")
        # Don't fail here as the user might want to install them later
    
    return True

def setup_environment():
    """Set up the Python environment"""
    print("🔧 Setting up environment...")
    
    databench_path = Path(__file__).parent / "databench"
    
    # Set PYTHONPATH
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(databench_path) not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{databench_path}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = str(databench_path)
        print(f"✅ PYTHONPATH set to include {databench_path}")
    
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"✅ Results directory ready: {results_dir}")

def install_dependencies():
    """Install missing dependencies"""
    print("📦 Installing dependencies...")
    
    requirements_files = [
        ("databench_requirements.txt", "Flask API dependencies"),
        ("databench/requirements.txt", "DataBench core dependencies")
    ]
    
    for req_file, description in requirements_files:
        req_path = Path(__file__).parent / req_file
        if req_path.exists():
            try:
                print(f"   Installing {description}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(req_path)
                ], check=True, capture_output=True, text=True)
                print(f"✅ Installed {description}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {req_file}")
                print(f"   Error: {e.stderr}")
                return False
        else:
            print(f"⚠️  Requirements file not found: {req_file}")
    
    return True

def test_databench():
    """Test that databench works"""
    print("🧪 Testing DataBench...")
    
    databench_path = Path(__file__).parent / "databench"
    script_path = databench_path / "scripts" / "evaluate.py"
    
    try:
        # Test with --help
        result = subprocess.run([
            sys.executable, str(script_path), "--help"
        ], capture_output=True, text=True, timeout=10, cwd=databench_path)
        
        if result.returncode == 0:
            print("✅ DataBench command line interface working")
            return True
        else:
            print(f"❌ DataBench test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ DataBench test timed out")
        return False
    except Exception as e:
        print(f"❌ DataBench test failed: {e}")
        return False

def start_server():
    """Start the DataBench API server"""
    print("🚀 Starting DataBench API server...")
    
    api_script = Path(__file__).parent / "databench_api.py"
    if not api_script.exists():
        print("❌ API script not found!")
        return False
    
    try:
        # Start the server
        print("\n" + "="*50)
        print("🌐 DataBench Web Interface Starting")
        print("="*50)
        print("   Server URL: http://localhost:5002/")
        print("   Health Check: http://localhost:5002/health")
        print("   API Documentation: See README-DATABENCH.md")
        print("")
        print("   Press Ctrl+C to stop the server")
        print("="*50)
        
        subprocess.run([sys.executable, str(api_script)], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🤖 DataBench Startup Script")
    print("="*40)
    print("This script will set up and start the DataBench web interface.")
    print("DataBench evaluates robotics datasets across 6 quality metrics.")
    print("")
    
    # Check if we can run
    if not check_dependencies():
        print("\n💡 Setup Required:")
        print("   1. Ensure databench folder is present with scripts/evaluate.py")
        print("   2. Install Flask dependencies: pip install -r databench_requirements.txt")
        print("   3. Install DataBench dependencies: pip install -r databench/requirements.txt")
        
        # Ask if user wants to auto-install
        response = input("\n🔧 Would you like to try auto-installing dependencies? (y/N): ")
        if response.lower() in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Auto-installation failed. Please install manually.")
                return 1
        else:
            return 1
    
    # Set up environment
    setup_environment()
    
    # Test databench
    if not test_databench():
        print("\n⚠️  DataBench test failed. Some features may not work.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            return 1
    
    # Show usage instructions
    print("\n📖 Usage Instructions:")
    print("   1. Enter a HuggingFace dataset path (e.g., 'gribok201/150episodes6')")
    print("   2. Select the metrics you want to evaluate")
    print("   3. Optionally set a subset size for faster testing")
    print("   4. Click 'Start Evaluation' and wait for results")
    print("")
    print("   Available metrics:")
    print("   - Action Consistency (a): Visual-text alignment")
    print("   - Visual Diversity (v): Scene variation analysis")
    print("   - High-Fidelity Vision (h): Quality assessment")
    print("   - Trajectory Quality (t): Data completeness")
    print("   - Dataset Coverage (c): Scale and diversity")
    print("   - Robot Action Quality (r): Motion smoothness")
    print("")
    
    # Start server
    if not start_server():
        return 1
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n🛑 Startup cancelled by user")
        sys.exit(0) 