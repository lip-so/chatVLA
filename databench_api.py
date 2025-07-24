#!/usr/bin/env python3
"""
DataBench Web API
Flask backend for the DataBench web interface
"""

import os
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configuration
DATABENCH_PATH = Path(__file__).parent / "databench"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Metric mapping
METRIC_CODES = {
    'a': 'action_consistency',
    'v': 'visual_diversity', 
    'h': 'hfv_overall_score',
    't': 'trajectory_quality',
    'c': 'dataset_coverage',
    'r': 'robot_action_quality'
}

METRIC_NAMES = {
    'a': 'Action Consistency',
    'v': 'Visual Diversity',
    'h': 'High-Fidelity Vision', 
    't': 'Trajectory Quality',
    'c': 'Dataset Coverage',
    'r': 'Robot Action Quality'
}

class DataBenchEvaluator:
    def __init__(self):
        self.databench_script = DATABENCH_PATH / "scripts" / "evaluate.py"
        
    def validate_request(self, data):
        """Validate the evaluation request"""
        errors = []
        
        if not data.get('dataset'):
            errors.append("Dataset path is required")
            
        if not data.get('metrics'):
            errors.append("At least one metric must be selected")
        else:
            # Validate metric codes
            invalid_metrics = [m for m in data['metrics'].split(',') if m not in METRIC_CODES]
            if invalid_metrics:
                errors.append(f"Invalid metrics: {', '.join(invalid_metrics)}")
                
        if data.get('subset'):
            try:
                subset = int(data['subset'])
                if subset <= 0 or subset > 10000:
                    errors.append("Subset size must be between 1 and 10000")
            except ValueError:
                errors.append("Subset size must be a valid number")
                
        return errors
        
    def build_command(self, data):
        """Build the databench evaluation command"""
        # Use correct argument name from databench
        cmd = [
            "python", str(self.databench_script),
            "--data", data['dataset'],  # Changed from --hf-dataset to --data
            "--metrics", data['metrics']
        ]
        
        if data.get('subset'):
            cmd.extend(["--subset", str(data['subset'])])
            
        # Generate output filename
        dataset_name = data['dataset'].replace('/', '_').replace(':', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"{dataset_name}_{timestamp}.json"
        cmd.extend(["--output", str(output_file)])
        
        return cmd, output_file
        
    def run_evaluation(self, data):
        """Run the databench evaluation"""
        try:
            # Validate request
            errors = self.validate_request(data)
            if errors:
                return {"error": "; ".join(errors)}, 400
                
            # Build command
            cmd, output_file = self.build_command(data)
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Set environment variables
            env = os.environ.copy()
            if data.get('token'):
                env['HF_TOKEN'] = data['token']
            env['PYTHONPATH'] = str(DATABENCH_PATH)
            
            # Run evaluation
            result = subprocess.run(
                cmd,
                cwd=DATABENCH_PATH,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Command failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout}")
                return {
                    "error": f"Evaluation failed: {result.stderr}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }, 500
                
            # Read results
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                return {"results": results, "output_file": str(output_file)}, 200
            else:
                return {"error": "Results file not found", "stdout": result.stdout}, 500
                
        except subprocess.TimeoutExpired:
            return {"error": "Evaluation timed out (max 1 hour)"}, 408
        except Exception as e:
            logger.exception("Evaluation failed")
            return {"error": str(e)}, 500

# Initialize evaluator
evaluator = DataBenchEvaluator()

@app.route('/', methods=['GET'])
def index():
    """Serve the main databench page"""
    return send_from_directory('.', 'databench.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "databench_path": str(DATABENCH_PATH),
        "databench_exists": DATABENCH_PATH.exists(),
        "script_exists": evaluator.databench_script.exists(),
        "results_dir": str(RESULTS_DIR),
        "python_path": os.environ.get('PYTHONPATH', 'Not set')
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate_dataset():
    """Main evaluation endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        logger.info(f"Evaluation request: {data}")
        
        result, status_code = evaluator.run_evaluation(data)
        return jsonify(result), status_code
        
    except Exception as e:
        logger.exception("API error")
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get available metrics"""
    return jsonify({
        "metrics": [
            {
                "code": code,
                "name": METRIC_NAMES[code],
                "description": "Detailed description would go here"
            }
            for code in METRIC_CODES.keys()
        ]
    })

@app.route('/api/results/<filename>', methods=['GET'])
def get_results(filename):
    """Get evaluation results"""
    try:
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            return jsonify({"error": "Results file not found"}), 404
            
        with open(filepath, 'r') as f:
            results = json.load(f)
            
        return jsonify(results)
        
    except Exception as e:
        logger.exception("Failed to read results")
        return jsonify({"error": str(e)}), 500

@app.route('/api/results', methods=['GET'])
def list_results():
    """List all available results"""
    try:
        results = []
        for file in RESULTS_DIR.glob("*.json"):
            stat = file.stat()
            results.append({
                "filename": file.name,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "size": stat.st_size
            })
            
        return jsonify({"results": sorted(results, key=lambda x: x['created'], reverse=True)})
        
    except Exception as e:
        logger.exception("Failed to list results")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Check if databench exists
    if not DATABENCH_PATH.exists():
        logger.error(f"DataBench not found at {DATABENCH_PATH}")
        exit(1)
        
    if not evaluator.databench_script.exists():
        logger.error(f"DataBench evaluation script not found at {evaluator.databench_script}")
        exit(1)
        
    # Set up environment
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(DATABENCH_PATH) not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{DATABENCH_PATH}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = str(DATABENCH_PATH)
        logger.info(f"PYTHONPATH set to include {DATABENCH_PATH}")
    
    # Get port from environment (for deployment platforms like Railway, Heroku, etc.)
    port = int(os.environ.get('PORT', 5002))
    
    logger.info(f"Starting DataBench API server on port {port}")
    logger.info(f"DataBench path: {DATABENCH_PATH}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Python path: {os.environ.get('PYTHONPATH')}")
    
    # Run with host 0.0.0.0 for external access and disable debug in production
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 