#!/usr/bin/env python3
"""
Standalone DataBench Test Server
Simple Flask server for testing DataBench functionality without Firebase authentication
"""

import os
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import DataBench components
from databench.api import DataBenchEvaluator, METRIC_CODES, METRIC_NAMES

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["*"])

# Initialize DataBench evaluator
databench_evaluator = DataBenchEvaluator()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "databench_available": True,
        "databench_path": str(databench_evaluator.databench_script.parent),
        "script_exists": databench_evaluator.databench_script.exists()
    })

@app.route('/api/databench/metrics', methods=['GET'])
def get_metrics():
    """Get available DataBench metrics"""
    metrics = {}
    for code, name in METRIC_NAMES.items():
        metrics[code] = {
            "name": name,
            "code": METRIC_CODES[code],
            "description": f"Evaluate {name.lower()} of robotics datasets"
        }
    return jsonify({"metrics": metrics})

@app.route('/api/databench/evaluate', methods=['POST'])
def evaluate_dataset():
    """Run DataBench evaluation - NO AUTHENTICATION REQUIRED (FOR TESTING)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        print(f"🔬 DataBench evaluation request: {data}")
        
        result, status_code = databench_evaluator.run_evaluation(data)
        return jsonify(result), status_code
        
    except Exception as e:
        print(f"❌ DataBench API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/databench/results/<filename>', methods=['GET'])
def get_results(filename):
    """Get evaluation results"""
    try:
        results_dir = Path(__file__).parent / "backend" / "databench" / "results"
        filepath = results_dir / filename
        
        if not filepath.exists():
            return jsonify({"error": "Results file not found"}), 404
            
        import json
        with open(filepath, 'r') as f:
            results = json.load(f)
            
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    
    print(f"""
    🧪 DataBench Test Server (No Authentication)
    ===========================================
    - DataBench API: Available
    - Authentication: DISABLED (for testing only)
    - Health endpoint: /health
    - Evaluate endpoint: /api/databench/evaluate
    - Metrics endpoint: /api/databench/metrics
    
    Starting server on http://localhost:{port}
    """)
    
    app.run(host='0.0.0.0', port=port, debug=True)