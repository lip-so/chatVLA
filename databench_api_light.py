import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Paths
DATABENCH_PATH = Path(__file__).parent / "databench"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    """Serve the main databench page"""
    return """
    <h1>DataBench API (Lightweight)</h1>
    <p>Status: Running in lightweight mode</p>
    <p>Note: Some advanced ML features are disabled due to deployment constraints.</p>
    <ul>
        <li><a href="/health">Health Check</a></li>
        <li><a href="/api/metrics">Available Metrics</a></li>
    </ul>
    """

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "mode": "lightweight",
            "databench_path": str(DATABENCH_PATH),
            "python_version": sys.version,
            "available_features": [
                "Basic dataset analysis",
                "Metadata extraction", 
                "Simple metrics"
            ],
            "disabled_features": [
                "Advanced ML analysis (requires PyTorch)",
                "Vision processing (requires full CV libraries)",
                "Heavy computations"
            ]
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get available metrics"""
    metrics = {
        "available": {
            "basic": "Basic dataset statistics",
            "metadata": "Dataset metadata analysis"
        },
        "disabled": {
            "action_consistency": "Requires PyTorch/transformers",
            "visual_diversity": "Requires CV libraries", 
            "high_fidelity_vision": "Requires PyTorch",
            "trajectory_quality": "Requires full ML stack",
            "dataset_coverage": "Requires embeddings",
            "robot_action_quality": "Requires ML analysis"
        },
        "note": "Running in lightweight mode due to deployment constraints"
    }
    return jsonify(metrics), 200

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Lightweight evaluation endpoint"""
    try:
        data = request.get_json()
        
        # Basic validation
        if not data or 'dataset' not in data:
            return jsonify({"error": "Dataset parameter required"}), 400
            
        dataset = data['dataset']
        
        # Return mock results for demonstration
        results = {
            "dataset": dataset,
            "status": "completed",
            "mode": "lightweight",
            "results": {
                "basic_stats": {
                    "total_episodes": "Unknown (requires full analysis)",
                    "avg_length": "Unknown (requires full analysis)",
                    "data_size": "Unknown (requires full analysis)"
                }
            },
            "message": "This is a lightweight demo. For full DataBench analysis, deploy with full ML dependencies or run locally.",
            "recommendation": "Use local setup for complete evaluation: python start_databench.py"
        }
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.exception("Lightweight evaluation failed")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment
    port = int(os.environ.get('PORT', 5002))
    
    logger.info(f"Starting DataBench API (Lightweight) on port {port}")
    logger.info("Note: Running in lightweight mode - some features disabled")
    
    # Run with host 0.0.0.0 for external access
    app.run(host='0.0.0.0', port=port, debug=False) 