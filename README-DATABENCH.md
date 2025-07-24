# DataBench Integration

This document explains how to set up and use the DataBench integration in the Tune Robotics website.

## Overview

DataBench is a comprehensive robotics dataset quality evaluation tool that assesses:
- **Action Consistency** (a): Visual-text alignment, action-observation consistency, temporal coherence
- **Visual Diversity** (v): Pairwise distances, clustering analysis, entropy measures  
- **High-Fidelity Vision** (h): Multi-view setup, resolution, environment quality, prompt clarity
- **Trajectory Quality** (t): Synchronization, frequency, data completeness
- **Dataset Coverage** (c): Scale, task diversity, visual variety, failure rates
- **Robot Action Quality** (r): Action smoothness, joint limits, physical feasibility

## Setup Instructions

### 1. Install DataBench Dependencies

```bash
# Navigate to the databench directory
cd databench

# Install databench requirements
pip install -r requirements.txt

# Fix numpy compatibility if needed
pip install "numpy<2.0" --upgrade

# Set Python path
export PYTHONPATH=$(pwd)
```

### 2. Install Web API Dependencies

```bash
# Install Flask API requirements
pip install -r databench_requirements.txt
```

### 3. Start the DataBench API Server

```bash
# Start the Flask backend
python databench_api.py
```

The API server will start on `http://localhost:5002`

### 4. Access the Web Interface

Open your web browser and navigate to:
- `http://localhost:5002/` - Direct API access
- Or serve `databench.html` through your web server

## Using the Web Interface

### 1. Dataset Configuration
- **HuggingFace Dataset Path**: Enter the dataset path (e.g., `gribok201/150episodes6`)
- **Subset Size**: Optional, specify number of episodes to evaluate (1-10000)
- **HuggingFace Token**: Required only for private datasets

### 2. Select Metrics
Click on the metric cards to select which evaluations to run:
- **a** - Action Consistency
- **v** - Visual Diversity  
- **h** - High-Fidelity Vision
- **t** - Trajectory Quality
- **c** - Dataset Coverage
- **r** - Robot Action Quality

### 3. Run Evaluation
Click "Start Evaluation" to begin the process. The evaluation can take anywhere from a few minutes to an hour depending on:
- Dataset size
- Selected metrics
- Subset size

### 4. View Results
Results are displayed with color-coded scores:
- **Green (0.8-1.0)**: ⭐⭐⭐⭐⭐ Excellent
- **Blue (0.6-0.8)**: ⭐⭐⭐⭐ Good  
- **Yellow (0.4-0.6)**: ⭐⭐⭐ Moderate
- **Orange (0.2-0.4)**: ⭐⭐ Poor
- **Red (0.0-0.2)**: ⭐ Very Poor

## API Endpoints

### Health Check
```bash
GET /health
```
Returns server status and configuration info.

### Evaluate Dataset
```bash
POST /api/evaluate
Content-Type: application/json

{
  "dataset": "gribok201/150episodes6",
  "metrics": "a,v,h,t,c,r",
  "subset": 100,
  "token": "hf_optional_token"
}
```

### Get Available Metrics
```bash
GET /api/metrics
```

### List Results
```bash
GET /api/results
```

### Get Specific Result
```bash
GET /api/results/<filename>
```

## Configuration

### Environment Variables
- `HF_TOKEN`: HuggingFace API token for private datasets
- `PYTHONPATH`: Should include the databench directory

### Custom Configuration
You can modify evaluation parameters by editing `databench/config.yaml`:

```yaml
# Example: Adjust quality thresholds
trajectory_quality:
  freq_thresholds:
    gold: 30     # ≥30Hz = excellent
    silver: 10   # ≥10Hz = good  
    bronze: 5    # ≥5Hz = acceptable

# Example: Modify robot action quality thresholds
robot_action_quality:
  thresholds:
    smoothness_threshold: 0.1
    velocity_threshold: 0.5
```

## Supported Dataset Formats

### HuggingFace Datasets
- Public datasets: No authentication required
- Private datasets: Requires HuggingFace token
- Automatic format detection
- Video streaming and caching

### Local Datasets (LeRobot Format)
```
dataset/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl  
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       └── episode_*.parquet
└── videos/
    └── chunk-000/
        └── observation.images.*/
            └── episode_*.mp4
```

## Example Usage

### Command Line (Direct)
```bash
# Evaluate all metrics on a HuggingFace dataset
python databench/scripts/evaluate.py \
  --hf-dataset gribok201/150episodes6 \
  --metrics a,v,h,t,c,r \
  --subset 100

# Evaluate specific metrics only
python databench/scripts/evaluate.py \
  --hf-dataset lerobot/pusht_image \
  --metrics a,v,h \
  --subset 50
```

### Web API (cURL)
```bash
# Start evaluation via API
curl -X POST http://localhost:5002/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "gribok201/150episodes6",
    "metrics": "a,v,h",
    "subset": 100
  }'

# Check health
curl http://localhost:5002/health
```

## Troubleshooting

### Common Issues

#### 1. NumPy Compatibility
```bash
pip install "numpy<2.0" --upgrade
```

#### 2. CUDA/GPU Issues
```bash
export CUDA_VISIBLE_DEVICES=""
```

#### 3. HuggingFace Authentication
```bash
huggingface-cli login
# OR
export HF_TOKEN="your_token_here"
```

#### 4. Memory Issues
- Use smaller subset sizes
- Evaluate fewer metrics at once
- Process datasets in chunks

#### 5. API Connection Issues
- Ensure Flask server is running on port 5002
- Check CORS settings if accessing from different domain
- Verify databench path in `databench_api.py`

### Performance Tips

1. **Start Small**: Use subset=10 for initial testing
2. **Selective Metrics**: Only run metrics you need
3. **Monitor Resources**: Large datasets can use significant RAM/GPU
4. **Batch Processing**: For multiple datasets, process sequentially

## File Structure

```
.
├── databench.html              # Web interface
├── databench_api.py           # Flask API server
├── databench_requirements.txt # API dependencies
├── results/                   # Evaluation results
├── databench/                 # DataBench source code
│   ├── scripts/
│   │   └── evaluate.py       # Main evaluation script
│   ├── config.yaml           # Configuration file
│   ├── requirements.txt      # DataBench dependencies
│   └── metrics/              # Metric implementations
└── README-DATABENCH.md       # This file
```

## Support

- **DataBench Issues**: Check the databench repository
- **Web Integration**: Contact yo@tunerobotics.xyz
- **API Documentation**: Visit `/health` endpoint for status

## License

DataBench is licensed under Apache 2.0. See the databench directory for full license details. 