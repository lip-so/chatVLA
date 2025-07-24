# 🧪 DataBench Integration Guide

**DataBench** is now fully integrated into the Tune Robotics website, providing comprehensive robotics dataset evaluation capabilities.

## 🚀 Quick Start

### 1. Start DataBench API
```bash
# In your project root
python databench_api.py
```

The API will start on `http://localhost:5002` with full ML functionality.

### 2. Open DataBench Web Interface
Visit `tunerobotics.xyz/databench.html` or open `databench.html` locally.

The interface automatically connects to your local DataBench server for real evaluation results.

### 3. Run Your First Evaluation
1. **Enter Dataset**: `gribok201/150episodes6` (example)
2. **Select Metrics**: Choose evaluation metrics you want
3. **Click "Start Evaluation"**: Real analysis with ML models
4. **View Results**: Comprehensive quality scores and insights

## 🔬 Full Functionality (Local Setup)

When you run `python databench_api.py` locally, you get:

✅ **Real ML Analysis**: PyTorch, transformers, sentence-transformers  
✅ **All Metrics**: Action consistency, visual diversity, trajectory quality  
✅ **HuggingFace Integration**: Private datasets with tokens  
✅ **Advanced Computer Vision**: CLIP models, optical flow  
✅ **Comprehensive Results**: Detailed breakdowns and insights  

## 🌐 Cloud Deployment (Coming Soon)

We're working on deploying the full DataBench with ML capabilities to Render for seamless cloud access.

**Current Status**: Railway free tier can't handle the ML dependencies (4GB limit)  
**Solution**: Render deployment with full PyTorch stack in progress  

## 📊 What DataBench Evaluates

### Available Metrics

| Code | Metric | Description |
|------|--------|-------------|
| `a` | Action Consistency | Visual-text alignment, temporal coherence |
| `v` | Visual Diversity | Scene variation, environmental coverage |
| `h` | High-Fidelity Vision | Resolution, frame rate, multi-view setup |
| `t` | Trajectory Quality | Synchronization, frequency, completeness |
| `c` | Dataset Coverage | Scale, task diversity, failure analysis |
| `r` | Robot Action Quality | Smoothness, joint limits, feasibility |

### Example Evaluation
```json
{
  "action_consistency": 0.847,
  "visual_diversity": 0.923,
  "trajectory_quality": 0.756,
  "dataset_coverage": 0.834,
  "overall_score": 0.840
}
```

## 🛠️ Setup Instructions

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt

# Or minimal requirements for basic functionality
pip install -r databench/requirements-minimal.txt
```

### Environment Setup
```bash
# Set Python path for databench imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/databench"

# Or the API will set it automatically
python databench_api.py
```

### Configuration
Edit `databench/config.yaml` to customize:
- Model settings
- Evaluation parameters  
- Output formats
- Cache settings

## 🔧 API Endpoints

### Health Check
```bash
curl http://localhost:5002/health
```

### Available Metrics
```bash
curl http://localhost:5002/api/metrics
```

### Run Evaluation
```bash
curl -X POST http://localhost:5002/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "gribok201/150episodes6",
    "metrics": "a,v,t",
    "subset": 5
  }'
```

## 🎯 Integration Details

### Frontend Features
- **Smart Progress Tracking**: Real-time evaluation status
- **Interactive Metric Selection**: Choose specific evaluations
- **Result Visualization**: Clean display of scores and insights
- **Error Handling**: Helpful messages for setup issues

### Backend Capabilities  
- **Automatic Dataset Detection**: HuggingFace Hub integration
- **Parallel Processing**: Multi-metric evaluation
- **Caching System**: Faster repeated evaluations
- **Comprehensive Logging**: Detailed evaluation logs

## 🐛 Troubleshooting

### Port Conflicts
```bash
# Kill any process using port 5002
lsof -ti:5002 | xargs kill -9

# Or use a different port
python databench_api.py --port 5003
```

### Missing Dependencies
```bash
# Install full ML stack
pip install torch transformers sentence-transformers

# Check installation
python -c "import torch; print(torch.__version__)"
```

### Dataset Access Issues
- **Private datasets**: Add HuggingFace token in the web interface
- **Network issues**: Check your internet connection
- **Quota limits**: Use smaller subset sizes for testing

## 📈 Performance Tips

1. **Start Small**: Use `subset: 5` for testing
2. **Cache Results**: Evaluations are cached for faster re-runs  
3. **Select Metrics**: Only evaluate metrics you need
4. **Local First**: Always faster than cloud deployment

## 🤝 Contributing

DataBench is actively developed. To contribute:

1. **Add New Metrics**: Extend `databench/metrics/`
2. **Improve UI**: Enhance the web interface
3. **Optimize Performance**: Speed up evaluations
4. **Add Datasets**: Support more robotics datasets

---

**Need Help?** Contact: yo@tunerobotics.xyz  
**Full Documentation**: See `databench/README.md` 