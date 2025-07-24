# DataBench Deployment Guide

DataBench can be deployed to various cloud platforms. The full ML functionality requires platforms that support larger Docker images.

## Quick Deploy Options

### Option 1: Render (Recommended for Full ML)
Best for full ML functionality with PyTorch, transformers, etc.

1. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository

2. **Auto-Deploy Setup**:
   - Render will detect `render.yaml` automatically
   - Uses the full `requirements.txt` with all ML dependencies
   - Starts the full `databench_api.py` with real evaluation

3. **Environment Variables**:
   ```
   PYTHONPATH=/opt/render/project/src/databench
   PYTHONUNBUFFERED=1
   PORT=10000
   ```

4. **Health Check**: `/health`

### Option 2: Railway (Limited - Lightweight Only)
Free tier has 4GB Docker image limit, so only lightweight version works.

### Option 3: Fly.io
Good for ML workloads, similar to Render.

### Option 4: Local Development
Full functionality always available locally:
```bash
python databench_api.py
```

## Files Included

- `render.yaml` - Render deployment configuration
- `databench_api.py` - Full ML evaluation API  
- `requirements.txt` - Complete ML dependencies
- `Procfile` - Railway/Heroku deployment
- `runtime.txt` - Python version specification

## Website Integration

The website (`databench.html`) connects directly to the deployed API for seamless evaluation without any "demo mode" confusion. 