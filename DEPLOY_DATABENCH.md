# 🌐 Deploy DataBench to Remote Server

This guide shows you how to host DataBench on a remote server instead of running it locally.

## 🚀 Quick Deploy Options

### Option 1: Railway (Recommended - Free & Easy)

1. **Create Railway Account**: Go to [railway.app](https://railway.app)

2. **Deploy from GitHub**:
   ```bash
   # Push your code to GitHub first
   git add .
   git commit -m "Add DataBench"
   git push origin main
   ```

3. **Railway Setup**:
   - Connect your GitHub repo
   - Railway will auto-detect Python
   - Set environment variables:
     ```
     PORT=5002
     PYTHONPATH=/app/databench
     ```

4. **Create `Procfile`**:
   ```bash
   web: python databench_api.py
   ```

5. **Update Port in `databench_api.py`**:
   ```python
   if __name__ == '__main__':
       port = int(os.environ.get('PORT', 5002))
       app.run(host='0.0.0.0', port=port)
   ```

### Option 2: Render (Free Tier Available)

1. **Create Render Account**: Go to [render.com](https://render.com)

2. **Create Web Service**:
   - Connect GitHub repo
   - Build Command: `pip install -r databench_requirements.txt && pip install -r databench/requirements.txt`
   - Start Command: `python databench_api.py`

3. **Environment Variables**:
   ```
   PYTHON_VERSION=3.9
   PYTHONPATH=/opt/render/project/src/databench
   ```

### Option 3: Heroku (Easy but Paid)

1. **Install Heroku CLI**
2. **Create Heroku App**:
   ```bash
   heroku create your-databench-app
   ```

3. **Configure Python**:
   ```bash
   echo "python-3.9.0" > runtime.txt
   echo "web: python databench_api.py" > Procfile
   ```

4. **Deploy**:
   ```bash
   git push heroku main
   ```

### Option 4: DigitalOcean App Platform

1. **Create DO Account**: $200 free credit available
2. **Create App from GitHub**
3. **Configure**:
   - Build Command: `pip install -r databench_requirements.txt`
   - Run Command: `python databench_api.py`

### Option 5: Google Cloud Run (Pay per use)

1. **Create `Dockerfile`**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r databench_requirements.txt
   RUN pip install -r databench/requirements.txt
   ENV PYTHONPATH=/app/databench
   EXPOSE 5002
   CMD ["python", "databench_api.py"]
   ```

2. **Deploy**:
   ```bash
   gcloud run deploy databench --source .
   ```

## 🔧 Required Files for Deployment

Create these files in your project root:

### `requirements-deploy.txt`
```
Flask>=2.3.0
Flask-CORS>=4.0.0
requests>=2.31.0
torch>=2.4.0,<2.5.0
transformers>=4.53.0,<4.54.0
sentence-transformers>=3.0.0,<3.1.0
numpy>=1.24.0,<2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
pandas>=2.0.0
pyarrow>=13.0.0
datasets>=2.15.0
huggingface_hub>=0.17.0
PyYAML>=6.0.1
```

### Updated `databench_api.py` (for deployment)
```python
import os

# ... existing code ...

if __name__ == '__main__':
    # Check if databench exists
    if not DATABENCH_PATH.exists():
        logger.error(f"DataBench not found at {DATABENCH_PATH}")
        exit(1)
        
    # Set up environment
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(DATABENCH_PATH) not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{DATABENCH_PATH}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = str(DATABENCH_PATH)
        
    # Get port from environment (for deployment platforms)
    port = int(os.environ.get('PORT', 5002))
    
    logger.info(f"Starting DataBench API server on port {port}")
    
    # Run with host 0.0.0.0 for external access
    app.run(host='0.0.0.0', port=port, debug=False)
```

## 🌍 Update Frontend to Use Remote Server

Update `databench.html` JavaScript:

```javascript
// Replace localhost with your deployed URL
const API_BASE_URL = 'https://your-app-name.railway.app'; // or your deployed URL

async callDataBenchAPI(formData) {
  const response = await fetch(`${API_BASE_URL}/api/evaluate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(formData)
  });
  // ... rest of the code
}
```

## ⚡ Quick Start (Railway Example)

1. **Update API for deployment**:
   ```bash
   # Add to databench_api.py
   port = int(os.environ.get('PORT', 5002))
   app.run(host='0.0.0.0', port=port, debug=False)
   ```

2. **Create Procfile**:
   ```
   web: python databench_api.py
   ```

3. **Deploy to Railway**:
   - Connect GitHub repo
   - Set `PYTHONPATH=/app/databench`
   - Deploy automatically

4. **Update frontend**:
   ```javascript
   const API_BASE_URL = 'https://your-app.railway.app';
   ```

## 💰 Cost Comparison

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| Railway | 500 hours/month | $5+/month | Hobby projects |
| Render | 750 hours/month | $7+/month | Small apps |
| Heroku | No free tier | $7+/month | Production |
| DigitalOcean | $200 credit | $5+/month | Scalable apps |
| Google Cloud | $300 credit | Pay per use | Enterprise |

## 🔒 Security Considerations

1. **Environment Variables**: Store sensitive data in env vars
2. **CORS**: Configure proper CORS origins
3. **Rate Limiting**: Add rate limiting for API endpoints
4. **HTTPS**: Most platforms provide HTTPS automatically

## 🐛 Troubleshooting

### Port Already in Use (Local)
```bash
# Kill process on port 5002
lsof -ti:5002 | xargs kill -9

# Or use different port
python databench_api.py --port 5003
```

### Memory Issues (Deployment)
- Use smaller model variants
- Implement model caching
- Add request timeouts

### Build Failures
- Check Python version compatibility
- Verify all requirements.txt files
- Check disk space limits

## 🎯 Recommended Setup

**For Development**: Railway (free, easy setup)
**For Production**: DigitalOcean or Google Cloud (more control)
**For Testing**: Local with different port

---

🤖 **Need help?** Contact yo@tunerobotics.xyz 