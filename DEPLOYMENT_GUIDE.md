# ChatVLA Deployment Guide

## Current Issue
Your website at https://tunerobotics.xyz is hosted on GitHub Pages (static hosting), but the DataBench functionality requires a backend API server. GitHub Pages can only serve static files, not run backend code, which is why you're getting HTTP 405 errors.

## Solution Architecture
You need a two-part deployment:
1. **Frontend** (GitHub Pages) - Already deployed at tunerobotics.xyz ✅
2. **Backend API** (Cloud Service) - Needs to be deployed ❌

## Quick Fix Instructions

### Step 1: Deploy Backend to Railway (Recommended)

1. **Create Railway Account**
   - Go to https://railway.app
   - Sign up with GitHub

2. **Deploy from GitHub**
   ```bash
   # In your terminal
   cd /Users/sofiia/chatVLA
   
   # Commit all recent changes
   git add .
   git commit -m "Fix deployment and add backend configuration"
   git push origin main
   ```

3. **Create New Project on Railway**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `chatVLA` repository
   - Railway will auto-detect the Dockerfile

4. **Set Environment Variables**
   - In Railway project settings, add:
   ```
   PORT=8080
   FLASK_ENV=production
   PYTHONPATH=/app:/app/backend
   ```

5. **Get Your Backend URL**
   - Once deployed, Railway will give you a URL like:
   - `https://chatvla-production-xxxx.up.railway.app`

### Step 2: Update Frontend Configuration

1. **Edit the config.js file**
   ```javascript
   // In /frontend/js/config.js, update line 26:
   return 'https://your-actual-railway-url.up.railway.app';
   ```
   Replace with your actual Railway URL from Step 1.5

2. **Commit and Push**
   ```bash
   git add frontend/js/config.js
   git commit -m "Update backend URL to Railway deployment"
   git push origin main
   ```

3. **Wait for GitHub Pages to Update**
   - Takes 1-2 minutes
   - Your site at tunerobotics.xyz will now connect to the backend

### Alternative: Deploy to Render

If Railway doesn't work, try Render:

1. **Create Render Account**
   - Go to https://render.com
   - Sign up with GitHub

2. **Create Web Service**
   - New → Web Service
   - Connect GitHub repository
   - Use these settings:
     - Build Command: `pip install -r requirements-minimal.txt`
     - Start Command: `gunicorn --bind 0.0.0.0:$PORT wsgi:app`

3. **Update config.js**
   - Replace the URL with your Render URL:
   - `https://chatvla.onrender.com`

## Verify Everything Works

1. **Check Backend Health**
   ```bash
   # Replace with your actual backend URL
   curl https://your-backend-url.up.railway.app/health
   ```
   Should return: `{"status":"healthy",...}`

2. **Test DataBench**
   - Go to https://tunerobotics.xyz/frontend/pages/databench.html
   - Open browser console (F12)
   - Should see: "✅ Backend is online and healthy"
   - Try running an evaluation

## Troubleshooting

### Still Getting 405 Error?
- Backend not deployed correctly
- Wrong URL in config.js
- Check Railway/Render logs for errors

### CORS Errors?
- Already configured in backend (`CORS(app, origins=["*"])`)
- Should work automatically

### Backend Crashes?
- Check logs in Railway/Render dashboard
- May need to use `Dockerfile.minimal` if memory issues
- Update railway.toml to use minimal Dockerfile:
  ```toml
  dockerfilePath = "./Dockerfile.minimal"
  ```

## File Structure
```
Your Deployment:
├── GitHub Pages (tunerobotics.xyz)
│   ├── index.html
│   ├── frontend/
│   │   ├── pages/databench.html
│   │   ├── js/config.js  ← UPDATE THIS WITH BACKEND URL
│   │   └── ...
│   └── (static files only)
│
└── Railway/Render (backend API)
    ├── backend/
    │   ├── api/main.py
    │   ├── databench/api.py
    │   └── ...
    ├── wsgi.py
    ├── simple_deploy.py
    └── requirements-minimal.txt
```

## Current Status
✅ Frontend deployed to GitHub Pages (tunerobotics.xyz)
✅ Backend code fixed and ready
✅ Configuration system in place
❌ Backend needs to be deployed to Railway/Render
❌ config.js needs to be updated with backend URL

## Next Steps
1. Deploy backend to Railway (5 minutes)
2. Update config.js with the Railway URL (1 minute)
3. Push changes to GitHub (1 minute)
4. Test DataBench functionality (2 minutes)

Total time: ~10 minutes to fix everything!

## Support
If you encounter issues:
1. Check Railway/Render deployment logs
2. Verify the backend URL is correct in config.js
3. Check browser console for error messages
4. Ensure all files are committed and pushed to GitHub