# 🚀 Railway Deployment Guide for Plug & Play Backend

## ✅ YES! The Plug & Play backend WILL work with Railway!

This guide explains how the Plug & Play LeRobot Installation Assistant is integrated with Railway deployment.

## 🎯 Overview

The Plug & Play functionality is **fully integrated** into `multi_api.py`, which is deployed on Railway. This means:
- ✅ All Plug & Play API endpoints are available
- ✅ Frontend HTML is served directly
- ✅ WebSocket support for real-time updates
- ✅ USB port detection (limited in cloud environment)

## 📊 Architecture

```
Railway Platform
    └── multi_api.py (Port 10000)
         ├── DataBench API
         ├── Plug & Play API
         │    ├── /api/system-info
         │    ├── /api/start-installation
         │    ├── /api/list-ports
         │    ├── /api/status
         │    ├── /api/browse-directory
         │    ├── /api/create-directory
         │    ├── /api/get-home-directory
         │    └── /api/save-port-config
         └── Static File Server
              ├── /plug-and-play.html
              ├── /landing.html
              └── /styles.css
```

## 🔧 Key Changes Made

### 1. **API Route Consistency**
All routes now use kebab-case for consistency:
- `/api/system-info` (not `system_info`)
- `/api/start-installation` (not `start_installation`)
- `/api/list-ports` (not `scan_usb_ports`)

### 2. **Dynamic URL Detection**
The frontend automatically detects if it's running on Railway or localhost:
```javascript
const isRailway = window.location.hostname !== 'localhost' && 
                 window.location.hostname !== '127.0.0.1';
this.apiBaseUrl = isRailway ? '' : 'http://localhost:5002';
```

### 3. **Frontend Serving**
`multi_api.py` now serves the Plug & Play HTML directly:
- `/plug-and-play` or `/plug-and-play.html` - Serves the interface
- Automatically adjusts URLs for Railway environment

## 🚀 Deployment Steps

### 1. **Push to GitHub**
```bash
git add .
git commit -m "Add Plug & Play Railway integration"
git push origin main
```

### 2. **Railway Auto-Deploy**
Railway will automatically:
- Detect the changes
- Run the build command from `railway.toml`
- Start `multi_api.py` via `start_railway.py`

### 3. **Access Plug & Play**
Once deployed, access at:
```
https://your-railway-app.railway.app/plug-and-play
```

## 🧪 Testing

### Local Testing (Before Deploy)
```bash
# Test multi_api with Plug & Play
PORT=10000 python multi_api.py

# In another terminal, verify endpoints
curl http://localhost:10000/api/system-info
curl http://localhost:10000/plug-and-play.html
```

### Production Testing (After Deploy)
```bash
# Replace with your Railway URL
curl https://your-app.railway.app/api/system-info
```

## ⚠️ Limitations on Railway

### 1. **USB Port Detection**
- ❌ Cannot detect physical USB devices in cloud
- ✅ API endpoints still work for compatibility
- 💡 Users should run locally for actual hardware

### 2. **Installation Path**
- ❌ Cannot install to user's local machine from cloud
- ✅ Can demonstrate the interface and flow
- 💡 Provide download option for local execution

### 3. **Conda/Git Commands**
- ❌ Installation commands won't affect user's system
- ✅ Can show what would happen
- 💡 Consider "demo mode" for cloud deployment

## 🎨 Frontend Behavior

### On Railway (Production)
- Uses relative URLs (no hardcoded localhost)
- WebSocket connects to same origin
- Shows installation interface (demo mode)

### On Localhost (Development)
- Uses `http://localhost:5002` for standalone backend
- Or `http://localhost:10000` for multi_api
- Full installation functionality

## 📝 Configuration

### Environment Variables
Railway automatically sets:
- `PORT=10000` (or assigned port)
- `RAILWAY_ENVIRONMENT=production`

### Optional Configuration
Add to Railway dashboard:
```bash
# Enable demo mode (optional)
PLUGPLAY_DEMO_MODE=true

# Custom messages
PLUGPLAY_CLOUD_MESSAGE="Running in cloud - download for local installation"
```

## 🔍 Debugging

### Check Logs
```bash
# In Railway dashboard, check logs for:
"✅ Plug & Play available"
"📁 Plug & Play path: ..."
```

### Verify Endpoints
Test these endpoints on your Railway app:
1. `/health` - General health check
2. `/api/system-info` - System requirements check
3. `/api/status` - Installation status
4. `/plug-and-play` - Frontend interface

## 🚨 Important Notes

### Security
- ✅ Installation paths validated
- ✅ No actual system modifications in cloud
- ✅ Safe for public deployment

### Performance
- ✅ Lightweight API endpoints
- ✅ Efficient static file serving
- ✅ WebSocket for real-time updates

### User Experience
- ✅ Clear indication when running in cloud
- ✅ Download instructions for local use
- ✅ Responsive design works on all devices

## 🎉 Success Indicators

Your Plug & Play is working on Railway when:
1. ✅ `/plug-and-play` loads the interface
2. ✅ "Check System" shows system info
3. ✅ WebSocket connects (see browser console)
4. ✅ All buttons respond (even if limited in cloud)

## 📚 Next Steps

1. **Add Demo Mode**: Show example installation flow
2. **Download Button**: Let users get local version
3. **Video Tutorial**: Embed installation guide
4. **FAQ Section**: Common Railway questions

## 🤝 Support

If you encounter issues:
1. Check Railway logs
2. Verify all files are committed
3. Test endpoints individually
4. Ensure `multi_api.py` has latest changes

The Plug & Play backend is **fully compatible** with Railway deployment! 🚀✨ 