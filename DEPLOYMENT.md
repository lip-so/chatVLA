# 🚀 Deployment Guide - Tune Robotics with Plug & Play

This guide covers how to deploy the Tune Robotics website with the integrated Plug & Play LeRobot Installation Assistant.

## 📋 Deployment Overview

The Tune Robotics site consists of:
- **Static Frontend**: Deployed via GitHub Pages to `tunerobotics.xyz`
- **Plug & Play Backend**: Flask server that runs locally for installation functionality

## 🌐 GitHub Pages Deployment (Static Frontend)

### Current Setup
- ✅ **Domain**: `tunerobotics.xyz` (configured via CNAME)
- ✅ **Source**: GitHub Pages serves from `main` branch
- ✅ **Static Files**: All HTML, CSS, JS files are served automatically

### Files Deployed Statically
- `landing.html` - Main website
- `plug-and-play.html` - Plug & Play interface (frontend only)
- `manifesto.html` - Manifesto page
- `Vision.html` - Vision page
- `styles.css` - Global styles
- All other static assets

## 🔧 Plug & Play Backend Deployment

### Local Development & User Setup
The Plug & Play installation assistant requires a local Flask backend for full functionality:

```bash
# Users run this locally to use the full installation system
python start-plug-and-play.py
```

### Production Backend Options

#### Option 1: Hybrid Approach (Recommended)
- **Static Frontend**: Served from GitHub Pages (`tunerobotics.xyz/plug-and-play.html`)
- **Local Backend**: Users run the Flask server locally when they need installation functionality
- **Benefits**: Simple deployment, secure local installation, no server costs

#### Option 2: Cloud Backend Deployment
Deploy the Flask backend to a cloud service:

**Heroku Example:**
```bash
# Create Procfile in Plug-and-play/backend/
echo "web: python app.py" > Plug-and-play/backend/Procfile

# Deploy to Heroku
cd Plug-and-play/backend
git init
git add .
git commit -m "Initial backend deploy"
heroku create tune-robotics-backend
git push heroku main
```

**Railway/Render/DigitalOcean**: Similar process with their respective CLI tools

#### Option 3: Serverless Functions
Convert the Flask app to serverless functions (Vercel, Netlify Functions, AWS Lambda)

## 📁 File Structure for Deployment

```
tunerobotics.xyz/
├── index.html (redirects to landing.html)
├── landing.html (main site with Plug & Play button)
├── plug-and-play.html (installation interface)
├── manifesto.html
├── Vision.html
├── styles.css
├── start-plug-and-play.py (local launcher)
├── README-PLUG-AND-PLAY.md (user instructions)
└── Plug-and-play/ (backend + installation logic)
    ├── backend/app.py (Flask server)
    ├── installers/ (installation scripts)
    └── frontend/ (original standalone interface)
```

## 🎯 User Experience Flow

### Website Visit (Static)
1. User visits `tunerobotics.xyz`
2. Sees main landing page with "Plug & Play" button
3. Clicks "Plug & Play" → loads `plug-and-play.html`
4. Page displays with installation interface

### Full Installation Functionality (Local)
1. User downloads/clones repository
2. Runs `python start-plug-and-play.py`
3. Flask backend starts locally
4. Browser opens to local installation interface
5. Real-time installation progress works fully

## ⚙️ GitHub Pages Configuration

### Verify Settings
1. Go to GitHub repository settings
2. Navigate to "Pages" section
3. Ensure:
   - ✅ Source: "Deploy from a branch"
   - ✅ Branch: `main` 
   - ✅ Folder: `/ (root)`
   - ✅ Custom domain: `tunerobotics.xyz`

### DNS Configuration
Ensure your DNS provider points to GitHub Pages:
```
CNAME record: tunerobotics.xyz → lip-so.github.io
```

## 🔄 Continuous Deployment

Every push to `main` branch automatically:
1. ✅ Triggers GitHub Pages rebuild
2. ✅ Updates `tunerobotics.xyz` within minutes
3. ✅ Serves latest version of all static files

## 🧪 Testing Deployment

### Test Static Site
```bash
# Check if static site is live
curl -I https://tunerobotics.xyz
curl -I https://tunerobotics.xyz/plug-and-play.html
```

### Test Local Backend Integration
```bash
# Clone and test locally
git clone https://github.com/lip-so/chatVLA.git
cd chatVLA
python start-plug-and-play.py
# Verify backend responds and UI works
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] All files committed to `main` branch
- [ ] CNAME file contains correct domain
- [ ] GitHub Pages enabled in repository settings
- [ ] DNS records point to GitHub Pages

### Post-Deployment Verification
- [ ] `tunerobotics.xyz` loads correctly
- [ ] "Plug & Play" button visible on landing page
- [ ] `tunerobotics.xyz/plug-and-play.html` loads
- [ ] Local backend launcher works: `python start-plug-and-play.py`
- [ ] Mobile responsiveness verified

## 🚨 Important Notes

### Security Considerations
- ✅ **Local Installation**: Backend runs locally, so user controls security
- ✅ **No Server Costs**: Static hosting is free
- ✅ **No Data Collection**: Installation happens on user's machine

### Limitations
- ⚠️ **Backend Functionality**: Requires local setup for full installation features
- ⚠️ **Port Dependencies**: Flask backend needs available port (5000)
- ⚠️ **Python Required**: Users need Python 3.7+ for local backend

### User Instructions
Users should:
1. Visit the static site to learn about Plug & Play
2. Clone/download the repository for full installation functionality
3. Follow `README-PLUG-AND-PLAY.md` for setup instructions

## 🎉 Success!

Your Tune Robotics website is now deployed with integrated Plug & Play functionality:
- **Static Frontend**: Immediately available at `tunerobotics.xyz`
- **Installation System**: Ready for local use by developers and robotics enthusiasts
- **Scalable Architecture**: Can be enhanced with cloud backend if needed

The hybrid approach provides the best of both worlds: simple deployment and powerful local installation capabilities! 🤖✨ 