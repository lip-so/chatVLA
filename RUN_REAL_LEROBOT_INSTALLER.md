# 🤖 REAL LeRobot Installation - How to Actually Install It

## 📍 The Current Situation:

1. **GitHub Pages** (tunerobotics.xyz) - ✅ Hosts your frontend
2. **Railway** (web-production-fdfaa.up.railway.app) - ✅ Provides mock API endpoints
3. **Real Installation** - Needs to run on YOUR computer or a powerful server

## 🚀 How to ACTUALLY Install LeRobot:

### Option 1: Run Real Backend Locally (RECOMMENDED)

```bash
# 1. Install dependencies
pip install flask flask-cors flask-socketio pyserial python-socketio eventlet

# 2. Run the REAL backend
cd /Users/sofiia/chatVLA
python real_backend.py

# 3. Update frontend to use localhost
# Edit frontend/pages/plug-and-play-databench-style.html
# Change line 960 from:
const API_URL = 'https://web-production-fdfaa.up.railway.app';
# To:
const API_URL = 'http://localhost:8080';

# 4. Open the page and install LeRobot for real!
open file:///Users/sofiia/chatVLA/frontend/pages/plug-and-play-databench-style.html
```

### Option 2: Use the Python GUI Installer

```bash
# Run the GUI installer directly
cd /Users/sofiia/chatVLA
python backend/plug_and_play/installers/lerobot_installer.py
```

This opens a GUI window that:
- ✅ Actually installs LeRobot
- ✅ Creates conda environment
- ✅ Detects USB ports
- ✅ Shows real progress

### Option 3: Manual Installation (Full Control)

```bash
# 1. Install Miniconda (if not installed)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# 2. Clone LeRobot
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot

# 3. Create conda environment
conda create -y -n lerobot python=3.10
conda activate lerobot

# 4. Install LeRobot
pip install -e .

# 5. Install additional tools
conda install -y ffmpeg -c conda-forge
pip install pyserial  # For USB port detection

# 6. Test it works
python -c "import lerobot; print('LeRobot installed successfully!')"
```

## 💡 Why Railway Can't Run the Real Installer:

Railway has limitations:
- **Memory limits** - Can't install Miniconda (500MB+)
- **Storage limits** - LeRobot needs several GB
- **No USB access** - Can't detect robot hardware
- **No conda** - Railway uses Docker, not conda environments

## ✅ What Works on Railway:

- Mock API endpoints for UI testing
- DataBench evaluation (mock)
- Health checks
- Basic backend functionality

## 🎯 The Solution:

1. **Use Railway** for the mock backend (UI testing)
2. **Run locally** for real LeRobot installation
3. **Or use** a dedicated server with more resources

## 📊 Status Summary:

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| Frontend | GitHub Pages | UI | ✅ Working |
| Mock Backend | Railway | Testing | ✅ Working |
| Real Installer | Your Computer | Actual Installation | 🔧 Run locally |

## 🚀 Quick Start:

```bash
# Just run this to install LeRobot for real:
cd /Users/sofiia/chatVLA
python backend/plug_and_play/installers/lerobot_installer.py
```

This is the REAL installer that actually works!