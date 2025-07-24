# 🤖 Plug & Play Integration Guide

**Plug & Play** is now fully integrated into the Tune Robotics website, providing automated LeRobot installation with real-time USB port detection and configuration.

## 🚀 Quick Start

### 1. Start Plug & Play Backend
```bash
# In your project root
python start-plug-and-play.py
```

The backend will start on `http://localhost:5000` with full installation capabilities.

### 2. Open Plug & Play Web Interface
Visit `tunerobotics.xyz/plug-and-play.html` or open `plug-and-play.html` locally.

The interface automatically connects to your local backend for real installation functionality.

### 3. Install LeRobot
1. **Choose Installation Directory**: Default paths provided for your OS
2. **Check System Status**: Automatic Git and Conda detection  
3. **Click "Start Installation"**: Automated LeRobot setup with real-time progress
4. **USB Detection**: Automatic detection and configuration of robotic arms
5. **Ready to Use**: Complete environment setup with `conda activate lerobot`

## 🔧 Full Functionality (Local Setup)

When you run `python start-plug-and-play.py`, you get:

✅ **Real Installation**: Complete LeRobot setup with all dependencies  
✅ **USB Port Detection**: Automatic robotic arm detection and configuration  
✅ **System Checks**: Git, Conda, Python compatibility verification  
✅ **Real-time Progress**: WebSocket-powered live installation tracking  
✅ **Error Handling**: Robust error detection and recovery  
✅ **Cross-platform**: Windows, macOS, and Linux support  

## 🛠️ What Plug & Play Does

### Installation Pipeline

1. **System Prerequisites Check**
   - Git installation and version
   - Conda/Miniconda availability  
   - Python version compatibility
   - Disk space verification

2. **Repository Setup**
   - Clone LeRobot from HuggingFace
   - Create dedicated conda environment
   - Install FFmpeg for video processing

3. **Dependency Installation** 
   - Install LeRobot framework
   - Configure PyTorch and ML dependencies
   - Set up robotic arm drivers

4. **Hardware Configuration**
   - Scan for connected USB devices
   - Detect robotic arms (Arduino, etc.)
   - Configure leader/follower arm pairs
   - Test communication protocols

5. **Environment Activation**
   - Verify installation integrity  
   - Generate usage instructions
   - Ready for immediate use

### USB Port Detection Features

| Feature | Description |
|---------|-------------|
| **Auto-Discovery** | Scans all USB ports for robotic hardware |
| **Device Identification** | Recognizes Arduino, servo controllers, sensors |
| **Smart Configuration** | Automatically configures leader/follower pairs |
| **Real-time Updates** | Live port scanning with connect/disconnect detection |
| **Cross-platform** | Works on Windows, macOS, and Linux |

## 📋 Setup Instructions

### Prerequisites
```bash
# Install backend dependencies
cd Plug-and-play/backend
pip install -r requirements.txt

# Required packages: flask, flask-socketio, flask-cors, pyserial
```

### Backend Configuration
The backend (`Plug-and-play/backend/app.py`) provides:
- **Flask API**: RESTful endpoints for installation control
- **WebSocket Support**: Real-time progress updates via Socket.IO
- **USB Detection**: Serial port scanning with device identification
- **Installation Management**: Threaded installation with progress tracking

### Environment Setup
```bash
# The backend automatically sets up paths and environment
# No manual configuration needed

# Optional: Set custom installation directory
export LEROBOT_INSTALL_PATH="/custom/path/lerobot"
```

## 🔧 API Endpoints

### System Information
```bash
curl http://localhost:5000/api/system_info
```

### Start Installation  
```bash
curl -X POST http://localhost:5000/api/start_installation \
  -H "Content-Type: application/json" \
  -d '{"installation_path": "/Users/username/lerobot"}'
```

### USB Port Scanning
```bash
curl http://localhost:5000/api/scan_usb_ports
```

### Cancel Installation
```bash
curl -X POST http://localhost:5000/api/cancel_installation
```

## 🎯 Integration Details

### Frontend Features
- **Smart Backend Detection**: Automatically detects local backend availability
- **Real-time Progress**: WebSocket-powered live installation tracking  
- **USB Management**: Live port detection with device information
- **Error Recovery**: Helpful error messages and recovery suggestions
- **Cross-platform UI**: Responsive design for all operating systems

### Backend Capabilities
- **Threaded Installation**: Non-blocking installation with progress callbacks
- **System Validation**: Comprehensive prerequisite checking
- **Hardware Detection**: Advanced USB device discovery and classification
- **Installation Recovery**: Automatic cleanup and retry mechanisms  
- **Logging System**: Detailed installation logs with error tracking

## 🐛 Troubleshooting

### Backend Connection Issues
```bash
# Check if backend is running
curl http://localhost:5000/api/status

# Start backend manually
cd Plug-and-play/backend
python app.py

# Or use the launcher
python start-plug-and-play.py
```

### Installation Failures
- **Git Issues**: Ensure Git is installed and accessible
- **Conda Issues**: Install Miniconda/Anaconda
- **Permission Issues**: Run with appropriate user permissions  
- **Network Issues**: Check internet connection for downloads
- **Disk Space**: Ensure sufficient space (>5GB recommended)

### USB Detection Issues
```bash
# Install serial port dependencies
pip install pyserial

# Check port permissions (Linux/macOS)
sudo usermod -a -G dialout $USER  # Linux
sudo dscl . append /Groups/_developer GroupMembership $USER  # macOS

# Windows: Install USB driver for your robotic arm
```

## 📈 Performance Tips

1. **Fast Installation**: Use SSD for installation directory
2. **Network Speed**: Use stable, fast internet for downloads
3. **USB Reliability**: Use high-quality USB cables for robotic arms
4. **System Resources**: Close unnecessary applications during installation
5. **Conda Channels**: Pre-configure conda channels for faster package resolution

## 🤝 Usage Examples

### Basic Installation
```bash
# Start backend
python start-plug-and-play.py

# Open browser to localhost:5000 or tunerobotics.xyz/plug-and-play.html
# Click "Start Installation" with default path
# Wait for completion and USB detection
```

### Custom Installation
```bash
# Set custom path in web interface
# Monitor real-time progress in log section
# Configure detected robotic arms
# Activate environment: conda activate lerobot
```

### USB Port Management
```bash
# Connect robotic arms via USB
# Click "Scan Ports" in web interface  
# Review detected devices
# Automatic configuration applied
```

## 🔄 Architecture

```
Website (plug-and-play.html)
    ↓ WebSocket + REST API
Backend (Plug-and-play/backend/app.py)
    ↓ System Commands
LeRobot Installation Pipeline
    ↓ USB Communication  
Robotic Hardware (Arduino, Servos, etc.)
```

## 📁 File Structure

```
Plug-and-play/
├── backend/
│   ├── app.py              # Main Flask backend
│   └── requirements.txt    # Backend dependencies
├── frontend/
│   ├── index.html          # Original frontend (reference)
│   ├── script.js           # Frontend JavaScript
│   └── styles.css          # Frontend styling
├── installers/             # Installation logic
├── utils/                  # Utility functions
└── RUN_ME.py              # Quick start script

# Website Integration
plug-and-play.html          # Integrated web interface
start-plug-and-play.py      # Backend launcher
```

## 🚀 Next Steps After Installation

1. **Activate Environment**: `conda activate lerobot`
2. **Test Installation**: `python -c "import lerobot; print('✅ LeRobot ready!')"`
3. **Connect Hardware**: Follow detected USB port configuration
4. **Run Examples**: Try LeRobot tutorials with your robotic arms
5. **Develop**: Start building your robotic applications

---

**Need Help?** Contact: yo@tunerobotics.xyz  
**Full Documentation**: See `Plug-and-play/README.md`  
**LeRobot Docs**: https://github.com/huggingface/lerobot 