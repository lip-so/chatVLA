# 🤖 Tune Robotics - Plug & Play Installation

The **Plug & Play Installation Assistant** is now integrated into the Tune Robotics website! This powerful tool makes it incredibly easy to install LeRobot with automatic USB port detection for your robotic arms.

## 🚀 Quick Start

### Option 1: Easy Launcher (Recommended)
```bash
python start-plug-and-play.py
```

This will automatically:
- ✅ Check your system requirements
- 📦 Install necessary dependencies  
- 🚀 Start the backend server
- 🌐 Open the web interface in your browser

### Option 2: Manual Setup
If you prefer to run things manually:

1. **Install backend dependencies:**
   ```bash
   pip install -r Plug-and-play/backend/requirements.txt
   ```

2. **Start the backend server:**
   ```bash
   cd Plug-and-play/backend
   python app.py
   ```

3. **Open the web interface:**
   - Open `plug-and-play.html` in your browser, or
   - Visit the main website and click "Plug & Play"

## 🌟 Features

- **🎯 Zero Configuration**: Just click start and everything is handled automatically
- **🔌 Smart USB Detection**: Automatically finds and configures your robotic arms
- **📊 Real-time Progress**: Watch the installation happen with live updates
- **🛡️ Error Recovery**: Smart error handling with helpful guidance
- **🌐 Modern Interface**: Beautiful web UI that matches the Tune Robotics design

## 📋 What Gets Installed

The assistant automatically handles:

1. **✅ Prerequisites Check** - Verifies Git and Conda are available
2. **📥 Repository Cloning** - Downloads LeRobot from HuggingFace  
3. **🏗️ Environment Setup** - Creates `conda create -n lerobot python=3.10`
4. **🎬 FFmpeg Installation** - Adds video processing capabilities
5. **🤖 LeRobot Installation** - Installs with `pip install -e .`
6. **🔌 USB Port Detection** - Automatically finds and configures robotic arms
7. **⚙️ Configuration** - Creates ready-to-use configuration files

## 🔧 System Requirements

**Required (checked automatically):**
- Python 3.7+
- Git
- Conda/Miniconda

**Don't have them?** The installer provides helpful links and guidance!

## 🔌 USB Port Detection

The system automatically:
- 🔍 Scans for USB serial ports
- 🤖 Identifies robotic arm connections  
- ⚙️ Creates configuration files
- 📝 Provides ready-to-use code

### Generated Configuration
After installation, you'll have a `lerobot_ports.py` file:
```python
LEADER_ARM_PORT = "/dev/cu.usbmodem14201"  
FOLLOWER_ARM_PORT = "/dev/cu.usbmodem14301"

# Ready to use in your code:
from lerobot_ports import LEADER_ARM_PORT, FOLLOWER_ARM_PORT
import serial

leader = serial.Serial(LEADER_ARM_PORT, baudrate=9600)
follower = serial.Serial(FOLLOWER_ARM_PORT, baudrate=9600)
```

## 🎯 After Installation  

### Activate Environment
```bash
conda activate lerobot
```

### Start Using Your Robots
```python
from lerobot_ports import LEADER_ARM_PORT, FOLLOWER_ARM_PORT
import serial

# Connect to arms (auto-configured!)
leader = serial.Serial(LEADER_ARM_PORT, baudrate=9600) 
follower = serial.Serial(FOLLOWER_ARM_PORT, baudrate=9600)

print("🤖 Robotic arms ready to use with Tune!")
```

## 🌐 Integration with Main Website

The Plug & Play system is fully integrated into the Tune Robotics website:

- **Navigation**: Access via "Plug & Play" button on the landing page
- **Design**: Matches the main website's dark theme and typography
- **Mobile-Friendly**: Responsive design that works on all devices
- **Seamless Experience**: Feels like a natural part of the website

## 🔧 Troubleshooting

### Common Issues

**❌ "Backend server not responding"**
- Make sure you ran `python start-plug-and-play.py`
- Check that port 5000 is available
- Try restarting the launcher script

**❌ "Conda not found"**
- Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Restart your terminal after installation

**❌ "No USB ports detected"**
- Connect your robotic arms via USB
- Try different USB cables
- Ensure drivers are installed for your robotic arms

## 🏗️ Architecture

The system consists of:

1. **Frontend**: `plug-and-play.html` - Integrated web interface
2. **Backend**: `Plug-and-play/backend/app.py` - Flask server with WebSocket
3. **Installer**: `Plug-and-play/installers/` - Core installation logic
4. **Utils**: `Plug-and-play/utils/` - USB detection and configuration tools

## 📞 Support

- 📖 **Full Documentation**: Check `Plug-and-play/README.md`
- 🔌 **Port Detection Guide**: See `Plug-and-play/docs/USAGE_PORT_DETECTION.md`
- 🤖 **LeRobot Issues**: https://github.com/huggingface/lerobot
- 💌 **Contact**: yo@tunerobotics.xyz

---

## 🎉 Success!

Once installation completes, your robotic arms are ready to use with Tune Robotics! The installer creates everything you need to start building amazing robotic applications.

**Now you can chat with your robot and make it do your dishes! 🤖✨** 