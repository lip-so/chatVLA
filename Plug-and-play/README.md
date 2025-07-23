# 🤖 LeRobot Installation Assistant

**The easiest way to install LeRobot with automatic USB port detection!**

Perfect for users without technical background - just run one command and get started with robotic arms.

---

## ✨ Features

- 🚀 **One-click installation** - No complex setup required
- 🔌 **Automatic USB port detection** - Finds your robotic arms automatically  
- 🌐 **Modern web interface** - User-friendly browser-based installer
- 🖥️ **Desktop GUI option** - Native application interface
- 🔄 **Real-time progress** - Watch installation happen live
- 🛠️ **Smart error handling** - Clear guidance when things go wrong
- 🌍 **Cross-platform** - Works on Windows, macOS, and Linux

---

## 🚀 Quick Start

### Option 1: Easy Launcher (Recommended)
```bash
python RUN_ME.py
```

This automatically:
1. ✅ Checks your system compatibility  
2. 🚀 Launches the best available installer
3. 📦 Installs LeRobot with all dependencies
4. 🔌 Detects and configures USB ports
5. 🎉 Gets you ready to use robotic arms!

### Option 2: Web Interface
```bash
python installers/fixed_run.py
```
Modern browser-based interface with real-time progress tracking.

### Option 3: Desktop Application  
```bash
python installers/lerobot_installer.py
```
Native GUI application with progress bars and detailed logging.

---

## 📋 Prerequisites

**Required (will be checked automatically):**
- Python 3.7+ 
- Git
- Conda/Miniconda

**Don't have them?** The installer will guide you through setup!

### Quick Installation Links:
- **Git**: https://git-scm.com/downloads
- **Miniconda**: https://docs.conda.io/en/latest/miniconda.html

---

## 🔌 USB Port Detection

### Automatic Detection
The installer automatically:
- 🔍 Scans for USB serial ports
- 🤖 Identifies robotic arm connections  
- ⚙️ Creates configuration files
- 📝 Provides ready-to-use code

### Manual Configuration
If you need to adjust ports later:
```bash
conda activate lerobot
python lerobot/find_port.py
```

### Generated Configuration
Creates `lerobot_ports.py`:
```python
LEADER_ARM_PORT = "/dev/cu.usbmodem14201"  
FOLLOWER_ARM_PORT = "/dev/cu.usbmodem14301"

# Use in your code:
from lerobot_ports import LEADER_ARM_PORT, FOLLOWER_ARM_PORT
import serial

leader = serial.Serial(LEADER_ARM_PORT, baudrate=9600)
follower = serial.Serial(FOLLOWER_ARM_PORT, baudrate=9600)
```

---

## 📁 Repository Structure

```
📦 LeRobot Installation Assistant
├── 🚀 RUN_ME.py              # Main launcher (start here!)
├── 📚 README.md              # This file
├── 📁 installers/            # Installation engines
│   ├── fixed_run.py          # Web interface launcher
│   ├── lerobot_installer.py  # Desktop GUI installer  
│   ├── installation_service.py # Core installation logic
│   └── error_handler.py      # Error handling utilities
├── 📁 frontend/              # Web interface files
│   ├── index.html           # Main web page
│   ├── styles.css           # Modern styling
│   ├── script.js            # Interactive features
│   └── port-detection.*     # USB port detection interface
├── 📁 backend/               # Web server
│   └── app.py               # Flask API server
├── 📁 utils/                 # Utilities  
│   └── lerobot/
│       └── find_port.py     # Interactive port detection
└── 📁 docs/                  # Documentation
    └── USAGE_PORT_DETECTION.md # Port detection guide
```

---

## 🎯 What Gets Installed

The installer automatically:

1. **Checks Prerequisites** - Verifies Git and Conda are available
2. **Clones Repository** - Downloads LeRobot from HuggingFace  
3. **Creates Environment** - Sets up `conda create -n lerobot python=3.10`
4. **Installs FFmpeg** - Adds video processing capabilities
5. **Installs LeRobot** - Installs in development mode with `pip install -e .`
6. **Detects USB Ports** - Automatically finds and configures robotic arms
7. **Creates Tools** - Copies utilities for future port detection

---

## 📖 After Installation  

### Activate Environment
```bash
conda activate lerobot
```

### Use Your Robotic Arms
```python
from lerobot_ports import LEADER_ARM_PORT, FOLLOWER_ARM_PORT
import serial

# Connect to arms (auto-configured!)
leader = serial.Serial(LEADER_ARM_PORT, baudrate=9600) 
follower = serial.Serial(FOLLOWER_ARM_PORT, baudrate=9600)

print("🤖 Robotic arms ready!")
```

### Run Interactive Port Detection
```bash
python lerobot/find_port.py
```

---

## 🔧 Troubleshooting

### Common Issues

**❌ "Conda not found"**
```bash
# Install Miniconda first:
# https://docs.conda.io/en/latest/miniconda.html
```

**❌ "Git not found"**  
```bash
# Install Git first:
# https://git-scm.com/downloads
```

**❌ No USB ports detected**
- Connect your robotic arms via USB
- Try different USB cables
- Run: `python utils/lerobot/find_port.py`

**❌ Installation fails**
- Check internet connection
- Ensure sufficient disk space (2GB+)  
- Run installer again (it resumes automatically)

### Manual Installation
If automatic installation fails:

```bash
# 1. Create environment
conda create -y -n lerobot python=3.10

# 2. Clone repository  
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 3. Activate and install
conda activate lerobot
conda install -y ffmpeg -c conda-forge
pip install -e .
pip install pyserial

# 4. Configure ports
python lerobot/find_port.py
```

---

## 🌟 Why Choose This Installer?

✅ **Beginner-Friendly** - No technical knowledge required  
✅ **Complete Solution** - Handles everything automatically  
✅ **Smart Detection** - Finds your robotic arms automatically  
✅ **Multiple Interfaces** - Web, desktop, and command line options  
✅ **Robust Error Handling** - Clear guidance when issues arise  
✅ **Cross-Platform** - Works everywhere Python runs  
✅ **Well-Documented** - Comprehensive guides and examples  

---

## 🤝 Support

- 📖 **Documentation**: Check the `docs/` folder
- 🔌 **Port Detection**: See `docs/USAGE_PORT_DETECTION.md` 
- 🐛 **LeRobot Issues**: https://github.com/huggingface/lerobot
- 💡 **General Help**: Try different installer options if one fails

---

## 🎉 Success!

**Your robotic arms are ready to use!**

The installer has:
- ✅ Installed LeRobot with all dependencies
- ✅ Configured your robotic arm USB ports  
- ✅ Created ready-to-use configuration files
- ✅ Set up tools for future port management

**Now start building amazing robotic applications! 🤖✨**