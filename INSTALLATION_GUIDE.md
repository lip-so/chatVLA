# LeRobot Installation Guide - Plug & Play System

## 🎯 Understanding the Architecture

The Tune Robotics Plug & Play system has **TWO components** that work together:

### 1. Production Website (tunerobotics.xyz)
- ✅ Provides the user interface
- ✅ Guides you through installation
- ✅ Shows progress and instructions
- ❌ **CANNOT install software on your computer** (browser security restriction)
- ❌ **CANNOT access your file system**

### 2. Local Installer Bridge (your computer)
- ✅ **PERFORMS the actual installation**
- ✅ Downloads LeRobot from GitHub
- ✅ Creates conda environment
- ✅ Installs all dependencies
- ✅ Has full access to your file system

## 🚨 Important: Why Production Seems "Inactive"

**The production website is ACTIVE and working correctly!**

However, it can only:
- Show you the interface
- Guide you through steps
- **SIMULATE** the installation process

It CANNOT actually install LeRobot on your machine because:
- Web browsers prevent websites from installing software (security feature)
- The production server runs on Railway's cloud, not your computer

## 📦 How to Install LeRobot

### Step 1: Get the Repository
```bash
git clone https://github.com/lip-so/chatVLA.git
cd chatVLA
```

### Step 2: Start the Local Installer
```bash
python3 local_installer_bridge.py
```

You'll see:
```
============================================================
🚀 TUNE ROBOTICS LOCAL INSTALLER BRIDGE
============================================================
This enables REAL installation of LeRobot on your machine.
The installer is now running at: http://localhost:7777
```

### Step 3: Visit the Website
Go to: https://tunerobotics.xyz/frontend/pages/plug-and-play-databench-style.html

The website will:
1. Detect your local installer automatically
2. Show a green banner: "✅ Connected to local installer"
3. Allow you to perform REAL installation

### Step 4: Complete Installation
1. Select your robot (Koch, SO-100, SO-101)
2. Choose installation path (default: ~/lerobot)
3. Click "Start Installation"
4. Watch real-time progress
5. Configure USB ports

## 🔍 How to Verify It's Working

### Check if Local Installer is Running:
```bash
curl http://localhost:7777/health
```

Should return:
```json
{
    "status": "online",
    "local": true,
    "can_install": true
}
```

### Check Production Website Status:
```bash
curl https://web-production-fdfaa.up.railway.app/health
```

Should return:
```json
{
    "status": "healthy",
    "services": {
        "plugplay": {
            "available": true
        }
    }
}
```

## ❓ FAQ

### Q: Why can't the website install directly?
**A:** Browser security prevents websites from:
- Installing software
- Accessing your file system
- Running system commands

This is a security feature, not a bug!

### Q: Is the production site broken?
**A:** No! It's working correctly. It provides:
- User interface
- Installation wizard
- Progress tracking

But it needs the local installer to actually install LeRobot.

### Q: What if I don't see the green banner?
**A:** The website shows an orange banner with instructions:
1. Make sure you've cloned the repository
2. Run `python3 local_installer_bridge.py`
3. Refresh the page

### Q: Can I install without the local bridge?
**A:** No. The local installer bridge is required because:
- Only local code can install software on your machine
- Web browsers cannot perform system installations
- This is by design for security

## 🛠️ Troubleshooting

### Local Installer Won't Start
```bash
# Check Python version (needs 3.8+)
python3 --version

# Install requirements
pip install flask flask-cors flask-socketio

# Try again
python3 local_installer_bridge.py
```

### Website Doesn't Detect Local Installer
1. Check installer is running on port 7777
2. Check firewall isn't blocking localhost:7777
3. Try accessing http://localhost:7777/health directly
4. Refresh the website page

### Installation Fails
Requirements for LeRobot:
- Git installed
- Conda/Miniconda installed
- Python 3.10+
- ~2GB disk space

## 📞 Support

Email: yo@tunerobotics.xyz

---

Remember: The production website provides the UI, but the local installer performs the actual installation. This is intentional and secure!
