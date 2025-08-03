# 🤖 LeRobot Plug & Play System

## ✅ WORKING SOLUTION: Web App Running Terminal Commands!

This system demonstrates **how to safely run terminal commands from a web app** using the **Local Backend + Web Frontend** approach.

### 🏗️ Architecture

```
┌─────────────────┐    HTTP/WebSocket    ┌──────────────────┐
│   Web Frontend  │ ◄─────────────────► │  Flask Backend   │
│  (Browser UI)   │                      │ (Terminal Access)│
└─────────────────┘                      └──────────────────┘
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │ System Commands  │
                                         │ git, conda, pip  │
                                         │ USB detection    │
                                         └──────────────────┘
```

### 🚀 How to Use

1. **Start the System**:
   ```bash
   cd backend/plug_and_play
   python launcher.py
   ```

2. **System will automatically**:
   - ✅ Start Flask backend on port 5003
   - ✅ Open web UI in your browser
   - ✅ Handle port conflicts and process management
   - ✅ Show real-time backend logs

3. **In the Web UI**:
   - Select your robot (Koch, SO-100, SO-101, etc.)
   - Choose installation type:
     - **Fresh Installation**: Downloads LeRobot from GitHub
     - **Use Existing**: Browse to select existing LeRobot folder
   - Follow the 3-step process

### 🔒 Security Features

The system includes a **secure command execution layer** that:

- ✅ **Whitelists allowed commands** (git, conda, pip, python)
- ✅ **Validates command arguments** with regex patterns
- ✅ **Blocks dangerous operations** (sudo, rm -rf /, etc.)
- ✅ **Restricts file system access** to allowed directories
- ✅ **Prevents command injection** and path traversal

### 🛠️ Terminal Commands Executed

The web app can safely run these commands on your device:

1. **Git Operations**:
   ```bash
   git clone https://github.com/huggingface/lerobot.git ~/lerobot
   ```

2. **Conda Environment Management**:
   ```bash
   conda create -n lerobot python=3.10 -y
   conda run -n lerobot pip install -e .
   ```

3. **Package Installation**:
   ```bash
   pip install pyserial dynamixel-sdk
   ```

4. **USB Port Detection**:
   ```python
   import serial.tools.list_ports
   # Detects connected robotic arms
   ```

### 📁 File Structure

```
backend/plug_and_play/
├── launcher.py           # 🚀 Main launcher (USE THIS)
├── working_api.py        # 🔧 Flask backend with WebSocket
├── system_commands.py    # 🔒 Secure command execution
└── USAGE.md             # 📖 This file

frontend/pages/
└── plug-and-play-databench-style.html  # 🎨 Web UI
```

### 🎯 Key Benefits

1. **User-Friendly**: No terminal knowledge required
2. **Secure**: Commands are validated and sandboxed
3. **Real-time**: Live installation progress via WebSocket
4. **Cross-platform**: Works on macOS, Linux, Windows
5. **Error Handling**: Proper error messages and recovery

### 🔧 Troubleshooting

**If backend won't start:**
```bash
# Kill any processes on port 5003
lsof -ti:5003 | xargs kill -9

# Restart the system
python launcher.py
```

**If frontend doesn't open:**
- Manually open: `frontend/pages/plug-and-play-databench-style.html`
- Backend will still work at: `http://localhost:5003`

### 🎉 Success!

You now have a **working web app that can run terminal commands** on the user's device safely and efficiently! The system demonstrates the best practices for this architecture.

---

**Note**: This approach works because users explicitly run the local backend - they're choosing to grant the web app terminal access by running the launcher script.