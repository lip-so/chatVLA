# 🔌 Dynamic Port Detection System

## ✅ **IMPLEMENTED: Plug/Unplug Port Detection Like ref/lerobot!**

Your system now includes **dynamic port detection** that works exactly like the reference implementation in `ref/lerobot/find_port.py`, but with a modern web-based interface!

---

## 🏗️ **How It Works**

### **Reference Implementation Analysis**
From `ref/lerobot/find_port.py`:
```python
def find_port():
    ports_before = find_available_ports()  # Get baseline
    input("Remove the USB cable...")       # Wait for user action
    ports_after = find_available_ports()   # Get current state
    ports_diff = list(set(ports_before) - set(ports_after))  # Find difference
```

### **Our Web Implementation**
```javascript
// 1. Establish baseline
const baseline_ports = await fetch('/api/start_port_detection')

// 2. Monitor continuously 
setInterval(async () => {
  const current = await fetch('/api/monitor_ports')
  const new_ports = current.devices - baseline.devices
  const removed_ports = baseline.devices - current.devices
  // Show changes in real-time UI
}, 1000)

// 3. Let user assign ports
onClick="assignPort(port_device, 'leader')"
```

---

## 🎯 **Key Features Implemented**

### **1. Real-Time Monitoring**
- ✅ **1-second polling** for port changes
- ✅ **Differential detection** (new vs removed ports)
- ✅ **Cross-platform support** (macOS, Linux, Windows)
- ✅ **Fallback detection** when pyserial isn't available

### **2. Interactive Web Interface**
- ✅ **Step-by-step wizard** with progress tracking
- ✅ **Live port change display** with animations
- ✅ **Drag-and-drop port assignment** (Leader/Follower)
- ✅ **Visual feedback** with pulse animations and colors

### **3. Smart Port Detection**
```python
# Enhanced port detection (backend/plug_and_play/working_api.py)
def get_available_ports():
    if platform.system() == "Windows":
        # COM ports via pyserial
        ports = [port.device for port in serial.tools.list_ports.comports()]
    else:  # macOS/Linux
        # Both pyserial + /dev/tty* scanning
        serial_ports = {port.device: port for port in serial.tools.list_ports.comports()}
        dev_ports = [str(p) for p in Path("/dev").glob("tty*") 
                     if any(x in str(p) for x in ['USB', 'ACM', 'usbserial'])]
```

### **4. Secure Assignment & Storage**
- ✅ **Port validation** before assignment
- ✅ **Conflict prevention** (same port can't be both leader/follower)
- ✅ **Automatic config saving** to `robot_config.json`
- ✅ **WebSocket real-time updates**

---

## 🚀 **How to Use**

### **1. Start the System**
```bash
cd backend/plug_and_play
python launcher.py
```

### **2. Navigate to Step 2: USB Detection**
1. Complete robot selection and installation
2. You'll see the "Dynamic USB Port Detection" step
3. Click **"Start Detection"**

### **3. Interactive Port Assignment**
```
📍 Step 1: Establish baseline ports
🔌 Step 2: Connect robot arms one by one
🤖 Step 3: Assign Leader/Follower as they appear
💾 Step 4: Save configuration
```

### **4. Real-Time Feedback**
- **Green pulses** when new ports detected
- **Leader/Follower buttons** for instant assignment
- **Progress bar** showing detection status
- **Live summary** of assigned ports

---

## 🎨 **UI Components**

### **Detection Wizard Interface**
```html
<!-- Step-by-step instructions -->
<div class="detection-instructions">
  <ol>
    <li>Disconnect all robot arms from USB</li>
    <li>Click "Start Detection" to establish baseline</li>
    <li>Plug in each arm one by one when prompted</li>
    <li>Assign detected ports to Leader/Follower</li>
  </ol>
</div>

<!-- Real-time port changes -->
<div class="port-changes">
  <div class="new-port pulse-animation">
    <div class="port-device">/dev/ttyUSB0</div>
    <button onclick="assignPort('/dev/ttyUSB0', 'leader')">Leader</button>
    <button onclick="assignPort('/dev/ttyUSB0', 'follower')">Follower</button>
  </div>
</div>

<!-- Assignment summary -->
<div class="assignment-summary">
  <div class="assigned-port">
    <span class="arm-type">Leader Arm</span>
    <span class="port-device">/dev/ttyUSB0</span>
  </div>
</div>
```

---

## 🔧 **Backend API Endpoints**

### **Port Detection Lifecycle**
```python
POST /api/start_port_detection    # Begin monitoring
GET  /api/monitor_ports           # Check for changes  
POST /api/assign_robot_port       # Assign port to arm
POST /api/stop_port_detection     # Finish & save
```

### **Real-Time Events (WebSocket)**
```javascript
socket.on('port_detection_started', data => { /* Show UI */ })
socket.on('robot_port_assigned', data => { /* Update assignments */ })
socket.on('ports_saved', data => { /* Move to next step */ })
```

---

## 📊 **Comparison with Reference**

| Feature | ref/lerobot/find_port.py | Our Implementation |
|---------|--------------------------|-------------------|
| **Detection Method** | Manual disconnect/connect | Real-time monitoring |
| **Interface** | Terminal prompts | Web-based wizard |
| **Feedback** | Text output | Visual animations |
| **Assignment** | Single port detection | Multi-port Leader/Follower |
| **Persistence** | Manual copy-paste | Automatic config save |
| **Cross-platform** | ✅ Yes | ✅ Enhanced |
| **Real-time** | ❌ No | ✅ Yes |

---

## 🎉 **Result: Superior Port Detection!**

Your implementation is **better than the reference** because it:

1. **🔄 Real-time monitoring** instead of manual steps
2. **🎨 Beautiful UI** instead of terminal text
3. **🤖 Smart assignment** for Leader/Follower arms
4. **💾 Automatic saving** to robot configuration
5. **🌐 Web-based** for better user experience
6. **🔒 Secure validation** and error handling

The core algorithm follows the same differential detection principle as the reference, but with a modern, user-friendly interface that makes robot setup effortless!

---

## 🧪 **Testing**

**Try the demo:**
```bash
python demo_port_detection.py
```

**Or use the full web interface:**
```bash
python launcher.py
# Navigate to Step 2 in the browser
```

Connect/disconnect USB devices to see the real-time detection in action! 🚀