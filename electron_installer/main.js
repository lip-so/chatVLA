const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const https = require('https');
const os = require('os');

let mainWindow;
let installerProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    icon: path.join(__dirname, 'icon.png'),
    autoHideMenuBar: true
  });

  mainWindow.loadFile('index.html');
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Handle installation
ipcMain.handle('start-installation', async () => {
  try {
    const homeDir = os.homedir();
    const installDir = path.join(homeDir, 'TuneRobotics');
    
    // Create installation directory
    if (!fs.existsSync(installDir)) {
      fs.mkdirSync(installDir, { recursive: true });
    }
    
    // Download chatVLA repository
    mainWindow.webContents.send('installation-progress', {
      step: 'Downloading LeRobot installer...',
      progress: 20
    });
    
    // Clone repository using git
    await runCommand('git', ['clone', 'https://github.com/lip-so/chatVLA.git', path.join(installDir, 'chatVLA')]);
    
    mainWindow.webContents.send('installation-progress', {
      step: 'Installing Python dependencies...',
      progress: 40
    });
    
    // Install Python dependencies
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    await runCommand(pythonCmd, ['-m', 'pip', 'install', 'flask', 'flask-cors', 'flask-socketio', 'pyserial', 'eventlet']);
    
    mainWindow.webContents.send('installation-progress', {
      step: 'Starting installation bridge...',
      progress: 60
    });
    
    // Start the local installer bridge
    const bridgePath = path.join(installDir, 'chatVLA', 'local_installer_bridge.py');
    installerProcess = spawn(pythonCmd, [bridgePath], {
      cwd: path.join(installDir, 'chatVLA')
    });
    
    // Wait for bridge to start
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    mainWindow.webContents.send('installation-progress', {
      step: 'Opening web installer...',
      progress: 80
    });
    
    // Load the Tune Robotics website
    mainWindow.loadURL('https://tunerobotics.xyz/frontend/pages/plug-and-play-databench-style.html');
    
    mainWindow.webContents.send('installation-progress', {
      step: 'Ready! Click "Install LeRobot" on the website.',
      progress: 100
    });
    
    return { success: true };
    
  } catch (error) {
    return { success: false, error: error.message };
  }
});

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const process = spawn(command, args);
    
    process.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed with code ${code}`));
      }
    });
    
    process.on('error', reject);
  });
}

// Cleanup on quit
app.on('before-quit', () => {
  if (installerProcess) {
    installerProcess.kill();
  }
});