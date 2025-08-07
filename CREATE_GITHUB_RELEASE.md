# 📦 CREATE GITHUB RELEASE FOR ONE-CLICK INSTALLERS

## Steps to Create the Release:

### 1. Build the Installers Locally
```bash
cd /Users/sofiia/chatVLA/electron_installer
npm install
npm run build-mac    # Creates .dmg file
npm run build-win    # Creates .exe file  
npm run build-linux  # Creates .AppImage file
```

### 2. Create GitHub Release
1. Go to: https://github.com/lip-so/chatVLA/releases/new
2. Tag version: `v1.0`
3. Release title: `Tune Robotics One-Click Installer v1.0`
4. Description:
```markdown
## 🤖 Tune Robotics One-Click LeRobot Installer

### Installation:
1. Download the installer for your platform
2. Run it
3. Click "Install LeRobot" on the website
4. Done! LeRobot is installed on your computer

### Downloads:
- 🍎 **Mac**: TuneRobotics-Mac.dmg
- 🪟 **Windows**: TuneRobotics-Windows.exe  
- 🐧 **Linux**: TuneRobotics-Linux.AppImage
```

5. Upload the built files from `electron_installer/dist/`

### 3. Update Website URLs
The website is already configured to download from:
- Mac: `https://github.com/lip-so/chatVLA/releases/download/v1.0/TuneRobotics-Mac.dmg`
- Windows: `https://github.com/lip-so/chatVLA/releases/download/v1.0/TuneRobotics-Windows.exe`
- Linux: `https://github.com/lip-so/chatVLA/releases/download/v1.0/TuneRobotics-Linux.AppImage`

## Alternative: Use Pre-built Test Installers

For testing, create placeholder installers:

```bash
# Create test installer that just downloads and runs the Python script
cat > TuneRobotics-QuickInstaller.sh << 'EOF'
#!/bin/bash
curl -fsSL https://raw.githubusercontent.com/lip-so/chatVLA/main/one_click_installer.py | python3
EOF

chmod +x TuneRobotics-QuickInstaller.sh
```

## How It Works Now:

1. User visits: https://tunerobotics.xyz/frontend/pages/plug-and-play-databench-style.html
2. Website auto-detects platform and downloads installer
3. User runs installer from Downloads folder
4. Installer starts local bridge automatically
5. User refreshes website
6. Website detects local bridge
7. User clicks "Install LeRobot"
8. REAL installation happens on their computer!

This is as close to TRUE one-click as technically possible!