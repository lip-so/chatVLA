#!/bin/bash
# One-line installer for Tune Robotics LeRobot
# Run: curl -fsSL https://raw.githubusercontent.com/lip-so/chatVLA/main/quick_install.sh | bash

set -e

echo "🤖 Tune Robotics - Automatic LeRobot Installer"
echo "=============================================="

# Check OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Windows;;
    MINGW*)     MACHINE=Windows;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Installing..."
    
    if [ "$MACHINE" = "Mac" ]; then
        # Install Python via Homebrew
        if ! command -v brew &> /dev/null; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python3
    elif [ "$MACHINE" = "Linux" ]; then
        sudo apt-get update && sudo apt-get install -y python3 python3-pip
    else
        echo "Please install Python 3 manually from https://python.org"
        exit 1
    fi
fi

# Install directory
INSTALL_DIR="$HOME/TuneRobotics"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download the one-click installer
echo "📦 Downloading installer..."
curl -fsSL https://raw.githubusercontent.com/lip-so/chatVLA/main/one_click_installer.py -o installer.py

# Install dependencies
echo "📚 Installing dependencies..."
python3 -m pip install --quiet flask flask-cors flask-socketio pyserial eventlet

# Create desktop shortcut
if [ "$MACHINE" = "Mac" ]; then
    # Create Mac app
    cat > "$HOME/Desktop/TuneRobotics.command" << 'EOF'
#!/bin/bash
cd "$HOME/TuneRobotics"
python3 installer.py
EOF
    chmod +x "$HOME/Desktop/TuneRobotics.command"
    echo "✅ Created desktop shortcut: TuneRobotics.command"
    
elif [ "$MACHINE" = "Linux" ]; then
    # Create Linux desktop entry
    cat > "$HOME/Desktop/tune-robotics.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Tune Robotics Installer
Comment=Install LeRobot with one click
Exec=python3 $HOME/TuneRobotics/installer.py
Icon=applications-robotics
Terminal=false
Categories=Development;Robotics;
EOF
    chmod +x "$HOME/Desktop/tune-robotics.desktop"
    echo "✅ Created desktop shortcut: tune-robotics.desktop"
fi

# Start the installer
echo ""
echo "=============================================="
echo "✅ Installation complete!"
echo "🚀 Starting Tune Robotics installer..."
echo "=============================================="
echo ""

# Run the installer
python3 installer.py