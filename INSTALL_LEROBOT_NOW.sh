#!/bin/bash

echo "🤖 REAL LeRobot Installer"
echo "========================"
echo ""
echo "This will ACTUALLY install LeRobot on your computer!"
echo ""
echo "Choose an option:"
echo "1) GUI Installer (Recommended)"
echo "2) Command Line Installation"
echo "3) Exit"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "Starting GUI installer..."
        python backend/plug_and_play/installers/lerobot_installer.py
        ;;
    2)
        echo "Starting command line installation..."
        echo ""
        read -p "Installation path (default: ~/lerobot): " install_path
        install_path=${install_path:-~/lerobot}
        
        echo "Installing to: $install_path"
        echo ""
        
        # Check prerequisites
        if ! command -v conda &> /dev/null; then
            echo "❌ Conda not found! Please install Miniconda first:"
            echo "   https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
        
        if ! command -v git &> /dev/null; then
            echo "❌ Git not found! Please install Git first:"
            echo "   https://git-scm.com/downloads"
            exit 1
        fi
        
        # Clone repository
        echo "📦 Cloning LeRobot repository..."
        git clone https://github.com/huggingface/lerobot.git "$install_path"
        
        # Create conda environment
        echo "🔧 Creating conda environment..."
        conda create -y -n lerobot python=3.10
        
        # Install LeRobot
        echo "📥 Installing LeRobot..."
        cd "$install_path"
        conda run -n lerobot pip install -e .
        
        # Install additional dependencies
        echo "📚 Installing additional dependencies..."
        conda install -y ffmpeg -c conda-forge -n lerobot
        conda run -n lerobot pip install pyserial
        
        echo ""
        echo "✅ LeRobot installed successfully!"
        echo ""
        echo "To use LeRobot:"
        echo "1. conda activate lerobot"
        echo "2. cd $install_path"
        echo "3. Start using LeRobot!"
        ;;
    3)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac