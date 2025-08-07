#!/bin/bash

echo "🔨 Building Tune Robotics One-Click Installers"
echo "=============================================="

cd electron_installer

# Install dependencies
echo "📦 Installing build dependencies..."
npm install

# Build for all platforms
echo "🖥️ Building for macOS..."
npm run build-mac

echo "🪟 Building for Windows..."
npm run build-win

echo "🐧 Building for Linux..."
npm run build-linux

echo ""
echo "✅ Build complete! Installers are in electron_installer/dist/"
echo ""
echo "Files created:"
echo "- Mac: dist/Tune Robotics Installer.dmg"
echo "- Windows: dist/Tune Robotics Installer.exe"
echo "- Linux: dist/Tune Robotics Installer.AppImage"