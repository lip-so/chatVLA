# Tune Robotics

Chat with your robot and make it do your tasks instantly.

## Overview

Tune Robotics enables natural language control of robots through three main components:

- **Vision**: Natural language understanding for robot control
- **DataBench**: Comprehensive dataset evaluation framework for robotics
- **Plug & Play**: Automated LeRobot installation and setup

## Features

### Vision
- Natural language to robot action translation
- Real-time robot control interface
- Multi-modal understanding capabilities

### DataBench
- 6 comprehensive metrics for robotics dataset evaluation:
  - Action Consistency
  - Visual Diversity
  - High-Fidelity Vision
  - Trajectory Quality
  - Dataset Coverage
  - Robot Action Quality

### Plug & Play
- **Local LeRobot installation** to user's machine
- Automated USB port detection and configuration
- Real-time installation progress tracking
- Step-by-step setup guidance

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip
- Git (for LeRobot installation)
- Conda/Miniconda (for LeRobot environment)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/lip-so/chatVLA.git
   cd chatVLA
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python backend/api/main.py
   ```

4. Open your browser to `http://localhost:5000`

## 📦 Installing LeRobot with Plug & Play

The Plug & Play system enables **LOCAL installation** of LeRobot on your machine through a web interface.

### How It Works

1. **Start the Local Installer Bridge** (on your machine):
   ```bash
   cd ~/chatVLA
   python3 local_installer_bridge.py
   ```
   This runs a local server on port 7777 that performs the actual installation.

2. **Visit the Web Interface**:
   - Go to https://tunerobotics.xyz/frontend/pages/plug-and-play-databench-style.html
   - The page will detect the local installer automatically
   - You'll see a green banner: "✅ Connected to local installer"

3. **Complete Installation**:
   - Select your robot type (Koch, SO-100, SO-101)
   - Choose installation path (default: ~/lerobot)
   - Click "Start Installation"
   - LeRobot will be downloaded and installed locally on your machine

### What Gets Installed

The local installer will:
- Clone LeRobot repository from GitHub
- Create conda environment 'lerobot' with Python 3.10
- Install all LeRobot dependencies
- Configure USB port detection for robot arms
- Set up the complete LeRobot pipeline

### Important Notes

⚠️ **The production website (tunerobotics.xyz) alone cannot install software on your machine.** You must run the local installer bridge to enable actual installation.

The system architecture:
- **Web Interface** (production): Provides UI and guidance
- **Local Installer** (your machine): Performs actual installation

## Project Structure

```
chatVLA/
├── frontend/           # Web interface
│   ├── index.html     # Main landing page
│   ├── css/           # Stylesheets
│   ├── js/            # JavaScript files
│   ├── pages/         # Additional HTML pages
│   └── assets/        # Images and other assets
│
├── backend/           # Server-side code
│   ├── api/           # Flask API server
│   ├── databench/     # DataBench evaluation system
│   └── plug_and_play/ # Plug & Play installation system
│
├── tests/             # Test suite
├── deployment/        # Deployment configurations
└── docs/              # Documentation
```

## 📧 Contact

- Email: yo@tunerobotics.xyz
- Website: [tunerobotics.xyz](https://tunerobotics.xyz)
