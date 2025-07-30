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
- Automated LeRobot installation
- USB port detection and configuration
- Step-by-step setup guidance

## Installation

### Prerequisites
- Python 3.8+
- pip

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatVLA.git
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
