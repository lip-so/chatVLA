# LeRobot Plug & Play System

Complete LeRobot pipeline with automated setup and robot selection.

## 🚀 Features

### Robot Selection
- **Visual Robot Selector**: Choose your robot from a beautiful UI
- **Supported Robots**:
  - Koch Follower (6-DOF leader-follower arm)
  - SO-100 Follower (5-DOF desktop arm)
  - SO-101 Follower (6-DOF precision arm)

### Full Pipeline Integration
1. **Setup & Calibration**: Automated motor detection and calibration
2. **Data Collection**: Record demonstrations with teleoperation
3. **Policy Training**: Train neural network policies on your data
4. **Deployment**: Deploy trained policies to your robot

### Automated Installation
- One-click installation for your selected robot
- Robot-specific dependency installation
- Automatic configuration file generation
- USB port detection wizard
- Pipeline script generation

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git
- Conda/Miniconda

### Quick Start

1. **Start the Backend**:
```bash
cd backend/plug_and_play
python start_enhanced.py
```

2. **Open the Web Interface**:
   - Navigate to `http://localhost:5002` in your browser
   - Or open `frontend/pages/plug-and-play.html`

3. **Select Your Robot**:
   - Choose your robot from the selection screen
   - Click "Continue with Selected Robot"

4. **Configure Installation**:
   - Choose installation directory
   - Click "Start Installation"
   - Follow the interactive USB port detection

5. **Use Your Robot**:
   After installation, navigate to your installation directory and run:
   ```bash
   ./run_lerobot.sh
   ```

## 📁 Generated Files

The system generates the following files in your installation directory:

```
lerobot/
├── robot_config.yaml          # Robot-specific configuration
├── calibrate_robot.py         # Calibration script
├── teleoperate.py            # Teleoperation script
├── record_dataset.py         # Data recording script
├── train_config.yaml         # Training configuration
├── deploy_policy.py          # Deployment script
├── run_lerobot.sh           # Main launcher
└── pipeline_scripts/        # Pipeline workflow scripts
    ├── setup_pipeline.sh
    ├── record_pipeline.sh
    ├── train_pipeline.sh
    └── deploy_pipeline.sh
```

## 🎮 Usage

### Interactive Menu
After installation, run `./run_lerobot.sh` for an interactive menu:

```
🤖 Welcome to LeRobot!
Robot: Koch Follower

Select an action:
1) Setup & Calibration
2) Data Collection
3) Train Policy
4) Deploy Policy
5) Full Pipeline
6) Exit
```

### Direct Commands

**Calibrate Robot**:
```bash
python calibrate_robot.py
```

**Record Dataset**:
```bash
python record_dataset.py
```

**Train Policy**:
```bash
python -m lerobot.train --config train_config.yaml
```

**Deploy Policy**:
```bash
python deploy_policy.py
```

## 🔧 API Endpoints

### Robot Management
- `GET /api/robot_types` - Get available robot types
- `GET /api/pipeline_workflow` - Get pipeline workflow stages

### Installation
- `POST /api/start_installation` - Start installation with robot selection
- `GET /api/status` - Get installation status

### Port Detection
- `GET /api/list_ports` - List available USB ports
- `POST /api/save_port_config` - Save port configuration

### Pipeline Actions
- `POST /api/run_pipeline_action` - Execute pipeline action

## 🐛 Troubleshooting

### USB Port Detection Issues
1. Ensure robots are properly connected
2. Check USB drivers are installed
3. Try different USB ports/cables
4. Run manual detection: `python -m serial.tools.list_ports`

### Installation Failures
1. Check internet connection
2. Verify Git and Conda are installed
3. Ensure sufficient disk space
4. Check installation logs

### Robot Connection Issues
1. Verify correct port configuration
2. Check motor IDs match configuration
3. Ensure proper power supply
4. Run calibration script

## 📚 Advanced Configuration

### Custom Robot Configuration
Edit `robot_config.yaml` to customize:
- Motor IDs and models
- Port assignments
- Control parameters

### Training Parameters
Modify `train_config.yaml` to adjust:
- Policy type (ACT, Diffusion, etc.)
- Batch size and learning rate
- Number of epochs
- Observation/action spaces

### Adding New Robots
1. Add robot definition to `ROBOT_CONFIGS` in `enhanced_api.py`
2. Create configuration template in `robot_configs.py`
3. Update frontend robot selection UI

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is part of Tune Robotics and follows the LeRobot license terms.