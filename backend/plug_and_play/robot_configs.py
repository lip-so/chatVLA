#!/usr/bin/env python3
"""
Robot-specific configuration generator for LeRobot Plug & Play.
Generates configuration files and setup scripts for each robot type.
"""

from pathlib import Path
from typing import Dict, Any
import yaml
import json


class RobotConfigGenerator:
    """Generates robot-specific configurations and scripts"""
    
    def __init__(self, robot_type: str, installation_path: Path):
        self.robot_type = robot_type
        self.installation_path = Path(installation_path)
        self.lerobot_path = self.installation_path / "lerobot"
        
    def generate_all_configs(self, leader_port: str = None, follower_port: str = None) -> Dict[str, str]:
        """Generate all configuration files for the robot"""
        configs = {}
        
        # Generate robot configuration
        configs['robot_config.yaml'] = self.generate_robot_config(leader_port, follower_port)
        
        # Generate calibration script
        configs['calibrate_robot.py'] = self.generate_calibration_script()
        
        # Generate teleoperation script
        configs['teleoperate.py'] = self.generate_teleoperation_script()
        
        # Generate recording script
        configs['record_dataset.py'] = self.generate_recording_script()
        
        # Generate training config
        configs['train_config.yaml'] = self.generate_training_config()
        
        # Generate deployment script
        configs['deploy_policy.py'] = self.generate_deployment_script()
        
        return configs
        
    def generate_robot_config(self, leader_port: str = None, follower_port: str = None) -> str:
        """Generate robot-specific configuration file"""
        configs = {
            'koch_follower': {
                'robot_type': 'koch_follower',
                'leader_config': {
                    'port': leader_port or '/dev/ttyUSB0',
                    'motors': {
                        'shoulder_pan': {'id': 1, 'model': 'xl430'},
                        'shoulder_lift': {'id': 2, 'model': 'xl430'},
                        'elbow_flex': {'id': 3, 'model': 'xl330'},
                        'wrist_flex': {'id': 4, 'model': 'xl330'},
                        'wrist_roll': {'id': 5, 'model': 'xl330'},
                        'gripper': {'id': 6, 'model': 'xl330'}
                    }
                },
                'follower_config': {
                    'port': follower_port or '/dev/ttyUSB1',
                    'motors': 'same_as_leader'
                },
                'use_degrees': False,
                'motor_bus': 'dynamixel'
            },
            'so100_follower': {
                'robot_type': 'so100_follower',
                'leader_config': {
                    'port': leader_port or '/dev/ttyUSB0',
                    'motors': {
                        'shoulder_pan': {'id': 1, 'model': 'sts3215'},
                        'shoulder_lift': {'id': 2, 'model': 'sts3215'},
                        'elbow_flex': {'id': 3, 'model': 'sts3215'},
                        'wrist_flex': {'id': 4, 'model': 'sts3215'},
                        'gripper': {'id': 5, 'model': 'sts3215'}
                    }
                },
                'follower_config': {
                    'port': follower_port or '/dev/ttyUSB1',
                    'motors': 'same_as_leader'
                },
                'use_degrees': False,
                'motor_bus': 'feetech'
            },
            'viperx': {
                'robot_type': 'viperx',
                'port': follower_port or '/dev/ttyUSB0',
                'motors': {
                    'waist': {'id': 1, 'model': 'mx64'},
                    'shoulder': {'id': 2, 'model': 'mx64'},
                    'elbow': {'id': 3, 'model': 'mx64'},
                    'forearm_roll': {'id': 4, 'model': 'mx64'},
                    'wrist_angle': {'id': 5, 'model': 'mx28'},
                    'wrist_rotate': {'id': 6, 'model': 'mx28'},
                    'gripper': {'id': 7, 'model': 'mx28'}
                },
                'use_degrees': True,
                'motor_bus': 'dynamixel'
            },
            'lekiwi': {
                'robot_type': 'lekiwi',
                'port': follower_port or '/dev/ttyUSB0',
                'remote_ip': '192.168.1.100',
                'motors': {
                    'arm_shoulder_pan': {'id': 1, 'model': 'sts3215'},
                    'arm_shoulder_lift': {'id': 2, 'model': 'sts3215'},
                    'arm_elbow_flex': {'id': 3, 'model': 'sts3215'},
                    'arm_wrist_flex': {'id': 4, 'model': 'sts3215'},
                    'arm_wrist_roll': {'id': 5, 'model': 'sts3215'},
                    'arm_gripper': {'id': 6, 'model': 'sts3215'},
                    'base_left_wheel': {'id': 7, 'model': 'sts3215'},
                    'base_back_wheel': {'id': 8, 'model': 'sts3215'},
                    'base_right_wheel': {'id': 9, 'model': 'sts3215'}
                },
                'use_degrees': False,
                'motor_bus': 'feetech'
            }
        }
        
        config = configs.get(self.robot_type, {})
        return yaml.dump(config, default_flow_style=False)
        
    def generate_calibration_script(self) -> str:
        """Generate robot calibration script"""
        return f'''#!/usr/bin/env python3
"""
Calibration script for {self.robot_type}
Auto-generated by LeRobot Plug & Play
"""

import yaml
from pathlib import Path
from lerobot.robots import make_robot

def main():
    # Load robot configuration
    config_path = Path(__file__).parent / "robot_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"Starting calibration for {config['robot_type']}...")
    
    # Create robot instance
    robot = make_robot(config)
    
    # Run calibration
    robot.calibrate()
    
    print("Calibration complete!")
    print("Calibration data saved to:", robot.calibration_path)

if __name__ == "__main__":
    main()
'''
        
    def generate_teleoperation_script(self) -> str:
        """Generate teleoperation script"""
        leader_required = self.robot_type in ['koch_follower', 'so100_follower', 'so101_follower', 'lekiwi']
        
        return f'''#!/usr/bin/env python3
"""
Teleoperation script for {self.robot_type}
Auto-generated by LeRobot Plug & Play
"""

import yaml
from pathlib import Path
from lerobot.teleoperate import teleoperate

def main():
    # Load robot configuration
    config_path = Path(__file__).parent / "robot_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"Starting teleoperation for {config['robot_type']}...")
    
    {'# This robot requires a leader arm for teleoperation' if leader_required else '# This robot can be teleoperated via gamepad or keyboard'}
    
    # Start teleoperation
    teleoperate(
        robot_type=config['robot_type'],
        config=config,
        {'leader_port=config["leader_config"]["port"],' if leader_required else ''}
        {'follower_port=config["follower_config"]["port"]' if leader_required else 'port=config["port"]'}
    )

if __name__ == "__main__":
    main()
'''
        
    def generate_recording_script(self) -> str:
        """Generate data recording script"""
        return f'''#!/usr/bin/env python3
"""
Data recording script for {self.robot_type}
Auto-generated by LeRobot Plug & Play
"""

import yaml
from pathlib import Path
from datetime import datetime
from lerobot.record import record

def main():
    # Load robot configuration
    config_path = Path(__file__).parent / "robot_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create dataset name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"{config['robot_type']}_dataset_{timestamp}"
    
    print(f"Recording dataset: {dataset_name}")
    print("Press 'q' to stop recording")
    
    # Start recording
    record(
        robot_type=config['robot_type'],
        config=config,
        dataset_name=dataset_name,
        fps=30,
        save_path=Path("datasets") / dataset_name
    )
    
    print(f"Dataset saved to: datasets/{dataset_name}")

if __name__ == "__main__":
    main()
'''
        
    def generate_training_config(self) -> str:
        """Generate training configuration"""
        training_configs = {
            'koch_follower': {
                'policy': 'act',
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 100,
                'observation_space': {
                    'joint_positions': 6,
                    'images': ['camera_top', 'camera_wrist']
                },
                'action_space': {
                    'joint_positions': 6
                }
            },
            'viperx': {
                'policy': 'diffusion',
                'batch_size': 16,
                'learning_rate': 5e-5,
                'num_epochs': 200,
                'observation_space': {
                    'joint_positions': 7,
                    'images': ['camera_front', 'camera_side']
                },
                'action_space': {
                    'joint_positions': 7
                }
            },
            'lekiwi': {
                'policy': 'act',
                'batch_size': 24,
                'learning_rate': 1e-4,
                'num_epochs': 150,
                'observation_space': {
                    'joint_positions': 9,
                    'images': ['camera_front'],
                    'base_velocity': 3
                },
                'action_space': {
                    'joint_positions': 6,
                    'base_velocity': 3
                }
            }
        }
        
        # Use default config if robot not in specific configs
        config = training_configs.get(self.robot_type, {
            'policy': 'act',
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'observation_space': {
                'joint_positions': 6,
                'images': ['camera_top']
            },
            'action_space': {
                'joint_positions': 6
            }
        })
        
        config['robot_type'] = self.robot_type
        config['device'] = 'cuda'  # Use GPU if available
        config['checkpoint_interval'] = 10
        
        return yaml.dump(config, default_flow_style=False)
        
    def generate_deployment_script(self) -> str:
        """Generate deployment script"""
        return f'''#!/usr/bin/env python3
"""
Policy deployment script for {self.robot_type}
Auto-generated by LeRobot Plug & Play
"""

import yaml
import torch
from pathlib import Path
from lerobot.deploy import deploy_policy

def main():
    # Load robot configuration
    config_path = Path(__file__).parent / "robot_config.yaml"
    with open(config_path) as f:
        robot_config = yaml.safe_load(f)
    
    # Load training configuration
    train_config_path = Path(__file__).parent / "train_config.yaml"
    with open(train_config_path) as f:
        train_config = yaml.safe_load(f)
    
    # Find latest checkpoint
    checkpoints = list(Path("checkpoints").glob("*.pth"))
    if not checkpoints:
        print("No trained checkpoints found!")
        print("Please train a policy first using train_policy.py")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Deploy policy
    deploy_policy(
        robot_type=robot_config['robot_type'],
        robot_config=robot_config,
        policy_type=train_config['policy'],
        checkpoint_path=latest_checkpoint,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

if __name__ == "__main__":
    main()
'''
        
    def save_configs(self, output_dir: Path = None):
        """Save all configuration files to disk"""
        if output_dir is None:
            output_dir = self.lerobot_path
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        configs = self.generate_all_configs()
        
        for filename, content in configs.items():
            filepath = output_dir / filename
            with open(filepath, 'w') as f:
                f.write(content)
                
            # Make Python scripts executable
            if filename.endswith('.py'):
                filepath.chmod(0o755)
                
        print(f"Configuration files saved to: {output_dir}")


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python robot_configs.py <robot_type> <installation_path>")
        print("Available robots: koch_follower, so100_follower, viperx, lekiwi")
        sys.exit(1)
        
    robot_type = sys.argv[1]
    installation_path = sys.argv[2]
    
    generator = RobotConfigGenerator(robot_type, installation_path)
    generator.save_configs()


if __name__ == "__main__":
    main()