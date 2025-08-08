#!/usr/bin/env python3

"""
LeRobot API Integration for Plug & Play
Provides backend endpoints for robot calibration, teleoperation, and recording
"""

import os
import sys
import threading
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
try:
    from flask import Blueprint, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Add ref path for LeRobot imports
ref_path = Path(__file__).parent.parent.parent / "ref"
sys.path.insert(0, str(ref_path))

logger = logging.getLogger(__name__)

# Check if LeRobot is available
LEROBOT_AVAILABLE = False
try:
    # Test import of a basic LeRobot module
    import lerobot.robots.so101_follower
    LEROBOT_AVAILABLE = True
    logger.info("✅ LeRobot modules are available")
except ImportError as e:
    logger.warning(f"⚠️ LeRobot modules not available: {e}")
    logger.warning("LeRobot API will run in simulation mode")

# Global SocketIO instance (set by main app)
socketio = None

def set_socketio(socket_instance):
    """Set the SocketIO instance for real-time communication"""
    global socketio
    socketio = socket_instance

# Create blueprint (only if Flask is available)
if FLASK_AVAILABLE:
    lerobot_bp = Blueprint('lerobot', __name__, url_prefix='/api/lerobot')
else:
    lerobot_bp = None

# Global state for active sessions
active_sessions = {
    'calibration': {},
    'teleoperation': {},
    'recording': {}
}

class LeRobotIntegration:
    """Main class for LeRobot integration"""
    
    def __init__(self):
        self.robots = {}
        self.teleoperators = {}
        self.cameras = {}
        self.datasets = {}
        self.active_threads = {}
        
    def emit_log(self, session_type: str, message: str, level: str = 'info'):
        """Emit log message via SocketIO"""
        if socketio:
            socketio.emit(f'{session_type}_log', {
                'message': message,
                'level': level,
                'timestamp': time.time()
            })
        else:
            logger.info(f"[{session_type.upper()}] {message}")
    
    def get_robot_config_class(self, robot_type: str, role: str = 'follower'):
        """Get the appropriate robot configuration class"""
        try:
            if robot_type == 'so101':
                if role == 'follower':
                    from lerobot.robots.so101_follower import SO101FollowerConfig
                    return SO101FollowerConfig
                else:
                    from lerobot.teleoperators.so101_leader import SO101LeaderConfig
                    return SO101LeaderConfig
            elif robot_type == 'so100':
                if role == 'follower':
                    from lerobot.robots.so100_follower import SO100FollowerConfig
                    return SO100FollowerConfig
                else:
                    from lerobot.teleoperators.so100_leader import SO100LeaderConfig
                    return SO100LeaderConfig
            elif robot_type == 'koch':
                if role == 'follower':
                    from lerobot.robots.koch_follower import KochFollowerConfig
                    return KochFollowerConfig
                else:
                    from lerobot.teleoperators.koch_leader import KochLeaderConfig
                    return KochLeaderConfig
            else:
                raise ValueError(f"Unsupported robot type: {robot_type}")
                
        except ImportError as e:
            logger.error(f"Failed to import robot config for {robot_type}: {e}")
            raise
    
    def get_robot_class(self, robot_type: str, role: str = 'follower'):
        """Get the appropriate robot class"""
        try:
            if robot_type == 'so101':
                if role == 'follower':
                    from lerobot.robots.so101_follower import SO101Follower
                    return SO101Follower
                else:
                    from lerobot.teleoperators.so101_leader import SO101Leader
                    return SO101Leader
            elif robot_type == 'so100':
                if role == 'follower':
                    from lerobot.robots.so100_follower import SO100Follower
                    return SO100Follower
                else:
                    from lerobot.teleoperators.so100_leader import SO100Leader
                    return SO100Leader
            elif robot_type == 'koch':
                if role == 'follower':
                    from lerobot.robots.koch_follower import KochFollower
                    return KochFollower
                else:
                    from lerobot.teleoperators.koch_leader import KochLeader
                    return KochLeader
            else:
                raise ValueError(f"Unsupported robot type: {robot_type}")
                
        except ImportError as e:
            logger.error(f"Failed to import robot class for {robot_type}: {e}")
            raise
    
    def calibrate_robot(self, session_id: str, robot_type: str, port: str, role: str = 'follower'):
        """Calibrate a robot"""
        try:
            self.emit_log('calibration', f'🔧 Starting {role} calibration for {robot_type}')
            
            # Get configuration and robot classes
            config_class = self.get_robot_config_class(robot_type, role)
            robot_class = self.get_robot_class(robot_type, role)
            
            # Create configuration
            config = config_class(
                port=port,
                id=f"{robot_type}_{role}_{session_id}"
            )
            
            self.emit_log('calibration', f'📡 Connecting to {role} on port {port}')
            
            # Create and connect robot
            robot = robot_class(config)
            robot.connect(calibrate=False)
            
            self.emit_log('calibration', f'🔗 Connected to {role} arm successfully')
            
            # Store robot instance
            self.robots[session_id] = robot
            
            self.emit_log('calibration', '⚙️ Starting calibration sequence...')
            
            # Perform calibration
            robot.calibrate()
            
            self.emit_log('calibration', f'✅ {role.title()} calibration completed successfully!', 'success')
            
            # Disconnect after calibration
            robot.disconnect()
            
            # Remove from active robots
            if session_id in self.robots:
                del self.robots[session_id]
                
        except Exception as e:
            error_msg = f"❌ Calibration error: {str(e)}"
            self.emit_log('calibration', error_msg, 'error')
            logger.error(f"Calibration failed: {e}")
            logger.error(traceback.format_exc())
            
            # Cleanup on error
            if session_id in self.robots:
                try:
                    self.robots[session_id].disconnect()
                except:
                    pass
                del self.robots[session_id]
    
    def start_teleoperation(self, session_id: str, leader_type: str, follower_type: str, 
                          leader_port: str, follower_port: str, use_cameras: bool = False):
        """Start teleoperation session"""
        try:
            self.emit_log('teleoperation', f'🤖 Initializing teleoperation: {leader_type} -> {follower_type}')
            
            # Setup camera configuration if requested
            camera_config = None
            if use_cameras:
                from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
                camera_config = {
                    "front": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30)
                }
                self.emit_log('teleoperation', '📷 Initializing camera feed (1920x1080 @ 30fps)...')
            
            # Create robot configurations
            follower_config_class = self.get_robot_config_class(follower_type, 'follower')
            leader_config_class = self.get_robot_config_class(leader_type, 'leader')
            
            robot_config = follower_config_class(
                port=follower_port,
                id=f"{follower_type}_follower_{session_id}",
                cameras=camera_config
            )
            
            teleop_config = leader_config_class(
                port=leader_port,
                id=f"{leader_type}_leader_{session_id}"
            )
            
            # Create robot instances
            follower_class = self.get_robot_class(follower_type, 'follower')
            leader_class = self.get_robot_class(leader_type, 'leader')
            
            robot = follower_class(robot_config)
            teleop_device = leader_class(teleop_config)
            
            self.emit_log('teleoperation', f'📡 Connecting to follower on {follower_port}...')
            robot.connect()
            
            self.emit_log('teleoperation', f'📡 Connecting to leader on {leader_port}...')
            teleop_device.connect()
            
            self.emit_log('teleoperation', '🔗 Both devices connected successfully')
            
            # Store instances
            self.robots[f"{session_id}_follower"] = robot
            self.teleoperators[f"{session_id}_leader"] = teleop_device
            
            self.emit_log('teleoperation', '🎮 Starting teleoperation loop...')
            
            # Start teleoperation loop
            def teleoperation_loop():
                try:
                    step_count = 0
                    while session_id in active_sessions['teleoperation']:
                        # Get action from leader
                        if use_cameras:
                            observation = robot.get_observation()
                        action = teleop_device.get_action()
                        
                        # Send action to follower
                        robot.send_action(action)
                        
                        step_count += 1
                        if step_count % 100 == 0:  # Log every 100 steps
                            self.emit_log('teleoperation', f'📊 Step {step_count} | Status: OK')
                        
                        time.sleep(0.01)  # 100Hz loop
                        
                except Exception as e:
                    self.emit_log('teleoperation', f'❌ Teleoperation loop error: {str(e)}', 'error')
                finally:
                    # Cleanup
                    try:
                        robot.disconnect()
                        teleop_device.disconnect()
                    except:
                        pass
                    
                    # Remove from storage
                    if f"{session_id}_follower" in self.robots:
                        del self.robots[f"{session_id}_follower"]
                    if f"{session_id}_leader" in self.teleoperators:
                        del self.teleoperators[f"{session_id}_leader"]
            
            # Start teleoperation in background thread
            thread = threading.Thread(target=teleoperation_loop, daemon=True)
            self.active_threads[session_id] = thread
            thread.start()
            
            self.emit_log('teleoperation', '✅ Teleoperation started successfully!', 'success')
            
        except Exception as e:
            error_msg = f"❌ Teleoperation error: {str(e)}"
            self.emit_log('teleoperation', error_msg, 'error')
            logger.error(f"Teleoperation failed: {e}")
            logger.error(traceback.format_exc())
    
    def start_recording(self, session_id: str, config: Dict[str, Any]):
        """Start dataset recording"""
        try:
            repo_id = config['repo_id']
            episodes = config['episodes']
            fps = config['fps']
            episode_time = config['episode_time']
            reset_time = config['reset_time']
            task_description = config['task_description']
            robot_type = config['robot_type']
            leader_port = config['leader_port']
            follower_port = config['follower_port']
            
            self.emit_log('recording', f'🎥 Starting recording: {episodes} episodes to {repo_id}')
            
            # Import required modules
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.datasets.utils import hw_to_dataset_features
            from lerobot.utils.control_utils import init_keyboard_listener
            from lerobot.utils.visualization_utils import _init_rerun
            from lerobot.record import record_loop
            
            # Setup camera configuration
            camera_config = {
                "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=fps)
            }
            
            # Create robot configurations
            follower_config_class = self.get_robot_config_class(robot_type, 'follower')
            leader_config_class = self.get_robot_config_class(robot_type, 'leader')
            
            robot_config = follower_config_class(
                port=follower_port,
                id=f"{robot_type}_follower_{session_id}",
                cameras=camera_config
            )
            
            teleop_config = leader_config_class(
                port=leader_port,
                id=f"{robot_type}_leader_{session_id}"
            )
            
            # Create robot instances
            follower_class = self.get_robot_class(robot_type, 'follower')
            leader_class = self.get_robot_class(robot_type, 'leader')
            
            robot = follower_class(robot_config)
            teleop = leader_class(teleop_config)
            
            self.emit_log('recording', f'📊 Configuration: {episodes} episodes, {fps}fps, {episode_time}s each')
            self.emit_log('recording', f'📁 Target repository: {repo_id}')
            
            # Configure dataset features
            action_features = hw_to_dataset_features(robot.action_features, "action")
            obs_features = hw_to_dataset_features(robot.observation_features, "observation")
            dataset_features = {**action_features, **obs_features}
            
            # Create dataset
            self.emit_log('recording', '🗃️ Creating LeRobot dataset structure...')
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=True,
                image_writer_threads=4,
            )
            
            # Initialize keyboard listener and rerun visualization
            self.emit_log('recording', '⌨️ Initializing keyboard controls...')
            _, events = init_keyboard_listener()
            _init_rerun(session_name="recording")
            
            # Connect devices
            self.emit_log('recording', f'📡 Connecting to {robot_type} robot on {follower_port}...')
            robot.connect()
            
            self.emit_log('recording', f'🎮 Connecting to teleoperator on {leader_port}...')
            teleop.connect()
            
            # Store instances
            self.robots[f"{session_id}_follower"] = robot
            self.teleoperators[f"{session_id}_leader"] = teleop
            self.datasets[session_id] = dataset
            
            def recording_loop():
                try:
                    episode_idx = 0
                    while (episode_idx < episodes and 
                           session_id in active_sessions['recording'] and 
                           not events["stop_recording"]):
                        
                        self.emit_log('recording', f'🎬 Recording episode {episode_idx + 1}/{episodes}: "{task_description}"')
                        
                        # Record episode
                        record_loop(
                            robot=robot,
                            events=events,
                            fps=fps,
                            teleop=teleop,
                            dataset=dataset,
                            control_time_s=episode_time,
                            single_task=task_description,
                            display_data=True,
                        )
                        
                        self.emit_log('recording', f'✅ Episode {episode_idx + 1} recorded successfully')
                        
                        # Reset environment if needed
                        if (episode_idx < episodes - 1 and 
                            not events["stop_recording"] and 
                            not events["rerecord_episode"]):
                            self.emit_log('recording', '⏸️ Reset environment for next episode...')
                            record_loop(
                                robot=robot,
                                events=events,
                                fps=fps,
                                teleop=teleop,
                                control_time_s=reset_time,
                                single_task=task_description,
                                display_data=True,
                            )
                        
                        if events["rerecord_episode"]:
                            self.emit_log('recording', "Re-recording episode")
                            events["rerecord_episode"] = False
                            events["exit_early"] = False
                            dataset.clear_episode_buffer()
                            continue
                        
                        dataset.save_episode()
                        episode_idx += 1
                    
                    # Push to hub
                    self.emit_log('recording', '💾 Saving dataset to disk...')
                    time.sleep(1)
                    
                    self.emit_log('recording', f'🚀 Pushing dataset to Hugging Face Hub: {repo_id}')
                    dataset.push_to_hub()
                    
                    self.emit_log('recording', '✅ Dataset recording completed and pushed to hub!', 'success')
                    
                except Exception as e:
                    self.emit_log('recording', f'❌ Recording error: {str(e)}', 'error')
                    logger.error(f"Recording failed: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    # Cleanup
                    try:
                        robot.disconnect()
                        teleop.disconnect()
                    except:
                        pass
                    
                    # Remove from storage
                    if f"{session_id}_follower" in self.robots:
                        del self.robots[f"{session_id}_follower"]
                    if f"{session_id}_leader" in self.teleoperators:
                        del self.teleoperators[f"{session_id}_leader"]
                    if session_id in self.datasets:
                        del self.datasets[session_id]
            
            # Start recording in background thread
            thread = threading.Thread(target=recording_loop, daemon=True)
            self.active_threads[session_id] = thread
            thread.start()
            
        except Exception as e:
            error_msg = f"❌ Recording setup error: {str(e)}"
            self.emit_log('recording', error_msg, 'error')
            logger.error(f"Recording setup failed: {e}")
            logger.error(traceback.format_exc())
    
    def stop_session(self, session_type: str, session_id: str):
        """Stop an active session"""
        try:
            if session_id in active_sessions[session_type]:
                del active_sessions[session_type][session_id]
            
            # Stop thread if exists
            if session_id in self.active_threads:
                # Thread will stop naturally when session is removed from active_sessions
                del self.active_threads[session_id]
            
            # Cleanup resources
            for key in list(self.robots.keys()):
                if session_id in key:
                    try:
                        self.robots[key].disconnect()
                    except:
                        pass
                    del self.robots[key]
            
            for key in list(self.teleoperators.keys()):
                if session_id in key:
                    try:
                        self.teleoperators[key].disconnect()
                    except:
                        pass
                    del self.teleoperators[key]
            
            if session_id in self.datasets:
                del self.datasets[session_id]
                
            self.emit_log(session_type, f'⏹️ {session_type.title()} session stopped', 'warning')
            
        except Exception as e:
            logger.error(f"Error stopping {session_type} session: {e}")

# Global integration instance
lerobot_integration = LeRobotIntegration()

# API Routes

if FLASK_AVAILABLE and lerobot_bp is not None:
    @lerobot_bp.route('/calibrate', methods=['POST'])
    def calibrate():
        """Start robot calibration"""
        try:
            data = request.get_json()
            role = data.get('role', 'follower')
            port = data.get('port')
            robot_type = data.get('robot_type', 'so101')
            
            if not port:
                return jsonify({"success": False, "error": "Port is required"}), 400
            
            session_id = f"cal_{int(time.time())}"
            active_sessions['calibration'][session_id] = {
                'role': role,
                'robot_type': robot_type,
                'port': port,
                'started': time.time()
            }
            
            # Start calibration in background thread
            thread = threading.Thread(
                target=lerobot_integration.calibrate_robot,
                args=(session_id, robot_type, port, role),
                daemon=True
            )
            thread.start()
            
            return jsonify({
                "success": True,
                "message": f"Calibration started for {role}",
                "session_id": session_id
            })
            
        except Exception as e:
            logger.error(f"Calibration endpoint failed: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @lerobot_bp.route('/start-teleop', methods=['POST'])
    def start_teleop():
        """Start teleoperation session"""
        try:
            data = request.get_json()
            leader_type = data.get('leader_type', 'so101')
            follower_type = data.get('follower_type', 'so101')
            leader_port = data.get('leader_port')
            follower_port = data.get('follower_port')
            use_cameras = data.get('use_cameras', False)
            
            if not leader_port or not follower_port:
                return jsonify({"success": False, "error": "Both leader and follower ports are required"}), 400
            
            session_id = f"teleop_{int(time.time())}"
            active_sessions['teleoperation'][session_id] = {
                'leader_type': leader_type,
                'follower_type': follower_type,
                'leader_port': leader_port,
                'follower_port': follower_port,
                'use_cameras': use_cameras,
                'started': time.time()
            }
            
            # Start teleoperation in background thread
            thread = threading.Thread(
                target=lerobot_integration.start_teleoperation,
                args=(session_id, leader_type, follower_type, leader_port, follower_port, use_cameras),
                daemon=True
            )
            thread.start()
            
            return jsonify({
                "success": True,
                "message": "Teleoperation started",
                "session_id": session_id
            })
            
        except Exception as e:
            logger.error(f"Teleoperation endpoint failed: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

@lerobot_bp.route('/stop-teleop', methods=['POST'])
def stop_teleop():
    """Stop teleoperation session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id and session_id in active_sessions['teleoperation']:
            lerobot_integration.stop_session('teleoperation', session_id)
        else:
            # Stop all teleoperation sessions
            for sid in list(active_sessions['teleoperation'].keys()):
                lerobot_integration.stop_session('teleoperation', sid)
        
        return jsonify({"success": True, "message": "Teleoperation stopped"})
        
    except Exception as e:
        logger.error(f"Stop teleoperation failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@lerobot_bp.route('/start-recording', methods=['POST'])
def start_recording():
    """Start dataset recording"""
    try:
        data = request.get_json()
        
        required_fields = ['repo_id', 'task_description', 'leader_port', 'follower_port']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"success": False, "error": f"{field} is required"}), 400
        
        session_id = f"record_{int(time.time())}"
        active_sessions['recording'][session_id] = {
            'config': data,
            'started': time.time()
        }
        
        # Start recording in background thread
        thread = threading.Thread(
            target=lerobot_integration.start_recording,
            args=(session_id, data),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Recording started",
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Recording endpoint failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@lerobot_bp.route('/stop-recording', methods=['POST'])
def stop_recording():
    """Stop dataset recording"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id and session_id in active_sessions['recording']:
            lerobot_integration.stop_session('recording', session_id)
        else:
            # Stop all recording sessions
            for sid in list(active_sessions['recording'].keys()):
                lerobot_integration.stop_session('recording', sid)
        
        return jsonify({"success": True, "message": "Recording stopped"})
        
    except Exception as e:
        logger.error(f"Stop recording failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@lerobot_bp.route('/sessions', methods=['GET'])
def get_sessions():
    """Get active sessions"""
    return jsonify({
        "success": True,
        "sessions": active_sessions
    })

@lerobot_bp.route('/health', methods=['GET'])
def health():
    """Health check for LeRobot integration"""
    try:
        # Test basic imports
        from lerobot.robots.so101_follower import SO101FollowerConfig
        
        return jsonify({
            "success": True,
            "message": "LeRobot integration is healthy",
            "active_sessions": {
                "calibration": len(active_sessions['calibration']),
                "teleoperation": len(active_sessions['teleoperation']),
                "recording": len(active_sessions['recording'])
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"LeRobot integration unhealthy: {str(e)}"
        }), 500
