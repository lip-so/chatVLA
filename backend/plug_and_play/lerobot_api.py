#!/usr/bin/env python3
"""
LeRobot API endpoints for direct robot control.
Provides calibration, teleoperation, and recording functionality.
"""

import logging
import threading
import time
import traceback
from pathlib import Path
from flask import Blueprint, request, jsonify

# Setup logging
logger = logging.getLogger(__name__)

# Try to import LeRobot components
try:
    # Add the ref/lerobot directory to the Python path
    import sys
    lerobot_path = Path(__file__).parent.parent.parent / "ref"
    sys.path.insert(0, str(lerobot_path))
    
    from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
    from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower
    from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower
    
    from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
    from lerobot.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
    from lerobot.teleoperators.koch_leader import KochLeaderConfig, KochLeader
    
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.record import record_loop
    from lerobot.utils.control_utils import init_keyboard_listener
    from lerobot.utils.visualization_utils import _init_rerun
    from lerobot.utils.utils import log_say
    
    LEROBOT_AVAILABLE = True
    logger.info("✅ LeRobot API loaded successfully")
except ImportError as e:
    LEROBOT_AVAILABLE = False
    logger.warning(f"⚠️ LeRobot not available: {e}")

# Create blueprint
lerobot_bp = Blueprint('lerobot', __name__, url_prefix='/api/lerobot')

# Global robot state management
robot_sessions = {}
socketio = None  # Will be set by main.py

def set_socketio(socket_instance):
    """Set the socketio instance for real-time communication"""
    global socketio
    socketio = socket_instance

@lerobot_bp.route('/calibrate', methods=['POST'])
def calibrate():
    """Calibrate robot using LeRobot API"""
    if not LEROBOT_AVAILABLE:
        # Provide simulation mode for production
        return jsonify({
            "success": True,
            "message": "Calibration completed (simulation mode)",
            "mode": "simulation",
            "note": "Running in simulation mode - LeRobot dependencies not available in production"
        })
    
    try:
        data = request.get_json()
        robot_type = data.get('robot_type', 'so101')
        role = data.get('role', 'follower')  # 'leader' or 'follower'
        port = data.get('port')
        
        if not port:
            return jsonify({
                "success": False,
                "error": "Port is required for calibration"
            }), 400
        
        # Start calibration in background thread
        thread = threading.Thread(
            target=_run_calibration,
            args=(robot_type, role, port)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Started {role} calibration for {robot_type}",
            "robot_type": robot_type,
            "role": role,
            "port": port
        })
        
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@lerobot_bp.route('/start-teleop', methods=['POST'])
def start_teleoperation():
    """Start teleoperation using LeRobot API"""
    if not LEROBOT_AVAILABLE:
        # Provide simulation mode for production
        import time
        session_id = f"teleop_sim_{int(time.time())}"
        robot_sessions[session_id] = {
            "type": "teleoperation",
            "status": "running",
            "mode": "simulation"
        }
        return jsonify({
            "success": True,
            "message": "Teleoperation started (simulation mode)",
            "session_id": session_id,
            "mode": "simulation",
            "note": "Running in simulation mode - LeRobot dependencies not available in production"
        })
    
    try:
        data = request.get_json()
        robot_type = data.get('robot_type', 'so101')
        leader_port = data.get('leader_port')
        follower_port = data.get('follower_port')
        session_id = data.get('session_id', f"teleop_{int(time.time())}")
        
        if not leader_port or not follower_port:
            return jsonify({
                "success": False,
                "error": "Both leader_port and follower_port are required"
            }), 400
        
        # Check if session already exists
        if session_id in robot_sessions:
            return jsonify({
                "success": False,
                "error": f"Teleoperation session {session_id} already running"
            }), 400
        
        # Start teleoperation in background thread
        robot_sessions[session_id] = {
            "type": "teleoperation",
            "status": "starting",
            "robot_type": robot_type,
            "leader_port": leader_port,
            "follower_port": follower_port
        }
        
        thread = threading.Thread(
            target=_run_teleoperation,
            args=(robot_type, leader_port, follower_port, session_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Started teleoperation session",
            "session_id": session_id,
            "robot_type": robot_type
        })
        
    except Exception as e:
        logger.error(f"Teleoperation error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@lerobot_bp.route('/stop-teleop', methods=['POST'])
def stop_teleoperation():
    """Stop teleoperation session"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        
        if session_id and session_id in robot_sessions:
            robot_sessions[session_id]["status"] = "stopping"
            del robot_sessions[session_id]
            
        return jsonify({
            "success": True,
            "message": "Teleoperation stop requested"
        })
        
    except Exception as e:
        logger.error(f"Stop teleoperation error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@lerobot_bp.route('/start-recording', methods=['POST'])
def start_recording():
    """Start dataset recording using LeRobot API"""
    if not LEROBOT_AVAILABLE:
        # Provide simulation mode for production
        import time
        session_id = f"record_sim_{int(time.time())}"
        robot_sessions[session_id] = {
            "type": "recording",
            "status": "running", 
            "mode": "simulation"
        }
        return jsonify({
            "success": True,
            "message": "Recording started (simulation mode)",
            "session_id": session_id,
            "mode": "simulation",
            "note": "Running in simulation mode - LeRobot dependencies not available in production"
        })
    
    try:
        data = request.get_json()
        robot_type = data.get('robot_type', 'so101')
        leader_port = data.get('leader_port')
        follower_port = data.get('follower_port')
        repo_id = data.get('repo_id')
        num_episodes = data.get('num_episodes', 5)
        fps = data.get('fps', 30)
        episode_time_sec = data.get('episode_time_sec', 60)
        reset_time_sec = data.get('reset_time_sec', 10)
        task_description = data.get('task_description', "Robot task")
        session_id = data.get('session_id', f"record_{int(time.time())}")
        
        if not leader_port or not follower_port:
            return jsonify({
                "success": False,
                "error": "Both leader_port and follower_port are required"
            }), 400
        
        if not repo_id:
            return jsonify({
                "success": False,
                "error": "repo_id is required for dataset recording"
            }), 400
        
        # Check if session already exists
        if session_id in robot_sessions:
            return jsonify({
                "success": False,
                "error": f"Recording session {session_id} already running"
            }), 400
        
        # Start recording in background thread
        robot_sessions[session_id] = {
            "type": "recording",
            "status": "starting",
            "robot_type": robot_type,
            "leader_port": leader_port,
            "follower_port": follower_port,
            "repo_id": repo_id,
            "num_episodes": num_episodes,
            "current_episode": 0
        }
        
        thread = threading.Thread(
            target=_run_recording,
            args=(robot_type, leader_port, follower_port, repo_id, 
                  num_episodes, fps, episode_time_sec, reset_time_sec, 
                  task_description, session_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Started recording session",
            "session_id": session_id,
            "robot_type": robot_type,
            "repo_id": repo_id,
            "num_episodes": num_episodes
        })
        
    except Exception as e:
        logger.error(f"Recording error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@lerobot_bp.route('/stop-recording', methods=['POST'])
def stop_recording():
    """Stop recording session"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        
        if session_id and session_id in robot_sessions:
            robot_sessions[session_id]["status"] = "stopping"
            
        return jsonify({
            "success": True,
            "message": "Recording stop requested"
        })
        
    except Exception as e:
        logger.error(f"Stop recording error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@lerobot_bp.route('/sessions', methods=['GET'])
def get_sessions():
    """Get active robot sessions"""
    return jsonify({
        "success": True,
        "sessions": robot_sessions
    })

# ============================================================================
# LeRobot Implementation Functions
# ============================================================================

def _get_robot_config(robot_type, port, role="follower", cameras=None):
    """Get robot configuration based on type and role"""
    robot_configs = {
        'so101': {
            'follower': SO101FollowerConfig,
            'leader': SO101LeaderConfig
        },
        'so100': {
            'follower': SO100FollowerConfig,
            'leader': SO100LeaderConfig
        },
        'koch': {
            'follower': KochFollowerConfig,
            'leader': KochLeaderConfig
        }
    }
    
    robot_classes = {
        'so101': {
            'follower': SO101Follower,
            'leader': SO101Leader
        },
        'so100': {
            'follower': SO100Follower,
            'leader': SO100Leader
        },
        'koch': {
            'follower': KochFollower,
            'leader': KochLeader
        }
    }
    
    if robot_type not in robot_configs:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    
    if role not in robot_configs[robot_type]:
        raise ValueError(f"Unsupported role: {role} for robot type: {robot_type}")
    
    config_class = robot_configs[robot_type][role]
    robot_class = robot_classes[robot_type][role]
    
    # Create configuration
    config_kwargs = {
        "port": port,
        "id": f"{robot_type}_{role}",
    }
    
    if cameras and role == "follower":
        config_kwargs["cameras"] = cameras
    
    config = config_class(**config_kwargs)
    
    return config, robot_class

def _run_calibration(robot_type, role, port):
    """Run calibration in background thread"""
    try:
        if socketio:
            socketio.emit('calibration_log', {
                'message': f'Starting {role} calibration for {robot_type}',
                'level': 'info'
            })
        
        config, robot_class = _get_robot_config(robot_type, port, role)
        robot = robot_class(config)
        
        if socketio:
            socketio.emit('calibration_log', {
                'message': f'Connecting to {role} on {port}...',
                'level': 'info'
            })
        
        robot.connect(calibrate=False)
        
        if socketio:
            socketio.emit('calibration_log', {
                'message': f'Starting calibration...',
                'level': 'info'
            })
        
        robot.calibrate()
        
        if socketio:
            socketio.emit('calibration_log', {
                'message': f'Calibration completed successfully!',
                'level': 'success'
            })
        
        robot.disconnect()
        
    except Exception as e:
        error_msg = f"Calibration failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        if socketio:
            socketio.emit('calibration_log', {
                'message': error_msg,
                'level': 'error'
            })

def _run_teleoperation(robot_type, leader_port, follower_port, session_id):
    """Run teleoperation in background thread"""
    try:
        if socketio:
            socketio.emit('teleoperation_log', {
                'message': f'Starting teleoperation session {session_id}',
                'level': 'info'
            })
        
        # Create robot configurations
        follower_config, follower_class = _get_robot_config(robot_type, follower_port, "follower")
        leader_config, leader_class = _get_robot_config(robot_type, leader_port, "leader")
        
        # Create robot instances
        robot = follower_class(follower_config)
        teleop_device = leader_class(leader_config)
        
        if socketio:
            socketio.emit('teleoperation_log', {
                'message': 'Connecting to robots...',
                'level': 'info'
            })
        
        robot.connect()
        teleop_device.connect()
        
        robot_sessions[session_id]["status"] = "running"
        
        if socketio:
            socketio.emit('teleoperation_log', {
                'message': 'Teleoperation started! Move the leader arm to control the follower.',
                'level': 'success'
            })
        
        # Main teleoperation loop
        while (session_id in robot_sessions and 
               robot_sessions[session_id]["status"] == "running"):
            
            action = teleop_device.get_action()
            robot.send_action(action)
            time.sleep(0.01)  # Small delay to prevent overwhelming
        
        if socketio:
            socketio.emit('teleoperation_log', {
                'message': 'Teleoperation stopped',
                'level': 'info'
            })
        
        robot.disconnect()
        teleop_device.disconnect()
        
        if session_id in robot_sessions:
            del robot_sessions[session_id]
        
    except Exception as e:
        error_msg = f"Teleoperation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        if socketio:
            socketio.emit('teleoperation_log', {
                'message': error_msg,
                'level': 'error'
            })
        
        if session_id in robot_sessions:
            del robot_sessions[session_id]

def _run_recording(robot_type, leader_port, follower_port, repo_id, 
                   num_episodes, fps, episode_time_sec, reset_time_sec, 
                   task_description, session_id):
    """Run dataset recording in background thread"""
    try:
        if socketio:
            socketio.emit('recording_log', {
                'message': f'Starting recording session {session_id}',
                'level': 'info'
            })
        
        # Create camera configuration (basic setup)
        camera_config = {
            "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=fps)
        }
        
        # Create robot configurations
        follower_config, follower_class = _get_robot_config(robot_type, follower_port, "follower", camera_config)
        leader_config, leader_class = _get_robot_config(robot_type, leader_port, "leader")
        
        # Create robot instances
        robot = follower_class(follower_config)
        teleop = leader_class(leader_config)
        
        if socketio:
            socketio.emit('recording_log', {
                'message': 'Setting up dataset...',
                'level': 'info'
            })
        
        # Configure dataset features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}
        
        # Create dataset
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        
        # Initialize keyboard listener and visualization
        _, events = init_keyboard_listener()
        _init_rerun(session_name="recording")
        
        if socketio:
            socketio.emit('recording_log', {
                'message': 'Connecting to robots...',
                'level': 'info'
            })
        
        robot.connect()
        teleop.connect()
        
        robot_sessions[session_id]["status"] = "running"
        
        episode_idx = 0
        while (episode_idx < num_episodes and 
               session_id in robot_sessions and
               robot_sessions[session_id]["status"] == "running" and
               not events["stop_recording"]):
            
            robot_sessions[session_id]["current_episode"] = episode_idx + 1
            
            if socketio:
                socketio.emit('recording_log', {
                    'message': f'Recording episode {episode_idx + 1} of {num_episodes}',
                    'level': 'info'
                })
            
            record_loop(
                robot=robot,
                events=events,
                fps=fps,
                teleop=teleop,
                dataset=dataset,
                control_time_s=episode_time_sec,
                single_task=task_description,
                display_data=True,
            )
            
            # Reset environment if not stopping
            if (session_id in robot_sessions and
                robot_sessions[session_id]["status"] == "running" and
                not events["stop_recording"] and 
                (episode_idx < num_episodes - 1 or events["rerecord_episode"])):
                
                if socketio:
                    socketio.emit('recording_log', {
                        'message': 'Reset the environment',
                        'level': 'info'
                    })
                
                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop=teleop,
                    control_time_s=reset_time_sec,
                    single_task=task_description,
                    display_data=True,
                )
            
            if events["rerecord_episode"]:
                if socketio:
                    socketio.emit('recording_log', {
                        'message': 'Re-recording episode',
                        'level': 'info'
                    })
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            dataset.save_episode()
            episode_idx += 1
        
        if socketio:
            socketio.emit('recording_log', {
                'message': 'Uploading dataset to Hugging Face Hub...',
                'level': 'info'
            })
        
        robot.disconnect()
        teleop.disconnect()
        dataset.push_to_hub()
        
        if socketio:
            socketio.emit('recording_log', {
                'message': f'Recording completed! Dataset uploaded to {repo_id}',
                'level': 'success'
            })
        
        if session_id in robot_sessions:
            del robot_sessions[session_id]
        
    except Exception as e:
        error_msg = f"Recording failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        if socketio:
            socketio.emit('recording_log', {
                'message': error_msg,
                'level': 'error'
            })
        
        if session_id in robot_sessions:
            del robot_sessions[session_id]
