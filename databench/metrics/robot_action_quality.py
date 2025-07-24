"""
Robot action quality metric for evaluating the quality of robot action and state data.

This metric analyzes various aspects of robot action data including smoothness,
joint limits compliance, state-action consistency, gripper behavior, and physical feasibility.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config

logger = logging.getLogger(__name__)

class RobotActionQualityMetric:
    """Evaluates the quality of robot action and state data."""
    
    def __init__(self, robot_type: str = "generic_6dof", config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the robot action quality metric.
        
        Args:
            robot_type: Type of robot to use for configuration (e.g., 'franka_panda', 'ur5', 'generic_6dof')
            config_overrides: Optional configuration overrides
        """
        # Load configuration
        self.config = get_config('robot_action_quality')
        self.general_config = get_config('general')
        
        # Apply any overrides
        if config_overrides:
            # This would require implementing override logic in the config classes
            # For now, we'll apply them manually
            pass
        
        # Set robot-specific configuration
        self.robot_type = robot_type
        self._setup_robot_config()
        
    def _setup_robot_config(self):
        """Setup robot-specific configuration parameters."""
        # Try to get robot-specific config
        from scripts.config_loader import get_config_loader
        loader = get_config_loader()
        robot_config = loader.get_robot_config(self.robot_type)
        
        if robot_config:
            logger.info(f"Using robot-specific configuration for {self.robot_type}")
            self.dof = robot_config.dof
            self.joint_limits_array = robot_config.joint_limits
            self.gripper_limits = robot_config.gripper_limits
            self.max_velocity = robot_config.max_velocity
            self.robot_config = robot_config # Store for use in _analyze_episode_gripper
        else:
            logger.warning(f"Robot type '{self.robot_type}' not found in config, using defaults")
            # Use default configuration based on detected DOF
            self._setup_default_config()
    
    def _setup_default_config(self):
        """Setup default configuration when robot-specific config is not available."""
        # Use 6DOF as default initially - will be updated dynamically from data
        if '6dof' in self.config.joint_limits:
            joint_config = self.config.joint_limits['6dof']
            self.joint_limits_dict = joint_config
            self.joint_limits_array = [list(joint_config[joint]) for joint in joint_config.keys()]
            self.dof = 6
        else:
            # Fallback to absolute default
            self.joint_limits_array = [self.config.default_joint_limits] * 6
            self.dof = 6
        
        self.gripper_limits = [0.0, 1.0]  # Default gripper limits
        self.max_velocity = 2.0  # Default max velocity
        
    def _setup_dynamic_config_from_data(self, actions: np.ndarray):
        """Setup configuration dynamically based on actual data shape."""
        if hasattr(self, 'robot_config') and self.robot_config:
            # Already have robot-specific config, don't override
            return
            
        detected_dof = actions.shape[1]
        if detected_dof != self.dof:
            logger.info(f"Detected {detected_dof} DOF from data (was expecting {self.dof}), updating configuration")
            self.dof = detected_dof
            
            # Extend or truncate joint limits as needed
            if detected_dof > len(self.joint_limits_array):
                # Extend with default limits
                default_limit = self.config.default_joint_limits if hasattr(self.config, 'default_joint_limits') else [-3.14, 3.14]
                while len(self.joint_limits_array) < detected_dof:
                    self.joint_limits_array.append(default_limit)
                logger.info(f"Extended joint limits to {detected_dof} DOF using default limits {default_limit}")
            elif detected_dof < len(self.joint_limits_array):
                # Truncate limits
                self.joint_limits_array = self.joint_limits_array[:detected_dof]
                logger.info(f"Truncated joint limits to {detected_dof} DOF")
        
    def compute(self, dataset: List[Dict], data_path: Path, embedding_manager) -> float:
        """
        Compute robot action quality metrics.
        
        Args:
            dataset: List of episode dictionaries
            data_path: Path to dataset directory
            embedding_manager: Embedding manager for processing
            
        Returns:
            Float score between 0 and 1
        """
        logger.info("Computing robot action quality...")
        
        # Check if this is a HuggingFace dataset
        is_hf_dataset = any(sample.get('data_path', '').startswith('hf_episode_') for sample in dataset)
        
        if is_hf_dataset:
            # Use direct episode data analysis for HuggingFace datasets
            return self._compute_hf_dataset_quality(dataset)
        else:
            # Use file-based analysis for local datasets
            return self._compute_local_dataset_quality(dataset, data_path)
    
    def _compute_hf_dataset_quality(self, dataset: List[Dict]) -> float:
        """Compute quality metrics for HuggingFace datasets using direct episode data."""
        smoothness_scores = []
        limits_scores = []
        consistency_scores = []
        gripper_scores = []
        feasibility_scores = []
        
        for sample in dataset:
            if 'actions' in sample:
                actions = sample['actions']
                if not isinstance(actions, np.ndarray):
                    actions = np.array(actions)
                
                # Smoothness analysis
                smoothness = self._analyze_smoothness_from_data(actions)
                if smoothness is not None:
                    smoothness_scores.append(smoothness)
                
                # Joint limits analysis
                limits = self._analyze_limits_from_data(actions)
                if limits is not None:
                    limits_scores.append(limits)
                
                # State-action consistency (if observations available)
                if 'observations' in sample:
                    observations = sample['observations']
                    if not isinstance(observations, np.ndarray):
                        observations = np.array(observations)
                    consistency = self._analyze_consistency_from_data(actions, observations)
                    if consistency is not None:
                        consistency_scores.append(consistency)
                
                # Gripper behavior analysis
                gripper = self._analyze_gripper_from_data(actions)
                if gripper is not None:
                    gripper_scores.append(gripper)
                
                # Physical feasibility
                feasibility = self._analyze_feasibility_from_data(actions)
                if feasibility is not None:
                    feasibility_scores.append(feasibility)
        
        return self._combine_quality_scores(
            smoothness_scores, limits_scores, consistency_scores, 
            gripper_scores, feasibility_scores
        )
    
    def _compute_local_dataset_quality(self, dataset: List[Dict], data_path: Path) -> float:
        """Compute quality metrics for local datasets using file-based analysis."""
        smoothness_scores = []
        limits_scores = []
        consistency_scores = []
        gripper_scores = []
        feasibility_scores = []
        
        for sample in dataset:
            episode_data_path = sample.get('data_path')
            if episode_data_path:
                # Smoothness analysis
                smoothness = self._analyze_episode_smoothness(episode_data_path, data_path)
                if smoothness is not None:
                    smoothness_scores.append(smoothness)
                
                # Joint limits analysis
                limits = self._analyze_episode_limits(episode_data_path, data_path)
                if limits is not None:
                    limits_scores.append(limits)
                
                # State-action consistency
                consistency = self._analyze_episode_consistency(episode_data_path, data_path)
                if consistency is not None:
                    consistency_scores.append(consistency)
                
                # Gripper behavior analysis
                gripper = self._analyze_episode_gripper(episode_data_path, data_path)
                if gripper is not None:
                    gripper_scores.append(gripper)
                
                # Physical feasibility
                feasibility = self._compute_episode_feasibility(episode_data_path, data_path)
                if feasibility is not None:
                    feasibility_scores.append(feasibility)
        
        return self._combine_quality_scores(
            smoothness_scores, limits_scores, consistency_scores, 
            gripper_scores, feasibility_scores
        )
    
    def _combine_quality_scores(self, smoothness_scores, limits_scores, consistency_scores, 
                               gripper_scores, feasibility_scores) -> float:
        """Combine individual quality scores into final score."""
        # Calculate individual metric scores
        smoothness_score = np.mean(smoothness_scores) if smoothness_scores else None
        limits_score = np.mean(limits_scores) if limits_scores else None
        consistency_score = np.mean(consistency_scores) if consistency_scores else None
        gripper_score = np.mean(gripper_scores) if gripper_scores else None
        feasibility_score = np.mean(feasibility_scores) if feasibility_scores else None
        
        # Log individual scores
        if smoothness_score is not None:
            logger.info(f"Action smoothness: {smoothness_score:.3f}")
        else:
            logger.warning("Action smoothness could not be evaluated")
        
        if limits_score is not None:
            logger.info(f"Joint limits conformity: {limits_score:.3f}")
        else:
            logger.warning("Joint limits conformity could not be evaluated")
        
        if consistency_score is not None:
            logger.info(f"State-action consistency: {consistency_score:.3f}")
        else:
            logger.warning("State-action consistency could not be evaluated")
        
        if gripper_score is not None:
            logger.info(f"Gripper behavior: {gripper_score:.3f}")
        else:
            logger.warning("Gripper behavior could not be evaluated")
        
        if feasibility_score is not None:
            logger.info(f"Physical feasibility: {feasibility_score:.3f}")
        else:
            logger.warning("Physical feasibility could not be evaluated")
        
        # Combine available scores
        available_scores = [score for score in [smoothness_score, limits_score, consistency_score, gripper_score, feasibility_score] if score is not None]
        
        if not available_scores:
            logger.error("No robot action quality metrics could be evaluated")
            return 0.0
        
        # Weight the scores
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # smoothness, limits, consistency, gripper, feasibility
        scores = [smoothness_score or 0.5, limits_score or 0.5, consistency_score or 0.5, gripper_score or 0.5, feasibility_score or 0.5]
        
        # Only use weights for available scores
        if len(available_scores) < 5:
            return float(np.mean(available_scores))
        
        final_score = sum(w * s for w, s in zip(weights, scores))
        return float(final_score)
    
    def _analyze_smoothness_from_data(self, actions: np.ndarray) -> Optional[float]:
        """Analyze action smoothness directly from action array."""
        try:
            if len(actions) < 3:
                return None
            
            smoothness_scores = []
            
            # For each joint/dimension
            for dim in range(actions.shape[1]):
                action_dim = actions[:, dim]
                
                # Calculate velocity and acceleration
                velocity = np.diff(action_dim)
                acceleration = np.diff(velocity)
                
                if len(acceleration) < 1:
                    continue
                
                # Smoothness metric: lower acceleration variance = smoother
                if np.std(acceleration) > 1e-6:
                    action_range = np.ptp(action_dim)
                    if action_range > 1e-6:
                        smoothness = 1.0 / (1.0 + np.std(acceleration) / action_range)
                    else:
                        smoothness = 1.0  # Constant action is perfectly smooth
                    smoothness_scores.append(smoothness)
            
            return np.mean(smoothness_scores) if smoothness_scores else None
            
        except Exception as e:
            logger.warning(f"Error analyzing smoothness from data: {e}")
            return None
    
    def _analyze_limits_from_data(self, actions: np.ndarray) -> Optional[float]:
        """Analyze joint limits compliance directly from action array."""
        try:
            if len(actions) == 0:
                return None
            
            # Use robot-specific limits if available
            if hasattr(self, 'joint_limits') and self.joint_limits:
                limits_scores = []
                
                for dim in range(min(actions.shape[1], len(self.joint_limits))):
                    action_dim = actions[:, dim]
                    min_limit, max_limit = self.joint_limits[dim]
                    
                    # Check violations
                    violations = np.sum((action_dim < min_limit) | (action_dim > max_limit))
                    compliance = 1.0 - (violations / len(action_dim))
                    limits_scores.append(compliance)
                
                return np.mean(limits_scores) if limits_scores else None
            else:
                # Generic limits analysis - check for extreme values
                action_ranges = np.ptp(actions, axis=0)
                action_stds = np.std(actions, axis=0)
                
                # Assume actions should be within reasonable range
                # Flag values that are more than 3 standard deviations from mean
                limits_scores = []
                for dim in range(actions.shape[1]):
                    action_dim = actions[:, dim]
                    mean_val = np.mean(action_dim)
                    std_val = np.std(action_dim)
                    
                    if std_val > 1e-6:
                        outliers = np.abs(action_dim - mean_val) > 3 * std_val
                        compliance = 1.0 - (np.sum(outliers) / len(action_dim))
                        limits_scores.append(compliance)
                
                return np.mean(limits_scores) if limits_scores else None
                
        except Exception as e:
            logger.warning(f"Error analyzing limits from data: {e}")
            return None
    
    def _analyze_consistency_from_data(self, actions: np.ndarray, observations: np.ndarray) -> Optional[float]:
        """Analyze state-action consistency directly from data arrays."""
        try:
            if len(actions) < 2 or len(observations) < 2:
                return None
            
            # Ensure same length
            min_len = min(len(actions), len(observations))
            actions = actions[:min_len]
            observations = observations[:min_len]
            
            consistency_scores = []
            
            # For each dimension, check if actions correlate with state changes
            for dim in range(min(actions.shape[1], observations.shape[1])):
                action_dim = actions[:, dim]
                obs_dim = observations[:, dim]
                
                # Calculate state changes
                state_changes = np.diff(obs_dim)
                action_commands = action_dim[:-1]
                
                # Check correlation
                if np.std(action_commands) > 1e-6 and np.std(state_changes) > 1e-6:
                    correlation = np.corrcoef(action_commands, state_changes)[0, 1]
                    if not np.isnan(correlation):
                        consistency = (correlation + 1) / 2  # Convert to [0, 1]
                        consistency_scores.append(consistency)
            
            return np.mean(consistency_scores) if consistency_scores else None
            
        except Exception as e:
            logger.warning(f"Error analyzing consistency from data: {e}")
            return None
    
    def _analyze_gripper_from_data(self, actions: np.ndarray) -> Optional[float]:
        """Analyze gripper behavior directly from action array."""
        try:
            if len(actions) == 0:
                return None
            
            # Assume last dimension is gripper (common convention)
            if actions.shape[1] > 0:
                gripper_actions = actions[:, -1]
                
                # Check for reasonable gripper behavior
                # Gripper should have discrete states (open/close)
                unique_values = np.unique(gripper_actions)
                
                # Good gripper behavior: few unique values, smooth transitions
                if len(unique_values) <= 10:  # Reasonable number of discrete states
                    # Check for smooth transitions
                    gripper_changes = np.diff(gripper_actions)
                    large_changes = np.abs(gripper_changes) > 0.5
                    
                    # Penalize too many abrupt changes
                    stability = 1.0 - (np.sum(large_changes) / len(gripper_changes))
                    return stability
                else:
                    # Too many unique values might indicate noisy gripper control
                    return 0.5
            
            return None
            
        except Exception as e:
            logger.warning(f"Error analyzing gripper from data: {e}")
            return None
    
    def _analyze_feasibility_from_data(self, actions: np.ndarray) -> Optional[float]:
        """Analyze physical feasibility directly from action array."""
        try:
            if len(actions) < 2:
                return None
            
            feasibility_scores = []
            
            # Check for physically reasonable action magnitudes
            action_magnitudes = np.linalg.norm(actions, axis=1)
            
            # Actions should not be too large (unrealistic)
            max_reasonable_action = 10.0  # Adjust based on robot type
            reasonable_actions = action_magnitudes <= max_reasonable_action
            magnitude_feasibility = np.mean(reasonable_actions)
            feasibility_scores.append(magnitude_feasibility)
            
            # Check for reasonable action changes (not too abrupt)
            action_changes = np.diff(actions, axis=0)
            change_magnitudes = np.linalg.norm(action_changes, axis=1)
            
            max_reasonable_change = 2.0  # Adjust based on robot type
            reasonable_changes = change_magnitudes <= max_reasonable_change
            change_feasibility = np.mean(reasonable_changes)
            feasibility_scores.append(change_feasibility)
            
            return np.mean(feasibility_scores) if feasibility_scores else None
            
        except Exception as e:
            logger.warning(f"Error analyzing feasibility from data: {e}")
            return None
    
    def _analyze_action_smoothness(self, dataset: List[Dict], data_path: Path) -> float:
        """Analyze action smoothness across episodes."""
        smoothness_scores = []
        
        # Sample episodes for analysis
        sample_size = min(20, len(dataset))
        for i in range(sample_size):
            episode = dataset[i]
            
            if 'data_path' in episode:
                try:
                    # Check if it's a HuggingFace dataset
                    if episode['data_path'].startswith('hf_episode_'):
                        # For HF datasets, compute smoothness from direct episode data
                        score = self._compute_hf_episode_smoothness(episode)
                    else:
                        # For local datasets, compute from file
                        score = self._compute_episode_smoothness(episode['data_path'], data_path)
                    
                    if score is not None:
                        smoothness_scores.append(score)
                except Exception as e:
                    logger.warning(f"Could not analyze smoothness for episode {i}: {e}")
        
        if not smoothness_scores:
            logger.warning("No episodes with valid data found for smoothness analysis")
            return None  # Cannot evaluate without data
        
        return float(np.mean(smoothness_scores))
    
    def _compute_hf_episode_smoothness(self, episode_data: Dict[str, Any]) -> Optional[float]:
        """Compute smoothness score for a HuggingFace episode."""
        try:
            if 'actions' not in episode_data:
                return None
            
            actions = episode_data['actions']
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            
            # Setup dynamic configuration based on actual data
            self._setup_dynamic_config_from_data(actions)
            
            if len(actions) < 3:
                return None
            
            smoothness_scores = []
            
            # For each joint dimension, compute smoothness
            for joint_idx in range(actions.shape[1]):
                action_joint = actions[:, joint_idx]
                
                # Compute first and second derivatives (velocity and acceleration)
                velocity = np.diff(action_joint)
                acceleration = np.diff(velocity)
                
                if len(acceleration) < 1:
                    continue
                
                # Compute smoothness metric (lower acceleration variance = smoother)
                if np.std(acceleration) > 1e-6:
                    # Normalize by action range to make scale-invariant
                    action_range = np.ptp(action_joint)
                    if action_range > 1e-6:
                        smoothness = 1.0 / (1.0 + np.std(acceleration) / action_range)
                    else:
                        smoothness = 1.0  # Constant action is perfectly smooth
                    smoothness_scores.append(smoothness)
            
            return np.mean(smoothness_scores) if smoothness_scores else None
            
        except Exception as e:
            logger.warning(f"Error computing HF episode smoothness: {e}")
            return None
    
    def _compute_episode_smoothness(self, episode_data_path: str, dataset_data_path: Path) -> Optional[float]:
        """Compute smoothness score for a single episode."""
        try:
            # Handle HuggingFace datasets - skip file-based loading
            if episode_data_path.startswith('hf_episode_'):
                logger.debug(f"Skipping smoothness check for HF dataset episode: {episode_data_path}")
                return None
            
            # Handle relative paths with data_path
            if dataset_data_path and not Path(episode_data_path).is_absolute():
                full_data_path = dataset_data_path / episode_data_path
            else:
                full_data_path = Path(episode_data_path)
                
            df = pd.read_parquet(str(full_data_path))
            
            # Check for array-format action column first
            if 'action' in df.columns:
                actions = np.stack(df['action'].values)  # Convert list of arrays to 2D array
                
                # Setup dynamic configuration based on actual data
                self._setup_dynamic_config_from_data(actions)
                
                if len(actions) < 3:
                    return None
                
                smoothness_scores = []
                
                # For each joint dimension, compute smoothness using jerk
                for joint_idx in range(actions.shape[1]):
                    action_joint = actions[:, joint_idx]
                    
                    # Compute derivatives: velocity, acceleration, jerk
                    velocity = np.diff(action_joint)
                    acceleration = np.diff(velocity)
                    jerk = np.diff(acceleration)
                    
                    if len(jerk) < 1:
                        continue
                    
                    # Smoothness metric based on jerk (lower jerk = smoother)
                    rms_jerk = np.sqrt(np.mean(jerk**2))
                    
                    # Normalize by action range to make scale-invariant
                    action_range = np.ptp(action_joint)
                    if action_range > 1e-6:
                        normalized_jerk = rms_jerk / action_range
                        # Convert to 0-1 score (lower jerk = higher score)
                        smoothness = 1.0 / (1.0 + normalized_jerk)
                    else:
                        smoothness = 1.0  # Constant action is perfectly smooth
                    
                    smoothness_scores.append(smoothness)
                
                return np.mean(smoothness_scores) if smoothness_scores else None
            
            # Fallback: Find individual action columns (legacy format)
            action_cols = [col for col in df.columns if col.startswith('action')]
            if not action_cols:
                return None
            
            smoothness_scores = []
            
            for action_col in action_cols:
                if action_col in df.columns:
                    actions = df[action_col].values
                    
                    if len(actions) < 3:
                        continue
                    
                    # Compute derivatives
                    velocity = np.diff(actions)
                    acceleration = np.diff(velocity)
                    jerk = np.diff(acceleration)
                    
                    if len(jerk) < 1:
                        continue
                    
                    # Smoothness metric
                    rms_jerk = np.sqrt(np.mean(jerk**2))
                    action_range = np.ptp(actions)
                    
                    if action_range > 1e-6:
                        normalized_jerk = rms_jerk / action_range
                        smoothness = 1.0 / (1.0 + normalized_jerk)
                    else:
                        smoothness = 1.0
                    
                    smoothness_scores.append(smoothness)
            
            return np.mean(smoothness_scores) if smoothness_scores else None
            
        except Exception as e:
            logger.warning(f"Error computing smoothness for {episode_data_path}: {e}")
            return None
    
    def _analyze_joint_limits(self, dataset: List[Dict], data_path: Path) -> float:
        """Analyze if actions stay within joint limits."""
        limits_scores = []
        
        # Sample episodes for analysis
        sample_size = min(20, len(dataset))
        for i in range(sample_size):
            episode = dataset[i]
            
            if 'data_path' in episode:
                try:
                    score = self._check_episode_limits(episode['data_path'], data_path)
                    if score is not None:
                        limits_scores.append(score)
                except Exception as e:
                    logger.warning(f"Could not analyze limits for episode {i}: {e}")
        
        if not limits_scores:
            logger.warning("No episodes with valid data found for joint limits analysis")
            return None  # Cannot evaluate without data
        
        return float(np.mean(limits_scores))
    
    def _check_episode_limits(self, episode_data_path: str, dataset_data_path: Path) -> Optional[float]:
        """Check if actions stay within limits for a single episode."""
        try:
            # Handle HuggingFace datasets - skip file-based loading
            if episode_data_path.startswith('hf_episode_'):
                logger.debug(f"Skipping limits check for HF dataset episode: {episode_data_path}")
                return None
            
            # Handle relative paths with data_path
            if dataset_data_path and not Path(episode_data_path).is_absolute():
                full_data_path = dataset_data_path / episode_data_path
            else:
                full_data_path = Path(episode_data_path)
                
            df = pd.read_parquet(str(full_data_path))
            
            # Check for array-format action column first
            if 'action' in df.columns:
                actions = np.stack(df['action'].values)  # Convert list of arrays to 2D array
                
                # Setup dynamic configuration based on actual data
                self._setup_dynamic_config_from_data(actions)
                
                violations = []
                
                # For each joint dimension, check against known limits
                for joint_idx in range(actions.shape[1]):
                    action_joint = actions[:, joint_idx]
                    
                    # Check against common robotic joint limits (degrees)
                    if joint_idx < len(self.joint_limits_array):
                        min_limit, max_limit = self.joint_limits_array[joint_idx]
                        
                        # Count violations
                        below_min = np.sum(action_joint < min_limit)
                        above_max = np.sum(action_joint > max_limit)
                        total_violations = below_min + above_max
                        
                        violation_rate = total_violations / len(action_joint)
                        violations.append(violation_rate)
                    else:
                        # For unknown joints, cannot evaluate without proper limits
                        logger.warning(f"No joint limits defined for action column, cannot evaluate physical validity")
                        continue  # Skip this joint instead of making assumptions
                
                if violations:
                    avg_violation_rate = np.mean(violations)
                    return 1.0 - avg_violation_rate  # Higher score for fewer violations
                else:
                    if len(actions) > 0:
                        logger.warning("No joint limits available for evaluation - cannot assess physical validity")
                        return None  # Cannot evaluate without limits
                    else:
                        return 1.0  # Perfect score if no violations
            
            # Fallback: Find individual action columns (legacy format)
            action_cols = [col for col in df.columns if col.startswith('action')]
            if not action_cols:
                return None
            
            violations = []
            
            for action_col in action_cols:
                if action_col in df.columns:
                    actions = df[action_col].values
                    
                    # Extract joint name from column
                    joint_name = action_col.split('.')[-1] if '.' in action_col else action_col.replace('action', '')
                    
                    if joint_name in self.joint_limits_dict:
                        min_limit, max_limit = self.joint_limits_dict[joint_name]
                        
                        # Count violations
                        below_min = np.sum(actions < min_limit)
                        above_max = np.sum(actions > max_limit)
                        total_violations = below_min + above_max
                        
                        violation_rate = total_violations / len(actions)
                        violations.append(violation_rate)
            
            if violations:
                avg_violation_rate = np.mean(violations)
                return 1.0 - avg_violation_rate
            else:
                return 1.0
            
        except Exception as e:
            logger.warning(f"Error checking limits for {episode_data_path}: {e}")
            return None
    
    def _analyze_state_action_consistency(self, dataset: List[Dict], data_path: Path) -> float:
        """Analyze consistency between states and actions."""
        consistency_scores = []
        
        # Sample episodes for analysis
        sample_size = min(20, len(dataset))
        for i in range(sample_size):
            episode = dataset[i]
            
            if 'data_path' in episode:
                try:
                    # Check if it's a HuggingFace dataset
                    if episode['data_path'].startswith('hf_episode_'):
                        # For HF datasets, compute consistency from direct episode data
                        score = self._compute_hf_episode_consistency(episode)
                    else:
                        # For local datasets, compute from file
                        score = self._compute_episode_consistency(episode['data_path'], data_path)
                    
                    if score is not None:
                        consistency_scores.append(score)
                except Exception as e:
                    logger.warning(f"Could not analyze consistency for episode {i}: {e}")
        
        if not consistency_scores:
            logger.warning("No episodes with valid data found for state-action consistency analysis")
            return None  # Cannot evaluate without data
        
        return float(np.mean(consistency_scores))
    
    def _compute_hf_episode_consistency(self, episode_data: Dict[str, Any]) -> Optional[float]:
        """Compute state-action consistency for a HuggingFace episode."""
        try:
            if 'actions' not in episode_data or 'observations' not in episode_data:
                return None
            
            actions = episode_data['actions']
            states = episode_data['observations']
            
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            if not isinstance(states, np.ndarray):
                states = np.array(states)
            
            # Setup dynamic configuration based on actual data
            self._setup_dynamic_config_from_data(actions)
            
            if len(actions) < 2 or len(states) < 2:
                return None
            
            # Ensure same length
            min_len = min(len(actions), len(states))
            actions = actions[:min_len]
            states = states[:min_len]
            
            consistency_scores = []
            
            # For each joint dimension, check if actions lead to expected state changes
            for joint_idx in range(min(actions.shape[1], states.shape[1])):
                action_joint = actions[:, joint_idx]
                state_joint = states[:, joint_idx]
                
                # Compute state changes
                state_changes = np.diff(state_joint)
                action_commands = action_joint[:-1]
                
                # Check correlation between actions and state changes
                if np.std(action_commands) > 1e-6 and np.std(state_changes) > 1e-6:
                    correlation = np.corrcoef(action_commands, state_changes)[0, 1]
                    if not np.isnan(correlation):
                        # Convert correlation to [0, 1] range
                        consistency = (correlation + 1) / 2
                        consistency_scores.append(consistency)
            
            return np.mean(consistency_scores) if consistency_scores else None
            
        except Exception as e:
            logger.warning(f"Error computing HF episode consistency: {e}")
            return None
    
    def _compute_episode_consistency(self, episode_data_path: str, dataset_data_path: Path) -> Optional[float]:
        """Compute state-action consistency for a single episode."""
        try:
            # Handle HuggingFace datasets - skip file-based loading
            if episode_data_path.startswith('hf_episode_'):
                logger.debug(f"Skipping consistency check for HF dataset episode: {episode_data_path}")
                return None
            
            # Handle relative paths with data_path
            if dataset_data_path and not Path(episode_data_path).is_absolute():
                full_data_path = dataset_data_path / episode_data_path
            else:
                full_data_path = Path(episode_data_path)
                
            df = pd.read_parquet(str(full_data_path))
            
            # Check for array-format action and state columns first
            if 'action' in df.columns and 'observation.state' in df.columns:
                actions = np.stack(df['action'].values)  # Convert list of arrays to 2D array
                states = np.stack(df['observation.state'].values)
                
                # Setup dynamic configuration based on actual data
                self._setup_dynamic_config_from_data(actions)
                
                if len(actions) < 2:
                    return None
                
                consistency_scores = []
                
                # For each joint dimension, check action-state consistency
                for joint_idx in range(actions.shape[1]):
                    action_joint = actions[:, joint_idx]
                    state_joint = states[:, joint_idx]
                    
                    # Check if actions and states are similar (they should be for good consistency)
                    if len(action_joint) == len(state_joint):
                        # Compute correlation between actions and states
                        if np.std(action_joint) > 1e-6 and np.std(state_joint) > 1e-6:
                            correlation = np.corrcoef(action_joint, state_joint)[0, 1]
                            if not np.isnan(correlation):
                                # Convert correlation to [0, 1] range
                                consistency = (correlation + 1) / 2
                                consistency_scores.append(consistency)
                        
                        # Also check if the difference between action and state is small
                        diff = np.abs(action_joint - state_joint)
                        action_range = np.ptp(action_joint)
                        if action_range > 1e-6:
                            relative_diff = np.mean(diff) / action_range
                            # Lower relative difference = higher consistency
                            diff_consistency = 1.0 / (1.0 + relative_diff)
                            consistency_scores.append(diff_consistency)
                
                return np.mean(consistency_scores) if consistency_scores else None
            
            # Fallback: Find individual action and state columns (legacy format)
            action_cols = [col for col in df.columns if col.startswith('action')]
            state_cols = [col for col in df.columns if col.startswith('observation.state')]
            
            if not action_cols or not state_cols:
                return None
            
            consistency_scores = []
            
            # Match action and state columns for each joint
            for action_col in action_cols:
                joint_name = action_col.split('.')[-1] if '.' in action_col else action_col.replace('action', '')
                
                # Find matching state column
                state_col = None
                for col in state_cols:
                    if joint_name in col:
                        state_col = col
                        break
                
                if state_col and action_col in df.columns and state_col in df.columns:
                    actions = df[action_col].values
                    states = df[state_col].values
                    
                    if len(actions) == len(states) and len(actions) > 1:
                        # Correlation check
                        if np.std(actions) > 1e-6 and np.std(states) > 1e-6:
                            correlation = np.corrcoef(actions, states)[0, 1]
                            if not np.isnan(correlation):
                                consistency = (correlation + 1) / 2
                                consistency_scores.append(consistency)
                        
                        # Difference check
                        diff = np.abs(actions - states)
                        action_range = np.ptp(actions)
                        if action_range > 1e-6:
                            relative_diff = np.mean(diff) / action_range
                            diff_consistency = 1.0 / (1.0 + relative_diff)
                            consistency_scores.append(diff_consistency)
            
            return np.mean(consistency_scores) if consistency_scores else None
            
        except Exception as e:
            logger.warning(f"Error computing consistency for {episode_data_path}: {e}")
            return None
    
    def _analyze_gripper_behavior(self, dataset: List[Dict], data_path: Path) -> float:
        """Analyze gripper opening/closing patterns."""
        gripper_scores = []
        
        # Sample episodes for analysis
        sample_size = min(20, len(dataset))
        for i in range(sample_size):
            episode = dataset[i]
            
            if 'data_path' in episode:
                try:
                    score = self._analyze_episode_gripper(episode['data_path'], data_path)
                    if score is not None:
                        gripper_scores.append(score)
                except Exception as e:
                    logger.warning(f"Could not analyze gripper for episode {i}: {e}")
        
        if not gripper_scores:
            logger.warning("No episodes with valid data found for gripper behavior analysis")
            return None  # Cannot evaluate without data
        
        return float(np.mean(gripper_scores))
    
    def _analyze_episode_gripper(self, episode_data_path: str, dataset_data_path: Path) -> Optional[float]:
        """Analyze gripper behavior in a single episode."""
        try:
            # Handle HuggingFace datasets - skip file-based loading
            if episode_data_path.startswith('hf_episode_'):
                logger.debug(f"Skipping gripper analysis for HF dataset episode: {episode_data_path}")
                return None
            
            # Handle relative paths with data_path
            if dataset_data_path and not Path(episode_data_path).is_absolute():
                full_data_path = dataset_data_path / episode_data_path
            else:
                full_data_path = Path(episode_data_path)
                
            df = pd.read_parquet(str(full_data_path))
            
            # Check for array-format action column first
            if 'action' in df.columns:
                actions = np.stack(df['action'].values)  # Convert list of arrays to 2D array
                
                # Setup dynamic configuration based on actual data
                self._setup_dynamic_config_from_data(actions)
                
                # Check if robot configuration specifies gripper joint
                gripper_joint = None
                if hasattr(self, 'robot_config') and self.robot_config:
                    gripper_joint = getattr(self.robot_config, 'gripper_joint', None)
                
                if gripper_joint is not None and gripper_joint < actions.shape[1]:
                    gripper_actions = actions[:, gripper_joint]
                elif actions.shape[1] >= 6:  # Common case: last dimension might be gripper
                    gripper_actions = actions[:, -1]  # Last dimension
                    logger.info("Assuming last dimension is gripper (no robot config specified)")
                else:
                    logger.warning("Cannot identify gripper joint - no robot configuration and insufficient action dimensions")
                    return None
                    
                # Analyze gripper behavior
                gripper_range = np.ptp(gripper_actions)
                
                # Check if gripper is binary (open/close) or continuous
                unique_values = np.unique(gripper_actions)
                
                if len(unique_values) <= 3:  # Likely binary or few discrete states
                    # For binary grippers, check for reasonable open/close patterns
                    changes = np.abs(np.diff(gripper_actions))
                    meaningful_changes = np.sum(changes > gripper_range * 0.1)
                    
                    # Good gripper behavior has some meaningful state changes
                    if len(gripper_actions) > 10:
                        change_rate = meaningful_changes / len(gripper_actions)
                        # Use configuration thresholds
                        min_change_rate = self.config.thresholds.get('gripper_change_rate_min', 0.05)
                        max_change_rate = self.config.thresholds.get('gripper_change_rate_max', 0.3)
                        if min_change_rate <= change_rate <= max_change_rate:
                            return 0.8  # Good binary gripper behavior
                        else:
                            return 0.2  # Poor gripper behavior
                    else:
                        logger.warning("Episode too short to evaluate gripper behavior")
                        return None
                
                else:  # Continuous gripper
                    # For continuous grippers, check for smoothness
                    velocity = np.abs(np.diff(gripper_actions))
                    avg_velocity = np.mean(velocity)
                    
                    # Smooth gripper control
                    if gripper_range > 1e-6:
                        normalized_velocity = avg_velocity / gripper_range
                        smoothness = 1.0 / (1.0 + normalized_velocity * 10)
                        return smoothness
                    else:
                        return 1.0  # Static gripper is perfectly smooth
                
                return self.general_config.default_score  # Neutral if no clear gripper dimension
            
            # Fallback: Find gripper column (legacy format)
            gripper_cols = [col for col in df.columns if 'gripper' in col.lower() or 'grip' in col.lower()]
            
            if not gripper_cols:
                return None
            
            gripper_col = gripper_cols[0]  # Use first gripper column
            gripper_values = df[gripper_col].values
            
            # Analyze gripper behavior
            gripper_range = np.ptp(gripper_values)
            unique_values = np.unique(gripper_values)
            
            if len(unique_values) <= 3:  # Binary gripper
                changes = np.abs(np.diff(gripper_values))
                meaningful_changes = np.sum(changes > gripper_range * 0.1)
                
                if len(gripper_values) > 10:
                    change_rate = meaningful_changes / len(gripper_values)
                    min_change_rate = self.config.thresholds.get('gripper_change_rate_min', 0.05)
                    max_change_rate = self.config.thresholds.get('gripper_change_rate_max', 0.3)
                    if min_change_rate <= change_rate <= max_change_rate:
                        return 0.8
                    else:
                        return 0.5
                else:
                    return 0.7
            
            else:  # Continuous gripper
                velocity = np.abs(np.diff(gripper_values))
                avg_velocity = np.mean(velocity)
                
                if gripper_range > 1e-6:
                    normalized_velocity = avg_velocity / gripper_range
                    smoothness = 1.0 / (1.0 + normalized_velocity * 10)
                    return smoothness
                else:
                    return 1.0
            
        except Exception as e:
            logger.warning(f"Error analyzing gripper for {episode_data_path}: {e}")
            return None
    
    def _analyze_physical_feasibility(self, dataset: List[Dict], data_path: Path) -> float:
        """Analyze physical feasibility of trajectories."""
        feasibility_scores = []
        
        # Sample episodes for analysis
        sample_size = min(20, len(dataset))
        for i in range(sample_size):
            episode = dataset[i]
            
            if 'data_path' in episode:
                try:
                    score = self._compute_episode_feasibility(episode['data_path'], data_path)
                    if score is not None:
                        feasibility_scores.append(score)
                except Exception as e:
                    logger.warning(f"Could not analyze feasibility for episode {i}: {e}")
        
        if not feasibility_scores:
            logger.warning("No episodes with valid data found for physical feasibility analysis")
            return None  # Cannot evaluate without data
        
        return float(np.mean(feasibility_scores))
    
    def _compute_episode_feasibility(self, episode_data_path: str, dataset_data_path: Path) -> Optional[float]:
        """Compute physical feasibility for a single episode."""
        try:
            # Handle HuggingFace datasets - skip file-based loading
            if episode_data_path.startswith('hf_episode_'):
                logger.debug(f"Skipping feasibility check for HF dataset episode: {episode_data_path}")
                return None
            
            # Handle relative paths with data_path
            if dataset_data_path and not Path(episode_data_path).is_absolute():
                full_data_path = dataset_data_path / episode_data_path
            else:
                full_data_path = Path(episode_data_path)
                
            df = pd.read_parquet(str(full_data_path))
            
            # Check for array-format action column first
            if 'action' in df.columns:
                actions = np.stack(df['action'].values)  # Convert list of arrays to 2D array
                
                # Setup dynamic configuration based on actual data
                self._setup_dynamic_config_from_data(actions)
                
                if len(actions) < 2:
                    return None
                
                feasibility_scores = []
                
                # For each joint dimension, check physical feasibility
                for joint_idx in range(actions.shape[1]):
                    action_joint = actions[:, joint_idx]
                    
                    # 1. Check for discontinuities (sudden large jumps)
                    velocities = np.abs(np.diff(action_joint))
                    action_range = np.ptp(action_joint)
                    
                    if action_range > 1e-6:
                        # Normalize velocities by action range
                        normalized_velocities = velocities / action_range
                        
                        # Flag large discontinuities (> threshold of range in one step)
                        discontinuity_threshold = self.config.thresholds.get('discontinuity_threshold', 0.2)
                        large_jumps = np.sum(normalized_velocities > discontinuity_threshold)
                        discontinuity_rate = large_jumps / len(velocities)
                        
                        discontinuity_score = 1.0 - discontinuity_rate
                        feasibility_scores.append(discontinuity_score)
                    
                    # 2. Check velocity limits (reasonable robot speeds)
                    if len(velocities) > 0:
                        max_velocity = np.max(normalized_velocities) if action_range > 1e-6 else 0
                        # Penalize extremely high velocities (> threshold of range per step)
                        velocity_threshold = self.config.thresholds.get('velocity_threshold', 0.5)
                        velocity_score = 1.0 / (1.0 + max(0, max_velocity - velocity_threshold) * 10)
                        feasibility_scores.append(velocity_score)
                    
                    # 3. Check for reasonable action ranges
                    if action_range > 0:
                        # Very large ranges might indicate unrealistic actions
                        # This is dataset-dependent, so we use a soft penalty
                        range_score = 1.0 / (1.0 + max(0, action_range - 360) / 100)  # Penalize > 360 degree ranges
                        feasibility_scores.append(range_score)
                
                return np.mean(feasibility_scores) if feasibility_scores else None
            
            # Fallback: Find individual action columns (legacy format)
            action_cols = [col for col in df.columns if col.startswith('action')]
            if not action_cols:
                return None
            
            feasibility_scores = []
            
            for action_col in action_cols:
                if action_col in df.columns:
                    actions = df[action_col].values
                    
                    if len(actions) < 2:
                        continue
                    
                    # Check discontinuities
                    velocities = np.abs(np.diff(actions))
                    action_range = np.ptp(actions)
                    
                    if action_range > 1e-6:
                        normalized_velocities = velocities / action_range
                        discontinuity_threshold = self.config.thresholds.get('discontinuity_threshold', 0.2)
                        large_jumps = np.sum(normalized_velocities > discontinuity_threshold)
                        discontinuity_rate = large_jumps / len(velocities)
                        discontinuity_score = 1.0 - discontinuity_rate
                        feasibility_scores.append(discontinuity_score)
                        
                        # Check velocity limits
                        max_velocity = np.max(normalized_velocities)
                        velocity_threshold = self.config.thresholds.get('velocity_threshold', 0.5)
                        velocity_score = 1.0 / (1.0 + max(0, max_velocity - velocity_threshold) * 10)
                        feasibility_scores.append(velocity_score)
                        
                        # Check action ranges
                        range_score = 1.0 / (1.0 + max(0, action_range - 360) / 100)
                        feasibility_scores.append(range_score)
            
            return np.mean(feasibility_scores) if feasibility_scores else None
            
        except Exception as e:
            logger.warning(f"Error computing feasibility for {episode_data_path}: {e}")
            return None
    
    def get_detailed_report(self, dataset: List[Dict], data_path: Path) -> Dict[str, Any]:
        """Generate a detailed report of action quality metrics."""
        report = {
            'overall_score': self.compute(dataset, data_path, None),
            'components': {
                'action_smoothness': self._analyze_action_smoothness(dataset, data_path),
                'joint_limits': self._analyze_joint_limits(dataset, data_path),
                'state_action_consistency': self._analyze_state_action_consistency(dataset, data_path),
                'gripper_behavior': self._analyze_gripper_behavior(dataset, data_path),
                'physical_feasibility': self._analyze_physical_feasibility(dataset, data_path)
            }
        }
        
        return report 