"""
Configuration loader for DataBench robotics dataset evaluation.

This module provides utilities to load and manage configuration parameters
from YAML files, with support for environment-specific overrides and
validation.
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """Base configuration class for metrics."""
    pass

@dataclass 
class VisualConfig(MetricConfig):
    """Configuration for visual processing parameters."""
    num_frames: int = 8
    sample_size: int = 1000
    video_timeout: int = 30
    supported_codecs: List[str] = field(default_factory=lambda: ["mp4", "avi", "mov", "mkv"])

@dataclass
class RobotActionQualityConfig(MetricConfig):
    """Configuration for robot action quality metric."""
    joint_limits: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    default_joint_limits: List[float] = field(default_factory=lambda: [-3.14, 3.14])
    thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class TrajectoryQualityConfig(MetricConfig):
    """Configuration for trajectory quality metric."""
    sync_thresholds: Dict[str, int] = field(default_factory=dict)
    freq_thresholds: Dict[str, int] = field(default_factory=dict)
    min_samples_for_frequency: int = 10
    completeness_threshold: float = 0.95

@dataclass
class DatasetCoverageConfig(MetricConfig):
    """Configuration for dataset coverage metric."""
    trajectory_thresholds: Dict[str, int] = field(default_factory=dict)
    task_thresholds: Dict[str, int] = field(default_factory=dict)
    min_visual_samples: int = 10
    failure_rate_threshold: float = 0.1

@dataclass
class ActionConsistencyConfig(MetricConfig):
    """Configuration for action consistency metric."""
    num_frames: int = 8
    min_episodes_for_cross_analysis: int = 2
    min_sequence_length: int = 5
    correlation_threshold: float = 0.1
    smoothness_window: int = 3
    weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class VisualDiversityConfig(MetricConfig):
    """Configuration for visual diversity metric."""
    num_frames: int = 8
    sample_size: int = 1000
    min_samples_for_clustering: int = 10
    n_clusters: int = 5
    random_state: int = 42
    weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class HighFidelityVisionConfig(MetricConfig):
    """Configuration for high fidelity vision metric."""
    resolution_thresholds: Dict[str, List[int]] = field(default_factory=dict)
    framerate_thresholds: Dict[str, int] = field(default_factory=dict)
    single_view_penalty: float = 0.3
    multi_view_bonus: float = 0.2
    blur_threshold: int = 100
    brightness_range: List[int] = field(default_factory=lambda: [20, 235])
    min_prompt_length: int = 10
    max_prompt_length: int = 200

@dataclass
class GeneralConfig:
    """General configuration parameters."""
    log_level: str = "INFO"
    max_workers: int = 4
    chunk_size: int = 100
    default_score: float = 0.5
    supported_video_extensions: List[str] = field(default_factory=list)
    supported_data_extensions: List[str] = field(default_factory=list)
    max_frames_in_memory: int = 1000

@dataclass
class RobotConfig:
    """Configuration for specific robot types."""
    dof: int
    joint_limits: List[List[float]]
    gripper_limits: List[float]
    max_velocity: float
    gripper_joint: Optional[int] = None  # Index of gripper joint in action array

class ConfigLoader:
    """Loads and manages configuration from YAML files."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file. If None, will look for
                        config.yaml in the current directory or parent directories.
        """
        self.config_path = self._find_config_file(config_path)
        self._config_data = None
        self._load_config()
    
    def _find_config_file(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Find the configuration file."""
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Look for config.yaml in current directory and parent directories
        current = Path.cwd()
        for _ in range(3):  # Look up to 3 levels up
            config_file = current / "config.yaml"
            if config_file.exists():
                return config_file
            current = current.parent
        
        raise FileNotFoundError("config.yaml not found in current directory or parent directories")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config_data = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def reload_config(self):
        """Reload configuration from file."""
        self._load_config()
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self._config_data.copy()
    
    def get_visual_config(self) -> VisualConfig:
        """Get visual processing configuration."""
        visual_data = self._config_data.get('visual', {})
        return VisualConfig(**visual_data)
    
    def get_robot_action_quality_config(self) -> RobotActionQualityConfig:
        """Get robot action quality configuration."""
        raq_data = self._config_data.get('robot_action_quality', {})
        return RobotActionQualityConfig(**raq_data)
    
    def get_trajectory_quality_config(self) -> TrajectoryQualityConfig:
        """Get trajectory quality configuration."""
        tq_data = self._config_data.get('trajectory_quality', {})
        return TrajectoryQualityConfig(**tq_data)
    
    def get_dataset_coverage_config(self) -> DatasetCoverageConfig:
        """Get dataset coverage configuration."""
        dc_data = self._config_data.get('dataset_coverage', {})
        return DatasetCoverageConfig(**dc_data)
    
    def get_action_consistency_config(self) -> ActionConsistencyConfig:
        """Get action consistency configuration."""
        ac_data = self._config_data.get('action_consistency', {})
        return ActionConsistencyConfig(**ac_data)
    
    def get_visual_diversity_config(self) -> VisualDiversityConfig:
        """Get visual diversity configuration."""
        vd_data = self._config_data.get('visual_diversity', {})
        return VisualDiversityConfig(**vd_data)
    
    def get_high_fidelity_vision_config(self) -> HighFidelityVisionConfig:
        """Get high fidelity vision configuration."""
        hfv_data = self._config_data.get('high_fidelity_vision', {})
        return HighFidelityVisionConfig(**hfv_data)
    
    def get_general_config(self) -> GeneralConfig:
        """Get general configuration."""
        general_data = self._config_data.get('general', {})
        return GeneralConfig(**general_data)
    
    def get_robot_config(self, robot_name: str) -> Optional[RobotConfig]:
        """
        Get configuration for a specific robot.
        
        Args:
            robot_name: Name of the robot (e.g., 'franka_panda', 'ur5')
            
        Returns:
            RobotConfig object or None if robot not found
        """
        robots_data = self._config_data.get('robots', {})
        robot_data = robots_data.get(robot_name)
        
        if robot_data:
            return RobotConfig(**robot_data)
        return None
    
    def get_available_robots(self) -> List[str]:
        """Get list of available robot configurations."""
        robots_data = self._config_data.get('robots', {})
        return list(robots_data.keys())
    
    def override_config(self, overrides: Dict[str, Any]):
        """
        Override configuration values at runtime.
        
        Args:
            overrides: Dictionary of configuration overrides using dot notation
                      (e.g., {'visual.num_frames': 16, 'general.log_level': 'DEBUG'})
        """
        for key, value in overrides.items():
            keys = key.split('.')
            current = self._config_data
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
            logger.info(f"Override applied: {key} = {value}")
    
    def validate_config(self) -> bool:
        """
        Validate the configuration for common issues.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        issues = []
        
        # Check required sections
        required_sections = ['visual', 'robot_action_quality', 'trajectory_quality', 
                           'dataset_coverage', 'action_consistency', 'visual_diversity',
                           'high_fidelity_vision', 'general']
        
        for section in required_sections:
            if section not in self._config_data:
                issues.append(f"Missing required section: {section}")
        
        # Check for positive numeric values where expected
        numeric_checks = [
            ('visual.num_frames', lambda x: x > 0),
            ('visual.sample_size', lambda x: x > 0),
            ('general.max_workers', lambda x: x > 0),
            ('general.default_score', lambda x: 0 <= x <= 1),
        ]
        
        for check_path, validation_func in numeric_checks:
            try:
                keys = check_path.split('.')
                current = self._config_data
                for k in keys:
                    current = current[k]
                
                if not validation_func(current):
                    issues.append(f"Invalid value for {check_path}: {current}")
            except (KeyError, TypeError):
                issues.append(f"Missing or invalid configuration: {check_path}")
        
        # Log issues
        if issues:
            for issue in issues:
                logger.error(f"Configuration validation error: {issue}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None):
        """
        Save current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration. If None, overwrites current file.
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

# Global configuration loader instance
_config_loader = None

def get_config_loader(config_path: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """
    Get the global configuration loader instance.
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader

def get_config(section: str) -> Any:
    """
    Convenience function to get configuration for a specific section.
    
    Args:
        section: Configuration section name
        
    Returns:
        Configuration object for the section
    """
    loader = get_config_loader()
    
    if section == 'visual':
        return loader.get_visual_config()
    elif section == 'robot_action_quality':
        return loader.get_robot_action_quality_config()
    elif section == 'trajectory_quality':
        return loader.get_trajectory_quality_config()
    elif section == 'dataset_coverage':
        return loader.get_dataset_coverage_config()
    elif section == 'action_consistency':
        return loader.get_action_consistency_config()
    elif section == 'visual_diversity':
        return loader.get_visual_diversity_config()
    elif section == 'high_fidelity_vision':
        return loader.get_high_fidelity_vision_config()
    elif section == 'general':
        return loader.get_general_config()
    else:
        raise ValueError(f"Unknown configuration section: {section}") 