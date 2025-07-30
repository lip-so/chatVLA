"""
Trajectory Quality Metric

Evaluates the quality of robot trajectories including:
- Time synchronization between different data modalities
- Action recording frequency and consistency
- Data completeness and missing values
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config

logger = logging.getLogger(__name__)

class TrajectoryQualityMetric:
    """Evaluates the quality of robot trajectory data."""
    
    def __init__(self):
        """Initialize the trajectory quality metric with configuration."""
        # Load configuration
        self.config = get_config('trajectory_quality')
        self.general_config = get_config('general')
        
        # Use configuration values
        self.sync_thresholds = self.config.sync_thresholds
        self.freq_thresholds = self.config.freq_thresholds
        self.min_samples_for_frequency = self.config.min_samples_for_frequency
        self.completeness_threshold = self.config.completeness_threshold
    
    def compute(self, dataset: List[Dict], data_path: Path, embedding_manager) -> float:
        # Analyze trajectory quality metrics
        sync_scores = []
        freq_scores = []
        completeness_scores = []
        
        # Check if this is a HuggingFace dataset
        is_hf_dataset = any(sample.get('data_path', '').startswith('hf_episode_') for sample in dataset)
        
        if is_hf_dataset:
            # For HuggingFace datasets, use direct episode data analysis
            for sample in dataset:
                if 'actions' in sample:
                    # Use episode data directly
                    sync_score = self._check_episode_sync_from_data(sample)
                    freq_score = self._check_episode_frequency_from_data(sample)
                    completeness_score = self._check_episode_completeness_from_data(sample)
                    
                    if sync_score is not None:
                        sync_scores.append(sync_score)
                    if freq_score is not None:
                        freq_scores.append(freq_score)
                    if completeness_score is not None:
                        completeness_scores.append(completeness_score)
        else:
            # For local datasets, use file-based analysis
            for sample in dataset:
                episode_data_path = sample.get('data_path')
                if episode_data_path:
                    sync_score = self._check_episode_sync(episode_data_path, data_path)
                    freq_score = self._check_episode_frequency(episode_data_path, data_path)
                    completeness_score = self._check_episode_completeness(episode_data_path, data_path)
                    
                    if sync_score is not None:
                        sync_scores.append(sync_score)
                    if freq_score is not None:
                        freq_scores.append(freq_score)
                    if completeness_score is not None:
                        completeness_scores.append(completeness_score)
        
        # Calculate final scores
        sync_quality = None
        if sync_scores:
            # Lower sync error = better quality
            avg_sync_error = np.mean(sync_scores)
            sync_quality = max(0.0, 1.0 - avg_sync_error / 100.0)  # Normalize assuming 100ms is poor
        
        freq_quality = None
        if freq_scores:
            # Higher frequency consistency = better quality
            freq_quality = np.mean(freq_scores)
        
        completeness_quality = None
        if completeness_scores:
            # Higher completeness = better quality
            completeness_quality = np.mean(completeness_scores)
        
        # Log results
        if sync_quality is not None:
            logger.info(f"Synchronization quality: {sync_quality:.3f}")
        else:
            logger.warning("Synchronization quality could not be evaluated")
        
        if freq_quality is not None:
            logger.info(f"Action frequency quality: {freq_quality:.3f}")
        else:
            logger.warning("Action frequency quality could not be evaluated")
        
        if completeness_quality is not None:
            logger.info(f"Data completeness: {completeness_quality:.3f}")
        else:
            logger.warning("Data completeness could not be evaluated")
        
        # Combine scores
        available_scores = [score for score in [sync_quality, freq_quality, completeness_quality] if score is not None]
        
        if not available_scores:
            logger.error("No trajectory quality metrics could be evaluated")
            return 0.0
        
        # Weight the scores
        weights = [0.3, 0.4, 0.3]  # sync, frequency, completeness
        scores = [sync_quality or 0.5, freq_quality or 0.5, completeness_quality or 0.5]
        
        # Only use weights for available scores
        if len(available_scores) < 3:
            return float(np.mean(available_scores))
        
        final_score = sum(w * s for w, s in zip(weights, scores))
        return float(final_score)
    
    def _analyze_synchronization(self, dataset: List[Dict], data_path: Path) -> Tuple[float, float]:
        """Analyze time synchronization across modalities."""
        sync_errors = []
        
        # Sample episodes for analysis
        sample_size = min(10, len(dataset))
        for i in range(sample_size):
            episode = dataset[i]
            
            if 'data_path' in episode:
                try:
                    sync_error = self._check_episode_sync(episode['data_path'], data_path)
                    if sync_error is not None:
                        sync_errors.append(sync_error)
                except Exception as e:
                    logger.warning(f"Could not analyze sync for episode {i}: {e}")
        
        if not sync_errors:
            logger.warning("No synchronization data available for evaluation")
            return None, None  # Cannot evaluate without sync data
        
        avg_sync_error = np.mean(sync_errors)
        
        # Score based on average synchronization error
        if avg_sync_error <= self.sync_thresholds['gold']:
            score = 1.0
        elif avg_sync_error <= self.sync_thresholds['silver']:
            score = 0.7
        elif avg_sync_error <= self.sync_thresholds['bronze']:
            score = 0.4
        else:
            score = 0.1
        
        return score, avg_sync_error
    
    def _check_episode_sync(self, episode_data_path: str, dataset_data_path: Path) -> float:
        """Check synchronization within a single episode."""
        try:
            # Handle HuggingFace datasets - skip sync check as they don't have file-based sync data
            if episode_data_path.startswith('hf_episode_'):
                logger.debug(f"Skipping sync check for HF dataset episode: {episode_data_path}")
                return None
            
            # Handle relative paths with data_path
            if dataset_data_path and not Path(episode_data_path).is_absolute():
                full_data_path = dataset_data_path / episode_data_path
            else:
                full_data_path = Path(episode_data_path)
                
            df = pd.read_parquet(str(full_data_path))
            
            # Look for timestamp columns
            timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
            
            if len(timestamp_cols) < 2:
                return None  # Need at least 2 modalities to check sync
            
            # Calculate time differences between modalities
            sync_errors = []
            
            for i in range(len(df)):
                timestamps = [df[col].iloc[i] for col in timestamp_cols]
                if all(pd.notna(ts) for ts in timestamps):
                    # Convert to milliseconds and find max deviation
                    timestamps = np.array(timestamps)
                    max_deviation = (np.max(timestamps) - np.min(timestamps)) * 1000  # to ms
                    sync_errors.append(max_deviation)
            
            return np.mean(sync_errors) if sync_errors else None
            
        except Exception as e:
            logger.warning(f"Error checking sync for {episode_data_path}: {e}")
            return None
    
    def _analyze_action_frequency(self, dataset: List[Dict], data_path: Path) -> Tuple[float, float]:
        """Analyze action recording frequency."""
        frequencies = []
        
        # Sample episodes for analysis
        sample_size = min(10, len(dataset))
        for i in range(sample_size):
            episode = dataset[i]
            
            if 'data_path' in episode:
                try:
                    freq = self._get_episode_frequency(episode['data_path'], data_path)
                    if freq is not None:
                        frequencies.append(freq)
                except Exception as e:
                    logger.warning(f"Could not analyze frequency for episode {i}: {e}")
        
        if not frequencies:
            logger.warning("No frequency data available for evaluation")
            return None, None  # Cannot evaluate without frequency data
        
        avg_frequency = np.mean(frequencies)
        
        # Score based on average frequency
        if avg_frequency >= self.freq_thresholds['gold']:
            score = 1.0
        elif avg_frequency >= self.freq_thresholds['silver']:
            score = 0.7
        elif avg_frequency >= self.freq_thresholds['bronze']:
            score = 0.4
        else:
            score = 0.1
        
        return score, avg_frequency
    
    def _get_episode_frequency(self, episode_data_path: str, dataset_data_path: Path) -> float:
        """Get the recording frequency of an episode."""
        try:
            # Handle HuggingFace datasets - skip frequency check as they don't have file-based frequency data
            if episode_data_path.startswith('hf_episode_'):
                logger.debug(f"Skipping frequency check for HF dataset episode: {episode_data_path}")
                return None
            
            # Handle relative paths with data_path
            if dataset_data_path and not Path(episode_data_path).is_absolute():
                full_data_path = dataset_data_path / episode_data_path
            else:
                full_data_path = Path(episode_data_path)
                
            df = pd.read_parquet(str(full_data_path))
            
            # Look for a main timestamp column
            timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
            
            if not timestamp_cols:
                return None
            
            # Use the first timestamp column
            timestamps = df[timestamp_cols[0]].dropna()
            
            if len(timestamps) < 2:
                return None
            
            # Calculate frequency
            time_diffs = np.diff(timestamps)
            avg_dt = np.mean(time_diffs)
            frequency = 1.0 / avg_dt if avg_dt > 0 else 0
            
            return frequency
            
        except Exception as e:
            logger.warning(f"Error getting frequency for {episode_data_path}: {e}")
            return None
    
    def _analyze_data_completeness(self, dataset: List[Dict], data_path: Path) -> float:
        completeness_scores = []
        
        # Sample episodes for analysis
        sample_size = min(10, len(dataset))
        for i in range(sample_size):
            episode = dataset[i]
            
            if 'data_path' in episode:
                try:
                    # Check if it's a HuggingFace dataset
                    if episode['data_path'].startswith('hf_episode_'):
                        # For HF datasets, check completeness from direct episode data
                        completeness = self._check_hf_episode_completeness(episode)
                    else:
                        # For local datasets, check from file
                        completeness = self._check_episode_completeness(episode['data_path'], data_path)
                    
                    if completeness is not None:
                        completeness_scores.append(completeness)
                except Exception as e:
                    logger.warning(f"Could not analyze completeness for episode {i}: {e}")
        
        if not completeness_scores:
            logger.warning("No completeness data available for evaluation")
            return None  # Cannot evaluate without data
        
        return np.mean(completeness_scores)
    
    def _check_hf_episode_completeness(self, episode_data: Dict[str, Any]) -> Optional[float]:
        """Check data completeness for a HuggingFace episode."""
        try:
            if 'actions' not in episode_data:
                return None
            
            actions = episode_data['actions']
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            
            # Check for missing/invalid values
            if actions.size == 0:
                return 0.0
            
            # Count valid (non-NaN, non-inf) values
            valid_values = np.isfinite(actions).sum()
            total_values = actions.size
            
            completeness = valid_values / total_values if total_values > 0 else 0.0
            return completeness
            
        except Exception as e:
            logger.warning(f"Error checking HF episode completeness: {e}")
            return None
    
    def _check_episode_completeness(self, episode_data_path: str, dataset_data_path: Path) -> float:
        """Check data completeness for a single episode."""
        try:
            # Handle HuggingFace datasets - skip file-based loading
            if episode_data_path.startswith('hf_episode_'):
                logger.debug(f"Skipping completeness check for HF dataset episode: {episode_data_path}")
                return None
            
            # Handle relative paths with data_path
            if dataset_data_path and not Path(episode_data_path).is_absolute():
                full_data_path = dataset_data_path / episode_data_path
            else:
                full_data_path = Path(episode_data_path)
                
            df = pd.read_parquet(str(full_data_path))
            
            # Check for missing values
            total_values = df.size
            missing_values = df.isnull().sum().sum()
            
            if total_values == 0:
                return None
            
            completeness = 1.0 - (missing_values / total_values)
            return completeness
            
        except Exception as e:
            logger.warning(f"Error checking completeness for {episode_data_path}: {e}")
            return None

    def _check_episode_sync_from_data(self, episode_data: Dict[str, Any]) -> float:
        """Check synchronization within a single episode using direct episode data."""
        try:
            # For HuggingFace datasets, we don't have multi-modal timestamps
            # So we'll check action-observation synchronization if available
            if 'actions' in episode_data and 'observations' in episode_data:
                actions = episode_data['actions']
                observations = episode_data['observations']
                
                if not isinstance(actions, np.ndarray):
                    actions = np.array(actions)
                if not isinstance(observations, np.ndarray):
                    observations = np.array(observations)
                
                # Check if lengths match (basic sync check)
                if len(actions) != len(observations):
                    length_diff = abs(len(actions) - len(observations))
                    max_length = max(len(actions), len(observations))
                    sync_error = (length_diff / max_length) * 100  # Percentage error
                    return sync_error
                else:
                    return 0.0  # Perfect sync
            
            return None  # No sync data available
            
        except Exception as e:
            logger.warning(f"Error checking sync from episode data: {e}")
            return None

    def _check_episode_frequency_from_data(self, episode_data: Dict[str, Any]) -> float:
        """Check action frequency consistency using direct episode data."""
        try:
            if 'actions' not in episode_data:
                return None
            
            actions = episode_data['actions']
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            
            if len(actions) < 2:
                return None
            
            # Assume uniform time steps (common in robotics datasets)
            # Check for consistency in action magnitudes as a proxy for frequency consistency
            action_magnitudes = np.linalg.norm(actions, axis=1)
            
            # Calculate coefficient of variation (std/mean) as consistency measure
            if np.mean(action_magnitudes) > 1e-6:
                cv = np.std(action_magnitudes) / np.mean(action_magnitudes)
                # Convert to consistency score (lower CV = higher consistency)
                consistency = 1.0 / (1.0 + cv)
                return consistency
            
            return 1.0  # If all actions are zero, that's perfectly consistent
            
        except Exception as e:
            logger.warning(f"Error checking frequency from episode data: {e}")
            return None

    def _check_episode_completeness_from_data(self, episode_data: Dict[str, Any]) -> float:
        """Check data completeness using direct episode data."""
        try:
            if 'actions' not in episode_data:
                return None
            
            actions = episode_data['actions']
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            
            # Check for NaN or invalid values
            if actions.size == 0:
                return 0.0
            
            valid_values = np.isfinite(actions).sum()
            total_values = actions.size
            
            completeness = valid_values / total_values
            return completeness
            
        except Exception as e:
            logger.warning(f"Error checking completeness from episode data: {e}")
            return None
    
    def get_quality_breakdown(self, dataset: List[Dict], data_path: Path) -> Dict[str, Any]:
        """Get detailed quality breakdown."""
        sync_score, avg_sync_error = self._analyze_synchronization(dataset, data_path)
        freq_score, avg_frequency = self._analyze_action_frequency(dataset, data_path)
        completeness_score = self._analyze_data_completeness(dataset, data_path)
        
        # Determine tiers
        sync_tier = self._get_sync_tier(avg_sync_error)
        freq_tier = self._get_freq_tier(avg_frequency)
        
        return {
            'synchronization': {
                'score': sync_score,
                'avg_error_ms': avg_sync_error,
                'tier': sync_tier
            },
            'frequency': {
                'score': freq_score,
                'avg_hz': avg_frequency,
                'tier': freq_tier
            },
            'completeness': {
                'score': completeness_score,
                'percentage': completeness_score * 100
            }
        }
    
    def _get_sync_tier(self, sync_error: float) -> str:
        """Get synchronization tier based on error."""
        if sync_error <= self.sync_thresholds['gold']:
            return 'Gold'
        elif sync_error <= self.sync_thresholds['silver']:
            return 'Silver'
        elif sync_error <= self.sync_thresholds['bronze']:
            return 'Bronze'
        else:
            return 'Poor'
    
    def _get_freq_tier(self, frequency: float) -> str:
        """Get frequency tier based on Hz."""
        if frequency >= self.freq_thresholds['gold']:
            return 'Gold'
        elif frequency >= self.freq_thresholds['silver']:
            return 'Silver'
        elif frequency >= self.freq_thresholds['bronze']:
            return 'Bronze'
        else:
            return 'Poor'