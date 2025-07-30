#!/usr/bin/env python3
"""
Dataset Analyzer

Analyzes a dataset to derive dataset-specific thresholds and parameters
for evaluation, rather than using hardcoded values.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from collections import Counter
import json

logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """Analyzes dataset to derive evaluation parameters dynamically."""
    
    def __init__(self, dataset: List[Dict], data_path: Path, embedding_manager=None):
        """
        Initialize the dataset analyzer.
        
        Args:
            dataset: List of episode data
            data_path: Path to the dataset
            embedding_manager: Optional embedding manager for video analysis
        """
        self.dataset = dataset
        self.data_path = data_path
        self.embedding_manager = embedding_manager
        self.analysis = {}
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis to derive evaluation parameters.
        
        Returns:
            Dictionary containing dataset-specific thresholds and parameters
        """
        logger.info("Analyzing dataset to derive evaluation parameters...")
        
        # Analyze different aspects of the dataset
        self.analysis['video_properties'] = self._analyze_video_properties()
        self.analysis['text_properties'] = self._analyze_text_properties()
        self.analysis['action_properties'] = self._analyze_action_properties()
        self.analysis['scale_properties'] = self._analyze_scale_properties()
        self.analysis['quality_thresholds'] = self._derive_quality_thresholds()
        
        logger.info("Dataset analysis complete")
        return self.analysis
    
    def _analyze_video_properties(self) -> Dict[str, Any]:
        """Analyze video properties to derive resolution and framerate thresholds."""
        if not self.embedding_manager:
            logger.warning("No embedding manager available for video analysis")
            return {}
        
        resolutions = []
        framerates = []
        camera_counts = []
        
        # Sample videos to analyze (limit to avoid excessive processing)
        sample_size = min(20, len(self.dataset))
        sample_indices = np.linspace(0, len(self.dataset) - 1, sample_size, dtype=int)
        
        for i in sample_indices:
            episode = self.dataset[i]
            
            # Count cameras
            if 'videos' in episode and isinstance(episode['videos'], dict):
                camera_counts.append(len(episode['videos']))
                
                # Analyze first video for technical properties
                video_path = next(iter(episode['videos'].values()))
                if self.data_path and isinstance(video_path, str):
                    video_full_path = self.data_path / video_path if not Path(video_path).is_absolute() else video_path
                    
                    from scripts.embed_utils import VideoLoader
                    video_info = VideoLoader.get_video_info(str(video_full_path))
                    
                    if video_info['exists']:
                        # Extract resolution
                        if 'x' in str(video_info['resolution']):
                            width, height = map(int, str(video_info['resolution']).split('x'))
                            resolutions.append(height)
                        
                        # Extract framerate
                        if video_info['fps'] != 'unknown':
                            framerates.append(float(video_info['fps']))
            
            elif episode.get('video'):
                camera_counts.append(1)
        
        # Derive thresholds based on actual data
        properties = {}
        
        if resolutions:
            properties['resolution_percentiles'] = {
                'p25': np.percentile(resolutions, 25),
                'p50': np.percentile(resolutions, 50),
                'p75': np.percentile(resolutions, 75),
                'p90': np.percentile(resolutions, 90)
            }
            properties['resolution_range'] = [min(resolutions), max(resolutions)]
        else:
            # Fallback only if no data available
            properties['resolution_percentiles'] = {'p25': 480, 'p50': 720, 'p75': 1080, 'p90': 1440}
            properties['resolution_range'] = [480, 1080]
        
        if framerates:
            properties['framerate_percentiles'] = {
                'p25': np.percentile(framerates, 25),
                'p50': np.percentile(framerates, 50),
                'p75': np.percentile(framerates, 75),
                'p90': np.percentile(framerates, 90)
            }
            properties['framerate_range'] = [min(framerates), max(framerates)]
        else:
            properties['framerate_percentiles'] = {'p25': 15, 'p50': 24, 'p75': 30, 'p90': 60}
            properties['framerate_range'] = [15, 30]
        
        if camera_counts:
            properties['camera_counts'] = Counter(camera_counts)
            properties['avg_cameras'] = np.mean(camera_counts)
            properties['max_cameras'] = max(camera_counts)
        else:
            properties['camera_counts'] = Counter([1])
            properties['avg_cameras'] = 1
            properties['max_cameras'] = 1
        
        return properties
    
    def _analyze_text_properties(self) -> Dict[str, Any]:
        """Analyze text properties to derive length and complexity thresholds."""
        prompt_lengths = []
        task_lengths = []
        combined_lengths = []
        unique_words = set()
        
        for episode in self.dataset:
            prompt = episode.get('prompt', '')
            task = episode.get('task', '')
            
            if prompt:
                prompt_words = prompt.split()
                prompt_lengths.append(len(prompt_words))
                unique_words.update(word.lower() for word in prompt_words)
            
            if task:
                task_words = task.split()
                task_lengths.append(len(task_words))
                unique_words.update(word.lower() for word in task_words)
            
            if prompt or task:
                combined_text = (prompt + ' ' + task).strip()
                combined_lengths.append(len(combined_text.split()))
        
        properties = {}
        
        if combined_lengths:
            properties['text_length_percentiles'] = {
                'p25': np.percentile(combined_lengths, 25),
                'p50': np.percentile(combined_lengths, 50),
                'p75': np.percentile(combined_lengths, 75),
                'p90': np.percentile(combined_lengths, 90)
            }
            properties['text_length_range'] = [min(combined_lengths), max(combined_lengths)]
            properties['vocabulary_size'] = len(unique_words)
            properties['avg_text_length'] = np.mean(combined_lengths)
        else:
            properties['text_length_percentiles'] = {'p25': 3, 'p50': 7, 'p75': 15, 'p90': 25}
            properties['text_length_range'] = [1, 50]
            properties['vocabulary_size'] = 0
            properties['avg_text_length'] = 0
        
        return properties
    
    def _analyze_action_properties(self) -> Dict[str, Any]:
        """Analyze action data properties to derive thresholds."""
        action_dimensions = []
        action_ranges = []
        action_frequencies = []
        
        sample_size = min(10, len(self.dataset))
        sample_indices = np.linspace(0, len(self.dataset) - 1, sample_size, dtype=int)
        
        for i in sample_indices:
            episode = self.dataset[i]
            
            if 'data_path' in episode:
                try:
                    if self.data_path and not Path(episode['data_path']).is_absolute():
                        data_file_path = self.data_path / episode['data_path']
                    else:
                        data_file_path = Path(episode['data_path'])
                    
                    if data_file_path.exists():
                        df = pd.read_parquet(str(data_file_path))
                        
                        # Analyze action dimensions
                        if 'action' in df.columns:
                            actions = np.stack(df['action'].values)
                            action_dimensions.append(actions.shape[1])
                            
                            # Analyze action ranges per dimension
                            for dim in range(actions.shape[1]):
                                action_ranges.append(np.ptp(actions[:, dim]))
                            
                            # Estimate frequency (assume timestamp or index)
                            if len(actions) > 1:
                                estimated_freq = len(actions) / (len(actions) / 30.0)  # Rough estimate
                                action_frequencies.append(estimated_freq)
                        
                except Exception as e:
                    logger.debug(f"Could not analyze action data for episode {i}: {e}")
                    continue
        
        properties = {}
        
        if action_dimensions:
            properties['action_dimensions'] = Counter(action_dimensions)
            properties['most_common_dof'] = Counter(action_dimensions).most_common(1)[0][0]
        
        if action_ranges:
            properties['action_range_percentiles'] = {
                'p25': np.percentile(action_ranges, 25),
                'p50': np.percentile(action_ranges, 50),
                'p75': np.percentile(action_ranges, 75),
                'p90': np.percentile(action_ranges, 90)
            }
        
        if action_frequencies:
            properties['action_frequency_percentiles'] = {
                'p25': np.percentile(action_frequencies, 25),
                'p50': np.percentile(action_frequencies, 50),
                'p75': np.percentile(action_frequencies, 75),
                'p90': np.percentile(action_frequencies, 90)
            }
        
        return properties
    
    def _analyze_scale_properties(self) -> Dict[str, Any]:
        """Analyze dataset scale properties."""
        properties = {
            'total_episodes': len(self.dataset),
            'scale_tier': self._determine_scale_tier(len(self.dataset))
        }
        
        # Analyze task diversity
        tasks = []
        for episode in self.dataset:
            task = episode.get('task', episode.get('prompt', 'unknown'))
            if task and task != 'unknown':
                tasks.append(task.lower().strip())
        
        if tasks:
            unique_tasks = len(set(tasks))
            properties['unique_tasks'] = unique_tasks
            properties['task_repetition'] = len(tasks) / unique_tasks if unique_tasks > 0 else 1
            properties['most_common_tasks'] = Counter(tasks).most_common(10)
        
        return properties
    
    def _determine_scale_tier(self, num_episodes: int) -> str:
        """Determine dataset scale tier based on episode count."""
        if num_episodes >= 1000:
            return 'large'
        elif num_episodes >= 100:
            return 'medium'
        elif num_episodes >= 20:
            return 'small'
        else:
            return 'tiny'
    
    def _derive_quality_thresholds(self) -> Dict[str, Any]:
        """Derive quality thresholds based on dataset analysis."""
        thresholds = {}
        
        # Video quality thresholds based on actual data
        video_props = self.analysis.get('video_properties', {})
        if 'resolution_percentiles' in video_props:
            res_p = video_props['resolution_percentiles']
            thresholds['resolution'] = {
                'excellent': res_p['p90'],    # Top 10%
                'good': res_p['p75'],         # Top 25%
                'acceptable': res_p['p50'],   # Median
                'poor': res_p['p25']          # Bottom 25%
            }
        
        if 'framerate_percentiles' in video_props:
            fps_p = video_props['framerate_percentiles']
            thresholds['framerate'] = {
                'excellent': fps_p['p90'],
                'good': fps_p['p75'],
                'acceptable': fps_p['p50'],
                'poor': fps_p['p25']
            }
        
        # Camera count thresholds based on actual data
        if 'max_cameras' in video_props:
            max_cams = video_props['max_cameras']
            avg_cams = video_props['avg_cameras']
            thresholds['camera_count'] = {
                'excellent': max_cams,
                'good': max(2, int(avg_cams * 1.2)),
                'acceptable': max(1, int(avg_cams)),
                'poor': 1
            }
        
        # Text quality thresholds based on actual data
        text_props = self.analysis.get('text_properties', {})
        if 'text_length_percentiles' in text_props:
            text_p = text_props['text_length_percentiles']
            thresholds['text_length'] = {
                'excellent': (text_p['p75'], text_p['p90']),  # Good range
                'good': (text_p['p50'], text_p['p75']),
                'acceptable': (text_p['p25'], text_p['p50']),
                'poor': (0, text_p['p25'])
            }
        
        # Scale-based thresholds
        scale_props = self.analysis.get('scale_properties', {})
        total_episodes = scale_props.get('total_episodes', 0)
        
        thresholds['dataset_scale'] = {
            'excellent': max(total_episodes, 100),
            'good': max(total_episodes * 0.8, 50),
            'acceptable': max(total_episodes * 0.5, 20),
            'poor': max(total_episodes * 0.2, 5)
        }
        
        return thresholds
    
    def get_adaptive_thresholds(self, metric_type: str) -> Dict[str, float]:
        """Get adaptive thresholds for a specific metric type."""
        quality_thresholds = self.analysis.get('quality_thresholds', {})
        
        if metric_type == 'resolution' and 'resolution' in quality_thresholds:
            res_t = quality_thresholds['resolution']
            return {
                'excellent_threshold': res_t['excellent'],
                'good_threshold': res_t['good'],
                'acceptable_threshold': res_t['acceptable'],
                'poor_threshold': res_t['poor']
            }
        
        elif metric_type == 'framerate' and 'framerate' in quality_thresholds:
            fps_t = quality_thresholds['framerate']
            return {
                'excellent_threshold': fps_t['excellent'],
                'good_threshold': fps_t['good'],
                'acceptable_threshold': fps_t['acceptable'],
                'poor_threshold': fps_t['poor']
            }
        
        elif metric_type == 'camera_count' and 'camera_count' in quality_thresholds:
            cam_t = quality_thresholds['camera_count']
            return {
                'excellent_threshold': cam_t['excellent'],
                'good_threshold': cam_t['good'],
                'acceptable_threshold': cam_t['acceptable'],
                'poor_threshold': cam_t['poor']
            }
        
        elif metric_type == 'text_length' and 'text_length' in quality_thresholds:
            text_t = quality_thresholds['text_length']
            return {
                'excellent_range': text_t['excellent'],
                'good_range': text_t['good'],
                'acceptable_range': text_t['acceptable'],
                'poor_range': text_t['poor']
            }
        
        # Fallback to defaults if analysis not available
        return self._get_fallback_thresholds(metric_type)
    
    def _get_fallback_thresholds(self, metric_type: str) -> Dict[str, float]:
        """Get fallback thresholds when dataset analysis is not available."""
        fallbacks = {
            'resolution': {
                'excellent_threshold': 1080,
                'good_threshold': 720,
                'acceptable_threshold': 480,
                'poor_threshold': 240
            },
            'framerate': {
                'excellent_threshold': 30,
                'good_threshold': 24,
                'acceptable_threshold': 15,
                'poor_threshold': 10
            },
            'camera_count': {
                'excellent_threshold': 4,
                'good_threshold': 3,
                'acceptable_threshold': 2,
                'poor_threshold': 1
            },
            'text_length': {
                'excellent_range': (10, 20),
                'good_range': (5, 15),
                'acceptable_range': (3, 10),
                'poor_range': (0, 5)
            }
        }
        
        return fallbacks.get(metric_type, {})
    
    def save_analysis(self, output_path: Path):
        """Save analysis results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.analysis, f, indent=2, default=str)
        logger.info(f"Dataset analysis saved to {output_path}")
    
    def load_analysis(self, input_path: Path):
        """Load analysis results from file."""
        with open(input_path, 'r') as f:
            self.analysis = json.load(f)
        logger.info(f"Dataset analysis loaded from {input_path}") 