"""
Dataset Coverage Metric

Evaluates dataset comprehensiveness including:
- Scale (number of trajectories/episodes)
- Task diversity and variety
- Visual scene diversity
- Failure case coverage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from collections import Counter

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config

logger = logging.getLogger(__name__)

class DatasetCoverageMetric:
    """Evaluates how comprehensive a dataset is across different dimensions."""
    
    def __init__(self):
        """Initialize the dataset coverage metric with configuration."""
        # Load configuration
        self.config = get_config('dataset_coverage')
        self.general_config = get_config('general')
        
        # Use configuration values
        self.trajectory_thresholds = self.config.trajectory_thresholds
        self.task_thresholds = self.config.task_thresholds
        self.min_visual_samples = self.config.min_visual_samples
        self.failure_rate_threshold = self.config.failure_rate_threshold
    
    def compute(self, dataset: List[Dict], data_path: Path, embedding_manager) -> float:
        """
        Compute dataset coverage score.
        
        Returns:
            Float score between 0 and 1, where 1 indicates highest coverage
        """
        if not dataset:
            return 0.0
        
        # Analyze different aspects of coverage
        scale_score = self._score_dataset_scale(dataset)
        task_diversity_score = self._analyze_task_diversity(dataset)
        visual_diversity_score = self._analyze_visual_diversity(dataset, embedding_manager, data_path)
        failure_rate_score = self._analyze_failure_rates(dataset)
        
        # Handle None values (metrics that couldn't be evaluated)
        available_scores = []
        available_weights = []
        
        # Scale score (always available)
        available_scores.append(scale_score)
        available_weights.append(0.2)
        
        # Task diversity score (always available)
        available_scores.append(task_diversity_score)
        available_weights.append(0.3)
        
        # Visual diversity score (may be None)
        if visual_diversity_score is not None:
            available_scores.append(visual_diversity_score)
            available_weights.append(0.3)
        else:
            logger.warning("Visual diversity could not be evaluated")
        
        # Failure rate score (may be None)
        if failure_rate_score is not None:
            available_scores.append(failure_rate_score)
            available_weights.append(0.2)
        else:
            logger.warning("Failure rate could not be evaluated")
        
        # Normalize weights for available scores
        total_weight = sum(available_weights)
        normalized_weights = [w / total_weight for w in available_weights]
        
        # Weighted combination of available scores
        total_score = sum(score * weight for score, weight in zip(available_scores, normalized_weights))
        
        # Log detailed breakdown
        logger.info(f"Dataset Coverage: {total_score:.3f}")
        logger.info(f"  - Scale: {len(dataset)} episodes ({scale_score:.3f})")
        logger.info(f"  - Task Diversity: {task_diversity_score:.3f}")
        if visual_diversity_score is not None:
            logger.info(f"  - Visual Diversity: {visual_diversity_score:.3f}")
        else:
            logger.info(f"  - Visual Diversity: Not evaluated")
        if failure_rate_score is not None:
            logger.info(f"  - Failure Rate: {failure_rate_score:.3f}")
        else:
            logger.info(f"  - Failure Rate: Not evaluated")
        
        return float(total_score)
    
    def _score_dataset_scale(self, dataset: List[Dict]) -> float:
        """Score based on number of trajectories."""
        num_trajectories = len(dataset)
        
        if num_trajectories >= self.trajectory_thresholds['gold']:
            return 1.0
        elif num_trajectories >= self.trajectory_thresholds['silver']:
            return 0.7
        elif num_trajectories >= self.trajectory_thresholds['bronze']:
            return 0.4
        else:
            # Linear interpolation for smaller datasets
            return min(0.4, num_trajectories / self.trajectory_thresholds['bronze'])
    
    def _analyze_task_diversity(self, dataset: List[Dict]) -> float:
        """Analyze diversity of tasks in the dataset."""
        tasks = []
        
        for episode in dataset:
            task = episode.get('task', episode.get('prompt', 'unknown'))
            if task and task != 'unknown':
                tasks.append(task.lower().strip())
        
        if not tasks:
            return 0.0
        
        # Count unique tasks
        unique_tasks = len(set(tasks))
        
        # Score based on number of unique tasks
        if unique_tasks >= self.task_thresholds['gold']:
            score = 1.0
        elif unique_tasks >= self.task_thresholds['silver']:
            score = 0.7
        elif unique_tasks >= self.task_thresholds['bronze']:
            score = 0.4
        else:
            score = 0.1
        
        # Bonus for task distribution balance
        task_counts = Counter(tasks)
        task_distribution = np.array(list(task_counts.values()))
        
        # Calculate entropy of task distribution
        task_probs = task_distribution / task_distribution.sum()
        entropy = -np.sum(task_probs * np.log2(task_probs + 1e-10))
        max_entropy = np.log2(len(task_counts))
        
        balance_bonus = (entropy / max_entropy) * 0.2 if max_entropy > 0 else 0
        
        return min(1.0, score + balance_bonus)
    
    def _analyze_visual_diversity(self, dataset: List[Dict], embedding_manager, data_path: Path = None) -> float:
        """Analyze visual diversity using image embeddings."""
        if not embedding_manager.clip_model:
            try:
                embedding_manager._load_clip()
            except Exception as e:
                logger.warning(f"Could not load CLIP model for visual diversity: {e}")
                return 0.5
        
        # Sample frames from dataset
        sample_size = min(100, len(dataset))
        sampled_episodes = np.random.choice(len(dataset), sample_size, replace=False)
        
        embeddings = []
        
        for i in sampled_episodes:
            episode = dataset[i]
            try:
                # Handle multiple video views if available
                if 'videos' in episode and isinstance(episode['videos'], dict):
                    # Use first available video view
                    video_path = next(iter(episode['videos'].values()))
                    if data_path and isinstance(video_path, str):
                        video_full_path = data_path / video_path if not Path(video_path).is_absolute() else video_path
                        frames = embedding_manager.load_visual_data(str(video_full_path))
                    else:
                        frames = embedding_manager.load_visual_data(video_path)
                elif episode.get('video'):
                    # Single video processing
                    video_path = episode['video']
                    if data_path and isinstance(video_path, str):
                        video_full_path = data_path / video_path if not Path(video_path).is_absolute() else video_path
                        frames = embedding_manager.load_visual_data(str(video_full_path))
                    else:
                        frames = embedding_manager.load_visual_data(video_path)
                else:
                    continue
                
                if frames:
                    # Use first frame as representative
                    frame_embedding = embedding_manager.encode_image_clip([frames[0]])
                    embeddings.append(frame_embedding.cpu().numpy()[0])
                    
            except Exception as e:
                logger.warning(f"Could not process visual data for episode {i}: {e}")
                continue
        
        if len(embeddings) < self.min_visual_samples:
            logger.warning("Not enough visual samples for diversity analysis")
            return 0.3
        
        # Calculate visual diversity
        embeddings = np.array(embeddings)
        
        # Method 1: Average pairwise distance
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_norm, embeddings_norm.T)
        
        # Get upper triangular part (excluding diagonal)
        mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
        pairwise_similarities = similarities[mask]
        
        # Diversity is 1 - average similarity
        diversity = 1.0 - np.mean(pairwise_similarities)
        
        return max(0.0, min(1.0, diversity))
    
    def _analyze_failure_rates(self, dataset: List[Dict]) -> float:
        """Analyze the failure rate in the dataset."""
        success_labels = []
        
        for episode in dataset:
            # Check for explicit success labels
            if 'success' in episode:
                success_labels.append(episode['success'])
            elif 'failed' in episode:
                success_labels.append(not episode['failed'])
            # Could also infer from task completion or other metrics
        
        if not success_labels:
            # No explicit success/failure labels available
            logger.warning("No success/failure labels found in dataset - cannot evaluate failure rate")
            return None
        
        failure_rate = (len(success_labels) - sum(success_labels)) / len(success_labels)
        
        # Score based on failure rate
        # Sweet spot is 5-15% failures (shows robustness without too much noise)
        if 0.05 <= failure_rate <= 0.15:
            score = 1.0
        elif 0.01 <= failure_rate <= 0.25:
            score = 0.7
        elif failure_rate > 0:
            score = 0.4
        else:
            # 0% failure rate is actually bad (too curated/brittle)
            score = 0.2
        
        return score
    
    def get_coverage_breakdown(self, dataset: List[Dict], data_path: Path, embedding_manager) -> Dict[str, Any]:
        """Get detailed coverage breakdown."""
        
        # Task analysis
        tasks = [episode.get('task', episode.get('prompt', 'unknown')) 
                for episode in dataset]
        tasks = [t for t in tasks if t and t != 'unknown']
        unique_tasks = len(set(task.lower().strip() for task in tasks))
        
        # Scale analysis
        num_trajectories = len(dataset)
        scale_tier = self._get_scale_tier(num_trajectories)
        
        # Task diversity tier
        if unique_tasks >= self.task_thresholds['gold']:
            task_tier = 'Gold'
        elif unique_tasks >= self.task_thresholds['silver']:
            task_tier = 'Silver'
        elif unique_tasks >= self.task_thresholds['bronze']:
            task_tier = 'Bronze'
        else:
            task_tier = 'Poor'
        
        # Failure rate analysis
        failure_rate = self._get_failure_rate(dataset)
        
        return {
            'scale': {
                'num_trajectories': num_trajectories,
                'tier': scale_tier
            },
            'task_diversity': {
                'unique_tasks': unique_tasks,
                'tier': task_tier,
                'task_list': list(set(task.lower().strip() for task in tasks))[:10]  # Top 10
            },
            'failure_rate': {
                'rate': failure_rate,
                'has_failure_data': failure_rate is not None
            }
        }
    
    def _get_scale_tier(self, num_trajectories: int) -> str:
        """Get scale tier based on number of trajectories."""
        if num_trajectories >= self.trajectory_thresholds['gold']:
            return 'Gold'
        elif num_trajectories >= self.trajectory_thresholds['silver']:
            return 'Silver'
        elif num_trajectories >= self.trajectory_thresholds['bronze']:
            return 'Bronze'
        else:
            return 'Small'
    
    def _get_failure_rate(self, dataset: List[Dict]) -> float:
        """Get failure rate from dataset."""
        success_labels = []
        
        for episode in dataset:
            if 'success' in episode:
                success_labels.append(episode['success'])
            elif 'failed' in episode:
                success_labels.append(not episode['failed'])
        
        if not success_labels:
            return None
        
        return (len(success_labels) - sum(success_labels)) / len(success_labels)