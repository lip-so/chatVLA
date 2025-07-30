"""
Visual Diversity Metric

Evaluates the visual diversity of a robotics dataset by analyzing:
- Pairwise frame similarities
- Clustering-based diversity
- Entropy of visual features
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config

logger = logging.getLogger(__name__)

class VisualDiversityMetric:
    """Evaluates visual diversity of a dataset."""
    
    def __init__(self):
        """Initialize the visual diversity metric with configuration."""
        # Load configuration
        self.config = get_config('visual_diversity')
        self.general_config = get_config('general')
        
        # Use configuration values
        self.num_frames = self.config.num_frames
        self.sample_size = self.config.sample_size
        self.min_samples_for_clustering = self.config.min_samples_for_clustering
        self.n_clusters = self.config.n_clusters
        self.random_state = self.config.random_state
        self.weights = self.config.weights
    
    def compute(self, dataset: List[Dict], data_path: Path, embedding_manager) -> float:
        """
        Compute visual diversity score for the dataset.
        
        Args:
            dataset: List of dataset samples with 'video' field
            data_path: Path to dataset directory (can be None for HF datasets)
            embedding_manager: EmbeddingManager instance
            
        Returns:
            Float score between 0 and 1, where 1 indicates high diversity
        """
        if not dataset:
            logger.warning("Empty dataset provided to visual diversity metric")
            return 0.0
        
        logger.info(f"Computing visual diversity for {len(dataset)} episodes")
        
        # Try multiple approaches to compute visual diversity
        # 1. Video-based diversity (traditional)
        video_diversity = self._compute_video_based_diversity(dataset, data_path, embedding_manager)
        
        # 2. Action-based diversity (proxy for visual diversity)
        action_diversity = self._compute_action_based_diversity(dataset, data_path)
        
        # 3. State-based diversity (proxy for environmental diversity)
        state_diversity = self._compute_state_based_diversity(dataset, data_path)
        
        # 4. Task-based diversity (proxy for scene diversity)
        task_diversity = self._compute_task_based_diversity(dataset)
        
        # Combine available diversity measures
        available_scores = []
        weights = []
        
        if video_diversity is not None:
            available_scores.append(video_diversity)
            weights.append(0.5)  # Video diversity is most important
            logger.info(f"Video diversity: {video_diversity:.3f}")
        
        if action_diversity is not None:
            available_scores.append(action_diversity)
            weights.append(0.3)  # Action diversity as proxy
            logger.info(f"Action diversity: {action_diversity:.3f}")
        
        if state_diversity is not None:
            available_scores.append(state_diversity)
            weights.append(0.2)  # State diversity as environmental proxy
            logger.info(f"State diversity: {state_diversity:.3f}")
        
        if task_diversity is not None:
            available_scores.append(task_diversity)
            weights.append(0.1)  # Task diversity as scene proxy
            logger.info(f"Task diversity: {task_diversity:.3f}")
        
        if not available_scores:
            logger.warning("No diversity measures could be computed")
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted combination
        final_diversity_raw = sum(score * weight for score, weight in zip(available_scores, normalized_weights))

        # Clamp to [0, 1] range so the metric always returns a valid score
        final_diversity = self._clamp_score(final_diversity_raw)
        
        logger.info(f"Visual diversity: {final_diversity:.3f} (computed from {len(available_scores)} measures, raw={final_diversity_raw:.3f})")
        return float(final_diversity)

    @staticmethod
    def _clamp_score(value: float) -> float:
        """Ensure a score is within [0, 1]."""
        if value is None or np.isnan(value):
            return 0.0
        return float(max(0.0, min(1.0, value)))
    
    def _compute_pairwise_diversity(self, features: np.ndarray) -> float:
        """Compute diversity based on pairwise distances."""
        if len(features) < 2:
            return 0.0
            
        # Sample to avoid computational explosion
        if len(features) > 1000:
            indices = np.random.choice(len(features), 1000, replace=False)
            features = features[indices]
        
        # Normalize features to have unit norm for better comparison
        features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        
        # Compute pairwise distances
        distances = []
        for i in range(len(features_normalized)):
            for j in range(i + 1, len(features_normalized)):
                dist = np.linalg.norm(features_normalized[i] - features_normalized[j])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Diversity is average pairwise distance
        avg_distance = np.mean(distances)
        
        # Normalize to [0, 1] range
        # For normalized vectors, max distance is 2 (opposite unit vectors)
        # Common distance range is [0, 2], so we divide by 2
        normalized_diversity = min(1.0, avg_distance / 2.0)
        
        return normalized_diversity
    
    def _compute_cluster_diversity(self, features: np.ndarray) -> float:
        """Compute diversity based on clustering."""
        if len(features) < self.min_samples_for_clustering:
            logger.debug(f"Insufficient samples for clustering: {len(features)} < {self.min_samples_for_clustering}")
            return 0.0
        
        try:
            # Determine optimal number of clusters
            max_clusters = min(8, len(features) // 2)
            if max_clusters < 2:
                return 0.0
            
            # Try to import and use clustering
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            best_score = -1
            best_k = 2
            
            for k in range(2, max_clusters + 1):
                try:
                    # Use simple initialization to avoid threadpool issues
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=1, max_iter=100)
                    cluster_labels = kmeans.fit_predict(features)
                    
                    # Check if we have valid clusters
                    unique_labels = np.unique(cluster_labels)
                    if len(unique_labels) < 2:
                        continue
                    
                    score = silhouette_score(features, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
                except Exception as e:
                    logger.debug(f"Clustering with k={k} failed: {e}")
                    continue
            
            if best_score == -1:
                logger.debug("All clustering attempts failed, skipping cluster diversity")
                return 0.0
            
            # Final clustering with best k
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=1, max_iter=100)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate silhouette score
            silhouette = silhouette_score(features, cluster_labels)
            
            # Convert to diversity score (0-1 scale)
            diversity = (silhouette + 1) / 2
            
            return diversity
            
        except Exception as e:
            logger.debug(f"Clustering failed due to system-level issue (this is OK): {e}")
            return 0.0
    
    def _compute_entropy_diversity(self, features: np.ndarray) -> float:
        """Compute diversity based on feature distribution entropy."""
        if len(features) < 2:
            return 0.0
        
        try:
            # Quantize features into bins for entropy calculation
            num_bins = 50
            feature_dim = features.shape[1]
            
            # Normalize features to [0, 1] range
            features_norm = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-10)
            
            # Compute entropy for each dimension
            entropies = []
            for dim in range(feature_dim):
                hist, _ = np.histogram(features_norm[:, dim], bins=num_bins, density=True)
                hist = hist + 1e-10  # Add small epsilon to avoid log(0)
                hist = hist / hist.sum()  # Normalize
                entropy = -np.sum(hist * np.log2(hist))
                entropies.append(entropy)
            
            # Average entropy across dimensions
            avg_entropy = np.mean(entropies)
            max_entropy = np.log2(num_bins)
            
            diversity = avg_entropy / max_entropy
            
            return diversity
            
        except Exception as e:
            logger.error(f"Error in entropy diversity computation: {e}")
            return 0.0
    
    def _compute_video_based_diversity(self, dataset: List[Dict], data_path: Path, embedding_manager) -> Optional[float]:
        """Compute diversity based on video content (traditional method)."""
        try:
            # Sample dataset if it's too large
            if len(dataset) > self.sample_size:
                np.random.seed(42)
                indices = np.random.choice(len(dataset), self.sample_size, replace=False)
                sampled_dataset = [dataset[i] for i in indices]
            else:
                sampled_dataset = dataset
            
            # Process samples
            all_features = []
            valid_samples = 0
            
            for i, sample in enumerate(sampled_dataset):
                try:
                    # Handle different video data formats
                    if 'videos' in sample and isinstance(sample['videos'], dict):
                        # Multi-view video data
                        view_features = []
                        for view_name, video_path in sample['videos'].items():
                            try:
                                if data_path and isinstance(video_path, str):
                                    full_video_path = data_path / video_path
                                    if full_video_path.exists():
                                        frames = embedding_manager.load_video_frames(str(full_video_path), self.num_frames)
                                    else:
                                        continue
                                else:
                                    # Handle HuggingFace video objects or other formats
                                    frames = embedding_manager.load_visual_data(video_path)
                                
                                if frames and len(frames) > 0:
                                    # Extract features from sampled frames
                                    sampled_frames = frames[::max(1, len(frames)//self.num_frames)][:self.num_frames]
                                    if len(sampled_frames) > 0:
                                        frame_features = embedding_manager.encode_image_clip(sampled_frames)
                                        view_feature = frame_features.mean(dim=0)
                                        view_features.append(view_feature)
                            except Exception as e:
                                logger.debug(f"Error processing view {view_name} for episode {i}: {e}")
                                continue
                        
                        # Combine features from all views
                        if view_features:
                            combined_feature = torch.stack(view_features).mean(dim=0)
                            all_features.append(combined_feature)
                            valid_samples += 1
                    
                    elif 'video' in sample and sample['video'] is not None:
                        # Single video data
                        video_path = sample['video']
                        try:
                            if data_path and isinstance(video_path, str):
                                full_video_path = data_path / video_path
                                if full_video_path.exists():
                                    frames = embedding_manager.load_video_frames(str(full_video_path), self.num_frames)
                                else:
                                    continue
                            else:
                                # Handle HuggingFace video objects or other formats
                                frames = embedding_manager.load_visual_data(video_path)
                            
                            if frames and len(frames) > 0:
                                # Extract features from sampled frames
                                sampled_frames = frames[::max(1, len(frames)//self.num_frames)][:self.num_frames]
                                if len(sampled_frames) > 0:
                                    frame_features = embedding_manager.encode_image_clip(sampled_frames)
                                    video_feature = frame_features.mean(dim=0)
                                    all_features.append(video_feature)
                                    valid_samples += 1
                        except Exception as e:
                            logger.debug(f"Error processing video for episode {i}: {e}")
                            continue
                    
                    # Alternative: Try to extract images from episode_data (HuggingFace format)
                    elif 'episode_data' in sample and sample['episode_data']:
                        try:
                            episode_data = sample['episode_data']
                            # Look for image observations in the episode data
                            image_keys = []
                            for key in episode_data[0].keys():
                                if 'image' in key.lower() or 'observation' in key.lower():
                                    image_keys.append(key)
                            
                            if image_keys:
                                # Sample frames from the episode
                                frame_indices = np.linspace(0, len(episode_data)-1, self.num_frames, dtype=int)
                                sampled_frames = []
                                
                                for frame_idx in frame_indices:
                                    frame_data = episode_data[frame_idx]
                                    # Try to get images from any available key
                                    for img_key in image_keys:
                                        if img_key in frame_data and frame_data[img_key] is not None:
                                            img_data = frame_data[img_key]
                                            try:
                                                # Handle different image formats
                                                processed_img = embedding_manager.load_visual_data(img_data)
                                                if processed_img:
                                                    sampled_frames.append(processed_img)
                                                    break
                                            except:
                                                continue
                                
                                if sampled_frames:
                                    # Flatten if needed (some might be nested)
                                    flat_frames = []
                                    for frame in sampled_frames:
                                        if isinstance(frame, list):
                                            flat_frames.extend(frame)
                                        else:
                                            flat_frames.append(frame)
                                    
                                    if flat_frames:
                                        frame_features = embedding_manager.encode_image_clip(flat_frames)
                                        video_feature = frame_features.mean(dim=0)
                                        all_features.append(video_feature)
                                        valid_samples += 1
                        except Exception as e:
                            logger.debug(f"Error processing episode_data for episode {i}: {e}")
                            continue
                    
                    if i % 100 == 0:
                        logger.debug(f"Processed {i}/{len(sampled_dataset)} samples")
                        
                except Exception as e:
                    logger.debug(f"Error processing sample {i}: {e}")
                    continue
            
            if valid_samples < 2:
                logger.debug(f"Insufficient video samples for diversity analysis: {valid_samples}")
                return None
            
            # Convert features to numpy array
            features_array = torch.stack(all_features).cpu().numpy()
            
            # Calculate different diversity metrics
            pairwise_diversity = self._compute_pairwise_diversity(features_array)
            cluster_diversity = self._compute_cluster_diversity(features_array)
            entropy_diversity = self._compute_entropy_diversity(features_array)
            
            # Weighted combination using configuration
            total_diversity = (
                self.weights.get('pairwise', 0.4) * pairwise_diversity +
                self.weights.get('cluster', 0.3) * cluster_diversity +
                self.weights.get('entropy', 0.3) * entropy_diversity
            )
            
            logger.debug(f"Video-based diversity: {total_diversity:.3f} (from {valid_samples} samples)")
            return float(total_diversity)
            
        except Exception as e:
            logger.debug(f"Video-based diversity computation failed: {e}")
            return None
    
    def _compute_action_based_diversity(self, dataset: List[Dict], data_path: Path) -> Optional[float]:
        """Compute diversity based on action sequences as a proxy for visual diversity."""
        try:
            action_sequences = []
            
            for sample in dataset:
                # Try to get actions from different sources
                actions = None
                
                # Method 1: Direct action data (HF datasets)
                if 'actions' in sample:
                    actions = sample['actions']
                
                # Method 2: From episode data
                elif 'episode_data' in sample:
                    episode_data = sample['episode_data']
                    if episode_data and len(episode_data) > 0:
                        actions = [item.get('action') for item in episode_data if 'action' in item]
                
                # Method 3: Load from data file (local datasets)
                elif 'data_path' in sample and data_path:
                    try:
                        data_file_path = sample['data_path']
                        if not data_file_path.startswith('hf_episode_'):
                            if not Path(data_file_path).is_absolute():
                                full_data_path = data_path / data_file_path
                            else:
                                full_data_path = Path(data_file_path)
                            
                            if full_data_path.exists():
                                import pandas as pd
                                df = pd.read_parquet(str(full_data_path))
                                if 'action' in df.columns:
                                    actions = df['action'].tolist()
                    except Exception as e:
                        logger.debug(f"Error loading action data from file: {e}")
                        continue
                
                # Convert actions to numpy array
                if actions:
                    if not isinstance(actions, np.ndarray):
                        try:
                            actions = np.array(actions)
                        except:
                            continue
                    
                    if len(actions) > 0 and actions.ndim == 2:
                        # Compute action statistics as diversity features
                        action_stats = np.concatenate([
                            np.mean(actions, axis=0),  # Mean actions
                            np.std(actions, axis=0),   # Action variability
                            np.max(actions, axis=0),   # Max actions
                            np.min(actions, axis=0)    # Min actions
                        ])
                        action_sequences.append(action_stats)
            
            if len(action_sequences) < 2:
                logger.debug(f"Insufficient action sequences for diversity analysis: {len(action_sequences)}")
                return None
            
            # Compute diversity of action statistics
            action_features = np.array(action_sequences)
            
            # Calculate pairwise distances
            pairwise_diversity = self._compute_pairwise_diversity(action_features)
            
            logger.debug(f"Action-based diversity: {pairwise_diversity:.3f} (from {len(action_sequences)} sequences)")
            return float(pairwise_diversity)
            
        except Exception as e:
            logger.debug(f"Action-based diversity computation failed: {e}")
            return None
    
    def _compute_state_based_diversity(self, dataset: List[Dict], data_path: Path) -> Optional[float]:
        """Compute diversity based on state observations as a proxy for environmental diversity."""
        try:
            state_sequences = []
            
            for sample in dataset:
                # Try to get states from different sources
                states = None
                
                # Method 1: Direct observation data (HF datasets)
                if 'observations' in sample:
                    states = sample['observations']
                elif 'observation.state' in sample:
                    states = sample['observation.state']
                
                # Method 2: From episode data
                elif 'episode_data' in sample:
                    episode_data = sample['episode_data']
                    if episode_data and len(episode_data) > 0:
                        states = [item.get('observation.state') for item in episode_data if 'observation.state' in item]
                
                # Method 3: Load from data file (local datasets)
                elif 'data_path' in sample and data_path:
                    try:
                        data_file_path = sample['data_path']
                        if not data_file_path.startswith('hf_episode_'):
                            if not Path(data_file_path).is_absolute():
                                full_data_path = data_path / data_file_path
                            else:
                                full_data_path = Path(data_file_path)
                            
                            if full_data_path.exists():
                                import pandas as pd
                                df = pd.read_parquet(str(full_data_path))
                                if 'observation.state' in df.columns:
                                    states = df['observation.state'].tolist()
                    except Exception as e:
                        logger.debug(f"Error loading state data from file: {e}")
                        continue
                
                # Convert states to numpy array
                if states:
                    if not isinstance(states, np.ndarray):
                        try:
                            states = np.array(states)
                        except:
                            continue
                    
                    if len(states) > 0 and states.ndim == 2:
                        # Compute state statistics as diversity features
                        state_stats = np.concatenate([
                            np.mean(states, axis=0),  # Mean states
                            np.std(states, axis=0),   # State variability
                            np.max(states, axis=0),   # Max states
                            np.min(states, axis=0)    # Min states
                        ])
                        state_sequences.append(state_stats)
            
            if len(state_sequences) < 2:
                logger.debug(f"Insufficient state sequences for diversity analysis: {len(state_sequences)}")
                return None
            
            # Compute diversity of state statistics
            state_features = np.array(state_sequences)
            
            # Calculate pairwise distances
            pairwise_diversity = self._compute_pairwise_diversity(state_features)
            
            logger.debug(f"State-based diversity: {pairwise_diversity:.3f} (from {len(state_sequences)} sequences)")
            return float(pairwise_diversity)
            
        except Exception as e:
            logger.debug(f"State-based diversity computation failed: {e}")
            return None
    
    def _compute_task_based_diversity(self, dataset: List[Dict]) -> Optional[float]:
        """Compute diversity based on task descriptions as a proxy for scene diversity."""
        try:
            tasks = []
            
            for sample in dataset:
                task = sample.get('task', sample.get('prompt', ''))
                if task and isinstance(task, str) and len(task.strip()) > 0:
                    tasks.append(task.strip().lower())
            
            if len(tasks) < 2:
                logger.debug(f"Insufficient task descriptions for diversity analysis: {len(tasks)}")
                return None
            
            # Count unique tasks
            unique_tasks = len(set(tasks))
            total_tasks = len(tasks)
            
            # Basic diversity: ratio of unique to total
            basic_diversity = unique_tasks / total_tasks
            
            # Enhanced diversity: consider task distribution
            from collections import Counter
            task_counts = Counter(tasks)
            
            # Calculate entropy of task distribution
            task_probs = np.array(list(task_counts.values())) / total_tasks
            entropy = -np.sum(task_probs * np.log2(task_probs + 1e-10))
            max_entropy = np.log2(len(task_counts))
            
            # Normalize entropy
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Combine basic diversity and entropy
            task_diversity = 0.5 * basic_diversity + 0.5 * normalized_entropy
            
            logger.debug(f"Task-based diversity: {task_diversity:.3f} (from {len(tasks)} tasks, {unique_tasks} unique)")
            return float(task_diversity)
            
        except Exception as e:
            logger.debug(f"Task-based diversity computation failed: {e}")
            return None
    
    def compute_per_category(self, dataset: List[Dict], data_path: Path, embedding_manager) -> Dict[str, float]:
        """
        Compute visual diversity for different categories if available.
        
        Returns:
            Dictionary mapping category names to diversity scores
        """
        # Group samples by category (if available)
        categories = {}
        for sample in dataset:
            category = sample.get('category', 'default')
            if category not in categories:
                categories[category] = []
            categories[category].append(sample)
        
        # Compute diversity for each category
        category_diversity = {}
        for category, samples in categories.items():
            if len(samples) >= 2:
                diversity = self.compute(samples, data_path, embedding_manager)
                category_diversity[category] = diversity
        
        return category_diversity
    
    def analyze_visual_clusters(self, dataset: List[Dict], data_path: Path, embedding_manager, num_clusters: int = 5) -> Dict[str, Any]:
        """
        Analyze visual clusters in the dataset.
        
        Returns:
            Dictionary with cluster analysis results
        """
        # Extract features (similar to compute method)
        all_features = []
        sample_info = []
        
        for sample in dataset:
            try:
                video_path = data_path / sample['video']
                frames = embedding_manager.load_video_frames(str(video_path), self.num_frames)
                
                if not frames:
                    continue
                
                frame_features = embedding_manager.encode_image_clip(frames)
                video_features = frame_features.mean(dim=0).cpu().numpy()
                
                all_features.append(video_features)
                sample_info.append({
                    'video': sample['video'],
                    'prompt': sample.get('prompt', ''),
                })
                
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue
        
        if len(all_features) < num_clusters:
            return {'error': 'Not enough samples for clustering analysis'}
        
        # Perform clustering
        features_array = np.array(all_features)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_array)
        
        # Analyze clusters
        cluster_analysis = {
            'num_clusters': num_clusters,
            'cluster_sizes': {},
            'cluster_examples': {}
        }
        
        for i in range(num_clusters):
            cluster_mask = labels == i
            cluster_size = np.sum(cluster_mask)
            cluster_analysis['cluster_sizes'][f'cluster_{i}'] = int(cluster_size)
            
            # Get a few examples from each cluster
            cluster_indices = np.where(cluster_mask)[0][:3]
            examples = [sample_info[idx] for idx in cluster_indices]
            cluster_analysis['cluster_examples'][f'cluster_{i}'] = examples
        
        return cluster_analysis