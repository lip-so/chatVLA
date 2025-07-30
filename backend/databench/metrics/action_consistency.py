"""
Action Consistency Metric

Evaluates the consistency of actions across different modalities and contexts:
- Visual-text consistency (CLIP/BLIP alignment)
- Action-observation consistency within episodes
- Cross-episode consistency for similar tasks
- Temporal consistency of action sequences
- Prompt-action correspondence verification (NEW)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import torch

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config

logger = logging.getLogger(__name__)

# Import the new prompt-action verifier
try:
    from .prompt_action_verifier import PromptActionVerifier
    PROMPT_VERIFIER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Prompt-action verifier not available: {e}")
    PromptActionVerifier = None
    PROMPT_VERIFIER_AVAILABLE = False

class ActionConsistencyMetric:
    """Evaluates action consistency across different modalities and contexts."""
    
    def __init__(self, enable_prompt_verification: bool = True, robot_type: str = None):
        """
        Initialize the action consistency metric with configuration.
        
        Args:
            enable_prompt_verification: Whether to enable the new prompt-action verification
            robot_type: Robot type for the prompt-action verifier
        """
        # Load configuration
        self.config = get_config('action_consistency')
        self.general_config = get_config('general')
        
        # Initialize DOF tracking for dynamic configuration
        self.detected_dof = None
        
        # Use configuration values
        self.num_frames = self.config.num_frames
        self.min_episodes_for_cross_analysis = self.config.min_episodes_for_cross_analysis
        self.min_sequence_length = self.config.min_sequence_length
        self.correlation_threshold = self.config.correlation_threshold
        self.smoothness_window = self.config.smoothness_window
        self.weights = self.config.weights
        
        # Initialize prompt-action verifier if enabled
        self.enable_prompt_verification = enable_prompt_verification and PROMPT_VERIFIER_AVAILABLE
        self.prompt_verifier = None
        
        if self.enable_prompt_verification:
            try:
                # Detect robot type from configuration or use default
                if robot_type is None:
                    robot_type = "generic_6dof"  # Default
                
                self.prompt_verifier = PromptActionVerifier(robot_type=robot_type)
                logger.info("Prompt-action verification enabled")
                
                # Update weights to include prompt verification
                self.weights = {
                    'visual_text': self.weights.get('visual_text', 0.25),
                    'action_observation': self.weights.get('action_observation', 0.25),
                    'cross_episode': self.weights.get('cross_episode', 0.15),
                    'temporal': self.weights.get('temporal', 0.15),
                    'prompt_action': 0.20  # New component
                }
            except Exception as e:
                logger.warning(f"Failed to initialize prompt-action verifier: {e}")
                self.enable_prompt_verification = False
                self.prompt_verifier = None
        elif enable_prompt_verification and not PROMPT_VERIFIER_AVAILABLE:
            logger.info("Prompt-action verification requested but dependencies not available")
    
    def compute(self, dataset: List[Dict], data_path: Path, embedding_manager) -> float:
        if not dataset:
            return 0.0
        
        # Compute different types of consistency
        visual_text_consistency = self._compute_visual_text_consistency(dataset, data_path, embedding_manager)
        action_observation_consistency = self._compute_action_observation_consistency(dataset, data_path)
        cross_episode_consistency = self._compute_cross_episode_consistency(dataset, data_path)
        temporal_consistency = self._compute_temporal_consistency(dataset, data_path, embedding_manager)
        
        # Compute prompt-action consistency if enabled
        prompt_action_consistency = None
        if self.enable_prompt_verification and self.prompt_verifier:
            prompt_action_consistency = self._compute_prompt_action_consistency(dataset, data_path)
        
        # Handle None values (metrics that couldn't be evaluated)
        available_scores = []
        available_weights = []
        
        if visual_text_consistency is not None:
            available_scores.append(visual_text_consistency)
            available_weights.append(self.weights.get('visual_text', 0.3))
        else:
            logger.warning("Visual-text consistency could not be evaluated")
        
        if action_observation_consistency is not None:
            available_scores.append(action_observation_consistency)
            available_weights.append(self.weights.get('action_observation', 0.3))
        else:
            logger.warning("Action-observation consistency could not be evaluated")
        
        if cross_episode_consistency is not None:
            available_scores.append(cross_episode_consistency)
            available_weights.append(self.weights.get('cross_episode', 0.2))
        else:
            logger.warning("Cross-episode consistency could not be evaluated")
        
        if temporal_consistency is not None:
            available_scores.append(temporal_consistency)
            available_weights.append(self.weights.get('temporal', 0.2))
        else:
            logger.warning("Temporal consistency could not be evaluated")
        
        if prompt_action_consistency is not None:
            available_scores.append(prompt_action_consistency)
            available_weights.append(self.weights.get('prompt_action', 0.2))
        else:
            if self.enable_prompt_verification:
                logger.warning("Prompt-action consistency could not be evaluated")
        
        if not available_scores:
            logger.error("No consistency metrics could be evaluated")
            return 0.0
        
        # Normalize weights for available scores
        total_weight = sum(available_weights)
        normalized_weights = [w / total_weight for w in available_weights]
        
        # Weighted combination of available consistency types
        total_consistency = sum(score * weight for score, weight in zip(available_scores, normalized_weights))
        
        logger.info(f"Action Consistency: {total_consistency:.3f}")
        if visual_text_consistency is not None:
            logger.info(f"  - Visual-Text: {visual_text_consistency:.3f}")
        else:
            logger.info(f"  - Visual-Text: Not evaluated")
        if action_observation_consistency is not None:
            logger.info(f"  - Action-Observation: {action_observation_consistency:.3f}")
        else:
            logger.info(f"  - Action-Observation: Not evaluated")
        if cross_episode_consistency is not None:
            logger.info(f"  - Cross-Episode: {cross_episode_consistency:.3f}")
        else:
            logger.info(f"  - Cross-Episode: Not evaluated")
        if temporal_consistency is not None:
            logger.info(f"  - Temporal: {temporal_consistency:.3f}")
        else:
            logger.info(f"  - Temporal: Not evaluated")
        if prompt_action_consistency is not None:
            logger.info(f"  - Prompt-Action: {prompt_action_consistency:.3f}")
        else:
            if self.enable_prompt_verification:
                logger.info(f"  - Prompt-Action: Not evaluated")
        
        return float(total_consistency)
    
    def _compute_prompt_action_consistency(self, dataset: List[Dict], data_path: Path) -> Optional[float]:
        """Compute prompt-action consistency using the new verification agent."""
        if not self.prompt_verifier:
            return None
        
        consistency_scores = []
        sample_size = min(10, len(dataset))  # Limit sample size for performance
        
        logger.info(f"Computing prompt-action consistency for {sample_size} episodes")
        
        for i in range(sample_size):
            try:
                episode = dataset[i]
                
                # Use the verification agent to analyze this episode
                result = self.prompt_verifier.verify_dataset_episode(episode, data_path)
                
                # Extract overall verification score
                verification_score = result.verification_scores.overall_verification_score
                consistency_scores.append(verification_score)
                
                # Log detailed results for first episode
                if i == 0:
                    logger.info(f"First episode prompt-action verification:")
                    logger.info(f"  - Overall Score: {verification_score:.3f}")
                    logger.info(f"  - Semantic Mapping: {result.verification_scores.semantic_mapping_accuracy:.3f}")
                    logger.info(f"  - Temporal Alignment: {result.verification_scores.temporal_alignment:.3f}")
                    logger.info(f"  - Task Completion: {result.verification_scores.task_completion_verification:.3f}")
                    
                    if result.issues_found:
                        logger.info(f"  - Issues Found: {len(result.issues_found)}")
                        for issue in result.issues_found[:3]:  # Show first 3 issues
                            logger.info(f"    * {issue['description']}")
            
            except Exception as e:
                logger.warning(f"Error verifying episode {i}: {e}")
                # Continue with other episodes
                continue
        
        if not consistency_scores:
            logger.warning("No episodes could be verified for prompt-action consistency")
            return None
        
        avg_consistency = np.mean(consistency_scores)
        logger.info(f"Prompt-action consistency computed from {len(consistency_scores)} episodes: {avg_consistency:.3f}")
        
        return avg_consistency

    def _compute_visual_text_consistency(self, dataset: List[Dict], data_path: Path, embedding_manager) -> float:
        """Compute visual-text consistency using CLIP similarity between prompts and video frames."""
        if not embedding_manager:
            logger.warning("No embedding manager available for visual-text consistency evaluation")
            return None
        
        # Try to ensure models are loaded
        try:
            embedding_manager.ensure_models_loaded()
        except Exception as e:
            logger.warning(f"Failed to load embedding models: {e}")
            return None
            
        consistency_scores = []
        valid_samples = 0
        
        # Sample episodes for analysis
        sample_size = min(20, len(dataset))
        logger.info(f"Computing visual-text consistency for {sample_size} episodes")
        
        for i in range(sample_size):
            sample = dataset[i]
            
            try:
                prompt = sample.get('prompt', sample.get('task', ''))
                if not prompt:
                    continue
                
                frames = None
                
                # Handle multiple video views if available
                if 'videos' in sample and isinstance(sample['videos'], dict):
                    if i == 0:  # Log only for first sample to avoid spam
                        logger.info(f"Processing episodes with {len(sample['videos'])} camera views: {list(sample['videos'].keys())}")
                    
                    view_similarities = []
                    for view_name, video_path in sample['videos'].items():
                        if video_path is None:
                            continue
                            
                        try:
                            # Handle different video data formats
                            if isinstance(video_path, dict):
                                # HF dataset format
                                frames = embedding_manager.load_visual_data(video_path)
                            elif isinstance(video_path, str):
                                # File path format
                                if data_path and not Path(video_path).is_absolute():
                                    video_full_path = data_path / video_path
                                    frames = embedding_manager.load_visual_data(str(video_full_path))
                                else:
                                    frames = embedding_manager.load_visual_data(video_path)
                            else:
                                # Direct video data (e.g., PIL Images, numpy arrays)
                                frames = embedding_manager.load_visual_data(video_path)
                            
                            if frames and len(frames) > 0:
                                # Sample frames for efficiency
                                sampled_frames = frames[::max(1, len(frames)//self.num_frames)][:self.num_frames]
                                if len(sampled_frames) > 0:
                                    # Compute CLIP similarity for this view
                                    similarity = embedding_manager.compute_clip_similarity([prompt], sampled_frames)
                                    view_similarities.append(similarity)
                                    logger.debug(f"  - View '{view_name}': {similarity:.3f} similarity")
                        except Exception as e:
                            logger.warning(f"Error processing view {view_name} for episode {i}: {e}")
                            continue
                    
                    if view_similarities:
                        # Average similarity across all views
                        avg_similarity = np.mean(view_similarities)
                        consistency_scores.append(avg_similarity)
                        valid_samples += 1
                        logger.debug(f"Episode {i}: Combined similarity from {len(view_similarities)} views: {avg_similarity:.3f}")
                
                # Handle single video
                elif 'video' in sample and sample['video'] is not None:
                    try:
                        video_data = sample['video']
                        
                        # Handle different video data formats
                        if isinstance(video_data, dict):
                            # HF dataset format
                            frames = embedding_manager.load_visual_data(video_data)
                        elif isinstance(video_data, str):
                            # File path format
                            if data_path and not Path(video_data).is_absolute():
                                video_full_path = data_path / video_data
                                frames = embedding_manager.load_visual_data(str(video_full_path))
                            else:
                                frames = embedding_manager.load_visual_data(video_data)
                        else:
                            # Direct video data
                            frames = embedding_manager.load_visual_data(video_data)
                        
                        if frames and len(frames) > 0:
                            # Sample frames for efficiency
                            sampled_frames = frames[::max(1, len(frames)//self.num_frames)][:self.num_frames]
                            if len(sampled_frames) > 0:
                                similarity = embedding_manager.compute_clip_similarity([prompt], sampled_frames)
                                consistency_scores.append(similarity)
                                valid_samples += 1
                                logger.debug(f"Episode {i}: Single video similarity: {similarity:.3f}")
                    except Exception as e:
                        logger.warning(f"Error processing single video for episode {i}: {e}")
                        continue
                
                # For HF datasets, try to extract images from the data if available
                elif 'images' in sample or 'image' in sample:
                    try:
                        images = sample.get('images', [sample.get('image')])
                        if images and len(images) > 0:
                            # Convert to PIL Images if needed
                            pil_images = []
                            for img in images:
                                if hasattr(img, 'convert'):  # Already PIL Image
                                    pil_images.append(img)
                                elif isinstance(img, np.ndarray):
                                    from PIL import Image
                                    pil_images.append(Image.fromarray(img))
                            
                            if pil_images:
                                similarity = embedding_manager.compute_clip_similarity([prompt], pil_images)
                                consistency_scores.append(similarity)
                                valid_samples += 1
                                logger.debug(f"Episode {i}: Image similarity: {similarity:.3f}")
                    except Exception as e:
                        logger.warning(f"Error processing images for episode {i}: {e}")
                        continue
                    
            except Exception as e:
                logger.warning(f"Error processing episode {i}: {e}")
                continue
        
        if not consistency_scores:
            logger.warning("No visual-text consistency scores computed - no valid video/image data found")
            return None
        
        avg_consistency = np.mean(consistency_scores)
        logger.info(f"Visual-text consistency computed from {valid_samples} episodes: {avg_consistency:.3f}")
        
        return float(avg_consistency)

    def _compute_action_observation_consistency(self, dataset: List[Dict], data_path: Path) -> float:
        """Compute action-observation consistency within episodes."""
        consistency_scores = []
        
        # Sample episodes for analysis
        sample_size = min(20, len(dataset))
        for i in range(sample_size):
            episode = dataset[i]
            
            try:
                # Try to analyze directly from episode data first (for HF datasets)
                score = self._analyze_episode_action_obs_consistency_from_data(episode)
                
                # Fallback to data_path method for local datasets
                if score is None and 'data_path' in episode:
                    score = self._analyze_episode_action_obs_consistency(episode['data_path'], data_path)
                
                if score is not None:
                    consistency_scores.append(score)
                    
            except Exception as e:
                logger.warning(f"Could not analyze action-obs consistency for episode {i}: {e}")
        
        if not consistency_scores:
            logger.warning("No episodes with valid data found for action-observation consistency")
            return None  # Cannot evaluate without data
        
        return float(np.mean(consistency_scores))
    
    def _analyze_episode_action_obs_consistency_from_data(self, episode_data: Dict[str, Any]) -> Optional[float]:
        """Analyze action-observation consistency directly from episode data (for HF datasets)."""
        try:
            if 'actions' not in episode_data or 'observations' not in episode_data:
                return None
            
            actions = episode_data['actions']
            observations = episode_data['observations']
            
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            if not isinstance(observations, np.ndarray):
                observations = np.array(observations)
            
            # Setup dynamic configuration based on actual data
            self._setup_dynamic_config_from_data(actions)
            
            if len(actions) < 2:
                return None
            
            consistency_scores = []
            
            # For each joint dimension, check if actions lead to expected state changes
            for joint_idx in range(actions.shape[1]):
                action_joint = actions[:, joint_idx]
                
                # Check if we have corresponding observation dimension
                if joint_idx < observations.shape[1]:
                    obs_joint = observations[:, joint_idx]
                    
                    # Compute state changes
                    state_changes = np.diff(obs_joint)
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
            logger.warning(f"Error analyzing action-obs consistency from episode data: {e}")
            return None
    
    def _setup_dynamic_config_from_data(self, actions: np.ndarray):
        """Setup configuration dynamically based on actual data shape."""
        detected_dof = actions.shape[1]
        if self.detected_dof is None or detected_dof != self.detected_dof:
            self.detected_dof = detected_dof
            logger.info(f"Detected {detected_dof} DOF from action data")
    
    def _analyze_episode_action_obs_consistency(self, episode_data_path: str, dataset_data_path: Path) -> Optional[float]:
        """Analyze action-observation consistency within a single episode."""
        try:
            # Handle relative paths with data_path
            if dataset_data_path and not Path(episode_data_path).is_absolute():
                full_data_path = dataset_data_path / episode_data_path
            else:
                full_data_path = Path(episode_data_path)
                
            df = pd.read_parquet(str(full_data_path))
            
            # Check for array-format action and state columns
            if 'action' in df.columns and 'observation.state' in df.columns:
                actions = np.stack(df['action'].values)  # Convert list of arrays to 2D array
                states = np.stack(df['observation.state'].values)
                
                # Setup dynamic configuration based on actual data
                self._setup_dynamic_config_from_data(actions)
                
                if len(actions) < 2:
                    return None
                
                consistency_scores = []
                
                # For each joint dimension, check if actions lead to expected state changes
                for joint_idx in range(actions.shape[1]):
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
            
            # Fallback: Try to find individual joint columns (legacy format)
            action_cols = [col for col in df.columns if col.startswith('action')]
            state_cols = [col for col in df.columns if col.startswith('observation.state')]
            
            if not action_cols or not state_cols:
                return None
            
            consistency_scores = []
            
            # For each joint, check if actions lead to expected state changes
            for action_col in action_cols:
                # Extract joint name
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
                    
                    if len(actions) < 2:
                        continue
                    
                    # Compute state changes
                    state_changes = np.diff(states)
                    action_commands = actions[:-1]
                    
                    # Check correlation between actions and state changes
                    if np.std(action_commands) > 1e-6 and np.std(state_changes) > 1e-6:
                        correlation = np.corrcoef(action_commands, state_changes)[0, 1]
                        if not np.isnan(correlation):
                            # Convert correlation to [0, 1] range
                            consistency = (correlation + 1) / 2
                            consistency_scores.append(consistency)
            
            return np.mean(consistency_scores) if consistency_scores else None
            
        except Exception as e:
            logger.warning(f"Error analyzing action-obs consistency for {episode_data_path}: {e}")
            return None

    def _compute_cross_episode_consistency(self, dataset: List[Dict], data_path: Path) -> float:
        """Compute cross-episode consistency by comparing action sequences for similar tasks."""
        if not dataset:
            return None
        
        # Group episodes by task/prompt for comparison
        task_groups = {}
        for episode in dataset:
            task = episode.get('task', episode.get('prompt', 'unknown'))
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append(episode)
        
        logger.info(f"Found {len(task_groups)} unique tasks for cross-episode consistency analysis")
        
        # If we have too few tasks, try semantic grouping
        if len(task_groups) < 2:
            logger.info("Too few unique tasks, attempting semantic grouping of prompts")
            task_groups = self._group_episodes_by_semantic_similarity(dataset)
        
        # If we still have only one group, compare episodes within that group
        if len(task_groups) == 1:
            task_name = list(task_groups.keys())[0]
            episodes = task_groups[task_name]
            logger.info(f"Single task group '{task_name}' with {len(episodes)} episodes - computing intra-task consistency")
            
            if len(episodes) >= self.min_episodes_for_cross_analysis:
                # For single task, compute consistency within the task
                consistency_score = self._compute_intra_task_consistency(episodes, data_path)
                if consistency_score is not None:
                    logger.info(f"Intra-task consistency: {consistency_score:.3f}")
                    return consistency_score
        
        consistency_scores = []
        
        for task, episodes in task_groups.items():
            if len(episodes) < self.min_episodes_for_cross_analysis:
                continue
            
            # Sample episodes for comparison (limit to avoid combinatorial explosion)
            sample_episodes = episodes[:min(10, len(episodes))]
            
            try:
                # Extract action sequences from each episode
                action_sequences = []
                for episode in sample_episodes:
                    # Try to extract actions directly from episode data first (for HF datasets)
                    actions = self._extract_action_sequence_from_episode(episode)
                    
                    # Fallback to data_path method for local datasets
                    if actions is None and 'data_path' in episode:
                        actions = self._extract_action_sequence(episode['data_path'], data_path)
                    
                    if actions is not None and len(actions) >= self.min_sequence_length:
                        action_sequences.append(actions)
                
                if len(action_sequences) >= self.min_episodes_for_cross_analysis:
                    # Compute pairwise similarities
                    similarities = []
                    for i in range(len(action_sequences)):
                        for j in range(i + 1, len(action_sequences)):
                            sim = self._compute_sequence_similarity(
                                action_sequences[i], 
                                action_sequences[j]
                            )
                            if sim is not None:
                                similarities.append(sim)
                    
                    if similarities:
                        task_consistency = np.mean(similarities)
                        consistency_scores.append(task_consistency)
                        logger.debug(f"Task '{task}': {task_consistency:.3f} consistency from {len(similarities)} comparisons")
                        
            except Exception as e:
                logger.warning(f"Error computing cross-episode consistency for task '{task}': {e}")
        
        if not consistency_scores:
            logger.warning("No cross-episode consistency scores computed - insufficient data or too few similar tasks")
            return None
        
        avg_consistency = np.mean(consistency_scores)
        logger.info(f"Cross-episode consistency computed from {len(consistency_scores)} task groups: {avg_consistency:.3f}")
        
        return avg_consistency
    
    def _group_episodes_by_semantic_similarity(self, dataset: List[Dict]) -> Dict[str, List[Dict]]:
        """Group episodes by semantic similarity of their prompts."""
        try:
            # Extract all prompts
            prompts = []
            for episode in dataset:
                prompt = episode.get('task', episode.get('prompt', ''))
                prompts.append(prompt)
            
            if not prompts:
                return {'default': dataset}
            
            # Use simple keyword-based grouping as fallback
            groups = {}
            for i, episode in enumerate(dataset):
                prompt = prompts[i].lower()
                
                # Simple keyword-based grouping
                if 'pick' in prompt or 'grab' in prompt or 'grasp' in prompt:
                    group_key = 'pick_tasks'
                elif 'place' in prompt or 'put' in prompt or 'drop' in prompt:
                    group_key = 'place_tasks'
                elif 'move' in prompt or 'push' in prompt or 'slide' in prompt:
                    group_key = 'move_tasks'
                elif 'open' in prompt or 'close' in prompt:
                    group_key = 'manipulation_tasks'
                elif 'clean' in prompt or 'wipe' in prompt:
                    group_key = 'cleaning_tasks'
                else:
                    group_key = 'other_tasks'
                
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(episode)
            
            # Filter out groups with too few episodes
            filtered_groups = {k: v for k, v in groups.items() if len(v) >= self.min_episodes_for_cross_analysis}
            
            if not filtered_groups:
                return {'default': dataset}
            
            logger.info(f"Semantic grouping created {len(filtered_groups)} groups: {list(filtered_groups.keys())}")
            return filtered_groups
            
        except Exception as e:
            logger.warning(f"Error in semantic grouping: {e}")
            return {'default': dataset}
    
    def _compute_intra_task_consistency(self, episodes: List[Dict], data_path: Path) -> Optional[float]:
        """Compute consistency between episodes of the same task."""
        try:
            # Extract action sequences from episodes
            action_sequences = []
            for episode in episodes:
                # Try to extract actions directly from episode data first (for HF datasets)
                actions = self._extract_action_sequence_from_episode(episode)
                
                # Fallback to data_path method for local datasets
                if actions is None and 'data_path' in episode:
                    actions = self._extract_action_sequence(episode['data_path'], data_path)
                
                if actions is not None and len(actions) >= self.min_sequence_length:
                    action_sequences.append(actions)
            
            if len(action_sequences) < self.min_episodes_for_cross_analysis:
                return None
            
            # Compute pairwise similarities between all episodes
            similarities = []
            for i in range(len(action_sequences)):
                for j in range(i + 1, len(action_sequences)):
                    sim = self._compute_sequence_similarity(
                        action_sequences[i], 
                        action_sequences[j]
                    )
                    if sim is not None:
                        similarities.append(sim)
            
            if not similarities:
                return None
            
            avg_similarity = np.mean(similarities)
            logger.debug(f"Intra-task consistency: {avg_similarity:.3f} from {len(similarities)} comparisons")
            return avg_similarity
            
        except Exception as e:
            logger.warning(f"Error computing intra-task consistency: {e}")
            return None
    
    def _extract_action_sequence(self, episode_data_path: str, dataset_data_path: Path) -> Optional[np.ndarray]:
        """Extract action sequence from an episode."""
        try:
            # Handle HuggingFace datasets
            if episode_data_path.startswith("hf_episode_"):
                # For HF datasets, the action data should be passed directly
                # This is a placeholder - the actual data should be passed via the episode dict
                logger.warning(f"HF episode data path {episode_data_path} requires direct action data passing")
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
                
                return actions
            
            # Fallback: Find individual action columns (legacy format)
            action_cols = [col for col in df.columns if col.startswith('action')]
            if not action_cols:
                return None
            
            # Stack all action dimensions
            actions = []
            for col in sorted(action_cols):  # Sort to ensure consistent ordering
                if col in df.columns:
                    actions.append(df[col].values)
            
            if actions:
                return np.stack(actions, axis=1)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting actions from {episode_data_path}: {e}")
            return None
    
    def _extract_action_sequence_from_episode(self, episode_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract action sequence directly from episode data (for HF datasets)."""
        try:
            if 'actions' in episode_data:
                actions = episode_data['actions']
                if not isinstance(actions, np.ndarray):
                    actions = np.array(actions)
                
                # Setup dynamic configuration based on actual data
                self._setup_dynamic_config_from_data(actions)
                
                return actions
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting actions from episode data: {e}")
            return None
    
    def _compute_sequence_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> Optional[float]:
        """Compute similarity between two action sequences."""
        try:
            if seq1 is None or seq2 is None:
                return None
            
            # Ensure both sequences are 2D arrays
            if seq1.ndim == 1:
                seq1 = seq1.reshape(-1, 1)
            if seq2.ndim == 1:
                seq2 = seq2.reshape(-1, 1)
            
            # Handle different sequence lengths by taking the shorter one
            min_len = min(len(seq1), len(seq2))
            if min_len < self.min_sequence_length:  # Use configuration parameter
                return None
            
            seq1_short = seq1[:min_len]
            seq2_short = seq2[:min_len]
            
            similarities = []
            
            # Compute similarity for each dimension (joint)
            for dim in range(seq1_short.shape[1]):
                s1 = seq1_short[:, dim]
                s2 = seq2_short[:, dim]
                
                # Normalize sequences to [0, 1] for comparison
                s1_range = np.ptp(s1)
                s2_range = np.ptp(s2)
                
                if s1_range > 1e-6:
                    s1_norm = (s1 - np.min(s1)) / s1_range
                else:
                    s1_norm = np.zeros_like(s1)
                
                if s2_range > 1e-6:
                    s2_norm = (s2 - np.min(s2)) / s2_range
                else:
                    s2_norm = np.zeros_like(s2)
                
                # Compute correlation
                if np.std(s1_norm) > self.correlation_threshold and np.std(s2_norm) > self.correlation_threshold:
                    correlation = np.corrcoef(s1_norm, s2_norm)[0, 1]
                    if not np.isnan(correlation):
                        similarity = (correlation + 1) / 2  # Convert to [0, 1]
                        similarities.append(similarity)
                
                # Also compute DTW-like distance for temporal patterns
                diff = np.abs(s1_norm - s2_norm)
                dtw_similarity = 1.0 / (1.0 + np.mean(diff))
                similarities.append(dtw_similarity)
            
            return np.mean(similarities) if similarities else None
            
        except Exception as e:
            logger.warning(f"Error computing sequence similarity: {e}")
            return None
    
    def _compute_temporal_consistency(self, dataset: List[Dict], data_path: Path, embedding_manager) -> float:
        """Compute temporal consistency by comparing similarity across different time segments."""
        if not dataset:
            return None
        
        temporal_scores = []
        
        # Sample episodes
        sample_size = min(10, len(dataset))
        for i in range(sample_size):
            sample = dataset[i]
            
            try:
                if embedding_manager and sample.get('video') is not None:
                    # Visual temporal consistency
                    if data_path and isinstance(sample['video'], str):
                        video_path = data_path / sample['video']
                        frames = embedding_manager.load_video_frames(str(video_path), self.num_frames * 2)
                    else:
                        frames = embedding_manager.load_visual_data(sample['video'])
                    
                    if frames and len(frames) >= 4:
                        prompt = sample['prompt']
                        
                        # Split frames into segments
                        mid_point = len(frames) // 2
                        first_half = frames[:mid_point]
                        second_half = frames[mid_point:]
                        
                        # Compute similarity for each segment
                        sim_first = embedding_manager.compute_clip_similarity([prompt], first_half)
                        sim_second = embedding_manager.compute_clip_similarity([prompt], second_half)
                        
                        # Temporal consistency is 1 - absolute difference
                        temporal_consistency = 1.0 - abs(sim_first - sim_second)
                        temporal_scores.append(temporal_consistency)
                        
                # Also check action temporal consistency
                action_temporal_score = self._analyze_action_temporal_consistency_from_data(sample)
                if action_temporal_score is None and 'data_path' in sample:
                    action_temporal_score = self._analyze_action_temporal_consistency(sample['data_path'], data_path)
                if action_temporal_score is not None:
                    temporal_scores.append(action_temporal_score)
                
            except Exception as e:
                logger.warning(f"Error computing temporal consistency for sample {i}: {e}")
        
        if not temporal_scores:
            logger.warning("No temporal consistency scores computed - insufficient data")
            return None  # Return None instead of default score
        
        return float(np.mean(temporal_scores))
    
    def _analyze_action_temporal_consistency(self, episode_data_path: str, dataset_data_path: Path) -> Optional[float]:
        """Analyze temporal consistency of actions within an episode."""
        try:
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
                
                consistency_scores = []
                
                # For each joint dimension, compute temporal smoothness
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
                        consistency_scores.append(smoothness)
                
                return np.mean(consistency_scores) if consistency_scores else None
            
            return None
            
        except Exception as e:
            logger.warning(f"Error analyzing action temporal consistency for {episode_data_path}: {e}")
            return None
    
    def _analyze_action_temporal_consistency_from_data(self, episode_data: Dict[str, Any]) -> Optional[float]:
        """Analyze temporal consistency of actions directly from episode data (for HF datasets)."""
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
            
            consistency_scores = []
            
            # For each joint dimension, compute temporal smoothness
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
                    consistency_scores.append(smoothness)
            
            return np.mean(consistency_scores) if consistency_scores else None
            
        except Exception as e:
            logger.warning(f"Error analyzing action temporal consistency from episode data: {e}")
            return None
    
    def compute_per_sample(self, dataset: List[Dict], data_path: Path, embedding_manager) -> List[float]:
        """
        Compute action consistency score for each sample individually.
        
        Returns:
            List of consistency scores for each sample
        """
        scores = []
        
        for sample in dataset:
            try:
                video_path = data_path / sample['video']
                prompt = sample['prompt']
                
                frames = embedding_manager.load_video_frames(str(video_path), self.num_frames)
                
                if not frames:
                    scores.append(0.0)
                    continue
                
                similarity = embedding_manager.compute_clip_similarity([prompt], frames)
                normalized_score = (similarity + 1) / 2
                scores.append(normalized_score)
                
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                scores.append(0.0)
        
        return scores
    
    def compute_temporal_consistency(self, dataset: List[Dict], data_path: Path, embedding_manager) -> float:
        """
        Compute temporal consistency by comparing similarity across different time segments.
        
        Returns:
            Float indicating how consistent the prompt-video alignment is across time
        """
        return self._compute_temporal_consistency(dataset, data_path, embedding_manager)