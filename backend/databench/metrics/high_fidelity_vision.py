"""
High-Fidelity Vision Metrics

Comprehensive evaluation of robotics dataset quality including:
- Multiple Views (camera count, wrist camera, naming)
- High-Resolution & Frame Rate 
- Environment Verification (setup, objects, workspace)
- Prompt Quality (clarity, specificity, consistency)
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Ensure cv2 is available for computer vision analysis
try:
    import cv2
except ImportError:
    cv2 = None
    logging.getLogger(__name__).warning("OpenCV not available - computer vision analysis will be limited")

logger = logging.getLogger(__name__)

class HighFidelityVisionMetric:
    """Evaluates high-fidelity vision aspects of robotics datasets."""
    
    def __init__(self, dataset_analyzer=None):
        """Initialize with optional dataset analyzer for adaptive thresholds."""
        self.dataset_analyzer = dataset_analyzer
        
    def _is_cv2_available(self) -> bool:
        """Check if OpenCV is available for computer vision analysis."""
        return cv2 is not None
        
    def compute(self, dataset: List[Dict], data_path: Path, embedding_manager) -> Dict[str, float]:
        """
        Compute comprehensive high-fidelity vision metrics.
        
        Returns:
            Dictionary with individual metric scores
        """
        if not dataset:
            return {}
        
        # Store data_path for use in other methods
        self.data_path = data_path
        
        # Get dataset-derived thresholds if analyzer available
        if self.dataset_analyzer and hasattr(self.dataset_analyzer, 'analysis'):
            self.adaptive_thresholds = {
                'resolution': self.dataset_analyzer.get_adaptive_thresholds('resolution'),
                'framerate': self.dataset_analyzer.get_adaptive_thresholds('framerate'),
                'camera_count': self.dataset_analyzer.get_adaptive_thresholds('camera_count'),
                'text_length': self.dataset_analyzer.get_adaptive_thresholds('text_length')
            }
            logger.info("Using dataset-derived adaptive thresholds for evaluation")
        else:
            # Use fallback thresholds
            from scripts.dataset_analyzer import DatasetAnalyzer
            temp_analyzer = DatasetAnalyzer([], data_path)
            self.adaptive_thresholds = {
                'resolution': temp_analyzer._get_fallback_thresholds('resolution'),
                'framerate': temp_analyzer._get_fallback_thresholds('framerate'),
                'camera_count': temp_analyzer._get_fallback_thresholds('camera_count'),
                'text_length': temp_analyzer._get_fallback_thresholds('text_length')
            }
            logger.warning("Using fallback thresholds - no dataset analysis available")
        
        results = {}

        # Discover dataset-level camera views once (for fallback)
        self.dataset_camera_views = self._discover_dataset_camera_views(data_path)
        
        # Multiple Views
        results['multiple_views'] = self._evaluate_multiple_views(dataset, data_path)
        
        # High Resolution & Frame Rate
        results['resolution_framerate'] = self._evaluate_resolution_framerate(dataset, data_path)
        
        # Environment Verification
        results['environment_verification'] = self._evaluate_environment_verification(dataset, data_path)
        
        # Prompt Quality
        results['prompt_quality'] = self._evaluate_prompt_quality(dataset, data_path)
        
        # Overall score (including environment and prompt verification)
        results['overall_score'] = np.mean([
            results['multiple_views'],
            results['resolution_framerate'],
            results['environment_verification'],
            results['prompt_quality']
        ])
        
        # Log detailed breakdown with explanations
        logger.info("High-Fidelity Vision Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.3f}")
        
        # Add specific warnings for poor scores using adaptive thresholds
        warning_thresholds = self._get_warning_thresholds()
        if results['multiple_views'] < warning_thresholds['multiple_views']:
            logger.warning("⚠️  POOR MULTI-VIEW SETUP:")
            logger.warning("   - Dataset needs at least 2 different cameras")
            logger.warning("   - Camera names should be meaningful (top, front, wrist, etc.)")
            logger.warning("   - Wrist camera is highly recommended for manipulation tasks")
            logger.warning("   - All cameras should have consistent resolution")
        
        if results['environment_verification'] < warning_thresholds['environment_verification']:
            logger.warning("⚠️  POOR ENVIRONMENT SETUP:")
            logger.warning("   - Environment should be properly set up for task")
            logger.warning("   - Objects should be relevant to the task")
            logger.warning("   - Workspace should be clean and organized")
            logger.warning("   - No unnecessary clutter or distractions")
        
        if results['prompt_quality'] < warning_thresholds['prompt_quality']:
            logger.warning("⚠️  POOR PROMPT QUALITY:")
            logger.warning("   - Prompts should be clear and descriptive")
            logger.warning("   - Task descriptions should match actual actions")
            logger.warning("   - Avoid generic or vague instructions")
            logger.warning("   - Prompts should be consistent within similar tasks")
        
        return results
    
    def _evaluate_multiple_views(self, dataset: List[Dict], data_path: Path) -> float:
        """Evaluate multiple camera views with proper HF dataset handling."""
        scores = []
        
        for episode in dataset[:min(50, len(dataset))]:  # Sample episodes
            episode_score = 0.0
            
            # Handle different dataset formats
            camera_count = self._count_cameras(episode)

            # Fallback: if episode provides <2 cameras but dataset-level views >1, use that
            if camera_count < 2 and self.dataset_camera_views:
                camera_count = len(self.dataset_camera_views)
            
            # For HuggingFace datasets, check if multiple views are available in repo
            if camera_count == 1 and 'video' in episode and isinstance(episode['video'], dict):
                # Check if HF repo has multiple video directories
                hf_camera_count = self._count_hf_cameras(episode)
                camera_count = max(camera_count, hf_camera_count)
            
            # Camera count scoring based on adaptive thresholds
            camera_thresholds = self.adaptive_thresholds['camera_count']
            
            if camera_count >= camera_thresholds['excellent_threshold']:
                episode_score += 1.0  # Excellent: meets or exceeds dataset maximum
            elif camera_count >= camera_thresholds['good_threshold']:
                episode_score += 0.8  # Good: above dataset average
            elif camera_count >= camera_thresholds['acceptable_threshold']:
                episode_score += 0.6  # Acceptable: meets dataset minimum
            else:
                # Single camera - partial credit for good quality
                if self._is_high_quality_single_view(episode):
                    episode_score += 0.2  # Some credit for good single view
                else:
                    episode_score = 0.0  # No credit for poor single view
                scores.append(episode_score)
                continue
                
            # Check for wrist camera (bonus points)
            has_wrist_camera = self._has_wrist_camera(episode)
            if has_wrist_camera:
                episode_score += 0.15  # Bonus for wrist camera
                
            # Camera naming evaluation
            naming_score = self._evaluate_camera_naming(episode)
            episode_score += naming_score * 0.15
            
            # Resolution consistency (if multiple views)
            if camera_count > 1:
                resolution_consistent = self._check_resolution_consistency(episode)
                if resolution_consistent:
                    episode_score += 0.05
                    
            scores.append(min(episode_score, 1.0))
        
        return float(np.mean(scores)) if scores else 0.0

    def _discover_dataset_camera_views(self, data_path: Path):
        """Discover camera views from meta/info.json for local or HF datasets."""
        camera_views = []
        
        try:
            import json, os
            
            # Step 1: Try to load info.json
            info_path = None
            repo_id = None
            
            if not data_path:
                # HuggingFace dataset
                if hasattr(self, 'dataset_analyzer') and self.dataset_analyzer and hasattr(self.dataset_analyzer, 'dataset'):
                    repo_id = self.dataset_analyzer.dataset[0].get('dataset_name', '') if self.dataset_analyzer.dataset else ''
                if repo_id:
                    try:
                        from huggingface_hub import hf_hub_download
                        info_path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset", revision="main")
                    except Exception as e:
                        logger.debug(f"Could not download info.json from HF: {e}")
                        info_path = None
            else:
                # Local dataset
                info_path = data_path / "meta" / "info.json"
                if not info_path.exists():
                    info_path = None
            
            # Step 2: Extract camera views from info.json if available
            if info_path and os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                # Method 1: Check for 'cameras' key (legacy format)
                if 'cameras' in info:
                    for cam in info['cameras']:
                        camera_views.append({'name': cam.get('name', ''), 'path': cam.get('path', '')})
                
                # Method 2: Check for 'camera_views' key (legacy format)
                elif 'camera_views' in info:
                    camera_views = info['camera_views']
                
                # Method 3: Extract from 'features' section (standard LeRobot format)
                elif 'features' in info:
                    for feature_name, feature_info in info['features'].items():
                        if feature_name.startswith('observation.images.'):
                            view_name = feature_name.split('observation.images.')[-1]
                            camera_views.append({
                                'name': view_name,
                                'path': feature_name,
                                'resolution': feature_info.get('shape', [0, 0, 0])[:2] if 'shape' in feature_info else None,
                                'framerate': feature_info.get('info', {}).get('video.fps', None) if 'info' in feature_info else None
                            })
                
                # If we found camera views from metadata, return them
                if camera_views:
                    logger.info(f"Found {len(camera_views)} camera views from metadata: {[v['name'] for v in camera_views]}")
                    return camera_views
            
            # Step 3: Fallback for HuggingFace - scan repository files
            if not data_path and not camera_views and repo_id:
                try:
                    from huggingface_hub import list_repo_files
                    files = list_repo_files(repo_id, repo_type="dataset")
                    view_dirs = set()
                    for f in files:
                        if f.startswith("videos/chunk-000/") and f.endswith(".mp4"):
                            parts = f.split('/')
                            if len(parts) >= 3:
                                view_dirs.add(parts[2])  # e.g., observation.images.top
                    for vd in view_dirs:
                        name = vd.split('.')[-1]
                        camera_views.append({'name': name, 'path': vd})
                    
                    if camera_views:
                        logger.info(f"Found {len(camera_views)} camera views from HF repo files: {[v['name'] for v in camera_views]}")
                        return camera_views
                except Exception as e:
                    logger.debug(f"Could not scan HF repo files: {e}")
            
            # Step 4: Fallback for local datasets - scan filesystem
            if data_path and not camera_views:
                videos_root = data_path / "videos" / "chunk-000"
                if videos_root.exists():
                    view_dirs = [p for p in videos_root.iterdir() if p.is_dir()]
                    for p in view_dirs:
                        # Path like observation.images.top -> name is 'top'
                        name = p.name.split('.')[-1]
                        camera_views.append({
                            'name': name, 
                            'path': p.relative_to(videos_root).as_posix()
                        })
                    
                    if camera_views:
                        logger.info(f"Found {len(camera_views)} camera views from filesystem: {[v['name'] for v in camera_views]}")
                        return camera_views
            
            # Step 5: Final fallback - try to infer from common patterns
            if not camera_views:
                logger.warning("No camera views discovered from metadata or filesystem")
                
        except Exception as e:
            logger.debug(f"Error in camera view discovery: {e}")
        
        return camera_views
    
    def _count_hf_cameras(self, episode: Dict) -> int:
        """Count cameras in HuggingFace dataset by checking video reference."""
        try:
            if 'video' in episode and isinstance(episode['video'], dict):
                # Check the filename for camera indicators
                filename = episode['video'].get('filename', '')
                if 'observation.images.top' in filename:
                    # This suggests there might be multiple views (top, front, etc.)
                    return 2  # Assume at least top and front
            elif 'videos' in episode and isinstance(episode['videos'], dict):
                return len(episode['videos'])
            elif 'video' in episode:
                return 1
        except Exception:
            pass
        return 1
    
    def _is_high_quality_single_view(self, episode: Dict) -> bool:
        """Check if single camera view is high quality."""
        try:
            # Check resolution against dataset-derived thresholds
            resolution = self._get_video_resolution(episode)
            resolution_thresholds = self.adaptive_thresholds['resolution']
            
            # Must meet at least acceptable threshold for dataset
            if resolution < resolution_thresholds['acceptable_threshold']:
                return False
                
            # Check if it's a meaningful viewpoint
            if 'videos' in episode:
                camera_names = list(episode['videos'].keys())
                return any(self._is_meaningful_camera_name(name) for name in camera_names)
            elif 'video' in episode and isinstance(episode['video'], dict):
                filename = episode['video'].get('filename', '')
                return any(keyword in filename.lower() for keyword in 
                          ['top', 'front', 'overhead', 'third_person'])
            
            return True  # Default to true for basic single view
        except Exception:
            return False
    
    def _evaluate_resolution_framerate(self, dataset: List[Dict], data_path: Path) -> float:
        """Evaluate resolution and frame rate."""
        scores = []
        
        for episode in dataset[:min(20, len(dataset))]:  # Sample episodes
            episode_score = 0.0
            
            # Check resolution using adaptive thresholds
            resolution = self._get_video_resolution(episode)
            resolution_thresholds = self.adaptive_thresholds['resolution']
            
            if resolution >= resolution_thresholds['excellent_threshold']:
                episode_score += 1.0  # Excellent resolution
            elif resolution >= resolution_thresholds['good_threshold']:
                episode_score += 0.8  # Good resolution
            elif resolution >= resolution_thresholds['acceptable_threshold']:
                episode_score += 0.6  # Acceptable resolution
            elif resolution >= resolution_thresholds['poor_threshold']:
                episode_score += 0.4  # Poor but usable resolution
            else:
                episode_score += 0.1  # Very poor resolution
                
            # Check frame rate using adaptive thresholds
            framerate = self._get_video_framerate(episode)
            framerate_thresholds = self.adaptive_thresholds['framerate']
            
            if framerate >= framerate_thresholds['excellent_threshold']:
                episode_score += 0.3  # Excellent frame rate
            elif framerate >= framerate_thresholds['good_threshold']:
                episode_score += 0.25  # Good frame rate
            elif framerate >= framerate_thresholds['acceptable_threshold']:
                episode_score += 0.2  # Acceptable frame rate
            elif framerate >= framerate_thresholds['poor_threshold']:
                episode_score += 0.1  # Poor frame rate
            # No bonus for very low frame rates
                
            scores.append(min(episode_score, 1.0))
        
        return float(np.mean(scores)) if scores else 0.0
    
    
    
    
    def _count_cameras(self, episode: Dict) -> int:
        """Count number of cameras in episode."""
        if 'videos' in episode and isinstance(episode['videos'], dict):
            return len(episode['videos'])
        elif 'video' in episode:
            return 1
        return 0
    
    def _has_wrist_camera(self, episode: Dict) -> bool:
        """Check if episode has wrist camera."""
        if 'videos' in episode and isinstance(episode['videos'], dict):
            camera_names = [key.lower() for key in episode['videos'].keys()]
            wrist_keywords = ['wrist', 'hand', 'gripper', 'end_effector']
            return any(keyword in name for name in camera_names for keyword in wrist_keywords)
        return False
    
    def _evaluate_camera_naming(self, episode: Dict) -> float:
        """Evaluate camera naming quality with strict requirements."""
        if 'videos' in episode and isinstance(episode['videos'], dict):
            camera_names = list(episode['videos'].keys())
            
            # Check if names are meaningful (not random)
            meaningful_names = 0
            for name in camera_names:
                if self._is_meaningful_camera_name(name):
                    meaningful_names += 1
            
            # Strict requirement: ALL camera names must be meaningful
            if meaningful_names == len(camera_names):
                return 1.0  # Perfect naming
            elif meaningful_names >= len(camera_names) * 0.8:
                return 0.7  # Most cameras named well
            elif meaningful_names >= len(camera_names) * 0.5:
                return 0.4  # Half the cameras named well
            else:
                return 0.1  # Poor naming
        return 0.0  # Single camera gets no points for naming
    
    def _is_meaningful_camera_name(self, name: str) -> bool:
        """Check if camera name is meaningful."""
        name_lower = name.lower()
        meaningful_keywords = [
            'top', 'front', 'side', 'left', 'right', 'overhead', 'wrist', 
            'hand', 'gripper', 'end_effector', 'third_person', 'first_person'
        ]
        return any(keyword in name_lower for keyword in meaningful_keywords)
    
    def _check_resolution_consistency(self, episode: Dict) -> bool:
        """Check if all videos have consistent resolution."""
        if 'videos' in episode and isinstance(episode['videos'], dict):
            resolutions = []
            for video_path in episode['videos'].values():
                res = self._get_video_resolution_from_path(video_path, self.data_path)
                if res > 0:
                    resolutions.append(res)
            
            # Check if all resolutions are the same
            return len(set(resolutions)) <= 1 if resolutions else False
        return True  # Single video is consistent
    
    def _get_video_resolution(self, episode: Dict) -> int:
        """Get highest resolution from episode videos."""
        max_resolution = 0
        
        # First, try to get resolution from metadata if available
        if hasattr(self, 'dataset_camera_views') and self.dataset_camera_views:
            for view in self.dataset_camera_views:
                if 'resolution' in view and view['resolution']:
                    # resolution is [height, width] from metadata
                    height = view['resolution'][0] if isinstance(view['resolution'], list) and len(view['resolution']) >= 2 else 0
                    max_resolution = max(max_resolution, height)
        
        # If we got resolution from metadata, use it
        if max_resolution > 0:
            return max_resolution
        
        # Otherwise, try to get from video files
        if 'videos' in episode and isinstance(episode['videos'], dict):
            for video_path in episode['videos'].values():
                res = self._get_video_resolution_from_path(video_path, self.data_path)
                max_resolution = max(max_resolution, res)
        elif 'video' in episode:
            max_resolution = self._get_video_resolution_from_path(episode['video'], self.data_path)
            
        return max_resolution
    
    def _get_video_resolution_from_path(self, video_path, data_path=None) -> int:
        """Get video resolution from file path or HF object."""
        try:
            if video_path is None:
                return 0
                
            if isinstance(video_path, dict):  # HF repo file
                # Try to get resolution from filename or metadata
                filename = video_path.get('filename', '')
                if filename:
                    # Check if we have metadata for this camera view
                    if hasattr(self, 'dataset_camera_views') and self.dataset_camera_views:
                        for view in self.dataset_camera_views:
                            if view['name'] in filename.lower() or view['path'] in filename:
                                if 'resolution' in view and view['resolution']:
                                    height = view['resolution'][0] if isinstance(view['resolution'], list) and len(view['resolution']) >= 2 else 0
                                    return height
                
                logger.debug(f"Cannot determine resolution for HuggingFace dataset file: {filename}")
                return 0
            
            if not isinstance(video_path, (str, Path)):
                return 0
                
            video_path_str = str(video_path)
            if not video_path_str or video_path_str == 'None':
                return 0
            
            # Handle relative paths with proper resolution
            if data_path:
                if not Path(video_path_str).is_absolute():
                    full_path = data_path / video_path_str
                else:
                    full_path = Path(video_path_str)
            else:
                full_path = Path(video_path_str)
            
            # Try to get resolution from metadata first
            if hasattr(self, 'dataset_camera_views') and self.dataset_camera_views:
                for view in self.dataset_camera_views:
                    if view['name'] in video_path_str.lower() or view['path'] in video_path_str:
                        if 'resolution' in view and view['resolution']:
                            height = view['resolution'][0] if isinstance(view['resolution'], list) and len(view['resolution']) >= 2 else 0
                            if height > 0:
                                return height
            
            # Use the robust video loader instead of OpenCV directly
            from scripts.embed_utils import VideoLoader
            
            # First try to get info without loading frames
            video_info = VideoLoader.get_video_info(str(full_path))
            
            if video_info['exists'] and video_info['resolution'] != 'unknown':
                # Parse resolution from "WIDTHxHEIGHT" format
                if 'x' in str(video_info['resolution']):
                    width, height = map(int, str(video_info['resolution']).split('x'))
                    return height
            
            # Fallback: try OpenCV directly
            import cv2
            cap = cv2.VideoCapture(str(full_path))
            if cap.isOpened():
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return height if height > 0 else 0
            
            cap.release()
            return 0
            
        except Exception as e:
            logger.debug(f"Could not get resolution from {video_path}: {e}")
            return 0
    
    def _get_video_framerate(self, episode: Dict) -> float:
        """Get video frame rate."""
        # First, try to get framerate from metadata if available
        if hasattr(self, 'dataset_camera_views') and self.dataset_camera_views:
            for view in self.dataset_camera_views:
                if 'framerate' in view and view['framerate']:
                    return float(view['framerate'])
        
        # Try to get from info.json metadata
        if hasattr(self, 'data_path') and self.data_path:
            try:
                info_path = self.data_path / "meta" / "info.json"
                if info_path.exists():
                    import json
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    
                    # Check global fps setting
                    if 'fps' in info:
                        return float(info['fps'])
                    
                    # Check features for fps
                    if 'features' in info:
                        for feature_name, feature_info in info['features'].items():
                            if feature_name.startswith('observation.images.'):
                                fps = feature_info.get('info', {}).get('video.fps', None)
                                if fps:
                                    return float(fps)
            except Exception as e:
                logger.debug(f"Could not get framerate from metadata: {e}")
        
        # Fallback to video file detection
        if 'videos' in episode and isinstance(episode['videos'], dict):
            video_path = next(iter(episode['videos'].values()))
        elif 'video' in episode:
            video_path = episode['video']
        else:
            return 0.0
            
        try:
            if isinstance(video_path, dict):  # HF repo file
                logger.debug(f"Cannot determine framerate for HuggingFace dataset file: {video_path.get('filename', 'unknown')}")
                return 0.0
            
            if not isinstance(video_path, (str, Path)):
                return 0.0
                
            video_path_str = str(video_path)
            if not video_path_str or video_path_str == 'None':
                return 0.0
            
            # Handle relative paths with proper resolution
            if hasattr(self, 'data_path') and self.data_path:
                if not Path(video_path_str).is_absolute():
                    full_path = self.data_path / video_path_str
                else:
                    full_path = Path(video_path_str)
            else:
                full_path = Path(video_path_str)
            
            # Use robust video loader
            from scripts.embed_utils import VideoLoader
            video_info = VideoLoader.get_video_info(str(full_path))
            
            if video_info['exists'] and video_info['fps'] != 'unknown':
                return float(video_info['fps'])
            
            # Fallback: try OpenCV directly
            import cv2
            cap = cv2.VideoCapture(str(full_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                return fps if fps > 0 else 0.0
            
            cap.release()
            return 0.0
            
        except Exception as e:
            logger.debug(f"Could not get framerate from {video_path}: {e}")
            return 0.0
    
    def _evaluate_static_objects(self, episode: Dict) -> float:
        """Evaluate static objects in wrist camera view."""
        logger.warning("Static object evaluation not implemented - requires computer vision analysis")
        return None  # Return None instead of placeholder score
    
    def _evaluate_environment_verification(self, dataset: List[Dict], data_path: Path) -> float:
        """Evaluate environment setup quality."""
        scores = []
        
        for episode in dataset[:min(30, len(dataset))]:  # Sample episodes
            episode_score = 0.0
            
            # Check if environment is properly set up
            env_setup_score = self._check_environment_setup(episode)
            if env_setup_score is not None:
                episode_score += env_setup_score * 0.3
            else:
                episode_score += 0.0 # No score for this metric
                
            # Check object relevance to task
            object_relevance_score = self._check_object_relevance(episode)
            if object_relevance_score is not None:
                episode_score += object_relevance_score * 0.3
            else:
                episode_score += 0.0 # No score for this metric
                
            # Check workspace organization
            workspace_score = self._check_workspace_organization(episode)
            if workspace_score is not None:
                episode_score += workspace_score * 0.2
            else:
                episode_score += 0.0 # No score for this metric
                
            # Check for distractions/clutter
            distraction_score = self._check_distractions(episode)
            if distraction_score is not None:
                episode_score += distraction_score * 0.2
            else:
                episode_score += 0.0 # No score for this metric
                
            scores.append(min(episode_score, 1.0))
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _evaluate_prompt_quality(self, dataset: List[Dict], data_path: Path) -> float:
        """Evaluate prompt and task description quality."""
        scores = []
        
        for episode in dataset[:min(50, len(dataset))]:  # Sample episodes
            episode_score = 0.0
            
            # Check prompt clarity and descriptiveness
            clarity_score = self._check_prompt_clarity(episode)
            episode_score += clarity_score * 0.3
            
            # Check task-action alignment
            alignment_score = self._check_task_alignment(episode)
            episode_score += alignment_score * 0.3
            
            # Check for generic/vague instructions
            specificity_score = self._check_prompt_specificity(episode)
            episode_score += specificity_score * 0.2
            
            # Check consistency within similar tasks
            consistency_score = self._check_prompt_consistency(episode, dataset)
            episode_score += consistency_score * 0.2
            
            scores.append(min(episode_score, 1.0))
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _check_environment_setup(self, episode: Dict) -> float:
        """Check if environment is properly set up for the task."""
        prompt = episode.get('prompt', '').lower()
        task = episode.get('task', '').lower()
        
        if not prompt and not task:
            logger.warning("No prompt or task description available for environment setup evaluation")
            return None
        
        # Look for indicators of proper setup
        setup_indicators = [
            'table', 'workspace', 'surface', 'platform', 'area', 'zone',
            'container', 'bin', 'box', 'tray', 'holder'
        ]
        
        if any(indicator in prompt or indicator in task for indicator in setup_indicators):
            return 0.8  # Good setup indicated by text
        else:
            return 0.2  # No clear setup indicators in text
    
    def _check_object_relevance(self, episode: Dict) -> float:
        """Check if objects in scene are relevant to the task."""
        prompt = episode.get('prompt', '').lower()
        task = episode.get('task', '').lower()
        
        if not prompt and not task:
            logger.warning("No prompt or task description available for object relevance evaluation")
            return None
        
        # Look for specific objects mentioned
        objects_mentioned = [
            'block', 'cube', 'sphere', 'cylinder', 'object', 'item', 'piece',
            'tool', 'cup', 'bottle', 'bowl', 'plate', 'box', 'container'
        ]
        
        if any(obj in prompt or obj in task for obj in objects_mentioned):
            return 0.8  # Specific objects mentioned
        else:
            return 0.2  # No specific objects mentioned
    
    def _check_workspace_organization(self, episode: Dict) -> float:
        """Check if workspace appears organized using computer vision and fallback analysis."""
        try:
            # Check if OpenCV is available
            if not self._is_cv2_available():
                logger.debug("OpenCV not available for workspace organization analysis")
                return self._fallback_workspace_organization(episode)
                
            # Try to get video path for analysis
            video_path = self._get_episode_video_path(episode)
            if not video_path:
                logger.debug("No video path available, using fallback workspace organization analysis")
                return self._fallback_workspace_organization(episode)
            
            # Load and analyze video frames
            frames = self._load_video_frames_for_analysis(video_path, max_frames=5)
            if not frames:
                logger.debug("Could not load video frames for workspace organization analysis")
                return None
            
            organization_scores = []
            
            for frame in frames:
                try:
                    # Analyze workspace organization in this frame
                    org_score = self._analyze_workspace_organization_in_frame(frame)
                    if org_score is not None:
                        organization_scores.append(org_score)
                except Exception as e:
                    logger.debug(f"Error analyzing workspace organization in frame: {e}")
                    continue
            
            if not organization_scores:
                logger.debug("No valid organization scores computed")
                return None
            
            # Return average organization score
            avg_score = np.mean(organization_scores)
            logger.debug(f"Workspace organization score: {avg_score:.3f}")
            return float(avg_score)
            
        except Exception as e:
            logger.warning(f"Workspace organization evaluation failed: {e}")
            return None
    
    def _check_distractions(self, episode: Dict) -> float:
        """Check for unnecessary distractions in the environment using computer vision and fallback analysis."""
        try:
            # Check if OpenCV is available
            if not self._is_cv2_available():
                logger.debug("OpenCV not available for distraction analysis")
                return self._fallback_distraction_detection(episode)
                
            # Try to get video path for analysis
            video_path = self._get_episode_video_path(episode)
            if not video_path:
                logger.debug("No video path available, using fallback distraction analysis")
                return self._fallback_distraction_detection(episode)
            
            # Load and analyze video frames
            frames = self._load_video_frames_for_analysis(video_path, max_frames=5)
            if not frames:
                logger.debug("Could not load video frames for distraction analysis")
                return None
            
            distraction_scores = []
            
            for frame in frames:
                try:
                    # Analyze distractions in this frame
                    distraction_score = self._analyze_distractions_in_frame(frame, episode)
                    if distraction_score is not None:
                        distraction_scores.append(distraction_score)
                except Exception as e:
                    logger.debug(f"Error analyzing distractions in frame: {e}")
                    continue
            
            if not distraction_scores:
                logger.debug("No valid distraction scores computed")
                return None
            
            # Return average distraction score (higher = less distracted)
            avg_score = np.mean(distraction_scores)
            logger.debug(f"Distraction score: {avg_score:.3f}")
            return float(avg_score)
            
        except Exception as e:
            logger.warning(f"Distraction detection failed: {e}")
            return None

    def _get_episode_video_path(self, episode: Dict) -> Optional[str]:
        """Get video path from episode data."""
        try:
            # Handle multiple video sources
            if 'videos' in episode and isinstance(episode['videos'], dict):
                # Use first available video (prefer 'front' or 'top' view)
                video_dict = episode['videos']
                preferred_views = ['front', 'top', 'overhead', 'third_person']
                
                # Try preferred views first
                for view in preferred_views:
                    if view in video_dict:
                        return self._resolve_video_path(video_dict[view])
                
                # Use first available view
                if video_dict:
                    return self._resolve_video_path(next(iter(video_dict.values())))
                    
            elif 'video' in episode:
                return self._resolve_video_path(episode['video'])
                
            return None
            
        except Exception as e:
            logger.debug(f"Error getting video path: {e}")
            return None

    def _resolve_video_path(self, video_path) -> Optional[str]:
        """Resolve video path to absolute path if needed."""
        try:
            if isinstance(video_path, dict):  # HF dataset file
                logger.debug("Cannot analyze HuggingFace dataset video files directly")
                return None
                
            if not isinstance(video_path, (str, Path)):
                return None
                
            video_path_str = str(video_path)
            if not video_path_str or video_path_str == 'None':
                return None
            
            # Handle relative paths
            if hasattr(self, 'data_path') and self.data_path:
                if not Path(video_path_str).is_absolute():
                    full_path = self.data_path / video_path_str
                    return str(full_path) if full_path.exists() else None
                else:
                    return video_path_str if Path(video_path_str).exists() else None
            else:
                return video_path_str if Path(video_path_str).exists() else None
                
        except Exception as e:
            logger.debug(f"Error resolving video path: {e}")
            return None

    def _load_video_frames_for_analysis(self, video_path: str, max_frames: int = 5) -> List[np.ndarray]:
        """Load video frames for computer vision analysis."""
        try:
            import cv2
            frames = []
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.debug(f"Could not open video: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return []
            
            # Sample frames evenly throughout video
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.debug(f"Error loading video frames: {e}")
            return []

    def _analyze_workspace_organization_in_frame(self, frame: np.ndarray) -> Optional[float]:
        """Analyze workspace organization in a single frame."""
        try:
            # Convert to HSV for better analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            height, width = frame.shape[:2]
            
            organization_factors = []
            
            # Factor 1: Color coherence (organized workspaces have consistent colors)
            color_coherence = self._analyze_color_coherence(hsv)
            organization_factors.append(color_coherence * 0.3)
            
            # Factor 2: Edge density and structure (organized spaces have clear structure)
            edge_structure = self._analyze_edge_structure(frame)
            organization_factors.append(edge_structure * 0.25)
            
            # Factor 3: Object density (not too crowded, not too empty)
            object_density = self._analyze_object_density(frame)
            organization_factors.append(object_density * 0.25)
            
            # Factor 4: Surface area analysis (clear working surfaces)
            surface_clarity = self._analyze_surface_clarity(frame, hsv)
            organization_factors.append(surface_clarity * 0.2)
            
            # Combine factors
            total_score = sum(organization_factors)
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            logger.debug(f"Error analyzing workspace organization: {e}")
            return None

    def _analyze_color_coherence(self, hsv: np.ndarray) -> float:
        """Analyze color coherence in the workspace."""
        try:
            # Sample pixels from the workspace (exclude edges which might be background)
            h, w = hsv.shape[:2]
            center_region = hsv[h//4:3*h//4, w//4:3*w//4]
            
            if center_region.size == 0:
                return 0.5
            
            # Analyze hue distribution
            hue_channel = center_region[:, :, 0].flatten()
            hue_hist, _ = np.histogram(hue_channel, bins=18, range=(0, 180))
            
            # Good workspaces have a few dominant colors (not too chaotic)
            hue_hist_norm = hue_hist / (hue_hist.sum() + 1e-10)
            entropy = -np.sum(hue_hist_norm * np.log2(hue_hist_norm + 1e-10))
            max_entropy = np.log2(18)
            
            # Lower entropy = more organized colors
            coherence = 1.0 - (entropy / max_entropy)
            return coherence
            
        except Exception:
            return 0.5

    def _analyze_edge_structure(self, frame: np.ndarray) -> float:
        """Analyze edge structure for organization assessment."""
        try:
            # Convert to grayscale and detect edges
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Analyze edge properties
            edge_density = np.sum(edges > 0) / edges.size
            
            # Look for horizontal and vertical structure
            kernel_h = np.ones((1, 5), np.uint8)
            kernel_v = np.ones((5, 1), np.uint8)
            
            horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
            vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
            
            h_density = np.sum(horizontal_edges > 0) / edges.size
            v_density = np.sum(vertical_edges > 0) / edges.size
            
            # Good organization has moderate edge density with some structure
            structure_score = (h_density + v_density) / edge_density if edge_density > 0 else 0
            
            # Optimal edge density is moderate (not too cluttered, not too empty)
            density_score = 1.0 - abs(edge_density - 0.1) / 0.1 if edge_density <= 0.2 else 0.5
            
            return 0.5 * structure_score + 0.5 * density_score
            
        except Exception:
            return 0.5

    def _analyze_object_density(self, frame: np.ndarray) -> float:
        """Analyze object density for organization assessment."""
        try:
            # Use simple blob detection to estimate object count
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Setup SimpleBlobDetector
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = 50
            params.maxThreshold = 200
            params.filterByArea = True
            params.minArea = 100
            params.maxArea = 5000
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray)
            
            # Calculate density score
            num_objects = len(keypoints)
            frame_area = frame.shape[0] * frame.shape[1]
            
            # Optimal density: 3-8 objects per 100k pixels
            optimal_range = (3, 8)
            normalized_count = (num_objects * 100000) / frame_area
            
            if optimal_range[0] <= normalized_count <= optimal_range[1]:
                return 1.0
            elif normalized_count < optimal_range[0]:
                return max(0.3, normalized_count / optimal_range[0])
            else:
                return max(0.3, optimal_range[1] / normalized_count)
                
        except Exception:
            return 0.5

    def _analyze_surface_clarity(self, frame: np.ndarray, hsv: np.ndarray) -> float:
        """Analyze clarity of working surfaces."""
        try:
            # Look for large, uniform regions that could be work surfaces
            h, w = frame.shape[:2]
            
            # Focus on lower half of image where tables usually are
            lower_region = hsv[h//2:, :]
            
            if lower_region.size == 0:
                return 0.5
            
            # Find regions with low saturation (likely surfaces)
            saturation = lower_region[:, :, 1]
            low_sat_mask = saturation < 100
            
            # Find large connected regions
            kernel = np.ones((5, 5), np.uint8)
            cleaned_mask = cv2.morphologyEx(low_sat_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            # Calculate ratio of clear surface area
            clear_surface_ratio = np.sum(cleaned_mask > 0) / cleaned_mask.size
            
            # Good workspaces have some clear surface area (20-60%)
            if 0.2 <= clear_surface_ratio <= 0.6:
                return 1.0
            elif clear_surface_ratio < 0.2:
                return clear_surface_ratio / 0.2
            else:
                return max(0.3, 0.6 / clear_surface_ratio)
                
        except Exception:
            return 0.5

    def _analyze_distractions_in_frame(self, frame: np.ndarray, episode: Dict) -> Optional[float]:
        """Analyze distractions in a single frame."""
        try:
            distraction_factors = []
            
            # Factor 1: Visual clutter (too many small objects)
            clutter_score = self._analyze_visual_clutter(frame)
            distraction_factors.append(clutter_score * 0.4)
            
            # Factor 2: Irrelevant objects (objects not mentioned in task)
            relevance_score = self._analyze_object_relevance(frame, episode)
            distraction_factors.append(relevance_score * 0.3)
            
            # Factor 3: Background distractions (busy backgrounds, multiple scenes)
            background_score = self._analyze_background_distractions(frame)
            distraction_factors.append(background_score * 0.3)
            
            # Combine factors (higher score = less distractions)
            total_score = sum(distraction_factors)
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            logger.debug(f"Error analyzing distractions: {e}")
            return None

    def _analyze_visual_clutter(self, frame: np.ndarray) -> float:
        """Analyze visual clutter in the frame."""
        try:
            # Use edge density as a proxy for clutter
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Use texture analysis for clutter detection
            # High texture variation indicates clutter
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize laplacian variance
            normalized_var = min(laplacian_var / 1000, 1.0)
            
            # Combine edge density and texture variation
            clutter_metric = 0.6 * edge_density + 0.4 * normalized_var
            
            # Lower clutter = higher score (less distracting)
            return max(0.0, 1.0 - clutter_metric)
            
        except Exception:
            return 0.5

    def _analyze_object_relevance(self, frame: np.ndarray, episode: Dict) -> float:
        """Analyze whether objects in frame are relevant to the task."""
        try:
            # Get task description
            prompt = episode.get('prompt', '').lower()
            task = episode.get('task', '').lower()
            task_text = (prompt + ' ' + task).lower()
            
            if not task_text.strip():
                return 0.5
            
            # Try to detect objects using simple computer vision
            # This is a simplified implementation without YOLO
            
            # Use color-based object detection for common objects
            relevant_objects = self._count_relevant_objects(frame, task_text)
            total_objects = self._count_total_objects(frame)
            
            if total_objects == 0:
                return 0.8  # Clean workspace
            
            # Higher relevance ratio = less distracting
            relevance_ratio = relevant_objects / total_objects
            return min(1.0, relevance_ratio + 0.3)  # Base score + relevance bonus
            
        except Exception:
            return 0.5

    def _count_relevant_objects(self, frame: np.ndarray, task_text: str) -> int:
        """Count objects that appear relevant to the task."""
        try:
            relevant_count = 0
            
            # Define color ranges for common task objects
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Check for specific object colors mentioned in task
            color_keywords = {
                'red': ([0, 50, 50], [10, 255, 255]),
                'blue': ([100, 50, 50], [130, 255, 255]),
                'green': ([35, 50, 50], [85, 255, 255]),
                'yellow': ([20, 50, 50], [35, 255, 255]),
                'orange': ([10, 50, 50], [20, 255, 255]),
                'purple': ([130, 50, 50], [160, 255, 255])
            }
            
            for color_name, (lower, upper) in color_keywords.items():
                if color_name in task_text:
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    if np.sum(mask > 0) > 500:  # Significant color presence
                        relevant_count += 1
            
            # Check for geometric shapes if mentioned
            if any(shape in task_text for shape in ['block', 'cube', 'box']):
                relevant_count += self._detect_rectangular_objects(frame)
            if any(shape in task_text for shape in ['ball', 'sphere', 'circle']):
                relevant_count += self._detect_circular_objects(frame)
            
            return relevant_count
            
        except Exception:
            return 0

    def _count_total_objects(self, frame: np.ndarray) -> int:
        """Count total number of distinct objects in frame."""
        try:
            # Use contour detection for object counting
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area to get meaningful objects
            min_area = 200
            max_area = frame.shape[0] * frame.shape[1] // 4
            
            object_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    object_count += 1
            
            return object_count
            
        except Exception:
            return 5  # Default estimate

    def _detect_rectangular_objects(self, frame: np.ndarray) -> int:
        """Detect rectangular objects in frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rect_count = 0
            for contour in contours:
                # Check if contour is roughly rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4 and cv2.contourArea(contour) > 500:
                    rect_count += 1
                    
            return min(rect_count, 3)  # Cap at reasonable number
            
        except Exception:
            return 0

    def _detect_circular_objects(self, frame: np.ndarray) -> int:
        """Detect circular objects in frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Use HoughCircles for circle detection
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                param1=50, param2=30, minRadius=10, maxRadius=100
            )
            
            if circles is not None:
                return min(len(circles[0]), 3)  # Cap at reasonable number
            return 0
            
        except Exception:
            return 0

    def _analyze_background_distractions(self, frame: np.ndarray) -> float:
        """Analyze background distractions in the frame."""
        try:
            # Convert to HSV for better analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = frame.shape[:2]
            
            # Analyze different regions of the frame
            background_scores = []
            
            # Top region (usually background)
            top_region = hsv[:h//3, :]
            top_score = self._analyze_region_uniformity(top_region)
            background_scores.append(top_score * 0.4)
            
            # Side regions
            left_region = hsv[:, :w//4]
            right_region = hsv[:, 3*w//4:]
            
            left_score = self._analyze_region_uniformity(left_region)
            right_score = self._analyze_region_uniformity(right_region)
            
            background_scores.append(left_score * 0.3)
            background_scores.append(right_score * 0.3)
            
            return sum(background_scores)
            
        except Exception:
            return 0.5

    def _analyze_region_uniformity(self, region: np.ndarray) -> float:
        """Analyze uniformity of a region (higher = less distracting)."""
        try:
            if region.size == 0:
                return 0.5
            
            # Calculate standard deviation of hue and saturation
            hue_std = np.std(region[:, :, 0])
            sat_std = np.std(region[:, :, 1])
            
            # Lower standard deviation = more uniform = less distracting
            hue_uniformity = max(0, 1.0 - hue_std / 50.0)
            sat_uniformity = max(0, 1.0 - sat_std / 50.0)
            
            return (hue_uniformity + sat_uniformity) / 2
            
        except Exception:
            return 0.5
    
    def _check_prompt_clarity(self, episode: Dict) -> float:
        """Check if prompt is clear and descriptive."""
        prompt = episode.get('prompt', '')
        task = episode.get('task', '')
        
        if not prompt and not task:
            return 0.0  # No prompt
        
        text = (prompt + ' ' + task).lower()
        word_count = len(text.split())
        
        # Use dataset-derived length thresholds
        text_thresholds = self.adaptive_thresholds['text_length']
        
        # Check if text length falls in good ranges
        excellent_range = text_thresholds.get('excellent_range', (10, 20))
        good_range = text_thresholds.get('good_range', (5, 15))
        acceptable_range = text_thresholds.get('acceptable_range', (3, 10))
        poor_range = text_thresholds.get('poor_range', (0, 5))
        
        if excellent_range[0] <= word_count <= excellent_range[1]:
            return 1.0  # Excellent length
        elif good_range[0] <= word_count <= good_range[1]:
            return 0.8  # Good length
        elif acceptable_range[0] <= word_count <= acceptable_range[1]:
            return 0.6  # Acceptable length
        elif word_count > excellent_range[1]:
            return 0.4  # Too verbose
        else:
            return 0.2  # Too short
    
    def _check_task_alignment(self, episode: Dict) -> float:
        """Check if task description aligns with likely actions."""
        # This is a simplified implementation
        # In practice, would analyze video content vs task description
        
        prompt = episode.get('prompt', '').lower()
        task = episode.get('task', '').lower()
        text = (prompt + ' ' + task).lower()
        
        # Look for action verbs that indicate clear tasks
        action_verbs = [
            'pick', 'place', 'move', 'push', 'pull', 'grab', 'lift', 'drop',
            'insert', 'remove', 'rotate', 'turn', 'open', 'close', 'stack',
            'unstack', 'pour', 'scoop', 'sweep', 'wipe', 'clean'
        ]
        
        action_count = sum(1 for verb in action_verbs if verb in text)
        
        if action_count >= 2:
            return 1.0  # Multiple clear actions
        elif action_count == 1:
            return 0.8  # One clear action
        else:
            return 0.4  # No clear actions
    
    def _check_prompt_specificity(self, episode: Dict) -> float:
        """Check if prompt avoids generic/vague instructions."""
        prompt = episode.get('prompt', '').lower()
        task = episode.get('task', '').lower()
        text = (prompt + ' ' + task).lower()
        
        # Generic/vague terms that should be avoided
        generic_terms = [
            'do something', 'perform task', 'complete action', 'execute',
            'manipulate object', 'interact with', 'handle item', 'work with'
        ]
        
        # Check for overly generic terms
        if any(term in text for term in generic_terms):
            return 0.2  # Too generic
        
        # Check for specific details
        specific_indicators = [
            'red', 'blue', 'green', 'yellow', 'left', 'right', 'top', 'bottom',
            'corner', 'center', 'first', 'second', 'large', 'small', 'big', 'tiny'
        ]
        
        specificity_count = sum(1 for indicator in specific_indicators if indicator in text)
        
        if specificity_count >= 2:
            return 1.0  # Very specific
        elif specificity_count == 1:
            return 0.8  # Somewhat specific
        else:
            return 0.6  # Neutral specificity
    
    def _check_prompt_consistency(self, episode: Dict, dataset: List[Dict]) -> float:
        """Check if prompt is consistent with similar tasks."""
        # This is a simplified implementation
        # In practice, would cluster similar tasks and check consistency
        
        current_prompt = episode.get('prompt', '').lower()
        current_task = episode.get('task', '').lower()
        
        if not current_prompt and not current_task:
            return 0.0
        
        # Look for similar tasks in dataset (simplified)
        similar_tasks = 0
        consistent_descriptions = 0
        
        for other_episode in dataset[:20]:  # Sample subset
            other_prompt = other_episode.get('prompt', '').lower()
            other_task = other_episode.get('task', '').lower()
            
            # Simple similarity check (shared keywords)
            current_words = set((current_prompt + ' ' + current_task).split())
            other_words = set((other_prompt + ' ' + other_task).split())
            
            if len(current_words.intersection(other_words)) >= 2:
                similar_tasks += 1
                # Check if descriptions are reasonably similar
                if len(current_words.intersection(other_words)) / len(current_words.union(other_words)) > 0.3:
                    consistent_descriptions += 1
        
        if similar_tasks == 0:
            return 0.8  # Unique task, can't check consistency
        
        consistency_ratio = consistent_descriptions / similar_tasks
        return consistency_ratio
    
    def _fallback_workspace_organization(self, episode: Dict) -> float:
        """Fallback workspace organization analysis using available data."""
        try:
            # Analyze based on task description and data patterns
            prompt = episode.get('prompt', '').lower()
            task = episode.get('task', '').lower()
            text = (prompt + ' ' + task).lower()
            
            organization_score = 0.5  # Default neutral score
            
            # Factor 1: Task complexity suggests organization level
            complexity_indicators = [
                'precise', 'careful', 'slowly', 'gently', 'accurately',
                'organized', 'clean', 'neat', 'tidy', 'structured'
            ]
            if any(word in text for word in complexity_indicators):
                organization_score += 0.2
            
            # Factor 2: Multi-step tasks usually require better organization
            step_indicators = ['first', 'then', 'next', 'after', 'finally', 'step']
            step_count = sum(1 for word in step_indicators if word in text)
            if step_count >= 2:
                organization_score += 0.15
            elif step_count >= 1:
                organization_score += 0.1
                
            # Factor 3: Specific object mentions suggest organized workspace
            specific_objects = [
                'table', 'container', 'box', 'bin', 'holder', 'platform',
                'surface', 'workspace', 'area', 'zone'
            ]
            object_count = sum(1 for obj in specific_objects if obj in text)
            if object_count >= 2:
                organization_score += 0.15
            elif object_count >= 1:
                organization_score += 0.1
                
            # Factor 4: Action trajectory analysis (if available)
            if 'episode_data' in episode or 'actions' in episode:
                trajectory_score = self._analyze_trajectory_smoothness(episode)
                if trajectory_score is not None:
                    # Smooth trajectories suggest organized workspace
                    organization_score += trajectory_score * 0.15
            
            # Clamp to valid range
            organization_score = min(1.0, max(0.1, organization_score))
            
            logger.debug(f"Fallback workspace organization score: {organization_score:.3f}")
            return float(organization_score)
            
        except Exception as e:
            logger.debug(f"Fallback workspace organization failed: {e}")
            return 0.5
    
    def _fallback_distraction_detection(self, episode: Dict) -> float:
        """Fallback distraction detection using available data."""
        try:
            # Analyze based on task description and data patterns
            prompt = episode.get('prompt', '').lower()
            task = episode.get('task', '').lower()
            text = (prompt + ' ' + task).lower()
            
            distraction_score = 0.7  # Default good score (low distractions)
            
            # Factor 1: Complex/confusing task descriptions suggest distractions
            confusion_indicators = [
                'complex', 'difficult', 'challenging', 'multiple', 'various',
                'cluttered', 'messy', 'chaotic', 'busy', 'crowded'
            ]
            confusion_count = sum(1 for word in confusion_indicators if word in text)
            if confusion_count >= 2:
                distraction_score -= 0.3
            elif confusion_count >= 1:
                distraction_score -= 0.2
                
            # Factor 2: Many different objects might indicate clutter
            object_types = [
                'block', 'cube', 'sphere', 'cylinder', 'box', 'container',
                'cup', 'bottle', 'tool', 'item', 'piece', 'object'
            ]
            object_variety = len(set(obj for obj in object_types if obj in text))
            if object_variety >= 4:
                distraction_score -= 0.2
            elif object_variety >= 6:
                distraction_score -= 0.3
                
            # Factor 3: Clear, focused tasks suggest less distractions
            focus_indicators = [
                'simple', 'single', 'one', 'only', 'just', 'clear',
                'focused', 'straightforward', 'direct'
            ]
            if any(word in text for word in focus_indicators):
                distraction_score += 0.2
                
            # Factor 4: Action trajectory analysis
            if 'episode_data' in episode or 'actions' in episode:
                trajectory_score = self._analyze_trajectory_smoothness(episode)
                if trajectory_score is not None:
                    # Smooth trajectories suggest fewer distractions
                    distraction_score += (1.0 - trajectory_score) * 0.1
            
            # Factor 5: Task duration analysis (if available)
            if 'length' in episode:
                duration = episode['length']
                # Very long episodes might indicate distractions/failures
                if duration > 1000:  # frames
                    distraction_score -= 0.1
                elif duration < 50:  # very short might indicate focus
                    distraction_score += 0.1
            
            # Clamp to valid range
            distraction_score = min(1.0, max(0.1, distraction_score))
            
            logger.debug(f"Fallback distraction score: {distraction_score:.3f}")
            return float(distraction_score)
            
        except Exception as e:
            logger.debug(f"Fallback distraction detection failed: {e}")
            return 0.6
            
    def _analyze_trajectory_smoothness(self, episode: Dict) -> Optional[float]:
        """Analyze trajectory smoothness as an indicator of workspace organization."""
        try:
            actions = None
            
            # Try to get actions from different sources
            if 'actions' in episode:
                actions = episode['actions']
            elif 'episode_data' in episode and episode['episode_data']:
                # Extract actions from episode data
                episode_data = episode['episode_data']
                if isinstance(episode_data, list) and len(episode_data) > 0:
                    actions = [item.get('action') for item in episode_data if 'action' in item]
            
            if not actions or len(actions) < 5:
                return None
            
            # Convert to numpy array
            if not isinstance(actions, np.ndarray):
                try:
                    actions = np.array(actions)
                except:
                    return None
            
            if actions.ndim != 2 or actions.shape[0] < 5:
                return None
                
            # Calculate smoothness (lower variation = smoother = more organized workspace)
            # Compute velocity (differences between consecutive actions)
            velocities = np.diff(actions, axis=0)
            
            # Compute acceleration (differences between consecutive velocities)
            accelerations = np.diff(velocities, axis=0)
            
            # Smoothness metric: inverse of acceleration variance
            acc_variance = np.mean(np.var(accelerations, axis=0))
            
            # Normalize to 0-1 scale (higher = smoother)
            smoothness = 1.0 / (1.0 + acc_variance * 10)  # Scale factor
            
            return float(smoothness)
            
        except Exception as e:
            logger.debug(f"Trajectory smoothness analysis failed: {e}")
            return None
    
    def _get_warning_thresholds(self) -> Dict[str, float]:
        """Get warning thresholds based on dataset analysis."""
        if self.dataset_analyzer and hasattr(self.dataset_analyzer, 'analysis'):
            # Use dataset-specific thresholds
            scale_props = self.dataset_analyzer.analysis.get('scale_properties', {})
            total_episodes = scale_props.get('total_episodes', 100)
            
            # Adjust warning thresholds based on dataset size
            if total_episodes >= 100:
                return {'multiple_views': 0.3, 'environment_verification': 0.5, 'prompt_quality': 0.5}
            elif total_episodes >= 20:
                return {'multiple_views': 0.2, 'environment_verification': 0.4, 'prompt_quality': 0.4}
            else:
                return {'multiple_views': 0.1, 'environment_verification': 0.3, 'prompt_quality': 0.3}
        else:
            return {'multiple_views': 0.3, 'environment_verification': 0.5, 'prompt_quality': 0.5}
    
    
