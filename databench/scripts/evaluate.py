#!/usr/bin/env python3
"""
DataBench: Comprehensive Robotics Dataset Evaluation

This script evaluates robotics datasets across multiple quality dimensions:
- Action consistency (visual-text alignment, temporal consistency)
- Visual diversity (scene variation, environmental coverage)
- High-fidelity vision (resolution, frame rate, multi-view)
- Trajectory quality (synchronization, frequency, completeness)
- Dataset coverage (scale, task diversity, failure analysis)
- Robot action quality (smoothness, limits, feasibility)

Supports both local LeRobot format datasets and HuggingFace Hub datasets.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import configuration system
try:
    from scripts.config_loader import get_config_loader, get_config
    from scripts.embed_utils import EmbeddingManager
    from scripts.dataset_analyzer import DatasetAnalyzer
except ImportError:
    # Fallback for when running from different directory
    sys.path.append(str(Path(__file__).parent))
    from config_loader import get_config_loader, get_config
    from embed_utils import EmbeddingManager
    try:
        from dataset_analyzer import DatasetAnalyzer
    except ImportError:
        # DatasetAnalyzer might not exist, create a minimal version
        class DatasetAnalyzer:
            def __init__(self, *args, **kwargs):
                pass

# Import all metric classes
from metrics.action_consistency import ActionConsistencyMetric
from metrics.visual_diversity import VisualDiversityMetric
from metrics.high_fidelity_vision import HighFidelityVisionMetric
from metrics.trajectory_quality import TrajectoryQualityMetric
from metrics.dataset_coverage import DatasetCoverageMetric
from metrics.robot_action_quality import RobotActionQualityMetric

logger = logging.getLogger(__name__)

# Metric code mapping
METRIC_MAPPING = {
    'a': 'action_consistency',
    'v': 'visual_diversity', 
    'h': 'high_fidelity_vision',
    't': 'trajectory_quality',
    'c': 'dataset_coverage',
    'r': 'robot_action_quality'
}

class RoboticsDatasetBenchmark:
    """Main benchmark class for evaluating robotics datasets."""
    
    def __init__(self, data_path: str, subset: Optional[int] = None, config_path: Optional[str] = None, 
                 config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the benchmark.
        
        Args:
            data_path: Path to the dataset or HuggingFace dataset name
            subset: Optional subset size for faster evaluation
            config_path: Optional path to custom configuration file
            config_overrides: Optional configuration overrides
        """
        self.data_path_str = data_path
        self.subset = subset
        
        # Initialize configuration
        try:
            self.config_loader = get_config_loader(config_path)
            if not self.config_loader.validate_config():
                logger.warning("Configuration validation failed, proceeding with defaults")
            
            # Apply any runtime overrides
            if config_overrides:
                self.config_loader.override_config(config_overrides)
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Proceeding without configuration system")
            self.config_loader = None
        
        # Load general configuration
        self.general_config = get_config('general') if self.config_loader else None
        
        # Setup logging level from configuration
        if self.general_config:
            log_level = getattr(logging, self.general_config.log_level.upper(), logging.INFO)
            logging.getLogger().setLevel(log_level)
        
        # Determine if it's a local path or HuggingFace dataset
        self.data_path = Path(data_path) if Path(data_path).exists() else None
        self.is_hf_dataset = self.data_path is None
        
        logger.info(f"Starting evaluation of dataset: {data_path}")
        if self.config_loader:
            logger.info(f"Using configuration from: {self.config_loader.config_path}")
        
    def get_robot_type_from_dataset(self, dataset: List[Dict]) -> str:
        """
        Attempt to determine robot type from dataset metadata.
        
        Args:
            dataset: Dataset episodes
            
        Returns:
            Robot type string for configuration lookup
        """
        # Try to determine robot type from dataset info
        if self.data_path and (self.data_path / "meta" / "info.json").exists():
            try:
                with open(self.data_path / "meta" / "info.json", 'r') as f:
                    info = json.load(f)
                    
                # Look for robot type indicators in metadata
                if 'robot_type' in info:
                    return info['robot_type']
                elif 'robot' in info:
                    robot_name = info['robot'].lower()
                    if 'franka' in robot_name or 'panda' in robot_name:
                        return 'franka_panda'
                    elif 'ur5' in robot_name:
                        return 'ur5'
                        
            except Exception as e:
                logger.debug(f"Could not read robot type from metadata: {e}")
        
        # Fallback: Try to infer from action data dimensions
        if dataset and len(dataset) > 0:
            first_episode = dataset[0]
            if 'data_path' in first_episode:
                try:
                    import pandas as pd
                    data_file = self.data_path / first_episode['data_path']
                    if data_file.exists():
                        df = pd.read_parquet(str(data_file))
                        if 'action' in df.columns:
                            action_shape = df['action'].iloc[0].shape if hasattr(df['action'].iloc[0], 'shape') else None
                            if action_shape and len(action_shape) > 0:
                                dof = action_shape[0]
                                if dof == 7:
                                    return 'franka_panda'  # Assume 7-DOF is Franka
                                elif dof == 6:
                                    return 'ur5'  # Assume 6-DOF is UR5
                                    
                except Exception as e:
                    logger.debug(f"Could not infer robot type from action data: {e}")
        
        # Default fallback
        return 'generic_6dof'
    
    def run_evaluation(self, metrics: List[str]) -> Dict[str, float]:
        """
        Run evaluation with specified metrics.
        
        Args:
            metrics: List of metric codes to run
            
        Returns:
            Dictionary of metric results
        """
        # Load dataset
        dataset = self.load_dataset()
        
        # Determine robot type for robot-specific metrics
        robot_type = self.get_robot_type_from_dataset(dataset)
        logger.info(f"Detected robot type: {robot_type}")
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager()
        
        # Map metric codes to classes
        metric_map = {
            'a': 'action_consistency',
            'v': 'visual_diversity', 
            'h': 'high_fidelity_vision',
            't': 'trajectory_quality',
            'c': 'dataset_coverage',
            'r': 'robot_action_quality'
        }
        
        results = {}
        
        for metric_code in metrics:
            if metric_code not in metric_map:
                logger.warning(f"Unknown metric code: {metric_code}")
                continue
                
            metric_name = metric_map[metric_code]
            logger.info(f"Running {metric_name} metric...")
            
            try:
                # Initialize metric with configuration
                if metric_name == 'action_consistency':
                    metric = ActionConsistencyMetric()
                    
                elif metric_name == 'visual_diversity':
                    metric = VisualDiversityMetric()
                    
                elif metric_name == 'high_fidelity_vision':
                    metric = HighFidelityVisionMetric()
                    
                elif metric_name == 'trajectory_quality':
                    metric = TrajectoryQualityMetric()
                    
                elif metric_name == 'dataset_coverage':
                    metric = DatasetCoverageMetric()
                    
                elif metric_name == 'robot_action_quality':
                    metric = RobotActionQualityMetric(robot_type=robot_type)
                
                # Compute metric
                score = metric.compute(dataset, self.data_path, embedding_manager)
                results[metric_name] = score
                
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                default_score = self.general_config.default_score if self.general_config else 0.5
                results[metric_name] = default_score
        
        return results
    
    def load_dataset(self) -> List[Dict]:
        """Load dataset from local path or HuggingFace."""
        if self.is_hf_dataset:
            return self._load_huggingface_dataset()
        else:
            return self._load_lerobot_dataset()
    
    def _load_lerobot_dataset(self) -> List[Dict]:
        """Load LeRobot dataset format."""
        import pandas as pd
        
        # Load meta information
        info_path = self.data_path / "meta" / "info.json"
        tasks_path = self.data_path / "meta" / "tasks.jsonl"
        episodes_path = self.data_path / "meta" / "episodes.jsonl"
        
        if not all([info_path.exists(), tasks_path.exists(), episodes_path.exists()]):
            raise FileNotFoundError("LeRobot dataset format files not found")
        
        # Load info
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Discover available camera views from the dataset info
        camera_views = self._discover_camera_views(info)
        logger.info(f"Discovered camera views: {camera_views}")
        
        # Load tasks
        tasks = {}
        with open(tasks_path, 'r') as f:
            for line in f:
                task_data = json.loads(line.strip())
                tasks[task_data['task_index']] = task_data['task']
        
        # Load episodes
        episodes = []
        with open(episodes_path, 'r') as f:
            for line in f:
                episode_data = json.loads(line.strip())
                episodes.append(episode_data)
        
        # Convert to expected format
        dataset = []
        for episode in episodes:
            episode_idx = episode['episode_index']
            task_text = episode['tasks'][0] if episode['tasks'] else "Unknown task"
            
            # Create video paths for all discovered camera views
            videos_dict = {}
            for view_info in camera_views:
                video_path = f"videos/chunk-000/{view_info['path']}/episode_{episode_idx:06d}.mp4"
                videos_dict[view_info['name']] = video_path
            
            # Create entry for this episode
            entry = {
                'episode_index': episode_idx,
                'task': task_text,
                'length': episode['length'],
                'video': None,  # Will be set to first available video
                'prompt': task_text,
                'videos': videos_dict,
                'data_path': f"data/chunk-000/episode_{episode_idx:06d}.parquet"
            }
            
            # Check which video files actually exist
            existing_videos = {}
            for view_name, video_path in entry['videos'].items():
                full_video_path = self.data_path / video_path
                if full_video_path.exists():
                    existing_videos[view_name] = video_path
            
            # Update videos dict with only existing files
            if existing_videos:
                entry['videos'] = existing_videos
                # Set primary video to first available view
                entry['video'] = list(existing_videos.values())[0]
                if episode_idx == 0:  # Log only for first episode
                    logger.info(f"Found {len(existing_videos)} camera views per episode: {list(existing_videos.keys())}")
            else:
                logger.warning(f"Episode {episode_idx}: No video files found")
            
            dataset.append(entry)
        
        if self.subset:
            dataset = dataset[:self.subset]
            logger.info(f"Using subset of {len(dataset)} samples")
            
        return dataset
    
    def _discover_camera_views(self, info: dict) -> List[dict]:
        """Discover all available camera views from dataset info and file system."""
        camera_views = []
        
        # Method 1: Extract from dataset features
        if 'features' in info:
            for feature_name, feature_info in info['features'].items():
                if 'observation.images.' in feature_name:
                    view_name = feature_name.split('observation.images.')[-1]
                    view_path = f"observation.images.{view_name}"
                    camera_views.append({
                        'name': view_name,
                        'path': view_path,
                        'source': 'metadata'
                    })
        
        # Method 2: Scan video directory structure
        videos_dir = self.data_path / "videos" / "chunk-000"
        if videos_dir.exists():
            for item in videos_dir.iterdir():
                if item.is_dir() and item.name.startswith('observation.images.'):
                    view_name = item.name.split('observation.images.')[-1]
                    # Check if we already have this view from metadata
                    if not any(view['name'] == view_name for view in camera_views):
                        camera_views.append({
                            'name': view_name,
                            'path': item.name,
                            'source': 'filesystem'
                        })
        
        # If no views found, try common defaults
        if not camera_views:
            logger.warning("No camera views discovered, using defaults")
            default_views = ['top', 'front', 'wrist', 'side', 'left', 'right']
            for view_name in default_views:
                view_path = f"observation.images.{view_name}"
                videos_path = self.data_path / "videos" / "chunk-000" / view_path
                if videos_path.exists():
                    camera_views.append({
                        'name': view_name,
                        'path': view_path,
                        'source': 'default'
                    })
        
        return camera_views
    
    def _load_huggingface_dataset(self) -> List[Dict]:
        """Load dataset from Hugging Face Hub."""
        from datasets import load_dataset
        from huggingface_hub import HfApi, login
        import datasets
        
        try:
            # First try loading without authentication (for public datasets)
            logger.info(f"Loading HuggingFace dataset: {self.data_path_str}")
            try:
                dataset = load_dataset(self.data_path_str, split='train', streaming=False)
                logger.info("Successfully loaded dataset without authentication")
            except Exception as e:
                # If loading fails, check if it's an authentication error
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['authentication', 'token', 'private', 'gated', 'access']):
                    logger.info("Dataset requires authentication, attempting to authenticate...")
                    
                    # Handle authentication - check multiple sources
                    token = None
                    
                    # Try environment variable first
                    import os
                    if 'HUGGINGFACE_TOKEN' in os.environ:
                        token = os.environ['HUGGINGFACE_TOKEN']
                        logger.info("Using HuggingFace token from environment variable")
                    
                    # Try to get token from HF cache
                    if not token:
                        try:
                            from huggingface_hub import HfFolder
                            token = HfFolder.get_token()
                            if token:
                                logger.info("Using cached HuggingFace token")
                        except Exception as e:
                            logger.debug(f"Failed to get cached token: {e}")
                            
                    # Try huggingface-cli login only if no token found
                    if not token:
                        try:
                            logger.info("No cached token found, attempting login...")
                            login()
                            from huggingface_hub import HfFolder  # import inside to ensure availability
                            token = HfFolder.get_token()
                        except Exception as e:
                            logger.debug(f"HF login failed: {e}")
                            logger.warning("No HuggingFace token available. Some datasets may not be accessible.")
                    
                    # Retry loading with authentication
                    if token:
                        try:
                            # Prefer new 'token' argument (datasets >= 2.15)
                            dataset = load_dataset(self.data_path_str, split='train', streaming=False, token=token)
                        except TypeError:
                            # Fall back to old 'use_auth_token' argument for older versions
                            dataset = load_dataset(self.data_path_str, split='train', streaming=False, use_auth_token=token)
                        except ValueError as ve:
                            if "Feature type" in str(ve) and "not found" in str(ve):
                                logger.error(f"Dataset {self.data_path_str} contains unsupported feature types: {ve}")
                                logger.error("This dataset may use custom feature types that are not supported by the current datasets library.")
                                raise ValueError(f"Unsupported dataset format: {ve}")
                            else:
                                raise
                    else:
                        # No token available; re-raise the original authentication error
                        raise e
                elif "Feature type" in str(e) and "not found" in str(e):
                    logger.error(f"Dataset {self.data_path_str} contains unsupported feature types: {e}")
                    logger.error("This dataset may use custom feature types that are not supported by the current datasets library.")
                    raise ValueError(f"Unsupported dataset format: {e}")
                else:
                    # Re-raise other errors
                    raise
            
            # Group dataset by episodes
            episodes_dict = {}
            for item in dataset:
                episode_idx = item.get('episode_index', 0)
                if episode_idx not in episodes_dict:
                    episodes_dict[episode_idx] = []
                episodes_dict[episode_idx].append(item)
            
            # Convert to our format
            converted_dataset = []
            for episode_idx, episode_items in episodes_dict.items():
                # Sort by frame index if available
                if 'frame_index' in episode_items[0]:
                    episode_items = sorted(episode_items, key=lambda x: x.get('frame_index', 0))
                
                # Extract task/prompt from first item
                first_item = episode_items[0]
                task = first_item.get('task', first_item.get('prompt', ''))
                
                # If no task description, try to extract from dataset name
                if not task:
                    dataset_name = self.data_path_str.split('/')[-1]  # Get dataset name
                    # Convert dataset name to readable task description
                    task = self._dataset_name_to_task_description(dataset_name)
                
                # Create episode entry
                entry = {
                    'episode_index': episode_idx,
                    'task': task,
                    'prompt': task,
                    'length': len(episode_items),
                    'dataset_name': self.data_path_str,
                }
                
                # Handle video data (from first item)
                if 'video' in first_item:
                    entry['video'] = first_item['video']  # Keep original video data
                elif 'videos' in first_item:
                    entry['videos'] = first_item['videos']  # Keep original videos dict
                
                # Collect action and observation data from all timesteps
                actions = []
                observations = []
                
                for item in episode_items:
                    if 'action' in item:
                        actions.append(item['action'])
                    if 'observation.state' in item:
                        observations.append(item['observation.state'])
                    elif 'observations' in item:
                        observations.append(item['observations'])
                
                # Store action and observation sequences
                if actions:
                    entry['actions'] = actions
                if observations:
                    entry['observations'] = observations
                
                # Store the full episode data for metrics that need it
                entry['episode_data'] = episode_items
                
                # Create a data_path identifier for HF datasets
                entry['data_path'] = f"hf_episode_{episode_idx}"
                
                converted_dataset.append(entry)
 
            # If no video information present, attempt to discover camera views from meta/info.json
            if not converted_dataset or ('videos' not in converted_dataset[0] and 'video' not in converted_dataset[0]):
                try:
                    from huggingface_hub import hf_hub_download
                    info_json_path = hf_hub_download(repo_id=self.data_path_str, filename="meta/info.json", repo_type="dataset")
                    import json as _json
                    with open(info_json_path, 'r') as _f:
                        info_data = _json.load(_f)
                    camera_views = self._discover_camera_views(info_data)
                    if camera_views:
                        logger.info(f"Discovered {len(camera_views)} camera views from info.json: {[v['name'] for v in camera_views]}")
                        for ep_entry in converted_dataset:
                            episode_idx = ep_entry['episode_index']
                            videos_dict = {}
                            for view in camera_views:
                                view_path = view['path']
                                # Construct relative video path as in LeRobot format
                                videos_dict[view['name']] = f"videos/chunk-000/{view_path}/episode_{episode_idx:06d}.mp4"
                            if videos_dict:
                                ep_entry['videos'] = videos_dict
                                # Primary video as first view
                                ep_entry['video'] = list(videos_dict.values())[0]
                except Exception as _e:
                    logger.debug(f"Could not attach camera views for HF dataset: {_e}")

            logger.info(f"Successfully loaded {len(converted_dataset)} episodes from HuggingFace dataset")
            return converted_dataset
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset {self.data_path_str}: {e}")
            raise
    
    def _dataset_name_to_task_description(self, dataset_name: str) -> str:
        """Convert a dataset name to a meaningful task description."""
        # Remove common prefixes and suffixes
        name = dataset_name.lower()
        name = name.replace('_dataset', '').replace('_data', '').replace('_test', '')
        name = name.replace('lse23clean', '').replace('clean', '')
        
        # Convert underscores to spaces
        name = name.replace('_', ' ')
        
        # Handle specific patterns
        if 'koch' in name:
            # Koch robot datasets
            if 'screwdriver' in name and 'attach' in name:
                return "Attach screwdriver to orange panel"
            elif 'bimanual' in name and 'folding' in name:
                return "Bimanual folding task"
            elif 'folding' in name:
                return "Folding task"
            else:
                return f"Koch robot {name}"
        
        elif 'so101' in name:
            # SO-101 robot datasets
            if 'table cleanup' in name:
                return "Table cleanup task"
            elif 'pen to holder' in name:
                return "Place pen in holder"
            elif 'block in bin' in name:
                return "Place block in bin"
            elif 'pickup' in name:
                return "Pick up object"
            elif 'place' in name:
                return "Place object"
            else:
                return f"SO-101 robot {name}"
        
        elif 'pusht' in name:
            return "Push T-shaped object"
        
        elif 'pick' in name and 'place' in name:
            return "Pick and place task"
        
        elif 'pick' in name:
            return "Pick up object"
        
        elif 'place' in name:
            return "Place object"
        
        elif 'push' in name:
            return "Push object"
        
        elif 'move' in name:
            return "Move object"
        
        # Default: clean up the name and return it
        words = name.split()
        if len(words) > 0:
            return ' '.join(words).title()
        
        return "Robot manipulation task"


class DatasetEvaluator:
    """Main dataset evaluator with adaptive thresholds."""
    
    def __init__(self, dataset, data_path, embedding_manager, subset=None):
        """Initialize evaluator with dataset analysis."""
        self.dataset = dataset
        self.data_path = data_path
        self.embedding_manager = embedding_manager
        self.subset = subset
        
        # Perform dataset analysis to derive adaptive thresholds
        logger.info("Analyzing dataset to derive evaluation parameters...")
        self.dataset_analyzer = DatasetAnalyzer(
            dataset=dataset[:subset] if subset else dataset,
            data_path=data_path,
            embedding_manager=embedding_manager
        )
        self.analysis = self.dataset_analyzer.analyze()
        logger.info("Dataset analysis complete")
    
    def run_metrics(self, metric_codes: List[str], robot_type: str) -> Dict[str, Any]:
        """Run evaluation metrics with adaptive thresholds."""
        results = {}
        
        for metric_code in metric_codes:
            metric_name = METRIC_MAPPING.get(metric_code)
            if not metric_name:
                logger.warning(f"Unknown metric code: {metric_code}")
                continue
                
            logger.info(f"Running {metric_name} metric...")
            
            try:
                # Initialize metric with dataset analyzer for adaptive thresholds
                if metric_name == 'action_consistency':
                    metric = ActionConsistencyMetric()
                    
                elif metric_name == 'visual_diversity':
                    metric = VisualDiversityMetric()
                    
                elif metric_name == 'high_fidelity_vision':
                    metric = HighFidelityVisionMetric(dataset_analyzer=self.dataset_analyzer)
                    
                elif metric_name == 'trajectory_quality':
                    metric = TrajectoryQualityMetric()
                    
                elif metric_name == 'dataset_coverage':
                    metric = DatasetCoverageMetric()
                    
                elif metric_name == 'robot_action_quality':
                    metric = RobotActionQualityMetric(robot_type=robot_type)
                
                else:
                    logger.error(f"Metric {metric_name} not implemented")
                    continue
                
                # Run metric with subset if specified
                dataset_subset = self.dataset[:self.subset] if self.subset else self.dataset
                result = metric.compute(dataset_subset, self.data_path, self.embedding_manager)
                results[metric_name] = result
                
            except Exception as e:
                logger.error(f"Error running {metric_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate robotics datasets')
    parser.add_argument('--data', required=True, help='Dataset path or HuggingFace dataset name')
    parser.add_argument('--metrics', default='a,v,h,t,c,r', help='Comma-separated metric codes')
    parser.add_argument('--subset', type=int, help='Evaluate on subset of episodes')
    parser.add_argument('--config', help='Path to custom configuration file')
    parser.add_argument('--config-override', action='append', help='Configuration overrides (key=value)')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s:%(name)s:%(message)s')
    
    # Parse configuration overrides
    config_overrides = {}
    if args.config_override:
        for override in args.config_override:
            if '=' in override:
                key, value = override.split('=', 1)
                # Try to parse as JSON first, then as string
                try:
                    config_overrides[key] = json.loads(value)
                except json.JSONDecodeError:
                    config_overrides[key] = value
            else:
                logger.warning(f"Invalid override format: {override}")
    
    # Parse metrics
    metrics = [m.strip() for m in args.metrics.split(',')]
    
    try:
        # Initialize benchmark
        benchmark = RoboticsDatasetBenchmark(
            data_path=args.data,
            subset=args.subset,
            config_path=args.config,
            config_overrides=config_overrides
        )
        
        # Load dataset
        dataset = benchmark.load_dataset()
        data_path = benchmark.data_path
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager()
        
        # Detect robot type
        robot_type = benchmark.get_robot_type_from_dataset(dataset)
        logger.info(f"Detected robot type: {robot_type}")
            
        # Initialize evaluator with dataset analysis
        evaluator = DatasetEvaluator(
            dataset=dataset,
            data_path=data_path,
            embedding_manager=embedding_manager,
            subset=args.subset
        )
            
        # Run evaluation with adaptive thresholds
        results = evaluator.run_metrics(metrics, robot_type)
        
        # Prepare output
        output_data = {
            'dataset_name': args.data,
            'version': '2025.07.08',
            'notes': f"Evaluated on {args.subset if args.subset else 'all'} samples",
            **results
        }
        
        # Save results
        output_file = args.output or f"results/{Path(args.data).name}_results.json"
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
            
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for key, value in output_data.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()