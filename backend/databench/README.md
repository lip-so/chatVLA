# Robotics Dataset Quality Evaluation

An evaluation tool for robotics datasets that assesses visual quality, action consistency, trajectory smoothness, and overall dataset coverage.
Supports both local datasets and datasets on Hugging Face.

## Quick Start

```bash
# Evaluate all metrics on a public HuggingFace dataset (no token needed)
python scripts/evaluate.py \
  --data gribok201/150episodes6 \
  --metrics a,v,h,t,c,r

# Evaluate specific metrics on a local LeRobot-format dataset
python scripts/evaluate.py \
  --data 150episodes6 \
  --metrics a,v,h \
  --subset 10

# (Optional) Search for similar episodes
# python scripts/evaluate.py --data gribok201/150episodes6 --search "grasp lego block"
```

## Features

- **6 Comprehensive Metrics**: Action consistency, visual diversity, high-fidelity vision, trajectory quality, dataset coverage, and robot action quality
- **Multi-Modal Analysis**: Evaluates visual data, action sequences, state observations, and text descriptions
- **Multiple Data Sources**: Local datasets (LeRobot format) and HuggingFace Hub datasets
- **Episode Search**: Find similar episodes using natural language queries with CLIP embeddings
- **Professional Scoring**: All metrics return values in the [0.0, 1.0] range (clamped) with detailed breakdowns

## Installation

### Direct Installation (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd databench

# Install dependencies
pip install -r requirements.txt

# Fix numpy compatibility if needed
pip install "numpy<2.0" --upgrade

# Set Python path for evaluation
export PYTHONPATH=$(pwd)
```

> **✅ Status**: All 6 evaluation features work perfectly! Docker functionality has been completely removed as it is not needed. DataBench runs excellently in native Python environments.

## Configuration System

DataBench uses a comprehensive YAML-based configuration system that allows you to customize evaluation parameters without modifying code. The configuration file `config.yaml` contains all metric thresholds, robot specifications, and processing parameters.

### Configuration File Structure

The configuration is organized into several sections:

- **visual**: Frame sampling, video processing parameters
- **robot_action_quality**: Joint limits, quality thresholds for different robot types
- **trajectory_quality**: Synchronization and frequency thresholds
- **dataset_coverage**: Scale and diversity thresholds
- **action_consistency**: Consistency analysis parameters and weights
- **visual_diversity**: Clustering and diversity calculation parameters
- **high_fidelity_vision**: Resolution, framerate, and quality thresholds
- **general**: Default values, logging, and processing parameters
- **robots**: Robot-specific configurations (joint limits, DOF, etc.)

### Using Custom Configuration

#### Method 1: Custom Config File
```bash
# Create your custom config
cp config.yaml my_config.yaml
# Edit my_config.yaml with your parameters

# Use with evaluation
python scripts/evaluate.py --data my_dataset --config my_config.yaml --metrics a,v,h,t,c,r
```

#### Method 2: Runtime Overrides
```bash
# Override specific values at runtime
python scripts/evaluate.py --data my_dataset \
  --config-override "visual.num_frames=16" \
  --config-override "robot_action_quality.thresholds.smoothness_threshold=0.05" \
  --config-override "trajectory_quality.freq_thresholds.gold=50" \
  --metrics a,v,h,t,c,r
```

### Robot-Specific Configuration

DataBench automatically detects robot types and applies appropriate configurations:

```yaml
robots:
  franka_panda:
    dof: 7
    joint_limits:
      - [-2.97, 2.97]   # joint1 limits in radians
      - [-1.76, 1.76]   # joint2 limits
      # ... more joints
    gripper_limits: [0.0, 0.08]  # in meters
    max_velocity: 1.5  # rad/s
    
  ur5:
    dof: 6
    joint_limits:
      - [-6.28, 6.28]   # base joint
      # ... more joints
    max_velocity: 3.15
```

Supported robot types:
- `franka_panda`: 7-DOF Franka Panda robot
- `ur5`: 6-DOF Universal Robots UR5
- `so101_follower`: 6-DOF SO101 follower robot
- `generic_6dof`: Generic 6-DOF manipulator (fallback)

### Key Configuration Parameters

#### Visual Processing
```yaml
visual:
  num_frames: 8          # Frames to sample from videos
  sample_size: 1000      # Max samples for diversity analysis
  video_timeout: 30      # Video loading timeout (seconds)
```

#### Quality Thresholds
```yaml
trajectory_quality:
  sync_thresholds:
    gold: 5      # ≤5ms synchronization = excellent
    silver: 20   # ≤20ms = good
    bronze: 50   # ≤50ms = acceptable
  freq_thresholds:
    gold: 30     # ≥30Hz = excellent
    silver: 10   # ≥10Hz = good
    bronze: 5    # ≥5Hz = acceptable
```

#### Robot Action Quality
```yaml
robot_action_quality:
  thresholds:
    smoothness_threshold: 0.1      # Max normalized jerk
    velocity_threshold: 0.5        # Max normalized velocity/step
    discontinuity_threshold: 0.2   # Max normalized jump/step
    gripper_change_rate_min: 0.05  # Min gripper state changes
    gripper_change_rate_max: 0.3   # Max gripper state changes
```

#### Metric Weights
```yaml
action_consistency:
  weights:
    visual_text: 0.3        # Weight for visual-text consistency
    action_observation: 0.3  # Weight for action-obs consistency
    cross_episode: 0.2      # Weight for cross-episode consistency
    temporal: 0.2           # Weight for temporal consistency
```

### Configuration Validation

The system automatically validates configuration on startup:
- Checks for required sections
- Validates numeric ranges
- Ensures positive values where needed
- Reports configuration issues

### Environment-Specific Configs

You can create environment-specific configurations:

```bash
# Development config with relaxed thresholds
python scripts/evaluate.py --data test_data --config configs/dev_config.yaml

# Production config with strict thresholds  
python scripts/evaluate.py --data production_data --config configs/prod_config.yaml

# Research config with detailed analysis
python scripts/evaluate.py --data research_data --config configs/research_config.yaml
```

### Programmatic Configuration

For advanced users, configurations can be modified programmatically:

```python
from scripts.config_loader import get_config_loader

# Load and modify config
loader = get_config_loader()
loader.override_config({
    'visual.num_frames': 16,
    'robot_action_quality.thresholds.smoothness_threshold': 0.05
})

# Use in evaluation
from scripts.evaluate import RoboticsDatasetBenchmark
benchmark = RoboticsDatasetBenchmark('my_dataset', config_overrides={
    'general.default_score': 0.6
})
```

## 📊 Available Metrics

| Code | Metric | Description | Requirements |
|------|--------|-------------|--------------|
| `a` | **Action Consistency** | Visual-text alignment, action-observation consistency, temporal consistency | Videos + Text |
| `v` | **Visual Diversity** | Pairwise distances, clustering analysis, entropy measures | Videos |
| `h` | **High-Fidelity Vision** | Multi-view setup, resolution, environment quality, prompt quality | Videos |
| `t` | **Trajectory Quality** | Synchronization, frequency, data completeness | Action data |
| `c` | **Dataset Coverage** | Scale, task diversity, visual variety, failure rates | Videos + Text |
| `r` | **Robot Action Quality** | Action smoothness, joint limits, physical feasibility | Action + State data |

## Usage Examples

### Basic Evaluation

```bash
# Evaluate all metrics
python scripts/evaluate.py --hf-dataset gribok201/150episodes6 --metrics a,v,h,t,c,r

# Evaluate specific metrics only
python scripts/evaluate.py --hf-dataset gribok201/150episodes6 --metrics a,v,h

# Use a subset for faster testing
python scripts/evaluate.py --hf-dataset gribok201/150episodes6 --metrics a,v --subset 10
```

### Local Dataset Evaluation

```bash
# Local LeRobot format dataset
python scripts/evaluate.py --data /path/to/dataset --metrics a,v,h,t,c,r

# With custom output file
python scripts/evaluate.py --data /path/to/dataset --metrics a,v --output my_results.json

# Quick test with small subset
python scripts/evaluate.py --data /path/to/dataset --metrics a,v --subset 5
```

### HuggingFace Datasets

```bash
# Public datasets
python scripts/evaluate.py --hf-dataset lerobot/pusht_image --metrics a,v,h,c

# Private datasets (requires authentication)
huggingface-cli login  # Run this first
python scripts/evaluate.py --hf-dataset private/dataset --metrics a,v,h,c

# With custom configuration
python scripts/evaluate.py --hf-dataset dataset/name --hf-config special --hf-split test --metrics a,v
```

### Episode Search

```bash
# Search for specific behaviors
python scripts/evaluate.py --hf-dataset gribok201/150episodes6 --search "grasp lego block"

# Get more results
python scripts/evaluate.py --hf-dataset gribok201/150episodes6 --search "pick up object" --top-k 10

# Search in local dataset
python scripts/evaluate.py --data /path/to/dataset --search "manipulation task"
```

## Understanding Results

### Sample Output
```json
{
  "dataset_name": "gribok201/150episodes6",
  "version": "2025.07.08",
  "notes": "Evaluated on 100 samples",
  "action_consistency": 0.634,
  "visual_diversity": 0.087,
  "hfv_multiple_views": 0.450,
  "hfv_resolution_framerate": 1.000,
  "hfv_environment_verification": 0.630,
  "hfv_prompt_quality": 0.740,
  "hfv_overall_score": 0.705,
  "trajectory_quality": 0.782,
  "dataset_coverage": 0.320,
  "robot_action_quality": 0.891
}
```

### Score Interpretation

| Score Range | Quality Level | Description |
|-------------|---------------|-------------|
| 0.8 - 1.0 | ⭐⭐⭐⭐⭐ Excellent | Professional quality, ready for deployment |
| 0.6 - 0.8 | ⭐⭐⭐⭐ Good | High quality with minor improvements needed |
| 0.4 - 0.6 | ⭐⭐⭐ Moderate | Acceptable quality, some issues to address |
| 0.2 - 0.4 | ⭐⭐ Poor | Significant improvements needed |
| 0.0 - 0.2 | ⭐ Very Poor | Major issues, requires substantial work |

### Detailed Metric Explanations

#### Action Consistency (0.634)
- **Visual-Text**: How well videos match task descriptions (CLIP similarity)
- **Action-Observation**: Correlation between commanded actions and observed states  
- **Cross-Episode**: Consistency of actions across episodes with same task
- **Temporal**: Consistency of visual-text alignment over time

#### Visual Diversity (0.087)
- **Pairwise**: Average distance between video representations
- **Cluster**: K-means clustering with silhouette analysis
- **Entropy**: Information-theoretic diversity measure

#### High-Fidelity Vision (0.705 overall)
- **Multiple Views** (0.450): Number and quality of camera angles
- **Resolution/Framerate** (1.000): Video technical quality
- **Environment** (0.630): Setup quality, lighting, object visibility
- **Prompt Quality** (0.740): Clarity and specificity of task descriptions

## Advanced Usage

### Environment Variables
```bash
# Set HuggingFace token for private datasets
export HF_TOKEN="your_token_here"

# Set custom cache directory
export HF_CACHE_DIR="/path/to/cache"

# Run evaluation
python scripts/evaluate.py --hf-dataset private/dataset --metrics a,v,h
```

### Custom Configuration
```bash
# Specify HuggingFace configuration
python scripts/evaluate.py --hf-dataset dataset/name \
  --hf-config special_config \
  --hf-split validation \
  --hf-cache-dir /custom/cache \
  --metrics a,v,h

# Use specific token
python scripts/evaluate.py --hf-dataset private/dataset \
  --hf-token your_token \
  --metrics a,v,h
```

## Supported Dataset Formats

### LeRobot Format (Local)
```
dataset/
├── meta/
│   ├── info.json          # Dataset metadata
│   ├── episodes.jsonl     # Episode information
│   └── tasks.jsonl        # Task descriptions
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # Action/state data
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.top/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.front/
            ├── episode_000000.mp4
            └── ...
```

### HuggingFace Datasets
- Automatically detects dataset format
- Supports both public and private datasets
- Handles video streaming and caching
- Compatible with datasets library

## Troubleshooting

### Common Issues

#### NumPy Compatibility Error
```bash
# Fix NumPy version conflicts
pip install "numpy<2.0" --upgrade
```

#### HuggingFace Authentication
```bash
# For private datasets
huggingface-cli login

# Or set token manually
export HF_TOKEN="your_token_here"
```

#### CUDA/GPU Issues
```bash
# Force CPU usage if GPU causes issues
export CUDA_VISIBLE_DEVICES=""
```

#### Memory Issues
```bash
# Use smaller subsets for large datasets
python scripts/evaluate.py --hf-dataset large/dataset --subset 50 --metrics a,v

# Process specific metrics only
python scripts/evaluate.py --hf-dataset large/dataset --metrics a --subset 100
```

### Video Loading Issues
- **OpenCV warnings**: Usually harmless, indicates video format compatibility
- **Empty frames**: Check video codec and file integrity
- **Path errors**: Ensure video files exist and paths are correct

### Performance Optimization
```bash
# Parallel processing (if supported)
python scripts/evaluate.py --hf-dataset dataset --metrics a,v --subset 100
```

## Output Files

Results are automatically saved to:
- Local datasets: `results/{dataset_name}_results.json`
- HuggingFace datasets: `results/{org}_{dataset}_results.json`
- Custom path: Use `--output custom_results.json`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new metrics to `metrics/` directory
4. Update `scripts/evaluate.py` to include new metrics
5. Add tests and docs
6. Submit a pull request

## License

This project is licensed under the Apache 2.0 License.

## Tune Robotics
www.tunerobotics.xyz

yo@tunerobotics.xyz