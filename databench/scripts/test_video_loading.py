#!/usr/bin/env python3
"""
Video Loading Diagnostic Tool

This script tests video loading capabilities and diagnoses codec support issues.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.embed_utils import VideoLoader
import argparse

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_video_file(video_path: str, num_frames: int = 8):
    """Test loading a specific video file."""
    print(f"\n{'='*60}")
    print(f"TESTING VIDEO: {video_path}")
    print(f"{'='*60}")
    
    # Get video info
    info = VideoLoader.get_video_info(video_path)
    
    print(f"File exists: {info['exists']}")
    print(f"File size: {info['size_mb']:.2f} MB")
    print(f"Codec: {info['codec']}")
    print(f"Resolution: {info['resolution']}")
    print(f"FPS: {info['fps']}")
    print(f"Duration: {info['duration']}")
    print(f"OpenCV readable: {info['opencv_readable']}")
    print(f"PyAV readable: {info['pyav_readable']}")
    print(f"FFmpeg readable: {info['ffmpeg_readable']}")
    
    if not info['exists']:
        print("❌ File does not exist!")
        return False
    
    # Test frame loading
    print(f"\nTesting frame extraction ({num_frames} frames)...")
    
    try:
        frames = VideoLoader.load_video_frames(video_path, num_frames)
        print(f"✅ Successfully loaded {len(frames)} frames")
        
        # Show frame info
        if frames:
            first_frame = frames[0]
            print(f"Frame size: {first_frame.size}")
            print(f"Frame mode: {first_frame.mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load frames: {e}")
        return False

def test_dataset_videos(dataset_path: str, num_test_videos: int = 5):
    """Test videos from a dataset directory."""
    dataset_path = Path(dataset_path)
    
    # Find video files
    video_dirs = []
    videos_base = dataset_path / "videos" / "chunk-000"
    
    if videos_base.exists():
        for item in videos_base.iterdir():
            if item.is_dir() and item.name.startswith('observation.images.'):
                video_dirs.append(item)
    
    if not video_dirs:
        print(f"No video directories found in {videos_base}")
        return
    
    print(f"Found {len(video_dirs)} camera views: {[d.name for d in video_dirs]}")
    
    # Test videos from first camera view
    test_dir = video_dirs[0]
    video_files = list(test_dir.glob("*.mp4"))[:num_test_videos]
    
    if not video_files:
        print(f"No video files found in {test_dir}")
        return
    
    print(f"\nTesting {len(video_files)} videos from {test_dir.name}...")
    
    success_count = 0
    for video_file in video_files:
        success = test_video_file(str(video_file), num_frames=4)
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(video_files)} videos loaded successfully")
    print(f"Success rate: {success_count/len(video_files)*100:.1f}%")
    print(f"{'='*60}")

def check_dependencies():
    """Check if all required video processing dependencies are available."""
    print("Checking video processing dependencies...")
    
    # OpenCV
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
    
    # PyAV
    try:
        import av
        print(f"✅ PyAV: {av.__version__}")
    except ImportError:
        print("❌ PyAV not available")
    
    # ImageIO
    try:
        import imageio
        print(f"✅ ImageIO: {imageio.__version__}")
    except ImportError:
        print("❌ ImageIO not available")
    
    # FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg: {version_line}")
        else:
            print("❌ FFmpeg not working")
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
    except Exception as e:
        print(f"❌ FFmpeg error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test video loading capabilities')
    parser.add_argument('--dataset', help='Path to dataset directory')
    parser.add_argument('--video', help='Path to specific video file')
    parser.add_argument('--num-frames', type=int, default=8, help='Number of frames to extract')
    parser.add_argument('--num-test-videos', type=int, default=5, help='Number of videos to test from dataset')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies only')
    
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
        return
    
    if args.video:
        test_video_file(args.video, args.num_frames)
    elif args.dataset:
        test_dataset_videos(args.dataset, args.num_test_videos)
    else:
        print("Please specify --video, --dataset, or --check-deps")
        parser.print_help()

if __name__ == "__main__":
    main() 