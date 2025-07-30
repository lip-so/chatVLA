#!/usr/bin/env python3
"""
Demonstration of the Prompt-Action Verification Agent

This script demonstrates the new prompt-action verification capabilities
that analyze whether task prompts correspond to actual robot actions.

The agent includes:
1. Action primitive detection from robot trajectories
2. NLP extraction from task prompts  
3. Semantic mapping between text and robot actions
4. Temporal alignment verification
5. Computer vision verification of object interactions
6. Task completion detection

Usage:
    python demo_prompt_action_verifier.py
"""

import numpy as np
import logging
from pathlib import Path
import sys

# Add metrics directory to path
sys.path.append('.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_basic_functionality():
    """Demonstrate basic functionality without requiring external dependencies."""
    print("=" * 80)
    print("PROMPT-ACTION VERIFICATION AGENT DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Test 1: Action primitive detection (works without spacy/yolo)
    print("🔬 Test 1: Action Primitive Detection")
    print("-" * 40)
    
    try:
        from metrics.action_primitives import ActionPrimitiveDetector
        
        # Create synthetic robot trajectory data
        robot_type = "so101_follower"
        detector = ActionPrimitiveDetector(robot_type)
        print(f"✅ ActionPrimitiveDetector initialized for {robot_type}")
        
        # Generate synthetic 6DOF robot actions (reach, grasp, lift, place sequence)
        time_steps = 180  # 6 seconds at 30Hz
        actions = np.zeros((time_steps, 6))
        
        # Reach phase (0-2s): movement in xyz
        actions[0:60, 0] = np.linspace(0, 0.3, 60)    # X movement
        actions[0:60, 1] = np.linspace(0, 0.2, 60)    # Y movement  
        actions[0:60, 2] = np.linspace(0, -0.1, 60)   # Z movement (down)
        
        # Grasp phase (2-3s): gripper closing
        actions[60:90, 5] = np.linspace(0, -0.8, 30)  # Gripper closing
        
        # Lift phase (3-5s): upward movement
        actions[90:150, 2] = np.linspace(-0.1, 0.2, 60)  # Z movement (up)
        
        # Place phase (5-6s): gripper opening
        actions[150:180, 5] = np.linspace(-0.8, 0, 30)  # Gripper opening
        
        timestamps = np.arange(time_steps) / 30.0
        
        # Detect primitives
        segments = detector.detect_primitives(actions, timestamps=timestamps)
        
        print(f"✅ Detected {len(segments)} action segments:")
        for i, segment in enumerate(segments):
            print(f"   {i+1}. {segment.primitive.value}: {segment.start_time:.1f}s-{segment.end_time:.1f}s "
                  f"(confidence: {segment.confidence:.2f})")
            print(f"      {segment.description}")
        
        summary = detector.get_primitive_summary(segments)
        print(f"✅ Primitive sequence: {' → '.join(summary['primitive_sequence'])}")
        print()
        
    except Exception as e:
        print(f"❌ Error in action primitive detection: {e}")
        print()
    
    # Test 2: Check component availability
    print("🔍 Test 2: Component Availability Check")
    print("-" * 40)
    
    dependencies = {
        'spacy': 'Natural Language Processing',
        'ultralytics': 'Computer Vision (YOLO)',
        'dtaidistance': 'Temporal Sequence Alignment',
        'clip': 'CLIP Model for Vision-Language'
    }
    
    available_components = []
    missing_components = []
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: {description}")
            available_components.append(dep)
        except ImportError:
            print(f"❌ {dep}: {description} (not installed)")
            missing_components.append(dep)
    
    print()
    
    # Test 3: Integration with action consistency metric
    print("🔗 Test 3: Integration with Action Consistency Metric")
    print("-" * 40)
    
    try:
        from metrics.action_consistency import ActionConsistencyMetric
        
        # Test without prompt verification (should work)
        metric_basic = ActionConsistencyMetric(enable_prompt_verification=False)
        print("✅ ActionConsistencyMetric (basic mode) initialized successfully")
        
        # Test with prompt verification (may fail if dependencies missing)
        try:
            metric_full = ActionConsistencyMetric(enable_prompt_verification=True, robot_type="so101_follower")
            print("✅ ActionConsistencyMetric (with prompt verification) initialized successfully")
        except Exception as e:
            print(f"⚠️  ActionConsistencyMetric (with prompt verification) failed: {e}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error testing ActionConsistencyMetric: {e}")
        print()

def demo_full_verification():
    """Demonstrate full verification if all dependencies are available."""
    print("🚀 Test 4: Full Prompt-Action Verification (if dependencies available)")
    print("-" * 40)
    
    try:
        from metrics.prompt_action_verifier import PromptActionVerifier
        
        # Initialize the full verification agent
        verifier = PromptActionVerifier(robot_type="so101_follower")
        print("✅ PromptActionVerifier initialized successfully")
        
        # Create test data
        prompt = "Grasp a lego block and put it in the bin"
        
        # Generate synthetic robot actions (pick and place sequence)
        time_steps = 240  # 8 seconds at 30Hz
        actions = np.zeros((time_steps, 6))
        
        # Approach phase (0-2s)
        actions[0:60, 0] = np.linspace(0, 0.2, 60)
        actions[0:60, 1] = np.linspace(0, 0.1, 60)
        actions[0:60, 2] = np.linspace(0, -0.05, 60)
        
        # Grasp phase (2-3s)  
        actions[60:90, 5] = np.linspace(0, -0.7, 30)
        
        # Lift phase (3-5s)
        actions[90:150, 2] = np.linspace(-0.05, 0.15, 60)
        
        # Move to bin (5-7s)
        actions[150:210, 0] = np.linspace(0.2, 0.4, 60)
        actions[150:210, 1] = np.linspace(0.1, 0.3, 60)
        
        # Release phase (7-8s)
        actions[210:240, 5] = np.linspace(-0.7, 0, 30)
        
        timestamps = np.arange(time_steps) / 30.0
        
        # Run verification
        result = verifier.verify_prompt_action_correspondence(
            prompt=prompt,
            actions=actions,
            timestamps=timestamps,
            video_path=None  # No video for this demo
        )
        
        print("✅ Verification completed successfully!")
        print()
        print("📊 Verification Scores:")
        scores = result.verification_scores
        print(f"   Overall Score: {scores.overall_verification_score:.3f}")
        print(f"   - Action Primitive Detection: {scores.action_primitive_detection:.3f}")
        print(f"   - Text Extraction Quality: {scores.text_extraction_quality:.3f}")
        print(f"   - Semantic Mapping Accuracy: {scores.semantic_mapping_accuracy:.3f}")
        print(f"   - Temporal Alignment: {scores.temporal_alignment:.3f}")
        print(f"   - Object Interaction Verification: {scores.object_interaction_verification:.3f}")
        print(f"   - Task Completion Verification: {scores.task_completion_verification:.3f}")
        print()
        
        print("🔍 Analysis Results:")
        print(f"   Text Actions Found: {len(result.text_components.actions)}")
        if result.text_components.actions:
            action_names = [a.verb for a in result.text_components.actions]
            print(f"   Actions: {', '.join(action_names)}")
        
        print(f"   Robot Primitives Detected: {len(result.detected_primitives)}")
        if result.detected_primitives:
            primitive_names = [p.primitive.value for p in result.detected_primitives]
            print(f"   Primitives: {', '.join(primitive_names)}")
        
        print(f"   Semantic Mappings: {len(result.semantic_mapping.mappings)}")
        print(f"   Completion Status: {result.completion_result.completion_status.value}")
        print()
        
        if result.issues_found:
            print("⚠️  Issues Found:")
            for issue in result.issues_found:
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue['severity'], "⚪")
                print(f"   {severity_icon} {issue['description']}")
            print()
        
        if result.recommendations:
            print("💡 Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"   {i}. {rec}")
            print()
        
        # Generate a report
        report = verifier.get_verification_report(result)
        print("📋 Full Verification Report:")
        print(report)
        
    except ImportError as e:
        print(f"⚠️  Full verification not available: {e}")
        print("   To enable full verification, install missing dependencies:")
        print("   pip install spacy ultralytics dtaidistance")
        print("   python -m spacy download en_core_web_sm")
        print()
    except Exception as e:
        print(f"❌ Error in full verification: {e}")
        print()

def demo_real_dataset():
    """Demonstrate verification on real dataset if available."""
    print("📁 Test 5: Real Dataset Verification")
    print("-" * 40)
    
    dataset_path = Path("150episodes6")
    if not dataset_path.exists():
        print("⚠️  150episodes6 dataset not found - skipping real dataset demo")
        print("   The agent can work with any robotics dataset in LeRobot format")
        print()
        return
    
    try:
        from metrics.prompt_action_verifier import PromptActionVerifier
        import pandas as pd
        import json
        
        # Load dataset metadata
        meta_path = dataset_path / "meta" / "episodes.jsonl"
        if not meta_path.exists():
            print("⚠️  Dataset metadata not found")
            return
        
        episodes = []
        with open(meta_path, 'r') as f:
            for line in f:
                episodes.append(json.loads(line))
        
        print(f"✅ Found {len(episodes)} episodes in dataset")
        
        # Verify first episode
        episode = episodes[0]
        print(f"   Verifying episode: {episode.get('episode_index', 'unknown')}")
        print(f"   Task: {episode.get('task', 'unknown')}")
        
        verifier = PromptActionVerifier(robot_type="so101_follower") 
        result = verifier.verify_dataset_episode(episode, dataset_path)
        
        print(f"✅ Episode verification completed!")
        print(f"   Overall Score: {result.verification_scores.overall_verification_score:.3f}")
        print(f"   Completion Status: {result.completion_result.completion_status.value}")
        
        if result.issues_found:
            print(f"   Issues Found: {len(result.issues_found)}")
        
        print()
        
    except ImportError:
        print("⚠️  Full verification dependencies not available")
        print()
    except Exception as e:
        print(f"❌ Error verifying real dataset: {e}")
        print()

def main():
    """Run the demonstration."""
    print("This demonstration shows the new Prompt-Action Verification Agent")
    print("that analyzes whether task descriptions match actual robot actions.")
    print()
    
    # Run basic functionality tests (should always work)
    demo_basic_functionality()
    
    # Run full verification demo (requires dependencies)
    demo_full_verification()
    
    # Try real dataset demo
    demo_real_dataset()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features of the Prompt-Action Verification Agent:")
    print("1. 🤖 Action Primitive Detection - Identifies high-level behaviors from trajectories")
    print("2. 📝 NLP Extraction - Extracts actionable components from prompts")
    print("3. 🔗 Semantic Mapping - Maps text actions to robot primitives")
    print("4. ⏰ Temporal Alignment - Verifies action sequences and timing")
    print("5. 👁️  Computer Vision - Verifies object interactions (requires YOLO)")
    print("6. ✅ Completion Detection - Determines if tasks were completed")
    print()
    print("The agent integrates seamlessly with the existing evaluation framework")
    print("and can be enabled/disabled based on available dependencies.")
    print()
    print("To install full dependencies:")
    print("  pip install spacy ultralytics dtaidistance")
    print("  python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main() 