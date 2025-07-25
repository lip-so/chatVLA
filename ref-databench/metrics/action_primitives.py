"""
Action Primitive Detection Module

Analyzes robot action trajectories to identify high-level behaviors and primitive actions
such as reaching, grasping, placing, pushing, pulling, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import scipy.signal
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)

class ActionPrimitive(Enum):
    """Enumeration of robotic action primitives."""
    REACH = "reach"
    GRASP = "grasp" 
    RELEASE = "release"
    PLACE = "place"
    PUSH = "push"
    PULL = "pull"
    LIFT = "lift"
    LOWER = "lower"
    ROTATE = "rotate"
    HOLD = "hold"
    MOVE = "move"
    APPROACH = "approach"
    RETREAT = "retreat"
    UNKNOWN = "unknown"

@dataclass
class ActionSegment:
    """Represents a detected action primitive segment."""
    primitive: ActionPrimitive
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float
    features: Dict[str, Any]
    description: str
    
    def __hash__(self):
        """Make ActionSegment hashable for use in sets and as dict keys."""
        return hash((self.primitive, self.start_time, self.end_time, self.start_frame, self.end_frame))
    
    def __eq__(self, other):
        """Define equality for ActionSegment objects."""
        if not isinstance(other, ActionSegment):
            return False
        return (self.primitive == other.primitive and 
                self.start_time == other.start_time and 
                self.end_time == other.end_time and
                self.start_frame == other.start_frame and
                self.end_frame == other.end_frame)

class ActionPrimitiveDetector:
    """Detects action primitives from robot trajectory data."""
    
    def __init__(self, robot_type: str = "generic_6dof"):
        """
        Initialize the action primitive detector.
        
        Args:
            robot_type: Type of robot for specialized detection rules
        """
        self.robot_type = robot_type
        self.gripper_threshold = 0.01  # Threshold for gripper state changes
        self.velocity_threshold = 0.05  # Threshold for significant movement
        self.position_threshold = 0.02  # Threshold for position changes
        self.smoothing_window = 5  # Window for smoothing signals
        
        # Robot-specific configurations
        self._setup_robot_config()
    
    def _setup_robot_config(self):
        """Setup robot-specific configuration parameters."""
        if self.robot_type == "franka_panda":
            self.dof = 7
            self.gripper_joint = 6  # Last joint is gripper
            self.position_joints = list(range(3))  # First 3 joints affect end-effector position most
        elif self.robot_type == "ur5":
            self.dof = 6
            self.gripper_joint = -1  # UR5 typically has separate gripper
            self.position_joints = list(range(3))
        elif self.robot_type == "so101_follower":
            self.dof = 6
            self.gripper_joint = 5  # Assume last joint is gripper
            self.position_joints = list(range(3))
        elif self.robot_type == "generic_6dof":
            # Generic configuration - use dynamic detection
            self.dof = None  # Will be detected from trajectory data
            self.gripper_joint = None  # Will be set to last joint
            self.position_joints = None  # Will be set to first 3 or half of joints
        else:  # Unknown configuration - will be set dynamically from data
            self.dof = None  # Will be detected from trajectory data
            self.gripper_joint = None  # Will be set to last joint
            self.position_joints = None  # Will be set to first 3 or half of joints
    
    def _setup_dynamic_config(self, trajectory: np.ndarray):
        """Setup configuration dynamically based on trajectory data shape."""
        detected_dof = trajectory.shape[1]
        
        # Update configuration if DOF has changed or was not set
        if self.dof is None or (self.robot_type in ["generic_6dof", "unknown_robot"] and detected_dof != self.dof):
            if self.dof is not None and detected_dof != self.dof:
                logger.info(f"Updating DOF configuration from {self.dof} to {detected_dof}")
            self.dof = detected_dof
            
            # Intelligent assumptions for unknown configurations
            if detected_dof == 1:
                # Single joint - probably not a gripper
                self.gripper_joint = -1
                self.position_joints = [0]
            elif detected_dof == 2:
                # Two joints - probably both position, no gripper
                self.gripper_joint = -1
                self.position_joints = [0, 1]
            elif detected_dof <= 6:
                # Standard arm configuration - assume last joint is gripper
                self.gripper_joint = detected_dof - 1
                self.position_joints = list(range(min(3, detected_dof - 1)))
            else:
                # High DOF robot (7+ joints) - assume last joint is gripper
                self.gripper_joint = detected_dof - 1
                # Use first half of joints for position (excluding gripper)
                num_position_joints = min(6, detected_dof - 1)
                self.position_joints = list(range(num_position_joints))
            
            logger.info(f"Detected {detected_dof} DOF robot - assuming gripper at joint {self.gripper_joint}, "
                       f"position joints: {self.position_joints}")
    
    def detect_primitives(self, actions: np.ndarray, states: Optional[np.ndarray] = None, 
                         timestamps: Optional[np.ndarray] = None) -> List[ActionSegment]:
        """
        Detect action primitives from trajectory data.
        
        Args:
            actions: Array of shape (T, DOF) containing robot actions
            states: Optional array of shape (T, DOF) containing robot states
            timestamps: Optional array of timestamps
            
        Returns:
            List of detected action segments
        """
        if len(actions) < 3:
            return []
        
        # Use states if available, otherwise use actions
        trajectory = states if states is not None else actions
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = np.arange(len(trajectory)) / 30.0  # Assume 30Hz
        
        # Setup dynamic configuration if needed
        self._setup_dynamic_config(trajectory)
        
        # Extract features for primitive detection
        features = self._extract_trajectory_features(trajectory, timestamps)
        
        # Detect different types of primitives
        segments = []
        
        # Detect movement patterns
        movement_segments = self._detect_movement_primitives(trajectory, timestamps, features)
        segments.extend(movement_segments)
        
        # Detect gripper actions
        if self.gripper_joint is not None and self.gripper_joint >= 0 and self.gripper_joint < trajectory.shape[1]:
            gripper_segments = self._detect_gripper_primitives(trajectory, timestamps)
            segments.extend(gripper_segments)
        else:
            logger.debug(f"Skipping gripper detection - gripper_joint: {self.gripper_joint}, trajectory shape: {trajectory.shape}")
        
        # Detect composite actions
        composite_segments = self._detect_composite_primitives(segments, trajectory, timestamps)
        segments.extend(composite_segments)
        
        # Sort segments by start time and resolve overlaps
        segments = self._resolve_overlapping_segments(segments)
        
        return segments
    
    def _extract_trajectory_features(self, trajectory: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features from trajectory for primitive detection."""
        features = {}
        
        # Compute velocities
        dt = np.diff(timestamps)
        dt = np.append(dt, dt[-1])  # Extend to match length
        velocities = np.gradient(trajectory, axis=0) / dt[:, np.newaxis]
        features['velocities'] = velocities
        
        # Compute accelerations
        accelerations = np.gradient(velocities, axis=0) / dt[:, np.newaxis]
        features['accelerations'] = accelerations
        
        # Compute speed (magnitude of velocity)
        speeds = np.linalg.norm(velocities, axis=1)
        features['speeds'] = speeds
        
        # Smooth signals for better detection
        if len(trajectory) > self.smoothing_window:
            features['smooth_positions'] = scipy.signal.savgol_filter(
                trajectory, self.smoothing_window, 3, axis=0
            )
            features['smooth_velocities'] = scipy.signal.savgol_filter(
                velocities, self.smoothing_window, 3, axis=0
            )
            features['smooth_speeds'] = scipy.signal.savgol_filter(
                speeds, self.smoothing_window, 3
            )
        else:
            features['smooth_positions'] = trajectory.copy()
            features['smooth_velocities'] = velocities.copy()
            features['smooth_speeds'] = speeds.copy()
        
        # Detect key points (local minima/maxima in speed)
        speed_peaks, _ = scipy.signal.find_peaks(features['smooth_speeds'], height=self.velocity_threshold)
        speed_valleys, _ = scipy.signal.find_peaks(-features['smooth_speeds'])
        features['speed_peaks'] = speed_peaks
        features['speed_valleys'] = speed_valleys
        
        return features
    
    def _detect_movement_primitives(self, trajectory: np.ndarray, timestamps: np.ndarray, 
                                  features: Dict[str, np.ndarray]) -> List[ActionSegment]:
        """Detect movement-based primitives (reach, move, approach, retreat)."""
        segments = []
        speeds = features['smooth_speeds']
        
        # Find movement phases (high speed) vs stationary phases (low speed)
        moving_mask = speeds > self.velocity_threshold
        stationary_mask = ~moving_mask
        
        # Find continuous movement segments
        movement_changes = np.diff(moving_mask.astype(int))
        movement_starts = np.where(movement_changes == 1)[0] + 1
        movement_ends = np.where(movement_changes == -1)[0] + 1
        
        # Handle edge cases
        if moving_mask[0]:
            movement_starts = np.insert(movement_starts, 0, 0)
        if moving_mask[-1]:
            movement_ends = np.append(movement_ends, len(moving_mask))
        
        # Create movement segments
        for start, end in zip(movement_starts, movement_ends):
            if end - start < 3:  # Skip very short movements
                continue
                
            # Analyze movement characteristics
            segment_positions = trajectory[start:end]
            segment_velocities = features['velocities'][start:end]
            
            # Determine movement type based on characteristics
            movement_distance = np.linalg.norm(segment_positions[-1] - segment_positions[0])
            avg_speed = np.mean(speeds[start:end])
            
            # Classify movement type
            if movement_distance > 0.1:  # Significant position change
                if avg_speed > 0.2:  # Fast movement
                    primitive = ActionPrimitive.REACH
                    description = f"Reaching movement (dist: {movement_distance:.3f}m)"
                else:  # Slow movement
                    primitive = ActionPrimitive.APPROACH
                    description = f"Approaching movement (dist: {movement_distance:.3f}m)"
            else:  # Small position change
                primitive = ActionPrimitive.MOVE
                description = f"Local movement (dist: {movement_distance:.3f}m)"
            
            confidence = min(1.0, avg_speed / 0.5)  # Confidence based on speed
            
            segment = ActionSegment(
                primitive=primitive,
                start_time=timestamps[start],
                end_time=timestamps[end-1],
                start_frame=start,
                end_frame=end-1,
                confidence=confidence,
                features={
                    'movement_distance': movement_distance,
                    'avg_speed': avg_speed,
                    'max_speed': np.max(speeds[start:end])
                },
                description=description
            )
            segments.append(segment)
        
        return segments
    
    def _detect_gripper_primitives(self, trajectory: np.ndarray, timestamps: np.ndarray) -> List[ActionSegment]:
        """Detect gripper-based primitives (grasp, release)."""
        segments = []
        
        if self.gripper_joint >= trajectory.shape[1]:
            return segments
        
        gripper_values = trajectory[:, self.gripper_joint]
        
        # Detect gripper state changes
        gripper_diff = np.abs(np.diff(gripper_values))
        significant_changes = gripper_diff > self.gripper_threshold
        
        change_indices = np.where(significant_changes)[0]
        
        for change_idx in change_indices:
            start_idx = max(0, change_idx - 2)
            end_idx = min(len(gripper_values), change_idx + 3)
            
            start_value = gripper_values[start_idx]
            end_value = gripper_values[end_idx-1]
            
            # Determine if gripper is closing (grasping) or opening (releasing)
            if end_value < start_value:  # Gripper closing
                primitive = ActionPrimitive.GRASP
                description = f"Grasping action (gripper: {start_value:.3f} → {end_value:.3f})"
            else:  # Gripper opening
                primitive = ActionPrimitive.RELEASE
                description = f"Releasing action (gripper: {start_value:.3f} → {end_value:.3f})"
            
            confidence = min(1.0, abs(end_value - start_value) / 0.05)  # Confidence based on change magnitude
            
            segment = ActionSegment(
                primitive=primitive,
                start_time=timestamps[start_idx],
                end_time=timestamps[end_idx-1],
                start_frame=start_idx,
                end_frame=end_idx-1,
                confidence=confidence,
                features={
                    'gripper_start': start_value,
                    'gripper_end': end_value,
                    'gripper_change': abs(end_value - start_value)
                },
                description=description
            )
            segments.append(segment)
        
        return segments
    
    def _detect_composite_primitives(self, existing_segments: List[ActionSegment], 
                                   trajectory: np.ndarray, timestamps: np.ndarray) -> List[ActionSegment]:
        """Detect composite primitives like pick, place, push, pull."""
        composite_segments = []
        
        # Sort segments by start time
        segments = sorted(existing_segments, key=lambda x: x.start_time)
        
        # Look for common patterns
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            # Check for pick pattern: reach + grasp
            if (current.primitive in [ActionPrimitive.REACH, ActionPrimitive.APPROACH] and 
                next_seg.primitive == ActionPrimitive.GRASP and
                next_seg.start_time - current.end_time < 2.0):  # Within 2 seconds
                
                # Check if there's a subsequent lift
                lift_segment = None
                if i + 2 < len(segments):
                    potential_lift = segments[i + 2]
                    if (potential_lift.start_time - next_seg.end_time < 1.0 and
                        potential_lift.primitive in [ActionPrimitive.MOVE, ActionPrimitive.REACH]):
                        
                        # Check if movement is upward (assuming z-axis is up)
                        if len(trajectory.shape) > 1 and trajectory.shape[1] >= 3:
                            z_start = trajectory[potential_lift.start_frame, 2]
                            z_end = trajectory[potential_lift.end_frame, 2]
                            if z_end > z_start + 0.02:  # Upward movement
                                lift_segment = potential_lift
                
                # Create pick composite action
                end_time = lift_segment.end_time if lift_segment else next_seg.end_time
                end_frame = lift_segment.end_frame if lift_segment else next_seg.end_frame
                
                pick_segment = ActionSegment(
                    primitive=ActionPrimitive.LIFT,  # Use LIFT to represent pick
                    start_time=current.start_time,
                    end_time=end_time,
                    start_frame=current.start_frame,
                    end_frame=end_frame,
                    confidence=min(current.confidence, next_seg.confidence),
                    features={
                        'components': ['reach', 'grasp'] + (['lift'] if lift_segment else []),
                        'has_lift': lift_segment is not None
                    },
                    description=f"Pick action (reach + grasp{' + lift' if lift_segment else ''})"
                )
                composite_segments.append(pick_segment)
            
            # Check for place pattern: move + release
            elif (current.primitive in [ActionPrimitive.MOVE, ActionPrimitive.APPROACH] and
                  next_seg.primitive == ActionPrimitive.RELEASE and
                  next_seg.start_time - current.end_time < 2.0):
                
                place_segment = ActionSegment(
                    primitive=ActionPrimitive.PLACE,
                    start_time=current.start_time,
                    end_time=next_seg.end_time,
                    start_frame=current.start_frame,
                    end_frame=next_seg.end_frame,
                    confidence=min(current.confidence, next_seg.confidence),
                    features={
                        'components': ['move', 'release']
                    },
                    description="Place action (move + release)"
                )
                composite_segments.append(place_segment)
        
        return composite_segments
    
    def _resolve_overlapping_segments(self, segments: List[ActionSegment]) -> List[ActionSegment]:
        """Resolve overlapping segments by keeping higher confidence ones."""
        if not segments:
            return segments
        
        # Sort by start time
        segments = sorted(segments, key=lambda x: x.start_time)
        
        # Remove overlaps, keeping higher confidence segments
        resolved = []
        for segment in segments:
            # Check for overlaps with existing resolved segments
            overlapping = [s for s in resolved if self._segments_overlap(s, segment)]
            
            if not overlapping:
                resolved.append(segment)
            else:
                # Keep segment with highest confidence among overlapping ones
                all_overlapping = overlapping + [segment]
                best_segment = max(all_overlapping, key=lambda x: x.confidence)
                
                # Remove overlapping segments and add the best one
                for overlap in overlapping:
                    if overlap in resolved:
                        resolved.remove(overlap)
                
                if best_segment not in resolved:
                    resolved.append(best_segment)
        
        return sorted(resolved, key=lambda x: x.start_time)
    
    def _segments_overlap(self, seg1: ActionSegment, seg2: ActionSegment) -> bool:
        """Check if two segments overlap in time."""
        return not (seg1.end_time <= seg2.start_time or seg2.end_time <= seg1.start_time)
    
    def get_primitive_sequence(self, segments: List[ActionSegment]) -> List[str]:
        """Get sequence of primitive names from segments."""
        return [segment.primitive.value for segment in sorted(segments, key=lambda x: x.start_time)]
    
    def get_primitive_summary(self, segments: List[ActionSegment]) -> Dict[str, Any]:
        """Get summary statistics of detected primitives."""
        if not segments:
            return {
                'total_segments': 0,
                'primitive_counts': {},
                'avg_confidence': 0.0,
                'total_duration': 0.0
            }
        
        primitive_counts = {}
        for segment in segments:
            prim = segment.primitive.value
            primitive_counts[prim] = primitive_counts.get(prim, 0) + 1
        
        total_duration = max(s.end_time for s in segments) - min(s.start_time for s in segments)
        avg_confidence = np.mean([s.confidence for s in segments])
        
        return {
            'total_segments': len(segments),
            'primitive_counts': primitive_counts,
            'avg_confidence': avg_confidence,
            'total_duration': total_duration,
            'primitive_sequence': self.get_primitive_sequence(segments)
        } 