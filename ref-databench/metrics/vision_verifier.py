"""
Vision Verifier Metric

This module provides functionality for verifying vision-related claims about a dataset.
"""

import os
import sys
import logging
import numpy as np
import torch
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import CLIPProcessor, CLIPModel  # Using transformers' CLIP instead of clip-by-openai

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required classes from other metrics modules
from .nlp_extractor import TaskComponents, ExtractedObject, SpatialRelation
from .action_primitives import ActionSegment, ActionPrimitive
from .semantic_mapper import SemanticMapping

logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """Represents a detected object in a video frame."""
    name: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    frame_idx: int
    timestamp: float
    properties: Dict[str, Any]  # color, size, etc.

@dataclass
class ObjectTrajectory:
    """Represents the trajectory of an object over time."""
    object_name: str
    detections: List[DetectedObject]
    start_time: float
    end_time: float
    center_trajectory: List[Tuple[float, float]]  # (x, y) centers over time
    motion_patterns: List[str]  # 'static', 'moving', 'picked_up', 'placed'

@dataclass
class InteractionEvent:
    """Represents an interaction between robot and object."""
    object_name: str
    interaction_type: str  # 'approach', 'contact', 'grasp', 'release'
    start_time: float
    end_time: float
    confidence: float
    evidence: Dict[str, Any]

@dataclass
class VerificationResult:
    """Result of vision-based verification."""
    detected_objects: List[DetectedObject]
    object_trajectories: List[ObjectTrajectory]
    interaction_events: List[InteractionEvent]
    object_verification_score: float
    interaction_verification_score: float
    spatial_verification_score: float
    overall_vision_score: float
    verification_details: Dict[str, Any]

class VisionVerifier:
    """Verifies prompt descriptions using computer vision analysis."""
    
    def __init__(self, yolo_model: str = "yolov8n.pt"):
        """
        Initialize the vision verifier.
        
        Args:
            yolo_model: Path to YOLO model for object detection
        """
        try:
            self.yolo = YOLO(yolo_model)
            logger.info(f"Loaded YOLO model: {yolo_model}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.yolo = None
        
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("Loaded CLIP model for object verification")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
        
        self._setup_object_mappings()
        self._setup_interaction_patterns()
    
    def _setup_object_mappings(self):
        """Setup mappings between text object names and vision classes."""
        # YOLO class names to common object names
        self.yolo_to_common = {
            'bottle': ['bottle', 'container'],
            'cup': ['cup', 'mug'],
            'bowl': ['bowl'],
            'banana': ['banana'],
            'apple': ['apple'],
            'orange': ['orange'],
            'book': ['book'],
            'laptop': ['laptop'],
            'mouse': ['mouse'],
            'keyboard': ['keyboard'],
            'cell phone': ['phone'],
            'remote': ['remote'],
            'scissors': ['scissors'],
            'teddy bear': ['bear', 'toy'],
            'hair drier': ['dryer'],
            'toothbrush': ['brush']
        }
        
        # Color mappings for object properties
        self.color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'green': ([35, 50, 50], [85, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'yellow': ([20, 50, 50], [35, 255, 255]),
            'orange': ([10, 50, 50], [20, 255, 255]),
            'purple': ([130, 50, 50], [160, 255, 255])
        }
    
    def _setup_interaction_patterns(self):
        """Setup patterns for detecting robot-object interactions."""
        self.interaction_thresholds = {
            'approach': 100,  # pixels
            'contact': 20,    # pixels
            'grasp': 10,      # pixels + object motion
            'release': 30     # pixels + object motion
        }
        
        self.motion_thresholds = {
            'static': 5,      # pixels movement
            'moving': 20,     # pixels movement
            'picked_up': 50   # pixels upward movement
        }
    
    def verify_prompt_actions(self, text_components: TaskComponents,
                            robot_segments: List[ActionSegment],
                            semantic_mappings: List[SemanticMapping],
                            video_path: str) -> VerificationResult:
        """
        Verify prompt descriptions using video analysis.
        
        Args:
            text_components: Extracted components from text
            robot_segments: Detected robot action segments
            semantic_mappings: Mappings between text and robot actions
            video_path: Path to video file
            
        Returns:
            VerificationResult containing verification analysis
        """
        if not self.yolo:
            logger.error("YOLO model not available for object detection")
            return self._create_empty_result()
        
        # Load and analyze video
        frames, timestamps = self._load_video_frames(video_path)
        if not frames:
            logger.error(f"Could not load video: {video_path}")
            return self._create_empty_result()
        
        # Detect objects in frames
        detected_objects = self._detect_objects_in_video(frames, timestamps)
        
        # Track objects over time
        object_trajectories = self._track_objects(detected_objects)
        
        # Detect interactions between robot and objects
        interaction_events = self._detect_interactions(
            object_trajectories, robot_segments, frames, timestamps
        )
        
        # Verify objects mentioned in text
        object_verification_score = self._verify_mentioned_objects(
            text_components.objects, detected_objects
        )
        
        # Verify interactions mentioned in text
        interaction_verification_score = self._verify_mentioned_interactions(
            text_components, interaction_events, semantic_mappings
        )
        
        # Verify spatial relationships
        spatial_verification_score = self._verify_spatial_relationships(
            text_components.spatial_relations, object_trajectories
        )
        
        # Calculate overall score
        overall_vision_score = self._calculate_overall_vision_score(
            object_verification_score, interaction_verification_score, spatial_verification_score
        )
        
        # Create verification details
        verification_details = self._create_verification_details(
            text_components, detected_objects, interaction_events, object_trajectories
        )
        
        return VerificationResult(
            detected_objects=detected_objects,
            object_trajectories=object_trajectories,
            interaction_events=interaction_events,
            object_verification_score=object_verification_score,
            interaction_verification_score=interaction_verification_score,
            spatial_verification_score=spatial_verification_score,
            overall_vision_score=overall_vision_score,
            verification_details=verification_details
        )
    
    def _load_video_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """Load video frames and timestamps."""
        frames = []
        timestamps = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames (every 5th frame to reduce computation)
                if frame_idx % 5 == 0:
                    frames.append(frame)
                    timestamps.append(frame_idx / fps)
                
                frame_idx += 1
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
        
        return frames, timestamps
    
    def _detect_objects_in_video(self, frames: List[np.ndarray], 
                                timestamps: List[float]) -> List[DetectedObject]:
        """Detect objects in all video frames."""
        detected_objects = []
        
        for frame_idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            try:
                results = self.yolo(frame, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for i, box in enumerate(boxes):
                            # Extract detection information
                            bbox = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Get class name
                            class_name = self.yolo.names[class_id]
                            
                            # Extract object properties (color, etc.)
                            properties = self._extract_object_properties(frame, bbox)
                            
                            detected_obj = DetectedObject(
                                name=class_name,
                                bbox=tuple(bbox),
                                confidence=confidence,
                                frame_idx=frame_idx,
                                timestamp=timestamp,
                                properties=properties
                            )
                            detected_objects.append(detected_obj)
            
            except Exception as e:
                logger.warning(f"Error detecting objects in frame {frame_idx}: {e}")
        
        return detected_objects
    
    def _extract_object_properties(self, frame: np.ndarray, 
                                  bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Extract properties (color, size) from detected object."""
        x1, y1, x2, y2 = bbox
        
        # Extract object region
        obj_region = frame[y1:y2, x1:x2]
        
        if obj_region.size == 0:
            return {}
        
        properties = {}
        
        # Detect dominant color
        dominant_color = self._detect_dominant_color(obj_region)
        if dominant_color:
            properties['color'] = dominant_color
        
        # Calculate size
        width = x2 - x1
        height = y2 - y1
        area = width * height
        properties['size'] = 'large' if area > 10000 else 'medium' if area > 2500 else 'small'
        properties['width'] = width
        properties['height'] = height
        properties['area'] = area
        
        return properties
    
    def _detect_dominant_color(self, region: np.ndarray) -> Optional[str]:
        """Detect dominant color in an image region."""
        if region.size == 0:
            return None
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Find most common color
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            percentage = np.sum(mask > 0) / mask.size
            
            if percentage > 0.3:  # At least 30% of the region
                return color_name
        
        return None
    
    def _track_objects(self, detected_objects: List[DetectedObject]) -> List[ObjectTrajectory]:
        """Track objects across frames to create trajectories."""
        # Group detections by object type
        object_groups = {}
        for obj in detected_objects:
            if obj.name not in object_groups:
                object_groups[obj.name] = []
            object_groups[obj.name].append(obj)
        
        trajectories = []
        
        for object_name, detections in object_groups.items():
            # Sort by timestamp
            detections.sort(key=lambda x: x.timestamp)
            
            # Create trajectory
            if detections:
                center_trajectory = []
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    center_trajectory.append((center_x, center_y))
                
                motion_patterns = self._analyze_motion_patterns(center_trajectory)
                
                trajectory = ObjectTrajectory(
                    object_name=object_name,
                    detections=detections,
                    start_time=detections[0].timestamp,
                    end_time=detections[-1].timestamp,
                    center_trajectory=center_trajectory,
                    motion_patterns=motion_patterns
                )
                trajectories.append(trajectory)
        
        return trajectories
    
    def _analyze_motion_patterns(self, trajectory: List[Tuple[float, float]]) -> List[str]:
        """Analyze motion patterns in object trajectory."""
        if len(trajectory) < 2:
            return ['static']
        
        patterns = []
        movements = []
        
        # Calculate movement between consecutive points
        for i in range(1, len(trajectory)):
            prev_x, prev_y = trajectory[i-1]
            curr_x, curr_y = trajectory[i]
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            distance = np.sqrt(dx**2 + dy**2)
            movements.append((distance, dx, dy))
        
        # Analyze overall movement
        total_movement = sum(m[0] for m in movements)
        avg_movement = total_movement / len(movements)
        
        if avg_movement < self.motion_thresholds['static']:
            patterns.append('static')
        elif avg_movement > self.motion_thresholds['moving']:
            patterns.append('moving')
            
            # Check for upward movement (being picked up)
            upward_movement = sum(m[2] for m in movements if m[2] < -10)  # Negative y is up
            if abs(upward_movement) > self.motion_thresholds['picked_up']:
                patterns.append('picked_up')
        
        return patterns if patterns else ['unknown']
    
    def _detect_interactions(self, object_trajectories: List[ObjectTrajectory],
                           robot_segments: List[ActionSegment],
                           frames: List[np.ndarray],
                           timestamps: List[float]) -> List[InteractionEvent]:
        """Detect interactions between robot and objects."""
        interactions = []
        
        # For each robot action segment
        for segment in robot_segments:
            # Find frames within this time segment
            segment_frames = [
                (i, frame, ts) for i, (frame, ts) in enumerate(zip(frames, timestamps))
                if segment.start_time <= ts <= segment.end_time
            ]
            
            if not segment_frames:
                continue
            
            # Look for object interactions in these frames
            for trajectory in object_trajectories:
                # Find trajectory points within this time segment
                segment_detections = [
                    det for det in trajectory.detections
                    if segment.start_time <= det.timestamp <= segment.end_time
                ]
                
                if not segment_detections:
                    continue
                
                # Analyze for interaction patterns
                interaction = self._analyze_robot_object_interaction(
                    segment, trajectory, segment_detections, segment_frames
                )
                
                if interaction:
                    interactions.append(interaction)
        
        return interactions
    
    def _analyze_robot_object_interaction(self, segment: ActionSegment,
                                        trajectory: ObjectTrajectory,
                                        detections: List[DetectedObject],
                                        frame_data: List[Tuple[int, np.ndarray, float]]) -> Optional[InteractionEvent]:
        """Analyze interaction between robot segment and object trajectory."""
        if not detections or not frame_data:
            return None
        
        # Simple heuristics for interaction detection
        interaction_type = None
        confidence = 0.0
        evidence = {}
        
        # Check for grasp pattern
        if segment.primitive == ActionPrimitive.GRASP:
            # Look for object becoming stationary or moving with robot
            if 'picked_up' in trajectory.motion_patterns:
                interaction_type = 'grasp'
                confidence = 0.8
                evidence['pattern'] = 'object_lifted'
        
        # Check for release pattern
        elif segment.primitive == ActionPrimitive.RELEASE:
            # Look for object starting to move independently
            if 'moving' in trajectory.motion_patterns:
                interaction_type = 'release'
                confidence = 0.7
                evidence['pattern'] = 'object_released'
        
        # Check for approach pattern
        elif segment.primitive in [ActionPrimitive.REACH, ActionPrimitive.APPROACH]:
            # Check if robot is getting closer to object
            interaction_type = 'approach'
            confidence = 0.6
            evidence['pattern'] = 'robot_approaching'
        
        if interaction_type:
            return InteractionEvent(
                object_name=trajectory.object_name,
                interaction_type=interaction_type,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=confidence,
                evidence=evidence
            )
        
        return None
    
    def _verify_mentioned_objects(self, text_objects: List[ExtractedObject],
                                detected_objects: List[DetectedObject]) -> float:
        """Verify that objects mentioned in text are present in video."""
        if not text_objects:
            return 1.0
        
        verified_count = 0
        
        for text_obj in text_objects:
            # Check if this object type was detected
            detected_names = {obj.name for obj in detected_objects}
            
            # Direct name match
            if text_obj.name in detected_names:
                verified_count += 1
                continue
            
            # Check for synonyms/variations
            found = False
            for detected_name in detected_names:
                if self._objects_match(text_obj.name, detected_name):
                    verified_count += 1
                    found = True
                    break
            
            if not found:
                # Check for color matches
                for detected_obj in detected_objects:
                    if (text_obj.properties and 
                        detected_obj.properties.get('color') in text_obj.properties):
                        verified_count += 0.5  # Partial credit for color match
                        break
        
        return verified_count / len(text_objects)
    
    def _objects_match(self, text_name: str, detected_name: str) -> bool:
        """Check if text object name matches detected object name."""
        # Direct match
        if text_name.lower() == detected_name.lower():
            return True
        
        # Check synonyms
        for yolo_class, synonyms in self.yolo_to_common.items():
            if detected_name == yolo_class and text_name.lower() in synonyms:
                return True
            if text_name.lower() == yolo_class and detected_name.lower() in synonyms:
                return True
        
        # Partial string match
        if text_name.lower() in detected_name.lower() or detected_name.lower() in text_name.lower():
            return True
        
        return False
    
    def _verify_mentioned_interactions(self, text_components: TaskComponents,
                                     interaction_events: List[InteractionEvent],
                                     semantic_mappings: List[SemanticMapping]) -> float:
        """Verify that interactions mentioned in text occurred in video."""
        if not text_components.actions:
            return 1.0
        
        verified_count = 0
        
        for text_action in text_components.actions:
            # Find corresponding robot action through semantic mapping
            robot_primitive = None
            for mapping in semantic_mappings:
                if mapping.text_action == text_action.verb:
                    robot_primitive = mapping.robot_primitive
                    break
            
            if not robot_primitive:
                continue
            
            # Check if expected interaction occurred
            expected_interaction = self._map_action_to_interaction(robot_primitive)
            if expected_interaction:
                # Look for this interaction in detected events
                for event in interaction_events:
                    if event.interaction_type == expected_interaction:
                        verified_count += 1
                        break
        
        return verified_count / len(text_components.actions) if text_components.actions else 1.0
    
    def _map_action_to_interaction(self, primitive: ActionPrimitive) -> Optional[str]:
        """Map robot primitive to expected interaction type."""
        mapping = {
            ActionPrimitive.GRASP: 'grasp',
            ActionPrimitive.RELEASE: 'release',
            ActionPrimitive.REACH: 'approach',
            ActionPrimitive.APPROACH: 'approach'
        }
        return mapping.get(primitive)
    
    def _verify_spatial_relationships(self, spatial_relations: List[SpatialRelation],
                                    object_trajectories: List[ObjectTrajectory]) -> float:
        """Verify spatial relationships mentioned in text."""
        if not spatial_relations:
            return 1.0
        
        verified_count = 0
        
        for relation in spatial_relations:
            # Find trajectories for the two objects
            obj1_trajectory = None
            obj2_trajectory = None
            
            for traj in object_trajectories:
                if self._objects_match(relation.object1, traj.object_name):
                    obj1_trajectory = traj
                if self._objects_match(relation.object2, traj.object_name):
                    obj2_trajectory = traj
            
            if obj1_trajectory and obj2_trajectory:
                # Check if spatial relationship holds
                if self._check_spatial_relationship(relation.relation, obj1_trajectory, obj2_trajectory):
                    verified_count += 1
        
        return verified_count / len(spatial_relations)
    
    def _check_spatial_relationship(self, relation: str, 
                                  traj1: ObjectTrajectory, traj2: ObjectTrajectory) -> bool:
        """Check if spatial relationship holds between two object trajectories."""
        if not traj1.center_trajectory or not traj2.center_trajectory:
            return False
        
        # Use final positions for spatial check
        x1, y1 = traj1.center_trajectory[-1]
        x2, y2 = traj2.center_trajectory[-1]
        
        # Simple spatial relationship checks
        if relation in ['on', 'above']:
            return y1 < y2  # obj1 is above obj2 (smaller y)
        elif relation in ['under', 'below']:
            return y1 > y2  # obj1 is below obj2
        elif relation in ['left']:
            return x1 < x2  # obj1 is left of obj2
        elif relation in ['right']:
            return x1 > x2  # obj1 is right of obj2
        elif relation in ['near', 'next to', 'close to']:
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            return distance < 100  # Arbitrary threshold
        
        return False
    
    def _calculate_overall_vision_score(self, object_score: float, 
                                      interaction_score: float, 
                                      spatial_score: float) -> float:
        """Calculate overall vision verification score."""
        weights = [0.4, 0.4, 0.2]  # object, interaction, spatial
        scores = [object_score, interaction_score, spatial_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def _create_verification_details(self, text_components: TaskComponents,
                                   detected_objects: List[DetectedObject],
                                   interaction_events: List[InteractionEvent],
                                   object_trajectories: List[ObjectTrajectory]) -> Dict[str, Any]:
        """Create detailed verification information."""
        return {
            'mentioned_objects': [obj.name for obj in text_components.objects],
            'detected_objects': list(set(obj.name for obj in detected_objects)),
            'mentioned_actions': [action.verb for action in text_components.actions],
            'detected_interactions': [event.interaction_type for event in interaction_events],
            'object_count': len(set(obj.name for obj in detected_objects)),
            'interaction_count': len(interaction_events),
            'trajectory_count': len(object_trajectories)
        }
    
    def _create_empty_result(self) -> VerificationResult:
        """Create empty result when vision analysis fails."""
        return VerificationResult(
            detected_objects=[],
            object_trajectories=[],
            interaction_events=[],
            object_verification_score=0.0,
            interaction_verification_score=0.0,
            spatial_verification_score=0.0,
            overall_vision_score=0.0,
            verification_details={}
        ) 