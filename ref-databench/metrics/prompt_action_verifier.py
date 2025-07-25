"""
Prompt-Action Verification Agent

Main agent that integrates all components to provide comprehensive verification
of whether task prompts correspond to actual robot actions performed.

This agent combines:
- Action primitive detection from trajectories  
- NLP extraction from prompts
- Semantic mapping between text and robot actions
- Temporal alignment verification
- Computer vision verification of object interactions
- Task completion detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .action_primitives import ActionPrimitiveDetector, ActionSegment
from .nlp_extractor import NLPExtractor, TaskComponents
from .semantic_mapper import SemanticMapper, MappingResult
from .temporal_aligner import TemporalAligner, AlignmentResult
from .vision_verifier import VisionVerifier, VerificationResult
from .completion_detector import CompletionDetector, CompletionResult, CompletionStatus

logger = logging.getLogger(__name__)

@dataclass
class VerificationScore:
    """Comprehensive verification scores."""
    action_primitive_detection: float
    text_extraction_quality: float
    semantic_mapping_accuracy: float
    temporal_alignment: float
    object_interaction_verification: float
    task_completion_verification: float
    overall_verification_score: float

@dataclass
class PromptActionVerificationResult:
    """Complete result of prompt-action verification."""
    # Input data
    prompt_text: str
    robot_type: str
    
    # Component results
    text_components: TaskComponents
    detected_primitives: List[ActionSegment]
    semantic_mapping: MappingResult
    temporal_alignment: AlignmentResult
    vision_verification: VerificationResult
    completion_result: CompletionResult
    
    # Verification scores
    verification_scores: VerificationScore
    
    # Summary and recommendations
    verification_summary: Dict[str, Any]
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]

class PromptActionVerifier:
    """Main agent for verifying prompt-action correspondence."""
    
    def __init__(self, robot_type: str = "generic_6dof", 
                 yolo_model: str = "yolov8n.pt",
                 spacy_model: str = "en_core_web_sm"):
        """
        Initialize the prompt-action verification agent.
        
        Args:
            robot_type: Type of robot for specialized analysis
            yolo_model: YOLO model path for object detection
            spacy_model: SpaCy model for NLP processing
        """
        self.robot_type = robot_type
        
        # Initialize component modules
        try:
            self.primitive_detector = ActionPrimitiveDetector(robot_type)
            self.nlp_extractor = NLPExtractor(spacy_model)
            self.semantic_mapper = SemanticMapper()
            self.temporal_aligner = TemporalAligner()
            self.vision_verifier = VisionVerifier(yolo_model)
            self.completion_detector = CompletionDetector()
            
            logger.info("Prompt-Action Verification Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize verification agent: {e}")
            raise
        
        self._setup_verification_thresholds()
    
    def _setup_verification_thresholds(self):
        """Setup thresholds for verification scoring."""
        self.thresholds = {
            'primitive_detection_min_confidence': 0.5,
            'text_extraction_min_confidence': 0.6,
            'semantic_mapping_min_coverage': 0.7,
            'temporal_alignment_min_score': 0.6,
            'vision_verification_min_score': 0.5,
            'completion_detection_min_score': 0.7,
            'overall_verification_threshold': 0.6
        }
        
        self.weights = {
            'action_primitive_detection': 0.15,
            'text_extraction_quality': 0.15,
            'semantic_mapping_accuracy': 0.25,
            'temporal_alignment': 0.20,
            'object_interaction_verification': 0.15,
            'task_completion_verification': 0.10
        }
    
    def verify_prompt_action_correspondence(self, prompt: str, 
                                          actions: np.ndarray,
                                          states: Optional[np.ndarray] = None,
                                          timestamps: Optional[np.ndarray] = None,
                                          video_path: Optional[str] = None) -> PromptActionVerificationResult:
        """
        Perform comprehensive verification of prompt-action correspondence.
        
        Args:
            prompt: Task description text
            actions: Robot action trajectory data (T, DOF)
            states: Optional robot state data (T, DOF)
            timestamps: Optional timestamps for trajectory
            video_path: Optional path to video file
            
        Returns:
            PromptActionVerificationResult with comprehensive analysis
        """
        logger.info(f"Starting prompt-action verification for: '{prompt[:50]}...'")
        
        try:
            # Step 1: Extract components from text
            logger.info("Step 1: Extracting components from text prompt")
            text_components = self.nlp_extractor.extract_components(prompt)
            
            # Step 2: Detect action primitives from robot trajectory
            logger.info("Step 2: Detecting action primitives from robot trajectory")
            detected_primitives = self.primitive_detector.detect_primitives(
                actions, states, timestamps
            )
            
            # Step 3: Map semantic actions to robot primitives
            logger.info("Step 3: Mapping semantic actions to robot primitives")
            semantic_mapping = self.semantic_mapper.map_actions(
                text_components, detected_primitives
            )
            
            # Step 4: Analyze temporal alignment
            logger.info("Step 4: Analyzing temporal alignment")
            temporal_alignment = self.temporal_aligner.analyze_temporal_alignment(
                text_components, detected_primitives, semantic_mapping.mappings
            )
            
            # Step 5: Verify object interactions using computer vision
            logger.info("Step 5: Verifying object interactions with computer vision")
            if video_path:
                vision_verification = self.vision_verifier.verify_prompt_actions(
                    text_components, detected_primitives, semantic_mapping.mappings, video_path
                )
            else:
                logger.warning("No video path provided, skipping vision verification")
                vision_verification = self._create_empty_vision_result()
            
            # Step 6: Detect task completion
            logger.info("Step 6: Detecting task completion")
            completion_result = self.completion_detector.detect_completion(
                text_components, detected_primitives, semantic_mapping,
                vision_verification, temporal_alignment
            )
            
            # Step 7: Calculate verification scores
            logger.info("Step 7: Calculating verification scores")
            verification_scores = self._calculate_verification_scores(
                text_components, detected_primitives, semantic_mapping,
                temporal_alignment, vision_verification, completion_result
            )
            
            # Step 8: Generate summary and recommendations
            logger.info("Step 8: Generating summary and recommendations")
            verification_summary = self._generate_verification_summary(
                text_components, semantic_mapping, temporal_alignment,
                vision_verification, completion_result, verification_scores
            )
            
            issues_found = self._identify_issues(
                semantic_mapping, temporal_alignment, vision_verification,
                completion_result, verification_scores
            )
            
            recommendations = self._generate_recommendations(
                issues_found, verification_scores
            )
            
            # Create comprehensive result
            result = PromptActionVerificationResult(
                prompt_text=prompt,
                robot_type=self.robot_type,
                text_components=text_components,
                detected_primitives=detected_primitives,
                semantic_mapping=semantic_mapping,
                temporal_alignment=temporal_alignment,
                vision_verification=vision_verification,
                completion_result=completion_result,
                verification_scores=verification_scores,
                verification_summary=verification_summary,
                issues_found=issues_found,
                recommendations=recommendations
            )
            
            logger.info(f"Verification completed. Overall score: {verification_scores.overall_verification_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            raise
    
    def verify_dataset_episode(self, episode_data: Dict[str, Any], 
                             data_path: Path) -> PromptActionVerificationResult:
        """
        Verify a single episode from a robotics dataset.
        
        Args:
            episode_data: Episode data dictionary
            data_path: Path to dataset
            
        Returns:
            PromptActionVerificationResult for the episode
        """
        # Extract prompt from episode
        prompt = episode_data.get('prompt', episode_data.get('task', ''))
        
        # Load action/state data
        actions, states, timestamps = self._load_trajectory_data(episode_data, data_path)
        
        # Get video path
        video_path = self._get_video_path(episode_data, data_path)
        
        return self.verify_prompt_action_correspondence(
            prompt, actions, states, timestamps, video_path
        )
    
    def _load_trajectory_data(self, episode_data: Dict[str, Any], 
                            data_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Load trajectory data from episode."""
        try:
            # First check if episode data already contains action arrays (HF datasets)
            if 'actions' in episode_data:
                actions = episode_data['actions']
                states = episode_data.get('observations', None)
                
                # Convert to numpy arrays if needed
                if not isinstance(actions, np.ndarray):
                    actions = np.array(actions)
                if states is not None and not isinstance(states, np.ndarray):
                    states = np.array(states)
                
                # Generate timestamps if not provided
                timestamps = None
                if len(actions) > 0:
                    timestamps = np.arange(len(actions)) / 30.0  # Assume 30Hz
                
                return actions, states, timestamps
            
            # Load parquet data file (local datasets)
            data_file_path = episode_data.get('data_path')
            if data_file_path and not data_file_path.startswith('hf_episode_'):
                if data_path is None:
                    raise ValueError("data_path is required for local datasets")
                
                if not Path(data_file_path).is_absolute():
                    full_data_path = data_path / data_file_path
                else:
                    full_data_path = Path(data_file_path)
                
                if full_data_path.exists():
                    df = pd.read_parquet(str(full_data_path))
                    
                    # Extract actions
                    if 'action' in df.columns:
                        actions = np.stack(df['action'].values)
                    else:
                        # Fallback: look for individual action columns
                        action_cols = [col for col in df.columns if col.startswith('action')]
                        if action_cols:
                            actions = df[action_cols].values
                        else:
                            raise ValueError("No action data found")
                    
                    # Extract states
                    states = None
                    if 'observation.state' in df.columns:
                        states = np.stack(df['observation.state'].values)
                    
                    # Extract timestamps
                    timestamps = None
                    if 'timestamp' in df.columns:
                        timestamps = df['timestamp'].values
                    elif len(actions) > 0:
                        timestamps = np.arange(len(actions)) / 30.0  # Assume 30Hz
                    
                    return actions, states, timestamps
                else:
                    raise FileNotFoundError(f"Data file not found: {full_data_path}")
            
            # For HF datasets with data_path starting with 'hf_episode_', return empty arrays
            # This indicates the episode data should be accessed directly from episode_data
            if data_file_path and data_file_path.startswith('hf_episode_'):
                logger.warning(f"HF episode data path {data_file_path} requires direct action data passing")
                return np.array([]), None, None
            
            raise ValueError("No valid trajectory data found")
            
        except Exception as e:
            logger.error(f"Error loading trajectory data: {e}")
            return np.array([]), None, None
    
    def _get_video_path(self, episode_data: Dict[str, Any], data_path: Path) -> Optional[str]:
        """Get video path from episode data."""
        if 'video' in episode_data and episode_data['video']:
            video_rel_path = episode_data['video']
            if not Path(video_rel_path).is_absolute():
                video_path = data_path / video_rel_path
            else:
                video_path = Path(video_rel_path)
            
            if video_path.exists():
                return str(video_path)
        
        # Check for multiple videos
        if 'videos' in episode_data and isinstance(episode_data['videos'], dict):
            for view_name, video_rel_path in episode_data['videos'].items():
                if not Path(video_rel_path).is_absolute():
                    video_path = data_path / video_rel_path
                else:
                    video_path = Path(video_rel_path)
                
                if video_path.exists():
                    return str(video_path)  # Return first available video
        
        return None
    
    def _calculate_verification_scores(self, text_components: TaskComponents,
                                     detected_primitives: List[ActionSegment],
                                     semantic_mapping: MappingResult,
                                     temporal_alignment: AlignmentResult,
                                     vision_verification: VerificationResult,
                                     completion_result: CompletionResult) -> VerificationScore:
        """Calculate comprehensive verification scores."""
        
        # Action primitive detection score
        primitive_score = 0.0
        if detected_primitives:
            avg_confidence = np.mean([seg.confidence for seg in detected_primitives])
            primitive_score = min(1.0, avg_confidence)
        
        # Text extraction quality score
        text_score = text_components.confidence
        
        # Semantic mapping accuracy score
        mapping_score = semantic_mapping.overall_score
        
        # Temporal alignment score
        temporal_score = temporal_alignment.overall_temporal_score
        
        # Object interaction verification score
        vision_score = vision_verification.overall_vision_score
        
        # Task completion verification score
        completion_score = completion_result.overall_completion_score
        
        # Calculate overall score using weights
        overall_score = (
            self.weights['action_primitive_detection'] * primitive_score +
            self.weights['text_extraction_quality'] * text_score +
            self.weights['semantic_mapping_accuracy'] * mapping_score +
            self.weights['temporal_alignment'] * temporal_score +
            self.weights['object_interaction_verification'] * vision_score +
            self.weights['task_completion_verification'] * completion_score
        )
        
        return VerificationScore(
            action_primitive_detection=primitive_score,
            text_extraction_quality=text_score,
            semantic_mapping_accuracy=mapping_score,
            temporal_alignment=temporal_score,
            object_interaction_verification=vision_score,
            task_completion_verification=completion_score,
            overall_verification_score=overall_score
        )
    
    def _generate_verification_summary(self, text_components: TaskComponents,
                                     semantic_mapping: MappingResult,
                                     temporal_alignment: AlignmentResult,
                                     vision_verification: VerificationResult,
                                     completion_result: CompletionResult,
                                     verification_scores: VerificationScore) -> Dict[str, Any]:
        """Generate comprehensive verification summary."""
        return {
            'prompt_analysis': {
                'actions_mentioned': len(text_components.actions),
                'objects_mentioned': len(text_components.objects),
                'spatial_relations': len(text_components.spatial_relations),
                'temporal_indicators': len(text_components.temporal_indicators),
                'text_confidence': text_components.confidence
            },
            'action_mapping': {
                'total_mappings': len(semantic_mapping.mappings),
                'coverage_score': semantic_mapping.coverage_score,
                'unmapped_text_actions': len(semantic_mapping.unmapped_text_actions),
                'unmapped_robot_primitives': len(semantic_mapping.unmapped_primitives)
            },
            'temporal_analysis': {
                'sequence_order_score': temporal_alignment.sequence_order_score,
                'timing_precision_score': temporal_alignment.timing_precision_score,
                'constraint_violations': len(temporal_alignment.constraint_violations)
            },
            'vision_analysis': {
                'objects_detected': len(set(obj.name for obj in vision_verification.detected_objects)),
                'interactions_detected': len(vision_verification.interaction_events),
                'object_verification_score': vision_verification.object_verification_score,
                'interaction_verification_score': vision_verification.interaction_verification_score
            },
            'completion_analysis': {
                'completion_status': completion_result.completion_status.value,
                'completion_confidence': completion_result.completion_confidence,
                'evidence_count': len(completion_result.evidence),
                'goal_criteria_met': len(completion_result.goal_criteria)
            },
            'overall_scores': {
                'action_primitive_detection': verification_scores.action_primitive_detection,
                'text_extraction_quality': verification_scores.text_extraction_quality,
                'semantic_mapping_accuracy': verification_scores.semantic_mapping_accuracy,
                'temporal_alignment': verification_scores.temporal_alignment,
                'object_interaction_verification': verification_scores.object_interaction_verification,
                'task_completion_verification': verification_scores.task_completion_verification,
                'overall_verification_score': verification_scores.overall_verification_score
            }
        }
    
    def _identify_issues(self, semantic_mapping: MappingResult,
                        temporal_alignment: AlignmentResult,
                        vision_verification: VerificationResult,
                        completion_result: CompletionResult,
                        verification_scores: VerificationScore) -> List[Dict[str, Any]]:
        """Identify issues in prompt-action correspondence."""
        issues = []
        
        # Check for low semantic mapping coverage
        if semantic_mapping.coverage_score < self.thresholds['semantic_mapping_min_coverage']:
            issues.append({
                'type': 'semantic_mapping',
                'severity': 'high',
                'description': f"Low action mapping coverage ({semantic_mapping.coverage_score:.2f})",
                'details': f"{len(semantic_mapping.unmapped_text_actions)} text actions unmapped"
            })
        
        # Check for temporal alignment issues
        if temporal_alignment.sequence_order_score < self.thresholds['temporal_alignment_min_score']:
            issues.append({
                'type': 'temporal_alignment',
                'severity': 'medium',
                'description': f"Poor temporal sequence alignment ({temporal_alignment.sequence_order_score:.2f})",
                'details': "Actions may not follow expected order"
            })
        
        # Check for temporal constraint violations
        if temporal_alignment.constraint_violations:
            issues.append({
                'type': 'temporal_constraints',
                'severity': 'medium',
                'description': f"{len(temporal_alignment.constraint_violations)} temporal constraint violations",
                'details': [v['violation_type'] for v in temporal_alignment.constraint_violations]
            })
        
        # Check for vision verification issues
        if vision_verification.object_verification_score < self.thresholds['vision_verification_min_score']:
            issues.append({
                'type': 'object_verification',
                'severity': 'medium',
                'description': f"Low object verification score ({vision_verification.object_verification_score:.2f})",
                'details': "Mentioned objects may not be present in video"
            })
        
        # Check for task completion issues
        if (completion_result.completion_status == CompletionStatus.FAILED or
            completion_result.overall_completion_score < self.thresholds['completion_detection_min_score']):
            issues.append({
                'type': 'task_completion',
                'severity': 'high',
                'description': f"Task appears incomplete or failed ({completion_result.completion_status.value})",
                'details': f"Completion score: {completion_result.overall_completion_score:.2f}"
            })
        
        # Check overall verification score
        if verification_scores.overall_verification_score < self.thresholds['overall_verification_threshold']:
            issues.append({
                'type': 'overall_verification',
                'severity': 'high',
                'description': f"Low overall verification score ({verification_scores.overall_verification_score:.2f})",
                'details': "Prompt may not accurately describe robot actions"
            })
        
        return issues
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]],
                                verification_scores: VerificationScore) -> List[str]:
        """Generate recommendations based on identified issues."""
        recommendations = []
        
        for issue in issues:
            if issue['type'] == 'semantic_mapping':
                recommendations.append(
                    "Improve prompt clarity by using more specific action verbs that match robot capabilities"
                )
            elif issue['type'] == 'temporal_alignment':
                recommendations.append(
                    "Add temporal indicators (first, then, after) to clarify action sequence"
                )
            elif issue['type'] == 'temporal_constraints':
                recommendations.append(
                    "Review action timing - some actions may be happening out of order"
                )
            elif issue['type'] == 'object_verification':
                recommendations.append(
                    "Ensure mentioned objects are clearly visible and properly labeled in prompts"
                )
            elif issue['type'] == 'task_completion':
                recommendations.append(
                    "Verify that the task was actually completed successfully before labeling"
                )
            elif issue['type'] == 'overall_verification':
                recommendations.append(
                    "Consider rewriting prompt to better match actual robot actions performed"
                )
        
        # Add general recommendations based on scores
        if verification_scores.text_extraction_quality < 0.7:
            recommendations.append("Use clearer, more descriptive language in prompts")
        
        if verification_scores.semantic_mapping_accuracy < 0.6:
            recommendations.append("Use action verbs that closely match robot primitive capabilities")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _create_empty_vision_result(self) -> VerificationResult:
        """Create empty vision result when video is not available."""
        from .vision_verifier import VerificationResult
        return VerificationResult(
            detected_objects=[],
            object_trajectories=[],
            interaction_events=[],
            object_verification_score=0.5,  # Neutral score
            interaction_verification_score=0.5,
            spatial_verification_score=0.5,
            overall_vision_score=0.5,
            verification_details={}
        )
    
    def get_verification_report(self, result: PromptActionVerificationResult) -> str:
        """Generate a human-readable verification report."""
        report = []
        
        report.append("=" * 60)
        report.append("PROMPT-ACTION VERIFICATION REPORT")
        report.append("=" * 60)
        report.append(f"Prompt: {result.prompt_text}")
        report.append(f"Robot Type: {result.robot_type}")
        report.append("")
        
        # Overall Score
        overall_score = result.verification_scores.overall_verification_score
        status = "✅ PASS" if overall_score >= 0.6 else "❌ FAIL"
        report.append(f"Overall Verification Score: {overall_score:.3f} {status}")
        report.append("")
        
        # Component Scores
        report.append("Component Scores:")
        scores = result.verification_scores
        report.append(f"  Action Primitive Detection: {scores.action_primitive_detection:.3f}")
        report.append(f"  Text Extraction Quality: {scores.text_extraction_quality:.3f}")
        report.append(f"  Semantic Mapping Accuracy: {scores.semantic_mapping_accuracy:.3f}")
        report.append(f"  Temporal Alignment: {scores.temporal_alignment:.3f}")
        report.append(f"  Object Interaction Verification: {scores.object_interaction_verification:.3f}")
        report.append(f"  Task Completion Verification: {scores.task_completion_verification:.3f}")
        report.append("")
        
        # Issues Found
        if result.issues_found:
            report.append("Issues Found:")
            for i, issue in enumerate(result.issues_found, 1):
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue['severity'], "⚪")
                report.append(f"  {i}. {severity_icon} {issue['description']}")
                if isinstance(issue['details'], list):
                    for detail in issue['details']:
                        report.append(f"     - {detail}")
                else:
                    report.append(f"     - {issue['details']}")
            report.append("")
        
        # Recommendations
        if result.recommendations:
            report.append("Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        # Summary Statistics
        summary = result.verification_summary
        report.append("Summary Statistics:")
        report.append(f"  Actions Mentioned: {summary['prompt_analysis']['actions_mentioned']}")
        report.append(f"  Objects Mentioned: {summary['prompt_analysis']['objects_mentioned']}")
        report.append(f"  Action Mappings: {summary['action_mapping']['total_mappings']}")
        report.append(f"  Objects Detected: {summary['vision_analysis']['objects_detected']}")
        report.append(f"  Completion Status: {summary['completion_analysis']['completion_status']}")
        
        return "\n".join(report) 