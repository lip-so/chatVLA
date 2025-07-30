"""
Task Completion Detector

Determines if the task described in a prompt was actually completed successfully
by analyzing the final state, goal achievement, and success indicators.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import cv2

from .action_primitives import ActionSegment, ActionPrimitive
from .nlp_extractor import TaskComponents, ExtractedAction
from .vision_verifier import VerificationResult, ObjectTrajectory, InteractionEvent
from .temporal_aligner import AlignmentResult
from .semantic_mapper import MappingResult

logger = logging.getLogger(__name__)

class CompletionStatus(Enum):
    """Task completion status."""
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class GoalCriteria:
    """Represents a goal criteria extracted from task description."""
    criteria_type: str  # 'object_location', 'state_change', 'interaction_completion'
    description: str
    target_object: Optional[str]
    target_location: Optional[str]
    expected_state: Optional[str]
    success_indicators: List[str]
    confidence: float

@dataclass
class CompletionEvidence:
    """Evidence for task completion."""
    evidence_type: str  # 'visual', 'temporal', 'action_sequence'
    description: str
    confidence: float
    supporting_data: Dict[str, Any]

@dataclass
class CompletionResult:
    """Result of task completion analysis."""
    completion_status: CompletionStatus
    completion_confidence: float
    goal_criteria: List[GoalCriteria]
    evidence: List[CompletionEvidence]
    success_indicators: Dict[str, float]
    failure_indicators: Dict[str, float]
    overall_completion_score: float

class CompletionDetector:
    """Detects task completion based on multi-modal analysis."""
    
    def __init__(self):
        """Initialize the completion detector."""
        self._setup_completion_patterns()
        self._setup_success_indicators()
    
    def _setup_completion_patterns(self):
        """Setup patterns for recognizing task completion."""
        self.completion_verbs = {
            'pick': {
                'goal': 'object_grasped_and_lifted',
                'indicators': ['object_picked_up', 'object_in_hand', 'upward_motion'],
                'final_state': 'object_elevated'
            },
            'place': {
                'goal': 'object_positioned_at_target',
                'indicators': ['object_released', 'object_stationary', 'target_reached'],
                'final_state': 'object_placed'
            },
            'move': {
                'goal': 'position_changed',
                'indicators': ['motion_completed', 'target_reached'],
                'final_state': 'new_position'
            },
            'stack': {
                'goal': 'objects_stacked',
                'indicators': ['vertical_arrangement', 'contact_between_objects'],
                'final_state': 'stacked_configuration'
            },
            'insert': {
                'goal': 'object_inside_container',
                'indicators': ['containment_achieved', 'object_disappeared'],
                'final_state': 'inserted'
            },
            'remove': {
                'goal': 'object_extracted',
                'indicators': ['object_separated', 'extraction_motion'],
                'final_state': 'removed'
            }
        }
        
        self.spatial_goals = {
            'in': 'containment',
            'on': 'surface_contact',
            'under': 'underneath_position',
            'next to': 'proximity',
            'behind': 'occlusion_or_depth',
            'in front of': 'forward_position'
        }
    
    def _setup_success_indicators(self):
        """Setup indicators for success and failure."""
        self.success_indicators = {
            'object_motion_stopped': 'Object becomes stationary after action',
            'target_location_reached': 'Object reaches intended location',
            'interaction_completed': 'Robot-object interaction finished',
            'stable_final_state': 'Final configuration is stable',
            'action_sequence_completed': 'All planned actions executed',
            'spatial_relationship_achieved': 'Desired spatial arrangement reached'
        }
        
        self.failure_indicators = {
            'object_dropped': 'Object falls or is dropped unexpectedly',
            'target_missed': 'Object does not reach intended location',
            'incomplete_sequence': 'Action sequence terminated early',
            'unstable_final_state': 'Final configuration is unstable',
            'collision_occurred': 'Unintended collision detected',
            'object_lost': 'Object disappears or moves out of view'
        }
    
    def detect_completion(self, text_components: TaskComponents,
                         robot_segments: List[ActionSegment],
                         semantic_mapping: MappingResult,
                         vision_result: VerificationResult,
                         temporal_result: AlignmentResult,
                         video_frames: Optional[List[np.ndarray]] = None) -> CompletionResult:
        """
        Detect task completion using multi-modal analysis.
        
        Args:
            text_components: Extracted components from text
            robot_segments: Detected robot action segments
            semantic_mapping: Semantic mapping results
            vision_result: Vision verification results
            temporal_result: Temporal alignment results
            video_frames: Optional video frames for analysis
            
        Returns:
            CompletionResult containing completion analysis
        """
        # Extract goal criteria from text
        goal_criteria = self._extract_goal_criteria(text_components)
        
        # Collect evidence from different modalities
        evidence = []
        
        # Visual evidence
        visual_evidence = self._analyze_visual_evidence(
            goal_criteria, vision_result, video_frames
        )
        evidence.extend(visual_evidence)
        
        # Action sequence evidence
        action_evidence = self._analyze_action_sequence_evidence(
            goal_criteria, robot_segments, semantic_mapping
        )
        evidence.extend(action_evidence)
        
        # Temporal evidence
        temporal_evidence = self._analyze_temporal_evidence(
            goal_criteria, temporal_result
        )
        evidence.extend(temporal_evidence)
        
        # Calculate success and failure indicators
        success_indicators = self._calculate_success_indicators(evidence, goal_criteria)
        failure_indicators = self._calculate_failure_indicators(evidence, goal_criteria)
        
        # Determine completion status
        completion_status, completion_confidence = self._determine_completion_status(
            success_indicators, failure_indicators, evidence
        )
        
        # Calculate overall completion score
        overall_score = self._calculate_overall_completion_score(
            completion_status, completion_confidence, success_indicators, failure_indicators
        )
        
        return CompletionResult(
            completion_status=completion_status,
            completion_confidence=completion_confidence,
            goal_criteria=goal_criteria,
            evidence=evidence,
            success_indicators=success_indicators,
            failure_indicators=failure_indicators,
            overall_completion_score=overall_score
        )
    
    def _extract_goal_criteria(self, text_components: TaskComponents) -> List[GoalCriteria]:
        """Extract goal criteria from text components."""
        criteria = []
        
        for action in text_components.actions:
            verb = action.verb
            
            if verb in self.completion_verbs:
                pattern = self.completion_verbs[verb]
                
                # Create goal criteria for this action
                target_object = action.objects[0] if action.objects else None
                
                goal = GoalCriteria(
                    criteria_type='action_completion',
                    description=f"Complete {verb} action" + (f" with {target_object}" if target_object else ""),
                    target_object=target_object,
                    target_location=action.location,
                    expected_state=pattern['final_state'],
                    success_indicators=pattern['indicators'],
                    confidence=action.confidence
                )
                criteria.append(goal)
        
        # Extract spatial goal criteria
        for relation in text_components.spatial_relations:
            spatial_goal = self.spatial_goals.get(relation.relation)
            if spatial_goal:
                goal = GoalCriteria(
                    criteria_type='spatial_relationship',
                    description=f"{relation.object1} {relation.relation} {relation.object2}",
                    target_object=relation.object1,
                    target_location=relation.object2,
                    expected_state=spatial_goal,
                    success_indicators=[f"{relation.relation}_achieved"],
                    confidence=relation.confidence
                )
                criteria.append(goal)
        
        # Extract completion criteria from success criteria in text
        for criteria_text in text_components.success_criteria:
            goal = GoalCriteria(
                criteria_type='explicit_goal',
                description=criteria_text,
                target_object=None,
                target_location=None,
                expected_state='goal_achieved',
                success_indicators=['explicit_goal_met'],
                confidence=0.9
            )
            criteria.append(goal)
        
        return criteria
    
    def _analyze_visual_evidence(self, goal_criteria: List[GoalCriteria],
                               vision_result: VerificationResult,
                               video_frames: Optional[List[np.ndarray]]) -> List[CompletionEvidence]:
        """Analyze visual evidence for task completion."""
        evidence = []
        
        # Check object trajectories for completion patterns
        for trajectory in vision_result.object_trajectories:
            # Check if object motion stopped (indicating completion)
            if 'static' in trajectory.motion_patterns:
                evidence.append(CompletionEvidence(
                    evidence_type='visual',
                    description=f"{trajectory.object_name} became stationary",
                    confidence=0.7,
                    supporting_data={'trajectory': trajectory.object_name, 'pattern': 'static'}
                ))
            
            # Check for pickup pattern
            if 'picked_up' in trajectory.motion_patterns:
                evidence.append(CompletionEvidence(
                    evidence_type='visual',
                    description=f"{trajectory.object_name} was picked up",
                    confidence=0.8,
                    supporting_data={'trajectory': trajectory.object_name, 'pattern': 'picked_up'}
                ))
        
        # Check interaction events for completion
        for interaction in vision_result.interaction_events:
            if interaction.interaction_type == 'grasp':
                evidence.append(CompletionEvidence(
                    evidence_type='visual',
                    description=f"Grasping interaction with {interaction.object_name}",
                    confidence=interaction.confidence,
                    supporting_data={'interaction': interaction.interaction_type, 'object': interaction.object_name}
                ))
            
            elif interaction.interaction_type == 'release':
                evidence.append(CompletionEvidence(
                    evidence_type='visual',
                    description=f"Release interaction with {interaction.object_name}",
                    confidence=interaction.confidence,
                    supporting_data={'interaction': interaction.interaction_type, 'object': interaction.object_name}
                ))
        
        # Analyze final state if video frames available
        if video_frames and len(video_frames) > 0:
            final_frame_evidence = self._analyze_final_frame(video_frames[-1], goal_criteria)
            evidence.extend(final_frame_evidence)
        
        return evidence
    
    def _analyze_action_sequence_evidence(self, goal_criteria: List[GoalCriteria],
                                        robot_segments: List[ActionSegment],
                                        semantic_mapping: MappingResult) -> List[CompletionEvidence]:
        """Analyze action sequence evidence for completion."""
        evidence = []
        
        # Check if all planned actions were executed
        mapped_actions = {mapping.text_action for mapping in semantic_mapping.mappings}
        
        if semantic_mapping.coverage_score > 0.8:
            evidence.append(CompletionEvidence(
                evidence_type='action_sequence',
                description="Most planned actions were executed",
                confidence=semantic_mapping.coverage_score,
                supporting_data={'coverage': semantic_mapping.coverage_score}
            ))
        
        # Check for complete action sequences
        for criteria in goal_criteria:
            if criteria.criteria_type == 'action_completion':
                # Look for complete action patterns
                if criteria.target_object and any(criteria.target_object in mapping.text_action 
                                                for mapping in semantic_mapping.mappings):
                    evidence.append(CompletionEvidence(
                        evidence_type='action_sequence',
                        description=f"Action sequence for {criteria.target_object} completed",
                        confidence=0.8,
                        supporting_data={'criteria': criteria.description}
                    ))
        
        # Check final robot state
        if robot_segments:
            final_segment = robot_segments[-1]
            if final_segment.primitive in [ActionPrimitive.PLACE, ActionPrimitive.RELEASE]:
                evidence.append(CompletionEvidence(
                    evidence_type='action_sequence',
                    description="Task ended with placement/release action",
                    confidence=0.7,
                    supporting_data={'final_action': final_segment.primitive.value}
                ))
        
        return evidence
    
    def _analyze_temporal_evidence(self, goal_criteria: List[GoalCriteria],
                                 temporal_result: AlignmentResult) -> List[CompletionEvidence]:
        """Analyze temporal evidence for completion."""
        evidence = []
        
        # Check if actions followed proper sequence
        if temporal_result.sequence_order_score > 0.8:
            evidence.append(CompletionEvidence(
                evidence_type='temporal',
                description="Actions followed expected temporal sequence",
                confidence=temporal_result.sequence_order_score,
                supporting_data={'sequence_score': temporal_result.sequence_order_score}
            ))
        
        # Check for timing consistency
        if temporal_result.timing_precision_score > 0.7:
            evidence.append(CompletionEvidence(
                evidence_type='temporal',
                description="Action timing was consistent with expectations",
                confidence=temporal_result.timing_precision_score,
                supporting_data={'timing_score': temporal_result.timing_precision_score}
            ))
        
        # Check for constraint violations (negative evidence)
        if temporal_result.constraint_violations:
            evidence.append(CompletionEvidence(
                evidence_type='temporal',
                description=f"Temporal constraint violations detected ({len(temporal_result.constraint_violations)})",
                confidence=0.8,
                supporting_data={'violations': len(temporal_result.constraint_violations)}
            ))
        
        return evidence
    
    def _analyze_final_frame(self, final_frame: np.ndarray, 
                           goal_criteria: List[GoalCriteria]) -> List[CompletionEvidence]:
        """Analyze final video frame for completion indicators."""
        evidence = []
        
        # Simple visual analysis of final state
        # This is a placeholder for more sophisticated analysis
        
        # Check for stable configuration (low motion)
        if final_frame is not None:
            evidence.append(CompletionEvidence(
                evidence_type='visual',
                description="Final frame available for analysis",
                confidence=0.5,
                supporting_data={'frame_available': True}
            ))
        
        return evidence
    
    def _calculate_success_indicators(self, evidence: List[CompletionEvidence],
                                    goal_criteria: List[GoalCriteria]) -> Dict[str, float]:
        """Calculate success indicator scores."""
        indicators = {}
        
        # Object motion stopped
        motion_evidence = [e for e in evidence if 'stationary' in e.description.lower()]
        indicators['object_motion_stopped'] = min(1.0, len(motion_evidence) * 0.3)
        
        # Interaction completed
        interaction_evidence = [e for e in evidence if e.evidence_type == 'visual' and 
                              any(word in e.description.lower() for word in ['grasp', 'release', 'pick'])]
        indicators['interaction_completed'] = min(1.0, len(interaction_evidence) * 0.4)
        
        # Action sequence completed
        sequence_evidence = [e for e in evidence if e.evidence_type == 'action_sequence']
        if sequence_evidence:
            indicators['action_sequence_completed'] = np.mean([e.confidence for e in sequence_evidence])
        else:
            indicators['action_sequence_completed'] = 0.0
        
        # Temporal consistency
        temporal_evidence = [e for e in evidence if e.evidence_type == 'temporal' and 'consistent' in e.description.lower()]
        if temporal_evidence:
            indicators['stable_final_state'] = np.mean([e.confidence for e in temporal_evidence])
        else:
            indicators['stable_final_state'] = 0.5
        
        return indicators
    
    def _calculate_failure_indicators(self, evidence: List[CompletionEvidence],
                                    goal_criteria: List[GoalCriteria]) -> Dict[str, float]:
        """Calculate failure indicator scores."""
        indicators = {}
        
        # Incomplete sequence
        violation_evidence = [e for e in evidence if 'violation' in e.description.lower()]
        indicators['incomplete_sequence'] = min(1.0, len(violation_evidence) * 0.5)
        
        # Target missed
        indicators['target_missed'] = 0.0  # Would need more sophisticated analysis
        
        # Object dropped
        indicators['object_dropped'] = 0.0  # Would need motion analysis
        
        # Unstable final state
        instability_evidence = [e for e in evidence if 'unstable' in e.description.lower()]
        indicators['unstable_final_state'] = min(1.0, len(instability_evidence) * 0.5)
        
        return indicators
    
    def _determine_completion_status(self, success_indicators: Dict[str, float],
                                   failure_indicators: Dict[str, float],
                                   evidence: List[CompletionEvidence]) -> Tuple[CompletionStatus, float]:
        """Determine overall completion status and confidence."""
        
        success_score = np.mean(list(success_indicators.values()))
        failure_score = np.mean(list(failure_indicators.values()))
        
        # Calculate overall confidence
        evidence_confidence = np.mean([e.confidence for e in evidence]) if evidence else 0.0
        
        # Determine status
        if success_score > 0.7 and failure_score < 0.3:
            return CompletionStatus.COMPLETED, min(success_score, evidence_confidence)
        elif success_score > 0.4 and failure_score < 0.6:
            return CompletionStatus.PARTIALLY_COMPLETED, evidence_confidence * 0.7
        elif failure_score > 0.6:
            return CompletionStatus.FAILED, failure_score
        else:
            return CompletionStatus.UNKNOWN, evidence_confidence * 0.5
    
    def _calculate_overall_completion_score(self, status: CompletionStatus, confidence: float,
                                          success_indicators: Dict[str, float],
                                          failure_indicators: Dict[str, float]) -> float:
        """Calculate overall completion score."""
        
        if status == CompletionStatus.COMPLETED:
            base_score = 1.0
        elif status == CompletionStatus.PARTIALLY_COMPLETED:
            base_score = 0.6
        elif status == CompletionStatus.FAILED:
            base_score = 0.2
        else:  # UNKNOWN
            base_score = 0.5
        
        # Adjust by confidence
        score = base_score * confidence
        
        # Apply success/failure indicator modifiers
        success_modifier = np.mean(list(success_indicators.values())) * 0.2
        failure_modifier = np.mean(list(failure_indicators.values())) * -0.2
        
        final_score = score + success_modifier + failure_modifier
        
        return max(0.0, min(1.0, final_score))
    
    def get_completion_summary(self, result: CompletionResult) -> Dict[str, Any]:
        """Get summary of completion detection results."""
        return {
            'completion_status': result.completion_status.value,
            'completion_confidence': result.completion_confidence,
            'overall_score': result.overall_completion_score,
            'goal_criteria_count': len(result.goal_criteria),
            'evidence_count': len(result.evidence),
            'success_indicators': result.success_indicators,
            'failure_indicators': result.failure_indicators,
            'evidence_summary': [
                {
                    'type': e.evidence_type,
                    'description': e.description,
                    'confidence': e.confidence
                }
                for e in result.evidence
            ]
        } 