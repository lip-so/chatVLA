"""
Semantic Action Mapper

Maps low-level robot action primitives to high-level semantic actions mentioned in prompts.
This module bridges the gap between physical robot movements and natural language descriptions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from enum import Enum
import difflib
from collections import defaultdict

from .action_primitives import ActionPrimitive, ActionSegment
from .nlp_extractor import TaskComponents, ExtractedAction, ActionType

logger = logging.getLogger(__name__)

@dataclass
class SemanticMapping:
    """Represents a mapping between text action and robot primitive."""
    text_action: str
    text_action_type: ActionType
    robot_primitive: ActionPrimitive
    robot_segment: ActionSegment
    confidence: float
    reasoning: str

@dataclass
class MappingResult:
    """Result of semantic mapping analysis."""
    mappings: List[SemanticMapping]
    unmapped_text_actions: List[ExtractedAction]
    unmapped_primitives: List[ActionSegment]
    coverage_score: float
    alignment_score: float
    temporal_consistency: float
    overall_score: float

class SemanticMapper:
    """Maps semantic actions from text to robot action primitives."""
    
    def __init__(self):
        """Initialize the semantic mapper."""
        self._setup_mapping_rules()
        self._setup_similarity_thresholds()
    
    def _setup_mapping_rules(self):
        """Setup rules for mapping text actions to robot primitives."""
        # Direct mappings: text verb -> robot primitive
        self.direct_mappings = {
            # Manipulation verbs
            'pick': [ActionPrimitive.REACH, ActionPrimitive.GRASP, ActionPrimitive.LIFT],
            'pickup': [ActionPrimitive.REACH, ActionPrimitive.GRASP, ActionPrimitive.LIFT],
            'grasp': [ActionPrimitive.GRASP],
            'grab': [ActionPrimitive.GRASP],
            'take': [ActionPrimitive.REACH, ActionPrimitive.GRASP],
            'lift': [ActionPrimitive.LIFT],
            'raise': [ActionPrimitive.LIFT],
            'place': [ActionPrimitive.PLACE],
            'put': [ActionPrimitive.PLACE],
            'set': [ActionPrimitive.PLACE],
            'drop': [ActionPrimitive.RELEASE],
            'release': [ActionPrimitive.RELEASE],
            'let': [ActionPrimitive.RELEASE],
            
            # Movement verbs
            'move': [ActionPrimitive.MOVE, ActionPrimitive.REACH],
            'reach': [ActionPrimitive.REACH],
            'approach': [ActionPrimitive.APPROACH],
            'retreat': [ActionPrimitive.RETREAT],
            'go': [ActionPrimitive.MOVE],
            
            # Interaction verbs
            'push': [ActionPrimitive.PUSH],
            'pull': [ActionPrimitive.PULL],
            'rotate': [ActionPrimitive.ROTATE],
            'turn': [ActionPrimitive.ROTATE],
            
            # Composite actions
            'insert': [ActionPrimitive.REACH, ActionPrimitive.PLACE],
            'remove': [ActionPrimitive.GRASP, ActionPrimitive.LIFT],
            'stack': [ActionPrimitive.LIFT, ActionPrimitive.PLACE],
            'unstack': [ActionPrimitive.GRASP, ActionPrimitive.LIFT]
        }
        
        # Primitive similarities (for fuzzy matching)
        self.primitive_similarities = {
            ActionPrimitive.REACH: [ActionPrimitive.MOVE, ActionPrimitive.APPROACH],
            ActionPrimitive.GRASP: [ActionPrimitive.REACH],
            ActionPrimitive.RELEASE: [ActionPrimitive.PLACE],
            ActionPrimitive.PLACE: [ActionPrimitive.MOVE, ActionPrimitive.RELEASE],
            ActionPrimitive.LIFT: [ActionPrimitive.MOVE, ActionPrimitive.GRASP],
            ActionPrimitive.MOVE: [ActionPrimitive.REACH, ActionPrimitive.APPROACH],
            ActionPrimitive.APPROACH: [ActionPrimitive.REACH, ActionPrimitive.MOVE],
            ActionPrimitive.RETREAT: [ActionPrimitive.MOVE]
        }
        
        # Temporal constraints for action sequences
        self.temporal_constraints = {
            'pick': {
                'sequence': ['reach', 'grasp', 'lift'],
                'max_gap': 3.0,  # seconds
                'required_order': True
            },
            'place': {
                'sequence': ['move', 'release'],
                'max_gap': 2.0,
                'required_order': True
            },
            'stack': {
                'sequence': ['lift', 'place'],
                'max_gap': 5.0,
                'required_order': True
            }
        }
    
    def _setup_similarity_thresholds(self):
        """Setup thresholds for similarity matching."""
        self.direct_match_threshold = 0.9
        self.fuzzy_match_threshold = 0.6
        self.temporal_window = 5.0  # seconds
        self.sequence_tolerance = 2.0  # seconds between sequence elements
    
    def map_actions(self, text_components: TaskComponents, 
                   robot_segments: List[ActionSegment]) -> MappingResult:
        """
        Map text actions to robot primitives.
        
        Args:
            text_components: Extracted components from text
            robot_segments: Detected robot action segments
            
        Returns:
            MappingResult containing mappings and analysis
        """
        mappings = []
        unmapped_text = text_components.actions.copy()
        unmapped_primitives = robot_segments.copy()
        
        # Phase 1: Direct mappings (exact matches)
        mappings.extend(self._find_direct_mappings(
            unmapped_text, unmapped_primitives, mappings
        ))
        
        # Phase 2: Sequence mappings (composite actions)
        mappings.extend(self._find_sequence_mappings(
            unmapped_text, unmapped_primitives, mappings
        ))
        
        # Phase 3: Fuzzy mappings (similar actions)
        mappings.extend(self._find_fuzzy_mappings(
            unmapped_text, unmapped_primitives, mappings
        ))
        
        # Remove mapped items from unmapped lists
        for mapping in mappings:
            if mapping.text_action in [a.verb for a in unmapped_text]:
                unmapped_text = [a for a in unmapped_text if a.verb != mapping.text_action]
            if mapping.robot_segment in unmapped_primitives:
                unmapped_primitives.remove(mapping.robot_segment)
        
        # Calculate scores
        coverage_score = self._calculate_coverage_score(text_components.actions, mappings)
        alignment_score = self._calculate_alignment_score(mappings)
        temporal_consistency = self._calculate_temporal_consistency(
            text_components, mappings, robot_segments
        )
        overall_score = self._calculate_overall_score(
            coverage_score, alignment_score, temporal_consistency
        )
        
        return MappingResult(
            mappings=mappings,
            unmapped_text_actions=unmapped_text,
            unmapped_primitives=unmapped_primitives,
            coverage_score=coverage_score,
            alignment_score=alignment_score,
            temporal_consistency=temporal_consistency,
            overall_score=overall_score
        )
    
    def _find_direct_mappings(self, text_actions: List[ExtractedAction], 
                            robot_segments: List[ActionSegment],
                            existing_mappings: List[SemanticMapping]) -> List[SemanticMapping]:
        """Find direct mappings between text actions and robot primitives."""
        mappings = []
        
        for text_action in text_actions:
            verb = text_action.verb
            if verb in self.direct_mappings:
                expected_primitives = self.direct_mappings[verb]
                
                # Look for matching robot primitives
                for expected_primitive in expected_primitives:
                    matching_segments = [
                        seg for seg in robot_segments 
                        if seg.primitive == expected_primitive and 
                        not self._is_segment_mapped(seg, existing_mappings + mappings)
                    ]
                    
                    if matching_segments:
                        # Choose the best matching segment (highest confidence)
                        best_segment = max(matching_segments, key=lambda x: x.confidence)
                        
                        confidence = min(text_action.confidence, best_segment.confidence)
                        
                        mapping = SemanticMapping(
                            text_action=verb,
                            text_action_type=text_action.action_type,
                            robot_primitive=expected_primitive,
                            robot_segment=best_segment,
                            confidence=confidence,
                            reasoning=f"Direct mapping: '{verb}' -> {expected_primitive.value}"
                        )
                        mappings.append(mapping)
                        break  # Only map to one primitive per text action in direct phase
        
        return mappings
    
    def _find_sequence_mappings(self, text_actions: List[ExtractedAction],
                              robot_segments: List[ActionSegment],
                              existing_mappings: List[SemanticMapping]) -> List[SemanticMapping]:
        """Find mappings for composite actions that require sequences."""
        mappings = []
        
        for text_action in text_actions:
            verb = text_action.verb
            if verb in self.temporal_constraints:
                constraint = self.temporal_constraints[verb]
                expected_sequence = constraint['sequence']
                max_gap = constraint['max_gap']
                
                # Find sequence of robot primitives
                sequence_segments = self._find_primitive_sequence(
                    expected_sequence, robot_segments, max_gap, existing_mappings + mappings
                )
                
                if sequence_segments and len(sequence_segments) >= len(expected_sequence) // 2:
                    # Create mapping for the composite action
                    avg_confidence = np.mean([seg.confidence for seg in sequence_segments])
                    confidence = min(text_action.confidence, avg_confidence)
                    
                    # Use the first segment as representative
                    representative_segment = sequence_segments[0]
                    
                    mapping = SemanticMapping(
                        text_action=verb,
                        text_action_type=text_action.action_type,
                        robot_primitive=representative_segment.primitive,
                        robot_segment=representative_segment,
                        confidence=confidence,
                        reasoning=f"Sequence mapping: '{verb}' -> {[seg.primitive.value for seg in sequence_segments]}"
                    )
                    mappings.append(mapping)
        
        return mappings
    
    def _find_fuzzy_mappings(self, text_actions: List[ExtractedAction],
                           robot_segments: List[ActionSegment],
                           existing_mappings: List[SemanticMapping]) -> List[SemanticMapping]:
        """Find fuzzy mappings using similarity measures."""
        mappings = []
        
        # Get already mapped text actions
        mapped_verbs = {m.text_action for m in existing_mappings}
        mapped_segments = {m.robot_segment for m in existing_mappings}
        
        for text_action in text_actions:
            if text_action.verb in mapped_verbs:
                continue
                
            best_mapping = None
            best_score = 0.0
            
            for robot_segment in robot_segments:
                if robot_segment in mapped_segments:
                    continue
                
                # Calculate similarity score
                similarity = self._calculate_action_similarity(
                    text_action, robot_segment
                )
                
                if similarity > best_score and similarity > self.fuzzy_match_threshold:
                    best_score = similarity
                    confidence = min(text_action.confidence, robot_segment.confidence) * similarity
                    
                    best_mapping = SemanticMapping(
                        text_action=text_action.verb,
                        text_action_type=text_action.action_type,
                        robot_primitive=robot_segment.primitive,
                        robot_segment=robot_segment,
                        confidence=confidence,
                        reasoning=f"Fuzzy mapping: '{text_action.verb}' -> {robot_segment.primitive.value} (similarity: {similarity:.2f})"
                    )
            
            if best_mapping:
                mappings.append(best_mapping)
                mapped_segments.add(best_mapping.robot_segment)
        
        return mappings
    
    def _find_primitive_sequence(self, expected_sequence: List[str], 
                               robot_segments: List[ActionSegment],
                               max_gap: float,
                               existing_mappings: List[SemanticMapping]) -> List[ActionSegment]:
        """Find a sequence of robot primitives matching expected sequence."""
        mapped_segments = {m.robot_segment for m in existing_mappings}
        available_segments = [s for s in robot_segments if s not in mapped_segments]
        
        # Sort segments by start time
        available_segments.sort(key=lambda x: x.start_time)
        
        # Try to find the sequence
        sequence_segments = []
        last_end_time = -float('inf')
        
        for expected_primitive_name in expected_sequence:
            expected_primitive = ActionPrimitive(expected_primitive_name)
            
            # Find next matching primitive after last_end_time
            matching_segment = None
            for segment in available_segments:
                if (segment.primitive == expected_primitive and
                    segment.start_time >= last_end_time and
                    segment.start_time - last_end_time <= max_gap):
                    matching_segment = segment
                    break
            
            if matching_segment:
                sequence_segments.append(matching_segment)
                last_end_time = matching_segment.end_time
                available_segments.remove(matching_segment)
            else:
                # Try similar primitives
                similar_primitives = self.primitive_similarities.get(expected_primitive, [])
                for similar_primitive in similar_primitives:
                    for segment in available_segments:
                        if (segment.primitive == similar_primitive and
                            segment.start_time >= last_end_time and
                            segment.start_time - last_end_time <= max_gap):
                            sequence_segments.append(segment)
                            last_end_time = segment.end_time
                            available_segments.remove(segment)
                            break
                    if sequence_segments and sequence_segments[-1].primitive == similar_primitive:
                        break
        
        return sequence_segments
    
    def _calculate_action_similarity(self, text_action: ExtractedAction, 
                                   robot_segment: ActionSegment) -> float:
        """Calculate similarity between text action and robot primitive."""
        # Check if text action directly maps to robot primitive
        if text_action.verb in self.direct_mappings:
            expected_primitives = self.direct_mappings[text_action.verb]
            if robot_segment.primitive in expected_primitives:
                return 1.0
        
        # Check similarity with similar primitives
        if text_action.verb in self.direct_mappings:
            expected_primitives = self.direct_mappings[text_action.verb]
            for expected in expected_primitives:
                similar_primitives = self.primitive_similarities.get(expected, [])
                if robot_segment.primitive in similar_primitives:
                    return 0.7
        
        # String similarity between action names
        text_name = text_action.verb
        primitive_name = robot_segment.primitive.value
        string_similarity = difflib.SequenceMatcher(None, text_name, primitive_name).ratio()
        
        # Action type compatibility
        type_compatibility = self._calculate_type_compatibility(
            text_action.action_type, robot_segment.primitive
        )
        
        # Combine similarities
        similarity = 0.3 * string_similarity + 0.7 * type_compatibility
        
        return similarity
    
    def _calculate_type_compatibility(self, text_type: ActionType, 
                                    robot_primitive: ActionPrimitive) -> float:
        """Calculate compatibility between text action type and robot primitive."""
        # Define which primitives belong to which action types
        type_mappings = {
            ActionType.MANIPULATION: {
                ActionPrimitive.GRASP, ActionPrimitive.RELEASE, ActionPrimitive.LIFT,
                ActionPrimitive.PLACE, ActionPrimitive.REACH
            },
            ActionType.LOCOMOTION: {
                ActionPrimitive.MOVE, ActionPrimitive.APPROACH, ActionPrimitive.RETREAT
            },
            ActionType.INTERACTION: {
                ActionPrimitive.PUSH, ActionPrimitive.PULL, ActionPrimitive.ROTATE
            }
        }
        
        if text_type in type_mappings and robot_primitive in type_mappings[text_type]:
            return 1.0
        
        # Check for partial compatibility
        manipulation_primitives = type_mappings[ActionType.MANIPULATION]
        locomotion_primitives = type_mappings[ActionType.LOCOMOTION]
        
        if (text_type == ActionType.MANIPULATION and 
            robot_primitive in locomotion_primitives):
            return 0.5  # Movement can be part of manipulation
        
        return 0.1  # Very low compatibility
    
    def _is_segment_mapped(self, segment: ActionSegment, 
                          mappings: List[SemanticMapping]) -> bool:
        """Check if a segment is already mapped."""
        return any(m.robot_segment == segment for m in mappings)
    
    def _calculate_coverage_score(self, text_actions: List[ExtractedAction], 
                                mappings: List[SemanticMapping]) -> float:
        """Calculate how many text actions were successfully mapped."""
        if not text_actions:
            return 1.0
        
        mapped_actions = {m.text_action for m in mappings}
        coverage = len(mapped_actions) / len(text_actions)
        return coverage
    
    def _calculate_alignment_score(self, mappings: List[SemanticMapping]) -> float:
        """Calculate quality of mappings based on confidence."""
        if not mappings:
            return 0.0
        
        return np.mean([m.confidence for m in mappings])
    
    def _calculate_temporal_consistency(self, text_components: TaskComponents,
                                      mappings: List[SemanticMapping],
                                      robot_segments: List[ActionSegment]) -> float:
        """Calculate temporal consistency between text and robot sequences."""
        if len(mappings) < 2:
            return 1.0  # Perfect if only one or no mappings
        
        # Sort mappings by text action order
        text_ordered_mappings = sorted(mappings, key=lambda m: next(
            (i for i, a in enumerate(text_components.actions) if a.verb == m.text_action),
            float('inf')
        ))
        
        # Sort mappings by robot segment start time
        robot_ordered_mappings = sorted(mappings, key=lambda m: m.robot_segment.start_time)
        
        # Calculate order consistency
        text_order = [m.text_action for m in text_ordered_mappings]
        robot_order = [m.text_action for m in robot_ordered_mappings]
        
        # Count how many pairs maintain order
        consistent_pairs = 0
        total_pairs = 0
        
        for i in range(len(text_order) - 1):
            for j in range(i + 1, len(text_order)):
                action_i, action_j = text_order[i], text_order[j]
                
                try:
                    robot_i = robot_order.index(action_i)
                    robot_j = robot_order.index(action_j)
                    
                    if robot_i < robot_j:  # Order preserved
                        consistent_pairs += 1
                    total_pairs += 1
                except ValueError:
                    continue
        
        if total_pairs == 0:
            return 1.0
        
        return consistent_pairs / total_pairs
    
    def _calculate_overall_score(self, coverage: float, alignment: float, 
                               temporal: float) -> float:
        """Calculate overall mapping quality score."""
        # Weight the different components
        weights = [0.4, 0.4, 0.2]  # coverage, alignment, temporal
        scores = [coverage, alignment, temporal]
        
        overall = sum(w * s for w, s in zip(weights, scores))
        return overall
    
    def get_mapping_summary(self, result: MappingResult) -> Dict[str, Any]:
        """Get summary of mapping results."""
        return {
            'total_mappings': len(result.mappings),
            'coverage_score': result.coverage_score,
            'alignment_score': result.alignment_score,
            'temporal_consistency': result.temporal_consistency,
            'overall_score': result.overall_score,
            'unmapped_text_actions': len(result.unmapped_text_actions),
            'unmapped_robot_primitives': len(result.unmapped_primitives),
            'mapping_details': [
                {
                    'text_action': m.text_action,
                    'robot_primitive': m.robot_primitive.value,
                    'confidence': m.confidence,
                    'reasoning': m.reasoning
                }
                for m in result.mappings
            ]
        } 