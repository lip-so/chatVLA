"""
Temporal Action Alignment Module

Verifies that actions mentioned in prompts happen in the right sequence and timing
relative to actual robot actions. This module analyzes temporal relationships
between text descriptions and robot trajectory segments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import difflib
from collections import defaultdict
from scipy import optimize
from dtaidistance import dtw

from .action_primitives import ActionSegment, ActionPrimitive
from .nlp_extractor import TaskComponents, ExtractedAction
from .semantic_mapper import SemanticMapping, MappingResult

logger = logging.getLogger(__name__)

@dataclass
class TemporalAlignment:
    """Represents temporal alignment between text and robot actions."""
    text_sequence: List[str]
    robot_sequence: List[str]
    alignment_pairs: List[Tuple[int, int]]  # (text_idx, robot_idx) pairs
    alignment_score: float
    temporal_gaps: List[float]  # Time gaps between aligned actions
    sequence_similarity: float
    timing_consistency: float

@dataclass
class TemporalConstraint:
    """Represents a temporal constraint from text."""
    constraint_type: str  # 'before', 'after', 'while', 'then', 'simultaneously'
    action1: str
    action2: str
    time_window: Optional[float]  # Expected time window (seconds)
    confidence: float

@dataclass
class AlignmentResult:
    """Result of temporal alignment analysis."""
    alignment: TemporalAlignment
    constraints: List[TemporalConstraint]
    constraint_violations: List[Dict[str, Any]]
    sequence_order_score: float
    timing_precision_score: float
    overall_temporal_score: float

class TemporalAligner:
    """Analyzes temporal alignment between text descriptions and robot actions."""
    
    def __init__(self):
        """Initialize the temporal aligner."""
        self._setup_temporal_patterns()
        self._setup_timing_expectations()
    
    def _setup_temporal_patterns(self):
        """Setup patterns for detecting temporal relationships in text."""
        self.temporal_keywords = {
            'sequence': ['first', 'second', 'third', 'then', 'next', 'after', 'subsequently'],
            'simultaneous': ['while', 'during', 'as', 'simultaneously', 'at the same time'],
            'before': ['before', 'prior to', 'preceding'],
            'after': ['after', 'following', 'once', 'when'],
            'immediate': ['immediately', 'right away', 'instantly', 'quickly'],
            'delayed': ['slowly', 'carefully', 'gradually', 'wait']
        }
        
        self.sequence_indicators = {
            'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4,
            '1st': 0, '2nd': 1, '3rd': 2, '4th': 3, '5th': 4,
            'initially': 0, 'finally': -1, 'lastly': -1
        }
    
    def _setup_timing_expectations(self):
        """Setup expected timing for different action types."""
        self.action_durations = {
            ActionPrimitive.REACH: (0.5, 2.0),      # (min, max) seconds
            ActionPrimitive.GRASP: (0.2, 1.0),
            ActionPrimitive.LIFT: (0.3, 1.5),
            ActionPrimitive.PLACE: (0.5, 2.0),
            ActionPrimitive.RELEASE: (0.1, 0.5),
            ActionPrimitive.MOVE: (0.5, 3.0),
            ActionPrimitive.APPROACH: (1.0, 3.0),
            ActionPrimitive.RETREAT: (0.5, 2.0)
        }
        
        self.transition_times = {
            (ActionPrimitive.REACH, ActionPrimitive.GRASP): (0.0, 0.5),
            (ActionPrimitive.GRASP, ActionPrimitive.LIFT): (0.0, 0.3),
            (ActionPrimitive.LIFT, ActionPrimitive.MOVE): (0.0, 0.5),
            (ActionPrimitive.MOVE, ActionPrimitive.PLACE): (0.0, 0.5),
            (ActionPrimitive.PLACE, ActionPrimitive.RELEASE): (0.0, 0.3)
        }
    
    def analyze_temporal_alignment(self, text_components: TaskComponents,
                                 robot_segments: List[ActionSegment],
                                 semantic_mappings: List[SemanticMapping]) -> AlignmentResult:
        """
        Analyze temporal alignment between text and robot actions.
        
        Args:
            text_components: Extracted components from text
            robot_segments: Detected robot action segments
            semantic_mappings: Mappings between text and robot actions
            
        Returns:
            AlignmentResult containing alignment analysis
        """
        # Extract temporal constraints from text
        constraints = self._extract_temporal_constraints(text_components)
        
        # Create alignment between text and robot sequences
        alignment = self._create_temporal_alignment(
            text_components, robot_segments, semantic_mappings
        )
        
        # Check constraint violations
        constraint_violations = self._check_constraint_violations(
            constraints, robot_segments, semantic_mappings
        )
        
        # Calculate scores
        sequence_order_score = self._calculate_sequence_order_score(alignment)
        timing_precision_score = self._calculate_timing_precision_score(
            alignment, robot_segments, semantic_mappings
        )
        overall_temporal_score = self._calculate_overall_temporal_score(
            sequence_order_score, timing_precision_score, constraints, constraint_violations
        )
        
        return AlignmentResult(
            alignment=alignment,
            constraints=constraints,
            constraint_violations=constraint_violations,
            sequence_order_score=sequence_order_score,
            timing_precision_score=timing_precision_score,
            overall_temporal_score=overall_temporal_score
        )
    
    def _extract_temporal_constraints(self, text_components: TaskComponents) -> List[TemporalConstraint]:
        """Extract temporal constraints from text components."""
        constraints = []
        text = text_components.raw_text.lower()
        actions = [a.verb for a in text_components.actions]
        
        # Look for explicit sequence indicators
        for indicator, order in self.sequence_indicators.items():
            if indicator in text:
                # Find action associated with this indicator
                words = text.split()
                for i, word in enumerate(words):
                    if word == indicator:
                        # Look for action verbs nearby
                        for j in range(max(0, i-3), min(len(words), i+4)):
                            if words[j] in actions:
                                constraint = TemporalConstraint(
                                    constraint_type='sequence',
                                    action1=words[j],
                                    action2='',  # Will be filled by sequence order
                                    time_window=None,
                                    confidence=0.9
                                )
                                constraints.append(constraint)
                                break
        
        # Look for temporal relationship keywords
        for category, keywords in self.temporal_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    # Find actions related by this temporal keyword
                    action_pairs = self._find_actions_related_by_keyword(
                        text, keyword, actions
                    )
                    
                    for action1, action2 in action_pairs:
                        time_window = self._estimate_time_window(category, action1, action2)
                        
                        constraint = TemporalConstraint(
                            constraint_type=category,
                            action1=action1,
                            action2=action2,
                            time_window=time_window,
                            confidence=0.7
                        )
                        constraints.append(constraint)
        
        return constraints
    
    def _find_actions_related_by_keyword(self, text: str, keyword: str, 
                                       actions: List[str]) -> List[Tuple[str, str]]:
        """Find pairs of actions related by a temporal keyword."""
        pairs = []
        words = text.split()
        
        for i, word in enumerate(words):
            if word == keyword:
                # Look for actions before and after the keyword
                before_action = None
                after_action = None
                
                # Search backwards for action
                for j in range(i-1, max(0, i-5), -1):
                    if words[j] in actions:
                        before_action = words[j]
                        break
                
                # Search forwards for action
                for j in range(i+1, min(len(words), i+6)):
                    if words[j] in actions:
                        after_action = words[j]
                        break
                
                if before_action and after_action:
                    pairs.append((before_action, after_action))
        
        return pairs
    
    def _estimate_time_window(self, constraint_type: str, action1: str, action2: str) -> Optional[float]:
        """Estimate expected time window for temporal constraint."""
        if constraint_type == 'immediate':
            return 0.5
        elif constraint_type == 'sequence':
            return 3.0
        elif constraint_type == 'simultaneous':
            return 0.0
        elif constraint_type == 'delayed':
            return 5.0
        else:
            return 2.0  # Default window
    
    def _create_temporal_alignment(self, text_components: TaskComponents,
                                 robot_segments: List[ActionSegment],
                                 semantic_mappings: List[SemanticMapping]) -> TemporalAlignment:
        """Create alignment between text and robot action sequences."""
        # Extract sequences
        text_sequence = [a.verb for a in text_components.actions]
        robot_sequence = [s.primitive.value for s in robot_segments]
        
        # Handle empty sequences
        if not text_sequence and not robot_sequence:
            return TemporalAlignment(
                text_sequence=text_sequence,
                robot_sequence=robot_sequence,
                alignment_pairs=[],
                alignment_score=1.0,  # Perfect alignment if both empty
                temporal_gaps=[],
                sequence_similarity=1.0,
                timing_consistency=1.0
            )
        
        if not text_sequence or not robot_sequence:
            return TemporalAlignment(
                text_sequence=text_sequence,
                robot_sequence=robot_sequence,
                alignment_pairs=[],
                alignment_score=0.0,  # No alignment possible
                temporal_gaps=[],
                sequence_similarity=0.0,
                timing_consistency=0.0
            )
        
        # Create mapping from semantic mappings
        text_to_robot_map = {}
        for mapping in semantic_mappings:
            text_to_robot_map[mapping.text_action] = mapping.robot_primitive.value
        
        # Align sequences using dynamic programming
        alignment_pairs = self._align_sequences(text_sequence, robot_sequence, text_to_robot_map)
        
        # Calculate alignment score - safe division
        max_len = max(len(text_sequence), len(robot_sequence))
        alignment_score = len(alignment_pairs) / max_len if max_len > 0 else 0.0
        
        # Calculate temporal gaps
        temporal_gaps = self._calculate_temporal_gaps(alignment_pairs, robot_segments)
        
        # Calculate sequence similarity using edit distance
        sequence_similarity = self._calculate_sequence_similarity(text_sequence, robot_sequence)
        
        # Calculate timing consistency
        timing_consistency = self._calculate_timing_consistency(
            alignment_pairs, text_sequence, robot_segments
        )
        
        return TemporalAlignment(
            text_sequence=text_sequence,
            robot_sequence=robot_sequence,
            alignment_pairs=alignment_pairs,
            alignment_score=alignment_score,
            temporal_gaps=temporal_gaps,
            sequence_similarity=sequence_similarity,
            timing_consistency=timing_consistency
        )
    
    def _align_sequences(self, text_sequence: List[str], robot_sequence: List[str],
                        text_to_robot_map: Dict[str, str]) -> List[Tuple[int, int]]:
        """Align text and robot sequences using dynamic programming."""
        m, n = len(text_sequence), len(robot_sequence)
        
        # Create scoring matrix
        dp = np.zeros((m + 1, n + 1))
        backtrack = np.zeros((m + 1, n + 1), dtype=int)
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                text_action = text_sequence[i - 1]
                robot_action = robot_sequence[j - 1]
                
                # Calculate match score
                if text_action in text_to_robot_map:
                    expected_robot_action = text_to_robot_map[text_action]
                    match_score = 2.0 if robot_action == expected_robot_action else 0.0
                else:
                    # Use string similarity as fallback
                    match_score = difflib.SequenceMatcher(None, text_action, robot_action).ratio()
                
                # Options: match, skip text, skip robot
                match = dp[i - 1][j - 1] + match_score
                skip_text = dp[i - 1][j] - 0.5  # Penalty for skipping
                skip_robot = dp[i][j - 1] - 0.5
                
                best_score = max(match, skip_text, skip_robot)
                dp[i][j] = best_score
                
                if best_score == match:
                    backtrack[i][j] = 0  # Match
                elif best_score == skip_text:
                    backtrack[i][j] = 1  # Skip text
                else:
                    backtrack[i][j] = 2  # Skip robot
        
        # Backtrack to find alignment
        alignment_pairs = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if backtrack[i][j] == 0:  # Match
                alignment_pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif backtrack[i][j] == 1:  # Skip text
                i -= 1
            else:  # Skip robot
                j -= 1
        
        alignment_pairs.reverse()
        return alignment_pairs
    
    def _calculate_temporal_gaps(self, alignment_pairs: List[Tuple[int, int]], 
                               robot_segments: List[ActionSegment]) -> List[float]:
        """Calculate time gaps between aligned robot actions."""
        gaps = []
        
        for i in range(len(alignment_pairs) - 1):
            current_robot_idx = alignment_pairs[i][1]
            next_robot_idx = alignment_pairs[i + 1][1]
            
            if current_robot_idx < len(robot_segments) and next_robot_idx < len(robot_segments):
                current_end = robot_segments[current_robot_idx].end_time
                next_start = robot_segments[next_robot_idx].start_time
                gap = next_start - current_end
                gaps.append(max(0, gap))
        
        return gaps
    
    def _calculate_sequence_similarity(self, text_sequence: List[str], 
                                     robot_sequence: List[str]) -> float:
        """Calculate similarity between text and robot sequences."""
        if not text_sequence or not robot_sequence:
            return 0.0
        
        # Use normalized edit distance
        max_len = max(len(text_sequence), len(robot_sequence))
        if max_len == 0:
            return 1.0  # Both sequences empty = perfect similarity
        edit_distance = self._edit_distance(text_sequence, robot_sequence)
        similarity = 1.0 - (edit_distance / max_len)
        return max(0.0, similarity)
    
    def _edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate edit distance between two sequences."""
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No cost for match
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1]   # Substitution
                    )
        
        return dp[m][n]
    
    def _calculate_timing_consistency(self, alignment_pairs: List[Tuple[int, int]],
                                    text_sequence: List[str], 
                                    robot_segments: List[ActionSegment]) -> float:
        """Calculate how consistent the timing is with expectations."""
        if len(alignment_pairs) < 2:
            return 1.0
        
        consistency_scores = []
        
        for i, (text_idx, robot_idx) in enumerate(alignment_pairs):
            if robot_idx < len(robot_segments):
                robot_segment = robot_segments[robot_idx]
                action_duration = robot_segment.end_time - robot_segment.start_time
                
                # Check if duration is within expected range
                expected_range = self.action_durations.get(robot_segment.primitive, (0.1, 5.0))
                min_duration, max_duration = expected_range
                
                if min_duration <= action_duration <= max_duration:
                    consistency_scores.append(1.0)
                else:
                    # Penalize based on how far outside the range
                    if action_duration < min_duration:
                        penalty = (min_duration - action_duration) / min_duration
                    else:
                        penalty = (action_duration - max_duration) / max_duration
                    
                    consistency_scores.append(max(0.0, 1.0 - penalty))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _check_constraint_violations(self, constraints: List[TemporalConstraint],
                                   robot_segments: List[ActionSegment],
                                   semantic_mappings: List[SemanticMapping]) -> List[Dict[str, Any]]:
        """Check for violations of temporal constraints."""
        violations = []
        
        # Create mapping from action names to robot segments
        action_to_segments = {}
        for mapping in semantic_mappings:
            action_to_segments[mapping.text_action] = mapping.robot_segment
        
        for constraint in constraints:
            if constraint.action1 in action_to_segments and constraint.action2 in action_to_segments:
                segment1 = action_to_segments[constraint.action1]
                segment2 = action_to_segments[constraint.action2]
                
                violation = self._check_single_constraint(constraint, segment1, segment2)
                if violation:
                    violations.append(violation)
        
        return violations
    
    def _check_single_constraint(self, constraint: TemporalConstraint,
                               segment1: ActionSegment, segment2: ActionSegment) -> Optional[Dict[str, Any]]:
        """Check a single temporal constraint."""
        if constraint.constraint_type == 'sequence':
            # Action1 should come before action2
            if segment1.start_time >= segment2.start_time:
                return {
                    'constraint': constraint,
                    'violation_type': 'order_violation',
                    'expected': f"{constraint.action1} before {constraint.action2}",
                    'actual': f"{constraint.action2} starts at {segment2.start_time:.2f}s, {constraint.action1} starts at {segment1.start_time:.2f}s",
                    'severity': 'high'
                }
        
        elif constraint.constraint_type == 'simultaneous':
            # Actions should overlap significantly
            overlap = min(segment1.end_time, segment2.end_time) - max(segment1.start_time, segment2.start_time)
            total_duration = max(segment1.end_time, segment2.end_time) - min(segment1.start_time, segment2.start_time)
            
            if total_duration > 0:
                overlap_ratio = overlap / total_duration
                if overlap_ratio < 0.5:  # Less than 50% overlap
                    return {
                        'constraint': constraint,
                        'violation_type': 'timing_violation',
                        'expected': f"{constraint.action1} and {constraint.action2} should be simultaneous",
                        'actual': f"Only {overlap_ratio:.1%} overlap",
                        'severity': 'medium'
                    }
        
        elif constraint.constraint_type == 'immediate':
            # Action2 should start very soon after action1 ends
            gap = segment2.start_time - segment1.end_time
            if gap > (constraint.time_window or 0.5):
                return {
                    'constraint': constraint,
                    'violation_type': 'timing_violation',
                    'expected': f"{constraint.action2} should start immediately after {constraint.action1}",
                    'actual': f"Gap of {gap:.2f}s between actions",
                    'severity': 'medium'
                }
        
        return None
    
    def _calculate_sequence_order_score(self, alignment: TemporalAlignment) -> float:
        """Calculate score for sequence order preservation."""
        if len(alignment.alignment_pairs) < 2:
            return 1.0
        
        correct_order_count = 0
        total_pairs = 0
        
        for i in range(len(alignment.alignment_pairs) - 1):
            for j in range(i + 1, len(alignment.alignment_pairs)):
                text_i, robot_i = alignment.alignment_pairs[i]
                text_j, robot_j = alignment.alignment_pairs[j]
                
                # Check if order is preserved
                if (text_i < text_j and robot_i < robot_j) or (text_i > text_j and robot_i > robot_j):
                    correct_order_count += 1
                
                total_pairs += 1
        
        return correct_order_count / total_pairs if total_pairs > 0 else 1.0
    
    def _calculate_timing_precision_score(self, alignment: TemporalAlignment,
                                        robot_segments: List[ActionSegment],
                                        semantic_mappings: List[SemanticMapping]) -> float:
        """Calculate score for timing precision."""
        return alignment.timing_consistency
    
    def _calculate_overall_temporal_score(self, sequence_score: float, timing_score: float,
                                        constraints: List[TemporalConstraint],
                                        violations: List[Dict[str, Any]]) -> float:
        """Calculate overall temporal alignment score."""
        # Base score from sequence and timing
        base_score = 0.6 * sequence_score + 0.4 * timing_score
        
        # Penalty for constraint violations
        if constraints:
            violation_penalty = len(violations) / len(constraints)
            violation_penalty *= 0.3  # Max 30% penalty
            base_score *= (1.0 - violation_penalty)
        
        return max(0.0, min(1.0, base_score))
    
    def get_alignment_summary(self, result: AlignmentResult) -> Dict[str, Any]:
        """Get summary of temporal alignment results."""
        return {
            'sequence_order_score': result.sequence_order_score,
            'timing_precision_score': result.timing_precision_score,
            'overall_temporal_score': result.overall_temporal_score,
            'alignment_pairs': len(result.alignment.alignment_pairs),
            'sequence_similarity': result.alignment.sequence_similarity,
            'timing_consistency': result.alignment.timing_consistency,
            'temporal_constraints': len(result.constraints),
            'constraint_violations': len(result.constraint_violations),
            'violation_details': [
                {
                    'type': v['violation_type'],
                    'expected': v['expected'],
                    'actual': v['actual'],
                    'severity': v['severity']
                }
                for v in result.constraint_violations
            ]
        } 