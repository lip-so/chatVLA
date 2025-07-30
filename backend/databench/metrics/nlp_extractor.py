"""
Natural Language Processing Extractor

Extracts actionable components from task prompts including:
- Action verbs and their types
- Target objects and their properties
- Spatial relationships and locations
- Temporal sequences and ordering
- Success criteria
"""

import re
import spacy
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions that can be extracted from prompts."""
    MANIPULATION = "manipulation"  # pick, place, grasp, release
    LOCOMOTION = "locomotion"     # move, go, approach, retreat
    INTERACTION = "interaction"   # push, pull, press, turn
    INSPECTION = "inspection"     # look, check, examine
    COMMUNICATION = "communication"  # signal, indicate

@dataclass
class ExtractedAction:
    """Represents an action extracted from text."""
    verb: str
    action_type: ActionType
    confidence: float
    objects: List[str]
    modifiers: List[str]
    location: Optional[str]
    order: int  # Position in sequence

@dataclass
class ExtractedObject:
    """Represents an object extracted from text."""
    name: str
    properties: List[str]  # color, size, shape, etc.
    location: Optional[str]
    is_target: bool  # Is this the main object being manipulated?
    confidence: float

@dataclass
class SpatialRelation:
    """Represents a spatial relationship."""
    relation: str  # in, on, under, next to, etc.
    object1: str
    object2: str
    confidence: float

@dataclass
class TaskComponents:
    """Complete analysis of a task prompt."""
    raw_text: str
    actions: List[ExtractedAction]
    objects: List[ExtractedObject]
    spatial_relations: List[SpatialRelation]
    temporal_indicators: List[str]
    success_criteria: List[str]
    confidence: float

class NLPExtractor:
    """Extracts actionable components from natural language task descriptions."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the NLP extractor.
        
        Args:
            model_name: SpaCy model to use for NLP processing
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"SpaCy model {model_name} not found. Using basic model.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("No SpaCy model available. NLP extraction will be limited.")
                self.nlp = None
        
        # Define action vocabularies
        self._setup_action_vocabularies()
        self._setup_object_patterns()
        self._setup_spatial_patterns()
        
    def _setup_action_vocabularies(self):
        """Setup vocabularies for different types of actions."""
        self.manipulation_verbs = {
            'pick', 'pickup', 'grasp', 'grab', 'take', 'lift', 'raise',
            'place', 'put', 'set', 'position', 'drop', 'release', 'let',
            'insert', 'remove', 'extract', 'pull', 'push', 'slide',
            'stack', 'unstack', 'pile', 'arrange', 'organize'
        }
        
        self.locomotion_verbs = {
            'move', 'go', 'travel', 'navigate', 'approach', 'reach',
            'retreat', 'back', 'return', 'come', 'advance', 'proceed'
        }
        
        self.interaction_verbs = {
            'push', 'pull', 'press', 'squeeze', 'turn', 'rotate', 'twist',
            'flip', 'switch', 'toggle', 'open', 'close', 'shut'
        }
        
        self.inspection_verbs = {
            'look', 'see', 'check', 'examine', 'inspect', 'observe',
            'find', 'locate', 'search', 'scan'
        }
        
        self.communication_verbs = {
            'signal', 'indicate', 'point', 'show', 'demonstrate'
        }
        
        # Create reverse mapping
        self.verb_to_type = {}
        for verb in self.manipulation_verbs:
            self.verb_to_type[verb] = ActionType.MANIPULATION
        for verb in self.locomotion_verbs:
            self.verb_to_type[verb] = ActionType.LOCOMOTION
        for verb in self.interaction_verbs:
            self.verb_to_type[verb] = ActionType.INTERACTION
        for verb in self.inspection_verbs:
            self.verb_to_type[verb] = ActionType.INSPECTION
        for verb in self.communication_verbs:
            self.verb_to_type[verb] = ActionType.COMMUNICATION
    
    def _setup_object_patterns(self):
        """Setup patterns for recognizing objects and their properties."""
        self.color_words = {
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'gray', 'grey', 'brown', 'silver', 'gold'
        }
        
        self.size_words = {
            'big', 'large', 'huge', 'giant', 'small', 'tiny', 'little',
            'medium', 'normal', 'regular', 'mini', 'micro'
        }
        
        self.shape_words = {
            'round', 'circular', 'square', 'rectangular', 'triangular',
            'spherical', 'cubic', 'cylindrical', 'oval', 'flat'
        }
        
        self.object_words = {
            'block', 'cube', 'box', 'container', 'ball', 'sphere',
            'cylinder', 'cup', 'bottle', 'plate', 'bowl', 'tool',
            'piece', 'item', 'object', 'thing', 'lego', 'brick',
            'bin', 'tray', 'holder', 'stand', 'base'
        }
        
        self.property_words = self.color_words | self.size_words | self.shape_words
    
    def _setup_spatial_patterns(self):
        """Setup patterns for spatial relationships."""
        self.spatial_prepositions = {
            'in', 'into', 'inside', 'within',
            'on', 'onto', 'upon', 'above', 'over',
            'under', 'below', 'beneath', 'underneath',
            'beside', 'next to', 'near', 'close to', 'by',
            'behind', 'in front of', 'ahead of',
            'left', 'right', 'center', 'middle',
            'top', 'bottom', 'corner', 'edge', 'side'
        }
        
        self.temporal_indicators = {
            'first', 'second', 'third', 'then', 'next', 'after', 'before',
            'finally', 'last', 'initially', 'subsequently', 'meanwhile'
        }
    
    def extract_components(self, text: str) -> TaskComponents:
        """
        Extract all actionable components from a text prompt.
        
        Args:
            text: Input text to analyze
            
        Returns:
            TaskComponents containing all extracted information
        """
        if not text or not text.strip():
            return TaskComponents(
                raw_text=text,
                actions=[],
                objects=[],
                spatial_relations=[],
                temporal_indicators=[],
                success_criteria=[],
                confidence=0.0
            )
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Extract different components
        actions = self._extract_actions(cleaned_text)
        objects = self._extract_objects(cleaned_text)
        spatial_relations = self._extract_spatial_relations(cleaned_text)
        temporal_indicators = self._extract_temporal_indicators(cleaned_text)
        success_criteria = self._extract_success_criteria(cleaned_text)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(actions, objects, spatial_relations)
        
        return TaskComponents(
            raw_text=text,
            actions=actions,
            objects=objects,
            spatial_relations=spatial_relations,
            temporal_indicators=temporal_indicators,
            success_criteria=success_criteria,
            confidence=confidence
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common contractions
        text = text.replace("'ll", " will")
        text = text.replace("'re", " are")
        text = text.replace("'ve", " have")
        text = text.replace("'d", " would")
        text = text.replace("n't", " not")
        
        return text
    
    def _extract_actions(self, text: str) -> List[ExtractedAction]:
        """Extract action verbs and their context from text."""
        actions = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            for token in doc:
                if token.lemma_ in self.verb_to_type:
                    action_type = self.verb_to_type[token.lemma_]
                    
                    # Find associated objects and modifiers
                    objects = []
                    modifiers = []
                    location = None
                    
                    # Look for direct objects
                    for child in token.children:
                        if child.dep_ in ['dobj', 'pobj']:
                            objects.append(child.text)
                        elif child.dep_ in ['advmod', 'amod']:
                            modifiers.append(child.text)
                        elif child.dep_ == 'prep':
                            # Check for location prepositions
                            for prep_child in child.children:
                                if prep_child.dep_ == 'pobj':
                                    location = f"{child.text} {prep_child.text}"
                    
                    # Calculate confidence based on context
                    confidence = self._calculate_action_confidence(token, objects, modifiers)
                    
                    action = ExtractedAction(
                        verb=token.lemma_,
                        action_type=action_type,
                        confidence=confidence,
                        objects=objects,
                        modifiers=modifiers,
                        location=location,
                        order=len(actions)
                    )
                    actions.append(action)
        else:
            # Fallback: simple word matching
            words = text.split()
            for i, word in enumerate(words):
                if word in self.verb_to_type:
                    action_type = self.verb_to_type[word]
                    
                    # Look for nearby objects (simple heuristic)
                    objects = []
                    for j in range(max(0, i-3), min(len(words), i+4)):
                        if words[j] in self.object_words:
                            objects.append(words[j])
                    
                    action = ExtractedAction(
                        verb=word,
                        action_type=action_type,
                        confidence=0.7,  # Lower confidence for fallback method
                        objects=objects,
                        modifiers=[],
                        location=None,
                        order=len(actions)
                    )
                    actions.append(action)
        
        return actions
    
    def _extract_objects(self, text: str) -> List[ExtractedObject]:
        """Extract objects and their properties from text."""
        objects = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Find noun phrases that might be objects
            for chunk in doc.noun_chunks:
                # Check if chunk contains known object words
                chunk_text = chunk.text.lower()
                chunk_words = chunk_text.split()
                
                if any(word in self.object_words for word in chunk_words):
                    # Extract properties
                    properties = []
                    for word in chunk_words:
                        if word in self.property_words:
                            properties.append(word)
                    
                    # Find the main object noun
                    object_name = chunk.root.text
                    
                    # Determine if this is a target object (appears near action verbs)
                    is_target = self._is_target_object(chunk, doc)
                    
                    confidence = self._calculate_object_confidence(chunk_text, properties, is_target)
                    
                    obj = ExtractedObject(
                        name=object_name,
                        properties=properties,
                        location=None,  # Will be filled by spatial relations
                        is_target=is_target,
                        confidence=confidence
                    )
                    objects.append(obj)
        else:
            # Fallback: pattern matching
            words = text.split()
            for i, word in enumerate(words):
                if word in self.object_words:
                    # Look for properties before the object
                    properties = []
                    for j in range(max(0, i-3), i):
                        if words[j] in self.property_words:
                            properties.append(words[j])
                    
                    obj = ExtractedObject(
                        name=word,
                        properties=properties,
                        location=None,
                        is_target=True,  # Assume all objects are targets in fallback
                        confidence=0.6
                    )
                    objects.append(obj)
        
        return objects
    
    def _extract_spatial_relations(self, text: str) -> List[SpatialRelation]:
        """Extract spatial relationships between objects."""
        relations = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            for token in doc:
                if token.text in self.spatial_prepositions:
                    # Find the objects this preposition relates
                    obj1, obj2 = self._find_related_objects(token, doc)
                    
                    if obj1 and obj2:
                        confidence = 0.8  # High confidence for explicit prepositions
                        
                        relation = SpatialRelation(
                            relation=token.text,
                            object1=obj1,
                            object2=obj2,
                            confidence=confidence
                        )
                        relations.append(relation)
        else:
            # Fallback: simple pattern matching
            for prep in self.spatial_prepositions:
                if prep in text:
                    # Simple heuristic to find nearby objects
                    pattern = rf"(\w+)\s+{re.escape(prep)}\s+(\w+)"
                    matches = re.finditer(pattern, text)
                    
                    for match in matches:
                        obj1, obj2 = match.groups()
                        if obj1 in self.object_words or obj2 in self.object_words:
                            relation = SpatialRelation(
                                relation=prep,
                                object1=obj1,
                                object2=obj2,
                                confidence=0.5
                            )
                            relations.append(relation)
        
        return relations
    
    def _extract_temporal_indicators(self, text: str) -> List[str]:
        """Extract temporal sequence indicators."""
        indicators = []
        
        words = text.split()
        for word in words:
            if word in self.temporal_indicators:
                indicators.append(word)
        
        # Also look for numbered sequences
        number_pattern = r'\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|\d+\.)\b'
        number_matches = re.findall(number_pattern, text)
        indicators.extend(number_matches)
        
        return indicators
    
    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria or completion indicators."""
        criteria = []
        
        # Look for completion phrases
        completion_patterns = [
            r'until\s+(.+)',
            r'so that\s+(.+)',
            r'in order to\s+(.+)',
            r'successfully\s+(.+)',
            r'completely\s+(.+)'
        ]
        
        for pattern in completion_patterns:
            matches = re.findall(pattern, text)
            criteria.extend(matches)
        
        return criteria
    
    def _calculate_confidence(self, actions: List[ExtractedAction], 
                            objects: List[ExtractedObject], 
                            spatial_relations: List[SpatialRelation]) -> float:
        """Calculate overall confidence in the extraction."""
        if not actions and not objects:
            return 0.0
        
        # Weight different components
        action_score = np.mean([a.confidence for a in actions]) if actions else 0.0
        object_score = np.mean([o.confidence for o in objects]) if objects else 0.0
        relation_score = np.mean([r.confidence for r in spatial_relations]) if spatial_relations else 0.0
        
        # Combine scores with weights
        weights = [0.4, 0.4, 0.2]  # Actions and objects more important
        scores = [action_score, object_score, relation_score]
        
        overall_confidence = sum(w * s for w, s in zip(weights, scores))
        
        # Bonus for having both actions and objects
        if actions and objects:
            overall_confidence *= 1.2
        
        return min(1.0, overall_confidence)
    
    def _calculate_action_confidence(self, token, objects: List[str], modifiers: List[str]) -> float:
        """Calculate confidence for an extracted action."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if action has objects
        if objects:
            confidence += 0.3
        
        # Boost confidence if action has modifiers
        if modifiers:
            confidence += 0.1
        
        # Boost confidence if token is a verb (when using spaCy)
        if hasattr(token, 'pos_') and token.pos_ == 'VERB':
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _calculate_object_confidence(self, chunk_text: str, properties: List[str], is_target: bool) -> float:
        """Calculate confidence for an extracted object."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if object has properties
        confidence += len(properties) * 0.1
        
        # Boost confidence if object is a target
        if is_target:
            confidence += 0.2
        
        # Boost confidence if object appears in known object words
        if any(word in self.object_words for word in chunk_text.split()):
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _is_target_object(self, chunk, doc) -> bool:
        """Determine if an object is a target (manipulated) object."""
        # Check if chunk appears near action verbs
        for token in doc:
            if token.lemma_ in self.verb_to_type:
                # Check if chunk is within reasonable distance of verb
                if abs(chunk.start - token.i) <= 5:  # Within 5 tokens
                    return True
        return False
    
    def _find_related_objects(self, prep_token, doc) -> Tuple[Optional[str], Optional[str]]:
        """Find objects related by a spatial preposition."""
        obj1, obj2 = None, None
        
        # Look for objects before and after the preposition
        for token in doc:
            if (token.pos_ == 'NOUN' and 
                abs(token.i - prep_token.i) <= 3):  # Within 3 tokens
                
                if token.i < prep_token.i and obj1 is None:
                    obj1 = token.text
                elif token.i > prep_token.i and obj2 is None:
                    obj2 = token.text
        
        return obj1, obj2
    
    def get_action_sequence(self, components: TaskComponents) -> List[str]:
        """Get ordered sequence of actions from components."""
        sorted_actions = sorted(components.actions, key=lambda x: x.order)
        return [action.verb for action in sorted_actions]
    
    def get_target_objects(self, components: TaskComponents) -> List[str]:
        """Get list of target objects from components."""
        return [obj.name for obj in components.objects if obj.is_target]
    
    def get_action_object_pairs(self, components: TaskComponents) -> List[Tuple[str, List[str]]]:
        """Get pairs of actions and their associated objects."""
        pairs = []
        for action in components.actions:
            pairs.append((action.verb, action.objects))
        return pairs 