"""
Record class for handling action records and action mapping.
"""

import torch
from typing import Dict, List, Tuple, Optional
from collections import Counter

from egtea_gaze.constants import NUM_ACTION_CLASSES
from logger import get_logger

logger = get_logger(__name__)

class Record:
    """
    Represents an action record from annotation files.
    
    Attributes:
        _data: Raw data from the annotation file
    """
    # Class-level action mapping
    _action_to_idx = None
    
    def __init__(self, row: List[str]):
        """
        Initialize a record from a row in the annotation file.
        
        Args:
            row: List of strings from a tab-separated line in the annotation file
        """
        self._data = row

    @property
    def path(self) -> str:
        """Video path."""
        return self._data[0]

    @property
    def start_frame(self) -> int:
        """Start frame of the action."""
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        """End frame of the action."""
        return int(self._data[2])

    @property
    def label(self) -> List[int]:
        """Action label as a list of integers."""
        return [int(x) for x in self._data[3:]]

    @property
    def verb_id(self) -> int:
        """Verb ID."""
        return self.label[0]

    @property
    def noun_id(self) -> int:
        """Noun ID."""
        return self.label[1]

    @property
    def num_frames(self) -> int:
        """Number of frames in the action clip."""
        return self.end_frame - self.start_frame + 1
    
    @property
    def action_tuple(self) -> Tuple[int, int]:
        """Get the action as a (verb_id, noun_id) tuple."""
        return (self.verb_id, self.noun_id)
    
    @property
    def action_idx(self) -> Optional[int]:
        """Get the index of this action in the action mapping."""
        if Record._action_to_idx is None:
            return None
        return Record._action_to_idx.get(self.action_tuple)
        
    @classmethod
    def set_action_mapping(cls, records: List["Record"], num_classes: int = NUM_ACTION_CLASSES) -> None:
        """Set the class-level action mapping.
        
        Args:
            records: List of action records
            num_classes: Number of action classes to include
        """
        # Count occurrences of each action
        action_counts = Counter([(r.verb_id, r.noun_id) for r in records])
        
        # Sort by frequency (descending) and take top N
        top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:num_classes]

        # Create mapping from action tuple to index
        cls._action_to_idx = {action: idx for idx, (action, _) in enumerate(top_actions)}
    
    @classmethod
    def get_action_mapping(cls) -> Dict[Tuple[int, int], int]:
        """Get the current action mapping.
        
        Returns:
            Dictionary mapping (verb, noun) tuples to class indices
        """
        return cls._action_to_idx or {}
    
    @classmethod
    def get_future_action_labels(cls, 
                             records: List["Record"], 
                             current_frame: int) -> Optional[Dict[str, torch.Tensor]]:
        """Create action label tensors for future actions.
        
        Args:
            records: List of action records for a video
            current_frame: Current frame number
            
        Returns:
            Dictionary containing:
            - 'next_action': Tensor of the next action class
            - 'future_actions': Binary tensor indicating which actions occur in the future
            - 'future_actions_ordered': Ordered tensor of future action classes
        """
        if cls._action_to_idx is None:
            return None
            
        num_action_classes = len(cls._action_to_idx)
        
        past_records = [record for record in records if record.end_frame <= current_frame]
        future_records = [record for record in records if record.start_frame > current_frame]
        
        if len(past_records) < 3 or len(future_records) < 3:
            return None
        
        future_records = sorted(future_records, key=lambda record: record.end_frame)
        
        future_actions = [
            record.action_idx
            for record in future_records if record.action_idx is not None
        ]
        
        if not future_actions:
            return None
            
        next_action_label = torch.tensor(future_actions[0], dtype=torch.long)
        future_action_labels_ordered = torch.tensor(future_actions, dtype=torch.long)
        
        future_action_labels = torch.zeros(num_action_classes, dtype=torch.long)
        future_action_labels[list(set(future_actions))] = 1
        
        return {
            'next_action': next_action_label,
            'future_actions': future_action_labels,
            'future_actions_ordered': future_action_labels_ordered
        } 

    def __str__(self) -> str:
        """String representation of the record."""
        return f"Record(path={self.path}, start_frame={self.start_frame}, end_frame={self.end_frame}, verb_id={self.verb_id}, noun_id={self.noun_id})" 