from typing import Dict, List, Tuple, Optional
import torch

from graph.io import Record
from logger import get_logger

logger = get_logger(__name__)

class ActionUtils:
    """Utilities for handling action-related operations in scene graphs."""
    
    @staticmethod
    def get_future_action_labels(
        records: List[Record], 
        current_frame: int, 
        action_to_class: Dict[Tuple[int, int], int]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Create action label tensors for future actions.
        
        Args:
            records: List of action records for a video
            current_frame: Current frame number
            action_to_class: Mapping from action tuples to class indices
            
        Returns:
            Dictionary containing:
            - 'next_action': Tensor of the next action class
            - 'future_actions': Binary tensor indicating which actions occur in the future
            - 'future_actions_ordered': Ordered tensor of future action classes
        """
        num_action_classes = len(action_to_class)
        
        past_records = [record for record in records if record.end_frame <= current_frame]
        future_records = [record for record in records if record.start_frame > current_frame]
        
        if len(past_records) < 3 or len(future_records) < 3:
            logger.debug(f"Not enough action data at frame {current_frame}: {len(past_records)} past, {len(future_records)} future")
            return None
        
        future_records = sorted(future_records, key=lambda record: record.end_frame)
        
        future_actions = [
            action_to_class[(record.label[0], record.label[1])] 
            for record in future_records if (record.label[0], record.label[1]) in action_to_class
        ]
        
        if not future_actions:
            logger.debug(f"No valid future actions at frame {current_frame}")
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