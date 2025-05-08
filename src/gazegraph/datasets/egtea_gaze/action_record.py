"""
Action record module for handling EGTEA Gaze+ action annotations.
"""

import os
import torch
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import json
from pathlib import Path

from gazegraph.datasets.egtea_gaze.constants import NUM_ACTION_CLASSES
from gazegraph.config.config_utils import get_config
from gazegraph.logger import get_logger

logger = get_logger(__name__)

class ActionRecord:
    """
    Represents an action record from EGTEA Gaze+ annotation files.
    
    Features:
    - Handles parsing of action annotation entries
    - Provides mapping between action IDs and human-readable names
    - Creates future action prediction labels from record lists
    """
    # Class-level action mapping
    _action_to_idx = None
    _is_initialized = False
    
    # Action name mappings
    _verb_id_to_name = {}
    _noun_id_to_name = {}
    _action_id_to_name = {}
    
    # Config access
    _config = None
    
    def __init__(self, row: List[str]):
        """
        Initialize a record from a row in the annotation file.
        
        Args:
            row: List of strings from a tab-separated line in the annotation file
        """
        self._data = row

    @property
    def video_name(self) -> str:
        """Video name."""
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
    def verb_name(self) -> Optional[str]:
        """Verb name."""
        return self._verb_id_to_name.get(self.verb_id)
        
    @property
    def noun_name(self) -> Optional[str]:
        """Noun name."""
        return self._noun_id_to_name.get(self.noun_id)
        
    @property
    def action_name(self) -> Optional[str]:
        """Full action name as 'verb noun'."""
        if self.verb_name and self.noun_name:
            return f"{self.verb_name} {self.noun_name}"
        return None

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
        if not self._is_initialized:
            logger.warning("Action mapping not initialized. Call initialize_action_mapping() first.")
            return None
        return self._action_to_idx.get(self.action_tuple)
    
    @classmethod
    def get_config(cls):
        """Get or initialize config."""
        if cls._config is None:
            cls._config = get_config()
        return cls._config
    
    @classmethod
    def load_name_mappings(cls, base_dir: Optional[str] = None) -> None:
        """Load verb and noun name mappings from index files.
        
        Args:
            base_dir: Base directory containing verb_idx.txt and noun_idx.txt.
                      If None, uses path from config.
        """
        # Use config path if base_dir not provided
        if base_dir is None:
            config = cls.get_config()
            base_dir = config.dataset.egtea.action_annotations
        
        # Load verb mapping
        verb_path = os.path.join(base_dir, "verb_idx.txt")
        if not os.path.exists(verb_path):
            raise FileNotFoundError(f"Verb index file not found at {verb_path}")
            
        with open(verb_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(' ')
                    if len(parts) >= 2:
                        verb_name = ' '.join(parts[:-1])
                        verb_id = int(parts[-1])
                        cls._verb_id_to_name[verb_id] = verb_name
        
        # Load noun mapping
        noun_path = os.path.join(base_dir, "noun_idx.txt")
        if not os.path.exists(noun_path):
            raise FileNotFoundError(f"Noun index file not found at {noun_path}")
            
        with open(noun_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(' ')
                    if len(parts) >= 2:
                        noun_name = ' '.join(parts[:-1])
                        noun_id = int(parts[-1])
                        cls._noun_id_to_name[noun_id] = noun_name
        
        # Load action mapping
        action_path = os.path.join(base_dir, "action_idx.txt")
        if not os.path.exists(action_path):
            raise FileNotFoundError(f"Action index file not found at {action_path}")
            
        with open(action_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(' ')
                    if len(parts) >= 2:
                        action_name = ' '.join(parts[:-1])
                        action_id = int(parts[-1])
                        cls._action_id_to_name[action_id] = action_name
                            
        logger.info(f"Loaded {len(cls._verb_id_to_name)} verbs, {len(cls._noun_id_to_name)} nouns, "
                   f"and {len(cls._action_id_to_name)} actions")
    
    @classmethod
    def _compute_action_mapping(cls, records: List["ActionRecord"], num_classes: int = NUM_ACTION_CLASSES) -> Dict[Tuple[int, int], int]:
        """Compute action mapping from training records.
        
        Args:
            records: List of action records
            num_classes: Number of action classes to include
            
        Returns:
            Dictionary mapping (verb_id, noun_id) to action index
        """
        # Count occurrences of each action
        action_counts = Counter([(r.verb_id, r.noun_id) for r in records])
        
        # Sort by frequency (descending) and take top N
        top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:num_classes]

        # Create mapping from action tuple to index
        return {action: idx for idx, (action, _) in enumerate(top_actions)}

    @classmethod
    def initialize_action_mapping(cls, train_records: List["ActionRecord"], num_classes: int = NUM_ACTION_CLASSES) -> None:
        """Initialize the action mapping from training records.
        
        This method should be called once at the start of training/evaluation to ensure
        consistent action mappings across the entire pipeline.
        
        Args:
            train_records: List of action records from the training set
            num_classes: Number of action classes to include
            
        Raises:
            ValueError: If train_records is empty or num_classes is invalid
            RuntimeError: If name mappings haven't been loaded
        """
        if not train_records:
            raise ValueError("Cannot initialize action mapping: no training records provided")
            
        if num_classes <= 0:
            raise ValueError(f"Invalid number of classes: {num_classes}")
            
        if not cls._verb_id_to_name or not cls._noun_id_to_name:
            raise RuntimeError("Name mappings must be loaded before initializing action mapping")
            
        cls._action_to_idx = cls._compute_action_mapping(train_records, num_classes)
        cls._is_initialized = True
        
        # Log mapping statistics
        unique_verbs = len(set(verb_id for (verb_id, _) in cls._action_to_idx.keys()))
        unique_nouns = len(set(noun_id for (_, noun_id) in cls._action_to_idx.keys()))
        logger.info(f"Initialized action mapping with {len(cls._action_to_idx)} actions "
                   f"({unique_verbs} unique verbs, {unique_nouns} unique nouns)")
    
    @classmethod
    def get_action_mapping(cls) -> Dict[Tuple[int, int], int]:
        """Get the current action mapping.
        
        Returns:
            Dictionary mapping (verb_id, noun_id) to action index
        """
        if not cls._is_initialized:
            logger.warning("Action mapping not initialized. Call initialize_action_mapping() first.")
            return {}
        return cls._action_to_idx.copy()
    
    @classmethod 
    def get_action_names(cls) -> Dict[int, str]:
        """Get mapping from action indices to human-readable names.
        
        Returns:
            Dictionary mapping action index to full action name
        """
        result = {}
        if cls._is_initialized and cls._action_to_idx:
            for (verb_id, noun_id), idx in cls._action_to_idx.items():
                verb_name = cls._verb_id_to_name.get(verb_id, f"verb_{verb_id}")
                noun_name = cls._noun_id_to_name.get(noun_id, f"noun_{noun_id}")
                result[idx] = f"{verb_name} {noun_name}"
        return result
    
    @classmethod
    def get_noun_label_mappings(cls) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Get noun label mappings.
        
        Returns:
            Tuple containing:
            - Dictionary mapping noun IDs to names
            - Dictionary mapping noun names to IDs
            
        Raises:
            RuntimeError: If name mappings haven't been loaded
        """
        if not cls._verb_id_to_name or not cls._noun_id_to_name:
            raise RuntimeError("Name mappings must be loaded before accessing noun labels")
            
        # Return copies to prevent modification of internal state
        id_to_name = dict(cls._noun_id_to_name)
        name_to_id = {name: id for id, name in id_to_name.items()}
        return id_to_name, name_to_id
    
    @classmethod
    def create_future_action_labels(cls, 
                                    records: List["ActionRecord"], 
                                    current_frame: int,
                                    num_action_classes: int = None) -> Optional[Dict[str, torch.Tensor]]:
        """Create action label tensors for future actions.
        
        Args:
            records: List of action records for a video
            current_frame: Current frame number
            num_action_classes: Number of action classes (optional, uses mapping length if not provided)
            
        Returns:
            Dictionary containing:
            - 'next_action': Tensor of the next action class
            - 'future_actions': Binary tensor indicating which actions occur in the future
            - 'future_actions_ordered': Ordered tensor of future action classes
            
        Raises:
            RuntimeError: If action mapping is not initialized
        """
        if not cls._is_initialized:
            raise RuntimeError("Action mapping not initialized. Call initialize_action_mapping() first.")
            
        if num_action_classes is None:
            num_action_classes = len(cls._action_to_idx)
        
        past_records = [record for record in records if record.end_frame <= current_frame]
        future_records = [record for record in records if record.start_frame > current_frame]
        future_records = sorted(future_records, key=lambda record: record.start_frame)
        
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
    
    @classmethod
    def load_records_from_file(cls, filepath: str) -> List["ActionRecord"]:
        """Load records from an annotation file.
        
        Args:
            filepath: Path to the annotation file
            
        Returns:
            List of ActionRecord objects
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Action record file not found at {filepath}")
            
        records = []
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        row = line.strip().split(' ')
                        if len(row) < 4:  # Need at least video_name, start, end, and one label
                            raise ValueError(f"Invalid record format at line {line_num}")
                        records.append(cls(row))
            logger.info(f"Loaded {len(records)} records from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load records from {filepath}: {e}")
            raise
        
        return records

    def __str__(self) -> str:
        """String representation of the record."""
        action = f"{self.verb_name} {self.noun_name}" if self.verb_name and self.noun_name else f"({self.verb_id}, {self.noun_id})"
        return f"ActionRecord(video_name={self.video_name}, start={self.start_frame}, end={self.end_frame}, action={action})" 