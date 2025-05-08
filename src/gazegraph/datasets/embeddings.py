"""
Node embedding module for creating semantic embeddings of graph nodes.
"""

import torch
from typing import Optional, Dict, List, Union

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.models.clip import ClipModel
from gazegraph.logger import get_logger

logger = get_logger(__name__)


class NodeEmbedder:
    """
    Handles creation of embeddings for various node types in scene graphs.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the node embedder.
        
        Args:
            device: Device to run models on ("cuda" or "cpu")
        """
        self.device = device
        self.clip_model = None
        
    def _get_clip_model(self) -> ClipModel:
        """Get or initialize the text embedding model."""
        if self.clip_model is None:
            logger.info(f"Initializing CLIP model on {self.device}")
            self.clip_model = ClipModel(device=self.device)
            self.clip_model.load()
        return self.clip_model
        
    def get_action_embedding(self, action_idx: int) -> Optional[torch.Tensor]:
        """
        Get embedding for an action using CLIP text embedding.
        
        Args:
            action_idx: The index of the action
            
        Returns:
            Tensor containing the action embedding, or None if the action is not found
        """
        # Get action name
        action_name = ActionRecord.get_action_name_by_idx(action_idx)
        if action_name is None:
            logger.warning(f"Action index {action_idx} not found in action mapping")
            return None
            
        # Get text embedding
        clip_model = self._get_clip_model()
        embedding = clip_model.encode_text([action_name])[0]  # List of 1 tensor -> single tensor
        
        return embedding 