"""
Node embedding module for creating semantic embeddings of graph nodes.
"""

import torch
from typing import Optional, Dict, List, Union

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.models.clip import ClipTextEmbeddingModel
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
        self.text_embedder = None
        
    def _get_text_embedder(self) -> ClipTextEmbeddingModel:
        """Get or initialize the text embedding model."""
        if self.text_embedder is None:
            logger.info(f"Initializing CLIP text embedder on {self.device}")
            self.text_embedder = ClipTextEmbeddingModel(device=self.device)
        return self.text_embedder
        
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
        embedder = self._get_text_embedder()
        embedding = embedder([action_name])[0]  # List of 1 tensor -> single tensor
        
        return embedding 