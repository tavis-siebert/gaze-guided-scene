"""
Node feature extraction strategies for graph datasets.

This module provides different strategies for extracting node features from graph checkpoints.
"""

import torch
from typing import Optional, Dict, Any, List, Tuple, Set, Union, Callable
from abc import ABC, abstractmethod
from collections import defaultdict

from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.datasets.node_embeddings import NodeEmbeddings
from gazegraph.graph.graph_tracer import GraphTracer
from gazegraph.datasets.egtea_gaze.video_processor import Video
from gazegraph.logger import get_logger
from gazegraph.config.config_utils import DotDict

logger = get_logger(__name__)


class NodeFeatureExtractor(ABC):
    """Base class for node feature extraction strategies."""
    
    def __init__(self, config: DotDict, node_embeddings: Optional[NodeEmbeddings] = None):
        self._temporal_feature_cache = {}
        self.config = config
        self.node_embeddings = node_embeddings
    
    def _extract_temporal_features(self, checkpoint: GraphCheckpoint, node_id: int) -> torch.Tensor:
        """Extract temporal features for a node.
        
        Args:
            checkpoint: GraphCheckpoint object
            node_id: ID of the node
            
        Returns:
            Tensor of temporal features
        """
        # Check if we already have cached temporal features for this node
        cache_key = (checkpoint.video_name, node_id, checkpoint.frame_number)
        if cache_key in self._temporal_feature_cache:
            return self._temporal_feature_cache[cache_key]
        
        node_data = checkpoint.nodes.get(node_id)
        if not node_data:
            logger.warning(f"Node ID {node_id} not found in checkpoint")
            # Get the device from the instance if available, otherwise use CPU
            device = getattr(self, 'device', 'cpu')
            return torch.zeros(5, device=device)  # Return zeros for missing nodes
        
        # Extract basic node information
        total_frames_visited = sum(end - start for start, end in node_data["visits"])
        num_visits = len(node_data["visits"])
        
        first_visit_frame = node_data["visits"][0][0] if node_data["visits"] else 0
        last_visit_frame = node_data["visits"][-1][1] if node_data["visits"] else 0
        
        # Normalize temporal features
        first_frame_normalized = first_visit_frame / checkpoint.non_black_frame_count
        last_frame_normalized = last_visit_frame / checkpoint.non_black_frame_count
        frame_fraction = checkpoint.frame_number / checkpoint.video_length
        
        # Get the device from the instance if available, otherwise use CPU
        device = getattr(self, 'device', 'cpu')
        
        # Create temporal features tensor on the appropriate device
        temporal_features = torch.tensor([
            total_frames_visited,
            num_visits,
            first_frame_normalized,
            last_frame_normalized,
            frame_fraction
        ], device=device)
        
        # Normalize first feature (total frames visited)
        if checkpoint.non_black_frame_count > 0:
            temporal_features[0] /= checkpoint.non_black_frame_count
        
        # Cache the temporal features
        self._temporal_feature_cache[cache_key] = temporal_features
        
        return temporal_features
    
    def _normalize_features(self, features_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize the features tensor. Always expects a 2D tensor [num_nodes, num_features].
        For empty input, returns empty tensor. For single-node or single-feature, handles gracefully.
        Currently normalizes the visit count (column 1) if present and max > 0.
        Args:
            features_tensor: [num_nodes, num_features] tensor
        Returns:
            Normalized tensor, same shape
        """
        if features_tensor.numel() == 0:
            return features_tensor
        # Ensure 2D shape
        if features_tensor.dim() == 1:
            features_tensor = features_tensor.unsqueeze(0)
        # Only normalize if at least two features
        if features_tensor.size(1) > 1:
            visit_col = features_tensor[:, 1]
            max_visit = visit_col.max()
            if max_visit > 0:
                features_tensor[:, 1] = visit_col / max_visit
        return features_tensor
    
    @abstractmethod
    def extract_features(self, checkpoint: GraphCheckpoint) -> torch.Tensor:
        """
        Extract node features from a checkpoint.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tensor of node features
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """
        Get the dimension of the node features.
        
        Returns:
            Dimension of the node features
        """
        pass

    def set_context(self, tracer: GraphTracer | None, video: Video | None):
        """
        Set the context for node feature extraction.
        
        Args:
            tracer: GraphTracer object
            video: Video object
        """
        self.tracer = tracer
        self.video = video


class OneHotNodeFeatureExtractor(NodeFeatureExtractor):
    """Extracts node features using one-hot encoding for object classes."""
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.device = device
    
    def extract_features(self, checkpoint: GraphCheckpoint) -> torch.Tensor:
        """
        Extract node features from a checkpoint using one-hot encoding for object classes.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tensor of node features
        """
        # Get device from temporal features if available
        device = getattr(self, 'device', 'cpu')
        
        features_list = []
        for node_id, node_data in checkpoint.nodes.items():
            # Get temporal features
            temporal_features = self._extract_temporal_features(checkpoint, node_id).flatten()
                
            # Create one-hot encoding for object class on the same device as temporal features
            class_idx = checkpoint.object_label_to_id.get(node_data["object_label"], 0)
            one_hot = torch.zeros(len(checkpoint.object_label_to_id), device=temporal_features.device)
            one_hot[class_idx] = 1
            
            # Combine features (now guaranteed to be on the same device)
            node_features = torch.cat([temporal_features, one_hot], dim=0)
            node_features = node_features.to(self.device)
            features_list.append(node_features)
        
        if not features_list:
            return torch.empty((0, self.feature_dim), device=self.device)
            
        # Stack all node features
        node_features_tensor = torch.stack(features_list)
        
        # Apply normalization
        return self._normalize_features(node_features_tensor)
    
    @property
    def feature_dim(self) -> int:
        """
        Get the dimension of the node features.
        
        Returns:
            Dimension of the node features (5 temporal features + num_object_classes)
        """
        # This will be dynamically calculated based on the checkpoint
        # 5 temporal features + variable number of object classes
        # The actual dimension is determined at runtime
        return -1  # Placeholder, actual value determined at runtime


class ROIEmbeddingNodeFeatureExtractor(NodeFeatureExtractor):
    """Extracts node features using ROI embeddings."""
    
    def __init__(self, device: str = "cuda", embedding_dim: int = 768, **kwargs):
        """
        Initialize the ROI embedding node feature extractor.
        
        Args:
            device: Device to run models on ("cuda" or "cpu")
            embedding_dim: Dimension of the embeddings
            node_embeddings: Optional NodeEmbeddings instance (for testing)
        """
        super().__init__(**kwargs)
        self.device = device
        self.embedding_dim = embedding_dim
        self.tracer = None
        self.video = None
        
        # Initialize node embeddings if not provided
        if self.node_embeddings is None:
            self.node_embeddings = NodeEmbeddings(self.config, device=device)
        
    def set_context(self, tracer: GraphTracer | None, video: Video | None):
        """
        Set the context for ROI embedding extraction.
        
        Args:
            tracer: GraphTracer object
            video: Video object
        """
        if tracer is None or video is None:
            raise ValueError("Tracer and video must be provided to ROI embedding node feature extractor")
        self.tracer = tracer
        self.video = video
    
    def _get_roi_embedding(self, checkpoint: GraphCheckpoint, node_id: int) -> torch.Tensor:
        """
        Get ROI embedding for a node. Delegates caching to NodeEmbeddings.
        
        Args:
            checkpoint: GraphCheckpoint object
            node_id: ID of the node
        Returns:
            ROI embedding tensor
        """
        node_data = checkpoint.nodes.get(node_id)
        if not node_data:
            logger.warning(f"Node ID {node_id} not found in checkpoint")
            return torch.zeros(self.embedding_dim)
        # Get embedding (caching is handled in NodeEmbeddings)
        roi_embedding = self.node_embeddings.get_object_node_embedding_roi(
            checkpoint, self.tracer, self.video, node_id
        )
        if roi_embedding is None:
            roi_embedding = torch.zeros(self.embedding_dim)
        return roi_embedding
    
    def extract_features(self, checkpoint: GraphCheckpoint) -> torch.Tensor:
        """
        Extract node features from a checkpoint using ROI embeddings.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tensor of node features
        """
        if self.tracer is None or self.video is None:
            raise ValueError("Tracer and video must be set before extracting ROI embeddings")
            
        features_list = []
        for node_id in checkpoint.nodes.keys():
            # Get temporal features
            temporal_features = self._extract_temporal_features(checkpoint, node_id).flatten()
            
            # Get ROI embedding for the node
            roi_embedding = self._get_roi_embedding(checkpoint, node_id).flatten()
            if temporal_features.device != roi_embedding.device:
                # Move to the device of roi_embedding (likely CUDA)
                temporal_features = temporal_features.to(roi_embedding.device)
            
            # Combine features
            node_features = torch.cat([temporal_features, roi_embedding], dim=0)
            node_features = node_features.to(self.device)
            features_list.append(node_features)
        
        if not features_list:
            return torch.empty((0, self.feature_dim), device=self.device)
            
        # Stack all node features
        node_features_tensor = torch.stack(features_list)
        
        # Apply normalization
        return self._normalize_features(node_features_tensor)
    
    @property
    def feature_dim(self) -> int:
        """
        Get the dimension of the node features.
        
        Returns:
            Dimension of the node features (5 temporal features + embedding_dim)
        """
        return 5 + self.embedding_dim  # 5 temporal features + embedding dimension


class ObjectLabelEmbeddingNodeFeatureExtractor(NodeFeatureExtractor):
    """Extracts node features using object label embeddings."""
    
    def __init__(self, device: str = "cuda", embedding_dim: int = 768, **kwargs):
        """
        Initialize the object label embedding node feature extractor.
        
        Args:
            device: Device to run models on ("cuda" or "cpu")
            embedding_dim: Dimension of the embeddings
            node_embeddings: Optional NodeEmbeddings instance (for testing)
        """
        super().__init__(**kwargs)
        self.device = device
        self.embedding_dim = embedding_dim
        
        if self.node_embeddings is None:
            self.node_embeddings = NodeEmbeddings(self.config, device=device)
    
    def _get_label_embedding(self, checkpoint: GraphCheckpoint, node_id: int) -> torch.Tensor:
        """
        Get label embedding for a node, using cache if available.
        
        Args:
            checkpoint: GraphCheckpoint object
            node_id: ID of the node
            
        Returns:
            Label embedding tensor
        """
        node_data = checkpoint.nodes.get(node_id)
        if not node_data:
            logger.warning(f"Node ID {node_id} not found in checkpoint")
            return torch.zeros(self.embedding_dim)
        
        # Get embedding from NodeEmbeddings (which handles caching internally)
        label_embedding = self.node_embeddings.get_object_node_embedding_label(
            checkpoint, node_id
        )
        
        # If label embedding is not available, use zeros
        if label_embedding is None:
            label_embedding = torch.zeros(self.embedding_dim)
            
        return label_embedding
    
    def extract_features(self, checkpoint: GraphCheckpoint) -> torch.Tensor:
        """
        Extract node features from a checkpoint using object label embeddings.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tensor of node features
        """
        features_list = []
        for node_id in checkpoint.nodes.keys():
            temporal_features = self._extract_temporal_features(checkpoint, node_id)
            label_embedding = self._get_label_embedding(checkpoint, node_id)
            # Ensure both are 1D and on the same device
            if temporal_features.device != label_embedding.device:
                temporal_features = temporal_features.to(label_embedding.device)
            temporal_features = temporal_features.flatten()
            label_embedding = label_embedding.flatten()
            node_features = torch.cat([temporal_features, label_embedding], dim=0)
            node_features = node_features.to(self.device)
            features_list.append(node_features)
        if not features_list:
            return torch.empty((0, self.feature_dim), device=self.device)
        node_features_tensor = torch.stack(features_list)
        return self._normalize_features(node_features_tensor)
    
    @property
    def feature_dim(self) -> int:
        """
        Get the dimension of the node features.
        
        Returns:
            Dimension of the node features (5 temporal features + embedding_dim)
        """
        return 5 + self.embedding_dim  # 5 temporal features + embedding dimension


class OneHotActionNodeFeatureExtractor(NodeFeatureExtractor):
    """Extracts node features using one-hot encoding for action classes.
    
    For action graphs, call extract_features(action_records: List[ActionRecord])
    to obtain a [num_nodes, num_action_classes] tensor, where each row is a one-hot vector.
    """
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
        self.action_mapping = ActionRecord.get_action_mapping()
        self.num_action_classes = len(self.action_mapping)

    def extract_features(self, checkpoint=None, action_records=None, *args, **kwargs) -> torch.Tensor:
        """
        Extract node features using one-hot encoding for action classes.

        Args:
            checkpoint: (Unused, for compatibility)
            action_records: List of ActionRecord objects. Each must have .action_idx set.
        Returns:
            Tensor of shape [num_nodes, num_action_classes]
        """
        if action_records is None:
            # For compatibility: return empty tensor
            return torch.empty((0, self.feature_dim), device=self.device)
        num_nodes = len(action_records)
        x = torch.zeros((num_nodes, self.num_action_classes), device=self.device)
        for i, rec in enumerate(action_records):
            if rec.action_idx is not None:
                x[i, rec.action_idx] = 1.0
        assert x.shape == (num_nodes, self.num_action_classes), f"Expected shape {(num_nodes, self.num_action_classes)}, got {x.shape}"
        return x


    @property
    def feature_dim(self) -> int:
        """Get the dimension of the node features (number of action classes)."""
        return self.num_action_classes


class ActionLabelEmbeddingNodeFeatureExtractor(NodeFeatureExtractor):
    """Extracts node features using action label embeddings.
    
    For action graphs, call extract_features(action_records: List[ActionRecord])
    to obtain a [num_nodes, embedding_dim] tensor, where each row is an embedding.
    """
    def __init__(self, device: str = "cuda", embedding_dim: int = 768, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.embedding_dim = embedding_dim
        if self.node_embeddings is None:
            self.node_embeddings = NodeEmbeddings(self.config, device=device)

    def extract_features(self, checkpoint=None, action_records=None, *args, **kwargs) -> torch.Tensor:
        """
        Extract node features using action label embeddings.

        Args:
            checkpoint: (Unused, for compatibility)
            action_records: List of ActionRecord objects. Each must have .action_idx set.
        Returns:
            Tensor of shape [num_nodes, embedding_dim]
        """
        if action_records is None:
            # For compatibility: return empty tensor
            return torch.empty((0, self.feature_dim), device=self.device)
        features = []
        for rec in action_records:
            emb = self.node_embeddings.get_action_embedding(rec.action_idx)
            features.append(emb)
        if features:
            x = torch.cat(features)
        else:
            x = torch.empty((0, self.embedding_dim), device=self.device)
        # assert x.shape == (len(action_records), self.embedding_dim), f"Expected shape ({len(action_records)}, {self.embedding_dim}), got {x.shape}"
        return x


    @property
    def feature_dim(self) -> int:
        """Get the dimension of the node features (embedding_dim)."""
        return self.embedding_dim


def get_node_feature_extractor(feature_type: str, device: str = "cuda", **kwargs) -> NodeFeatureExtractor:
    """
    Factory function to get the appropriate node feature extractor.
    
    Args:
        feature_type: Type of node features to extract ("one-hot", "roi-embeddings", "object-label-embeddings",
                                                     "action-label-embedding", "action-one-hot")
        device: Device to run models on ("cuda" or "cpu")
        **kwargs: Additional arguments to pass to the node feature extractor
        
    Returns:
        NodeFeatureExtractor instance
    """
    if feature_type == "one-hot":
        return OneHotNodeFeatureExtractor(**kwargs)
    elif feature_type == "roi-embeddings":
        return ROIEmbeddingNodeFeatureExtractor(device=device, **kwargs)
    elif feature_type == "object-label-embeddings":
        return ObjectLabelEmbeddingNodeFeatureExtractor(device=device, **kwargs)
    elif feature_type == "action-label-embedding":
        return ActionLabelEmbeddingNodeFeatureExtractor(device=device, **kwargs)
    elif feature_type == "action-one-hot":
        return OneHotActionNodeFeatureExtractor(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown node feature type: {feature_type}")
