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

logger = get_logger(__name__)


class NodeFeatureExtractor(ABC):
    """Base class for node feature extraction strategies."""
    
    def __init__(self):
        # Initialize cache for temporal features
        self._temporal_feature_cache = {}
    
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
            return torch.zeros(5)  # Return zeros for missing nodes
        
        # Extract basic node information
        total_frames_visited = sum(end - start for start, end in node_data["visits"])
        num_visits = len(node_data["visits"])
        
        first_visit_frame = node_data["visits"][0][0] if node_data["visits"] else 0
        last_visit_frame = node_data["visits"][-1][1] if node_data["visits"] else 0
        
        # Normalize temporal features
        first_frame_normalized = first_visit_frame / checkpoint.non_black_frame_count
        last_frame_normalized = last_visit_frame / checkpoint.non_black_frame_count
        frame_fraction = checkpoint.frame_number / checkpoint.video_length
        
        # Create temporal features tensor
        temporal_features = torch.tensor([
            total_frames_visited,
            num_visits,
            first_frame_normalized,
            last_frame_normalized,
            frame_fraction
        ])
        
        # Normalize first feature (total frames visited)
        if checkpoint.non_black_frame_count > 0:
            temporal_features[0] /= checkpoint.non_black_frame_count
        
        # Cache the temporal features
        self._temporal_feature_cache[cache_key] = temporal_features
        
        return temporal_features
    
    def _normalize_features(self, features_tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization to the features tensor.
        
        Args:
            features_tensor: Tensor of node features
            
        Returns:
            Normalized tensor of node features
        """
        if features_tensor.shape[0] == 0:
            return features_tensor
        
        # Normalize visit count if needed
        if features_tensor[:, 1].max() > 0:
            features_tensor[:, 1] /= features_tensor[:, 1].max()
        
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


class OneHotNodeFeatureExtractor(NodeFeatureExtractor):
    """Extracts node features using one-hot encoding for object classes."""
    
    def __init__(self):
        super().__init__()
    
    def extract_features(self, checkpoint: GraphCheckpoint) -> torch.Tensor:
        """
        Extract node features from a checkpoint using one-hot encoding for object classes.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tensor of node features
        """
        features_list = []
        for node_id, node_data in checkpoint.nodes.items():
            # Get temporal features
            temporal_features = self._extract_temporal_features(checkpoint, node_id)
                
            # Create one-hot encoding for object class
            class_idx = checkpoint.labels_to_int.get(node_data["object_label"], 0)
            one_hot = torch.zeros(checkpoint.num_object_classes)
            one_hot[class_idx] = 1
            
            # Combine features
            node_features = torch.cat([temporal_features, one_hot])
            features_list.append(node_features)
        
        if not features_list:
            return torch.tensor([])
            
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
    
    def __init__(self, device: str = "cuda", embedding_dim: int = 512):
        """
        Initialize the ROI embedding node feature extractor.
        
        Args:
            device: Device to run models on ("cuda" or "cpu")
            embedding_dim: Dimension of the embeddings
        """
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.node_embeddings = NodeEmbeddings(device=device)
        self.tracer = None
        self.video = None
        
        # Cache for ROI embeddings - keyed by (video_name, node_id)
        self.roi_embedding_cache = {}
        
    def set_context(self, tracer: GraphTracer, video: Video):
        """
        Set the context for ROI embedding extraction.
        
        Args:
            tracer: Graph tracer to retrieve detections
            video: Video processor for frame access
        """
        self.tracer = tracer
        self.video = video
    
    def _get_roi_embedding(self, checkpoint: GraphCheckpoint, node_id: int) -> torch.Tensor:
        """
        Get ROI embedding for a node, using cache if available.
        
        Args:
            checkpoint: GraphCheckpoint object
            node_id: ID of the node
            
        Returns:
            ROI embedding tensor
        """
        # Check if we already have a cached embedding for this node
        cache_key = (checkpoint.video_name, node_id)
        if cache_key in self.roi_embedding_cache:
            return self.roi_embedding_cache[cache_key]
        
        # Get new embedding
        roi_embedding = self.node_embeddings.get_object_node_embedding_roi(
            checkpoint, self.tracer, self.video, node_id
        )
        
        # If ROI embedding is not available, use zeros
        if roi_embedding is None:
            roi_embedding = torch.zeros(self.embedding_dim)
        
        # Cache the embedding
        self.roi_embedding_cache[cache_key] = roi_embedding
        
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
            temporal_features = self._extract_temporal_features(checkpoint, node_id)
            
            # Get ROI embedding for the node
            roi_embedding = self._get_roi_embedding(checkpoint, node_id)
            
            # Combine features
            node_features = torch.cat([temporal_features, roi_embedding])
            features_list.append(node_features)
        
        if not features_list:
            return torch.tensor([])
            
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
    
    def __init__(self, device: str = "cuda", embedding_dim: int = 512):
        """
        Initialize the object label embedding node feature extractor.
        
        Args:
            device: Device to run models on ("cuda" or "cpu")
            embedding_dim: Dimension of the embeddings
        """
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.node_embeddings = NodeEmbeddings(device=device)
        
        # Cache for label embeddings - keyed by object_label
        self.label_embedding_cache = {}
    
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
        
        object_label = node_data["object_label"]
        
        # Check if we already have a cached embedding for this label
        if object_label in self.label_embedding_cache:
            return self.label_embedding_cache[object_label]
        
        # Get new embedding
        label_embedding = self.node_embeddings.get_object_node_embedding_label(
            checkpoint, node_id
        )
        
        # If label embedding is not available, use zeros
        if label_embedding is None:
            label_embedding = torch.zeros(self.embedding_dim)
        
        # Cache the embedding
        self.label_embedding_cache[object_label] = label_embedding
        
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
            # Get temporal features
            temporal_features = self._extract_temporal_features(checkpoint, node_id)
            
            # Get label embedding for the node
            label_embedding = self._get_label_embedding(checkpoint, node_id)
            
            # Combine features
            node_features = torch.cat([temporal_features, label_embedding])
            features_list.append(node_features)
        
        if not features_list:
            return torch.tensor([])
            
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


def get_node_feature_extractor(feature_type: str, device: str = "cuda", **kwargs) -> NodeFeatureExtractor:
    """
    Factory function to get the appropriate node feature extractor.
    
    Args:
        feature_type: Type of node features to extract ("one-hot", "roi-embeddings", or "object-label-embeddings")
        device: Device to run models on ("cuda" or "cpu")
        **kwargs: Additional arguments to pass to the node feature extractor
        
    Returns:
        NodeFeatureExtractor instance
    """
    if feature_type == "one-hot":
        return OneHotNodeFeatureExtractor()
    elif feature_type == "roi-embeddings":
        return ROIEmbeddingNodeFeatureExtractor(device=device, **kwargs)
    elif feature_type == "object-label-embeddings":
        return ObjectLabelEmbeddingNodeFeatureExtractor(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown node feature type: {feature_type}")
