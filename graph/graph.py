from typing import Dict, List, Optional, Set, Tuple, Any
import torch
import numpy as np
from collections import deque
import math
from collections import defaultdict

from graph.node import Node, VisitRecord
from graph.edge import Edge
from graph.utils import AngleUtils, GraphTraversal
from graph.visualizer import GraphVisualizer
from egtea_gaze.utils import resolution
from logger import get_logger

# Type aliases for better readability
Position = Tuple[int, int]
EdgeFeature = torch.Tensor
EdgeIndex = List[List[int]]

# Initialize logger for this module
logger = get_logger(__name__)

class Graph:
    """
    A scene graph representing objects and their spatial relationships.
    
    The graph consists of nodes (objects) connected by edges (spatial relationships).
    Each node represents an object in the scene, and edges represent the spatial
    relationships between objects.
    """
    
    def __init__(self):
        """Initialize an empty graph with a root node."""
        self.root = Node(id=-1, object_label='root')
        self.current_node = self.root
        self.num_nodes = 0
        self.edges = []
        
    def get_all_nodes(self) -> List[Node]:
        """Get all nodes in the graph."""
        return GraphTraversal.get_all_nodes(self.root)
    
    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """
        Find a node by its ID.
        
        Args:
            node_id: The ID of the node to find
            
        Returns:
            The node if found, None otherwise
        """
        if self.root.id == node_id:
            return self.root
            
        for node in self.get_all_nodes():
            if node.id == node_id:
                return node
        return None
    
    def get_node_by_label(self, label: str) -> Optional[Node]:
        """
        Find a node by its label.
        
        Args:
            label: The label of the node to find
            
        Returns:
            The first node with the given label if found, None otherwise
        """
        for node in self.get_all_nodes():
            if node.object_label == label:
                return node
        return None
    
    def add_node(
        self, 
        label: str, 
        visit: VisitRecord, 
        keypoints: List[Any], 
        descriptors: List[Any]
    ) -> Node:
        """
        Add a new node to the graph.
        
        Args:
            label: The object label
            visit: The visit period [start_frame, end_frame]
            keypoints: The keypoints from feature detector
            descriptors: The descriptors from feature detector
            
        Returns:
            The newly created node
        """
        node = Node.create(
            node_id=self.num_nodes,
            label=label,
            visit=visit,
            keypoints=keypoints,
            descriptors=descriptors
        )
        self.num_nodes += 1
        return node
    
    def add_edge(
        self, 
        source_node: Node, 
        target_node: Node, 
        prev_pos: Position, 
        curr_pos: Position, 
        num_bins: int = 8
    ) -> None:
        """
        Add an edge between two nodes.
        
        Args:
            source_node: The source node
            target_node: The target node
            prev_pos: The previous position (x,y)
            curr_pos: The current position (x,y)
            num_bins: The number of angle bins
        """
        if not source_node.has_neighbor(target_node):
            # Create bidirectional edges
            forward_edge, backward_edge = Edge.create_bidirectional_edges(
                source_node, target_node, prev_pos, curr_pos, num_bins
            )
            
            # Add forward edge to source node
            source_node.add_edge(forward_edge)
            self.edges.append(forward_edge)
            
            # Add backward edge to target node if exists (not connecting to root)
            if backward_edge:
                target_node.add_edge(backward_edge)
                self.edges.append(backward_edge)
    
    def update_graph(
        self, 
        label_counts: Dict[str, int], 
        visit: VisitRecord, 
        keypoints: List[Any], 
        descriptors: List[Any], 
        prev_gaze_pos: Position, 
        curr_gaze_pos: Position, 
        num_bins: int = 8, 
        inlier_thresh: float = 0.3
    ) -> Node:
        """
        Update the graph with a new observation.
        
        Args:
            label_counts: Dictionary of object labels and their counts
            visit: First and last frame of the object fixation
            keypoints: SIFT keypoints for the object
            descriptors: SIFT descriptors for the object
            prev_gaze_pos: Previous gaze position (x,y)
            curr_gaze_pos: Current gaze position (x,y)
            num_bins: Number of angle bins
            inlier_thresh: Inlier threshold for RANSAC
            
        Returns:
            The next node (either existing or newly created)
        """
        # Find most likely object label
        most_likely_label = max(label_counts, key=label_counts.get)
        
        # Try to find matching node
        matching_node = Node.find_matching(
            self.current_node, keypoints, descriptors, most_likely_label, inlier_thresh
        )
        next_node = Node.merge(visit, matching_node)
        
        # Create new node if no match found
        if next_node is None:
            next_node = self.add_node(most_likely_label, visit, keypoints, descriptors)
        
        # Connect nodes if not already connected and not self-loop
        if next_node != self.current_node and not self.current_node.has_neighbor(next_node):
            self.add_edge(
                self.current_node,
                next_node,
                prev_gaze_pos,
                curr_gaze_pos,
                num_bins
            )
        
        # Update current node
        self.current_node = next_node
        return next_node
    
    def print_graph(self, use_degrees: bool = True) -> None:
        """
        Print the graph structure.
        
        Args:
            use_degrees: Whether to display angles in degrees (True) or radians (False)
        """
        if self.num_nodes == 0:
            logger.info("Graph is empty.")
            return
            
        logger.info(f"Graph with {self.num_nodes} nodes:")
        GraphVisualizer.print_levels(self.root, use_degrees)
    
    def get_features_tensor(
        self,
        video_length: int,
        current_frame: int,
        relative_frame: int,
        timestamp_fraction: float,
        labels_to_int: Dict[str, int],
        num_object_classes: int,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get graph features as tensors for model input.
        
        Args:
            video_length: Total length of the video
            current_frame: Current frame number
            relative_frame: Relative frame number
            timestamp_fraction: Fraction of video at current timestamp
            labels_to_int: Mapping from object labels to class indices
            num_object_classes: Number of object classes
            normalize: Whether to normalize the node features
            
        Returns:
            Tuple of (node_features, edge_indices, edge_features)
        """
        # Collect node features
        nodes = []
        for node in sorted(self.get_all_nodes(), key=lambda n: n.id):
            if node.id >= 0:  # Skip root node
                features_tensor = node.get_features_tensor(
                    video_length,
                    current_frame,
                    relative_frame,
                    timestamp_fraction,
                    labels_to_int,
                    num_object_classes
                )
                nodes.append(features_tensor)
        
        # Handle empty graph case
        if not nodes:
            return torch.tensor([]), torch.tensor([[],[]], dtype=torch.long), torch.tensor([])
        
        # Stack node features
        node_features = torch.stack(nodes)
        
        # Normalize node features if requested
        if normalize:
            # Normalize visit duration by relative frame number
            node_features[:, 0] /= relative_frame
            
            # Normalize number of visits by maximum value
            if node_features[:, 1].max() > 0:
                node_features[:, 1] /= node_features[:, 1].max()
            
            # Set timestamp fraction
            node_features[:, 4] = timestamp_fraction
        
        # Get edge features using Edge's static method
        edge_indices, edge_features = Edge.get_edges_tensor(self.edges)
        
        return node_features, edge_indices, edge_features 