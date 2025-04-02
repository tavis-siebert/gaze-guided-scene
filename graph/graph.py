from typing import Dict, List, Optional, Set, Tuple, Any
import torch
import numpy as np
from collections import deque, defaultdict
import math

from graph.node import Node, VisitRecord
from graph.edge import Edge
from graph.utils import AngleUtils, GraphTraversal
from graph.visualizer import GraphVisualizer
from egtea_gaze.utils import resolution
from logger import get_logger

# Type aliases for better readability
GazePosition = Tuple[int, int]
EdgeFeature = torch.Tensor
EdgeIndex = List[List[int]]
NodeId = int
EdgeId = Tuple[NodeId, NodeId]  # (source_id, target_id)

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
        # Create the root node (special node with ID -1)
        self.root = Node(id=-1, object_label='root')
        self.current_node = self.root
        
        # Track nodes by ID for efficient lookup
        self.nodes: Dict[NodeId, Node] = {-1: self.root}
        self.num_nodes = 0  # Counter for normal (non-root) nodes
        
        # Store all edges in the graph
        self.edges: List[Edge] = []
        
        # Adjacency mapping for efficient edge lookups
        # Maps node ID to list of neighbor node IDs
        self.adjacency = defaultdict(list)
        
        # The tracer will be set from outside (in build_graph.py)
        self.tracer = None
        
    def get_all_nodes(self) -> List[Node]:
        """
        Get all nodes in the graph including the root.
        
        Returns:
            List of all nodes in the graph
        """
        return list(self.nodes.values())
        
    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """
        Find a node by its ID.
        
        Args:
            node_id: The ID of the node to find
            
        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(node_id)
    
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
        # Store the node in our nodes dictionary
        self.nodes[node.id] = node
        self.num_nodes += 1
        return node
    
    def has_neighbor(self, source_id: NodeId, target_id: NodeId) -> bool:
        """
        Check if two nodes are connected.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            True if the nodes are connected, False otherwise
        """
        return target_id in self.adjacency[source_id]
    
    def get_node_neighbors(self, node_id: NodeId) -> List[NodeId]:
        """
        Get all neighbors of a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of neighbor node IDs
        """
        return self.adjacency[node_id]
    
    def add_edge(
        self, 
        source_node: Node, 
        target_node: Node, 
        prev_gaze_pos: GazePosition, 
        curr_gaze_pos: GazePosition, 
        num_bins: int = 8
    ) -> Tuple[Optional[Edge], Optional[Edge]]:
        """
        Add an edge between two nodes.
        
        Args:
            source_node: The source node
            target_node: The target node
            prev_gaze_pos: The previous gaze position (x,y)
            curr_gaze_pos: The current gaze position (x,y)
            num_bins: The number of angle bins
            
        Returns:
            Tuple of (forward_edge, backward_edge) - backward_edge may be None
        """
        if not self.has_neighbor(source_node.id, target_node.id):
            # Check if source is root node
            is_root = source_node.id == self.root.id
            
            # Create bidirectional edges
            forward_edge, backward_edge = Edge.create_bidirectional_edges(
                source_id=source_node.id,
                target_id=target_node.id,
                is_root=is_root,
                prev_gaze_pos=prev_gaze_pos,
                curr_gaze_pos=curr_gaze_pos,
                num_bins=num_bins
            )
            
            # Add forward edge
            self.edges.append(forward_edge)
            self.adjacency[source_node.id].append(target_node.id)
            
            # Add backward edge if exists (not connecting to root)
            if backward_edge:
                self.edges.append(backward_edge)
                self.adjacency[target_node.id].append(source_node.id)
                
            return forward_edge, backward_edge
        
        return None, None
    
    def update_graph(
        self, 
        label_counts: Dict[str, int], 
        visit: VisitRecord, 
        keypoints: List[Any], 
        descriptors: List[Any], 
        prev_gaze_pos: GazePosition, 
        curr_gaze_pos: GazePosition, 
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
        matching_node = self._find_matching_node(
            keypoints, descriptors, most_likely_label, inlier_thresh
        )
        next_node = Node.merge(visit, matching_node)
        
        # Create new node if no match found
        if next_node:
            frame_number = visit[1]  # Use end frame of the visit
            self.tracer.log_node_updated(
                frame_number,
                next_node.id,
                next_node.object_label,
                next_node.get_features(),
                visit
            )
            logger.info(f"Node {next_node.id} updated with new visit at frames {visit}")
        else:
            next_node = self.add_node(most_likely_label, visit, keypoints, descriptors)
        
        # Connect nodes if not already connected and not self-loop
        if (next_node != self.current_node and 
            not self.has_neighbor(self.current_node.id, next_node.id)):
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
    
    def _find_matching_node(
        self,
        keypoints: List[Any], 
        descriptors: List[Any], 
        label: str, 
        inlier_thresh: float
    ) -> Optional[Node]:
        """
        Find a matching node by searching from current node.
        
        Args:
            keypoints: SIFT keypoints for the object
            descriptors: SIFT descriptors for the object
            label: Object label to match
            inlier_thresh: Inlier threshold for RANSAC
            
        Returns:
            Matching node if found, None otherwise
        """
        # Use Node's static method but pass graph instance
        return Node.find_matching(
            graph=self,
            curr_node=self.current_node, 
            keypoints=keypoints, 
            descriptors=descriptors, 
            label=label, 
            inlier_thresh=inlier_thresh
        )
    
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
        GraphVisualizer.print_levels(self.root, use_degrees, self.edges, self)
    
    def get_features_tensor(
        self,
        video_length: int,
        current_frame: int,
        relative_frame: int,
        timestamp_fraction: float,
        labels_to_int: Dict[str, int],
        num_object_classes: int,
        node_data: Optional[Dict[int, torch.Tensor]] = None
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
            node_data: Optional dictionary of pre-computed node features
            
        Returns:
            Tuple of (node_features, edge_indices, edge_features)
        """
        # If node_data is provided, use it
        if node_data is not None and node_data:
            # Stack node features
            node_features = torch.stack(list(node_data.values()))
            
            # Normalize visit duration by relative frame number
            node_features[:, 0] /= relative_frame
            
            # Normalize number of visits by maximum value
            if node_features[:, 1].max() > 0:
                node_features[:, 1] /= node_features[:, 1].max()
            
            # Set timestamp fraction
            node_features[:, 4] = timestamp_fraction
        else:
            # Collect node features (skipping root node)
            nodes = []
            for node in self.nodes.values():
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
            
            # Normalize visit duration by relative frame number
            node_features[:, 0] /= relative_frame
            
            # Normalize number of visits by maximum value
            if node_features[:, 1].max() > 0:
                node_features[:, 1] /= node_features[:, 1].max()
        
        # Extract edge data in the original format
        edge_index = [[], []]
        edge_attrs = []
        
        # Process all edges (skipping edges to/from root)
        for edge in self.edges:
            # Skip edges connecting to root
            if edge.source_id < 0 or edge.target_id < 0:
                continue
                
            # Add edge indices
            edge_index[0].append(edge.source_id)
            edge_index[1].append(edge.target_id)
            
            # Add edge features
            edge_attrs.append(edge.get_features_tensor())
        
        # Convert to tensors
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) if edge_index[0] else torch.tensor([[],[]], dtype=torch.long)
        edge_attr_tensor = torch.stack(edge_attrs) if edge_attrs else torch.tensor([])
            
        return node_features, edge_index_tensor, edge_attr_tensor
    
    def get_edge(self, source_id: NodeId, target_id: NodeId) -> Optional[Edge]:
        """
        Get an edge between two nodes if it exists.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            The edge if found, None otherwise
        """
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None