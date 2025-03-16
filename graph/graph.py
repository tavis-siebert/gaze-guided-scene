from typing import Dict, List, Optional, Set, Tuple, Any
import torch
import numpy as np
from collections import deque

from graph.node import Node, VisitRecord, NodeManager
from graph.utils import AngleUtils, GraphTraversal
from graph.visualizer import GraphVisualizer
from egtea_gaze.utils import resolution

# Type aliases for better readability
Position = Tuple[int, int]
EdgeFeature = torch.Tensor
EdgeIndex = List[List[int]]

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
        self.edge_data: List[EdgeFeature] = []
        self.edge_index: EdgeIndex = [[], []]
        
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
        node = NodeManager.create_node(
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
        angle, distance = self._calculate_edge_features(prev_pos, curr_pos, num_bins)
        
        if not source_node.has_neighbor(target_node):
            self._add_bidirectional_edge(
                source_node,
                target_node,
                angle,
                distance,
                prev_pos,
                curr_pos
            )
    
    def _calculate_edge_features(
        self, 
        prev_pos: Position, 
        curr_pos: Position, 
        num_bins: int
    ) -> Tuple[float, float]:
        """
        Calculate edge features between two positions.
        
        Args:
            prev_pos: Previous position (x,y)
            curr_pos: Current position (x,y)
            num_bins: Number of angle bins
            
        Returns:
            Tuple of (angle, distance)
        """
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos
        dx, dy = curr_x - prev_x, curr_y - prev_y
        
        angle = AngleUtils.get_angle_bin(dx, dy, num_bins)
        distance = np.sqrt(dx**2 + dy**2)
        
        return angle, distance
    
    def _normalize_positions(
        self, 
        prev_pos: Position, 
        curr_pos: Position
    ) -> Tuple[float, float, float, float]:
        """Normalize positions by resolution for edge features."""
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos
        
        return (
            prev_x / resolution[0], 
            prev_y / resolution[1], 
            curr_x / resolution[0], 
            curr_y / resolution[1]
        )
    
    def _create_edge_feature(
        self, 
        prev_x: float, 
        prev_y: float, 
        curr_x: float, 
        curr_y: float
    ) -> EdgeFeature:
        """Create an edge feature tensor from normalized positions."""
        return torch.tensor([prev_x, prev_y, curr_x, curr_y])
    
    def _update_edge_indices(
        self, 
        source_id: int, 
        target_id: int
    ) -> None:
        """Update edge indices for a bidirectional edge."""
        self.edge_index[0].extend([source_id, target_id])
        self.edge_index[1].extend([target_id, source_id])
    
    def _add_bidirectional_edge(
        self, 
        curr_node: Node, 
        next_node: Node, 
        angle: float, 
        distance: float,
        prev_pos: Position,
        curr_pos: Position
    ) -> None:
        """
        Add bidirectional edges between two nodes with appropriate features.
        
        Args:
            curr_node: Source node
            next_node: Target node
            angle: Angle between nodes
            distance: Distance between nodes
            prev_pos: Previous position (x,y)
            curr_pos: Current position (x,y)
        """
        # Add forward edge
        curr_node.add_neighbor(next_node, angle, distance)
        
        # Add backward edge only if not connecting to root
        if curr_node.object_label != 'root':
            # Calculate opposite angle
            opposite_angle = AngleUtils.get_opposite_angle(angle)
            next_node.add_neighbor(curr_node, opposite_angle, distance)
            
            # Normalize positions and create edge feature
            norm_prev_x, norm_prev_y, norm_curr_x, norm_curr_y = self._normalize_positions(prev_pos, curr_pos)
            edge_feature = self._create_edge_feature(norm_prev_x, norm_prev_y, norm_curr_x, norm_curr_y)
            
            # Add edge features for both directions (same feature for both)
            self.edge_data.append(edge_feature)
            self.edge_data.append(edge_feature)
            
            # Update edge indices
            self._update_edge_indices(curr_node.id, next_node.id)
    
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
        matching_node = NodeManager.find_matching_node(
            self.current_node, keypoints, descriptors, most_likely_label, inlier_thresh
        )
        next_node = NodeManager.merge_node(visit, matching_node)
        
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
            print("Graph is empty.")
            return
            
        print(f"Graph with {self.num_nodes} nodes:")
        GraphVisualizer.print_levels(self.root, use_degrees)
    
    def extract_features(
        self,
        node_data: Dict[int, torch.Tensor],
        relative_frame: int,
        timestamp_fraction: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract node and edge features from the graph.
        
        Args:
            node_data: Dictionary mapping node IDs to feature tensors
            relative_frame: Relative frame number for normalization
            timestamp_fraction: Fraction of video at current timestamp
            
        Returns:
            Tuple of (node_features, edge_indices, edge_features)
        """
        # Handle empty graph case
        if not node_data:
            return torch.tensor([]), torch.tensor([[],[]], dtype=torch.long), torch.tensor([])
            
        # Stack node features and normalize
        node_features = torch.stack(list(node_data.values()))
        normalized = Node.normalize_features(node_features, relative_frame, timestamp_fraction)
        
        # Extract edge features and indices
        edge_features = torch.stack(self.edge_data) if self.edge_data else torch.tensor([])
        edge_indices = torch.tensor(
            self.edge_index, dtype=torch.long
        ) if self.edge_index[0] else torch.tensor([[],[]], dtype=torch.long)
        
        return normalized, edge_indices, edge_features
    
    def to_pytorch_geometric(self) -> Dict:
        """
        Convert the graph to a PyTorch Geometric compatible format.
        
        Returns:
            Dictionary with node features, edge indices, and edge features
        """
        # Collect node features
        node_features = self._collect_node_features()
        
        if not node_features:
            return {"x": None, "edge_index": None, "edge_attr": None}
            
        return {
            "x": torch.stack(node_features) if node_features else None,
            "edge_index": torch.tensor(self.edge_index, dtype=torch.long) if self.edge_index[0] else None,
            "edge_attr": torch.stack(self.edge_data) if self.edge_data else None
        }
    
    def _collect_node_features(self) -> List[torch.Tensor]:
        """Collect features for all nodes in the graph."""
        node_features = []
        for node in sorted(self.get_all_nodes(), key=lambda n: n.id):
            if node.id >= 0:  # Skip root node
                # Example node features: [visit_duration, num_visits, first_frame, last_frame, label_one_hot]
                visit_duration = node.get_visit_duration()
                num_visits = len(node.visits)
                first_frame = node.get_first_visit_frame() or 0
                last_frame = node.get_last_visit_frame() or 0
                
                # This is a placeholder - in a real implementation, you'd create a proper one-hot encoding
                # based on your label vocabulary
                label_feature = torch.zeros(10)  # Assuming 10 possible labels
                label_feature[0] = 1  # Placeholder
                
                node_feature = torch.tensor([visit_duration, num_visits, first_frame, last_frame])
                node_features.append(torch.cat([node_feature, label_feature]))
        
        return node_features 