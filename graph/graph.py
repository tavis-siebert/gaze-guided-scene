from typing import Dict, List, Optional, Set, Tuple, Any
import torch
import numpy as np
from collections import deque

from graph.node import Node, VisitRecord
from graph.utils import (
    GraphTraversal, 
    AngleUtils, 
    EdgeManager, 
    NodeManager,
    GraphVisualizer,
    Position,
    EdgeFeature,
    EdgeIndex
)

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
        angle, distance = EdgeManager.calculate_edge_features(prev_pos, curr_pos, num_bins)
        
        if not source_node.has_neighbor(target_node):
            EdgeManager.add_bidirectional_edge(
                source_node,
                target_node,
                angle,
                distance,
                self.edge_data,
                self.edge_index,
                prev_pos,
                curr_pos
            )
    
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
    
    def to_pytorch_geometric(self) -> Dict:
        """
        Convert the graph to a PyTorch Geometric compatible format.
        
        Returns:
            Dictionary with node features, edge indices, and edge features
        """
        # Collect node features
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
        
        if not node_features:
            return {"x": None, "edge_index": None, "edge_attr": None}
            
        return {
            "x": torch.stack(node_features) if node_features else None,
            "edge_index": torch.tensor(self.edge_index, dtype=torch.long) if self.edge_index[0] else None,
            "edge_attr": torch.stack(self.edge_data) if self.edge_data else None
        } 