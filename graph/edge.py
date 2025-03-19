from typing import Tuple, List, Optional, Any, Dict
import torch
import numpy as np

from graph.utils import AngleUtils
from egtea_gaze.utils import resolution

# Type aliases for better readability
Position = Tuple[int, int]
EdgeFeature = torch.Tensor
NodeId = int

class Edge:
    """
    Represents a directed edge in the scene graph.
    
    Each edge connects two nodes and stores information about the spatial
    relationship between them, such as angle, distance, and relative positions.
    
    Attributes:
        source_id: The ID of the source node
        target_id: The ID of the target node
        angle: The angle between source and target
        distance: The distance between source and target
        prev_pos: Previous position (x,y)
        curr_pos: Current position (x,y)
    """
    def __init__(
        self,
        source_id: NodeId,
        target_id: NodeId,
        angle: float,
        distance: float,
        prev_pos: Position,
        curr_pos: Position
    ):
        """
        Initialize a new Edge.
        
        Args:
            source_id: The ID of the source node
            target_id: The ID of the target node
            angle: The angle between source and target
            distance: The distance between source and target
            prev_pos: Previous position (x,y)
            curr_pos: Current position (x,y)
        """
        self.source_id = source_id
        self.target_id = target_id
        self.angle = angle
        self.distance = distance
        self.prev_pos = prev_pos
        self.curr_pos = curr_pos
    
    @staticmethod
    def calculate_edge_features(
        prev_pos: Position, 
        curr_pos: Position, 
        num_bins: int
    ) -> Tuple[float, float]:
        """
        Calculate edge features (angle and distance) between two positions.
        
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
    
    @staticmethod
    def create_bidirectional_edges(
        source_id: NodeId,
        target_id: NodeId,
        is_root: bool,
        prev_pos: Position,
        curr_pos: Position,
        num_bins: int = 8
    ) -> Tuple['Edge', Optional['Edge']]:
        """
        Create bidirectional edges between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            is_root: Whether the source node is the root node
            prev_pos: Previous position (x,y)
            curr_pos: Current position (x,y)
            num_bins: Number of angle bins
            
        Returns:
            Tuple of (forward_edge, backward_edge)
            backward_edge is None if source is the root node
        """
        angle, distance = Edge.calculate_edge_features(prev_pos, curr_pos, num_bins)
        
        # Create forward edge
        forward_edge = Edge(source_id, target_id, angle, distance, prev_pos, curr_pos)
        
        # Create backward edge only if not connecting to root
        backward_edge = None
        if not is_root:
            backward_edge = Edge.create_opposite(
                source_id=target_id,
                target_id=source_id,
                angle=AngleUtils.get_opposite_angle(angle),
                distance=distance,
                prev_pos=curr_pos,  # Swapped for opposite direction
                curr_pos=prev_pos   # Swapped for opposite direction
            )
        
        return forward_edge, backward_edge
    
    @staticmethod
    def create_opposite(
        source_id: NodeId,
        target_id: NodeId,
        angle: float,
        distance: float,
        prev_pos: Position,
        curr_pos: Position
    ) -> 'Edge':
        """
        Create an edge in the opposite direction.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            angle: The angle for the new edge
            distance: The distance for the new edge
            prev_pos: Previous position for the new edge
            curr_pos: Current position for the new edge
            
        Returns:
            A new Edge object representing the opposite direction
        """
        return Edge(source_id, target_id, angle, distance, prev_pos, curr_pos)
    
    def get_features(self) -> Dict[str, float]:
        """
        Get a dictionary of human-readable features for this edge.
        
        Returns:
            Dictionary with feature keys and values
        """
        prev_x, prev_y = self.prev_pos
        curr_x, curr_y = self.curr_pos
        
        # Normalize by resolution
        norm_prev_x = prev_x / resolution[0]
        norm_prev_y = prev_y / resolution[1]
        norm_curr_x = curr_x / resolution[0] 
        norm_curr_y = curr_y / resolution[1]
        
        return {
            "normalized_prev_x": norm_prev_x,
            "normalized_prev_y": norm_prev_y,
            "normalized_curr_x": norm_curr_x,
            "normalized_curr_y": norm_curr_y,
            "angle": self.angle,
            "distance": self.distance
        }
    
    def get_features_tensor(self) -> EdgeFeature:
        """
        Get features for this edge as a tensor.
        
        Returns:
            Feature tensor for the edge
        """
        features = self.get_features()
        
        return torch.tensor([
            features["normalized_prev_x"], 
            features["normalized_prev_y"], 
            features["normalized_curr_x"], 
            features["normalized_curr_y"]
        ])
    
    @staticmethod
    def get_edges_tensor(edges: List['Edge']) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list of edges to tensor format.
        
        Args:
            edges: List of Edge objects
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        if not edges:
            return torch.tensor([[],[]], dtype=torch.long), torch.tensor([])
        
        # Extract edge indices
        edge_index = [[], []]
        for edge in edges:
            edge_index[0].append(edge.source_id)
            edge_index[1].append(edge.target_id)
            
        # Extract edge features
        edge_attr = [edge.get_features_tensor() for edge in edges]
        
        return torch.tensor(edge_index, dtype=torch.long), torch.stack(edge_attr)
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two edges are equal based on their endpoints.
        
        Args:
            other: The other edge to compare with
            
        Returns:
            True if the edges have the same source and target, False otherwise
        """
        if not isinstance(other, Edge):
            return False
        return (self.source_id == other.source_id and 
                self.target_id == other.target_id)
    
    def __hash__(self) -> int:
        """
        Hash function for Edge based on source and target IDs.
        
        Returns:
            Hash value
        """
        return hash((self.source_id, self.target_id))
    
    def __repr__(self) -> str:
        """
        String representation of the edge.
        
        Returns:
            String representation
        """
        return f"Edge({self.source_id} → {self.target_id}, angle={self.angle:.2f}, dist={self.distance:.2f})" 