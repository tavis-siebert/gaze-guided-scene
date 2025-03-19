from typing import Tuple, List, Optional, Any, Dict
import torch
import numpy as np

from graph.node import Node
from graph.utils import AngleUtils
from egtea_gaze.utils import resolution

# Type aliases for better readability
Position = Tuple[int, int]
EdgeFeature = torch.Tensor

class Edge:
    """
    Represents a directed edge in the scene graph.
    
    Each edge connects two nodes and stores information about the spatial
    relationship between them, such as angle, distance, and relative positions.
    
    Attributes:
        source: The source node
        target: The target node
        angle: The angle between source and target
        distance: The distance between source and target
        features: Normalized position features as a tensor
    """
    def __init__(
        self,
        source: Node,
        target: Node,
        angle: float,
        distance: float,
        prev_pos: Position,
        curr_pos: Position
    ):
        """
        Initialize a new Edge.
        
        Args:
            source: The source node
            target: The target node
            angle: The angle between source and target
            distance: The distance between source and target
            prev_pos: Previous position (x,y)
            curr_pos: Current position (x,y)
        """
        self.source = source
        self.target = target
        self.angle = angle
        self.distance = distance
        self.features = self._compute_features(prev_pos, curr_pos)
    
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
        source: Node,
        target: Node,
        prev_pos: Position,
        curr_pos: Position,
        num_bins: int = 8
    ) -> Tuple['Edge', Optional['Edge']]:
        """
        Create bidirectional edges between two nodes.
        
        Args:
            source: Source node
            target: Target node
            prev_pos: Previous position (x,y)
            curr_pos: Current position (x,y)
            num_bins: Number of angle bins
            
        Returns:
            Tuple of (forward_edge, backward_edge)
            backward_edge is None if source is the root node
        """
        angle, distance = Edge.calculate_edge_features(prev_pos, curr_pos, num_bins)
        
        # Create forward edge
        forward_edge = Edge(source, target, angle, distance, prev_pos, curr_pos)
        
        # Create backward edge only if not connecting to root
        backward_edge = None
        if source.object_label != 'root':
            backward_edge = forward_edge.get_opposite()
        
        return forward_edge, backward_edge
    
    def _compute_features(self, prev_pos: Position, curr_pos: Position) -> EdgeFeature:
        """
        Compute edge features from positions.
        
        Args:
            prev_pos: Previous position (x,y)
            curr_pos: Current position (x,y)
            
        Returns:
            Tensor of normalized position features
        """
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos
        
        # Normalize by resolution
        norm_prev_x = prev_x / resolution[0]
        norm_prev_y = prev_y / resolution[1]
        norm_curr_x = curr_x / resolution[0] 
        norm_curr_y = curr_y / resolution[1]
        
        return torch.tensor([norm_prev_x, norm_prev_y, norm_curr_x, norm_curr_y])
    
    def get_features_tensor(self) -> EdgeFeature:
        """
        Get features for this edge as a tensor.
        
        Returns:
            Feature tensor for the edge
        """
        return self.features
    
    @property
    def source_id(self) -> int:
        """Get the source node ID."""
        return self.source.id
    
    @property
    def target_id(self) -> int:
        """Get the target node ID."""
        return self.target.id
    
    def get_opposite(self) -> 'Edge':
        """
        Create an edge in the opposite direction.
        
        Returns:
            A new Edge object with source and target swapped and opposite angle
        """
        opposite_angle = AngleUtils.get_opposite_angle(self.angle)
        return Edge(
            self.target, 
            self.source, 
            opposite_angle, 
            self.distance,
            # We reuse the same positions but swap direction for consistency
            self._get_position_from_features(True),  # curr_pos becomes prev_pos
            self._get_position_from_features(False)  # prev_pos becomes curr_pos
        )
    
    def _get_position_from_features(self, is_prev: bool) -> Position:
        """Extract positions from features for creating opposite edges."""
        if is_prev:
            return (
                int(self.features[0].item() * resolution[0]),
                int(self.features[1].item() * resolution[1])
            )
        else:
            return (
                int(self.features[2].item() * resolution[0]),
                int(self.features[3].item() * resolution[1])
            )
    
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