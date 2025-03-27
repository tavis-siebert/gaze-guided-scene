from typing import Tuple, List, Optional, Any, Dict
import torch
import numpy as np

from graph.utils import AngleUtils
from egtea_gaze.utils import resolution

# Type aliases for better readability
GazePosition = Tuple[int, int]  # Updated from Position to GazePosition
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
        prev_gaze_pos: Previous gaze position (x,y)
        curr_gaze_pos: Current gaze position (x,y)
        num_bins: Number of angle bins for discretization
    """
    def __init__(
        self,
        source_id: NodeId,
        target_id: NodeId,
        prev_gaze_pos: GazePosition,
        curr_gaze_pos: GazePosition,
        num_bins: int = 8
    ):
        """
        Initialize a new Edge.
        
        Args:
            source_id: The ID of the source node
            target_id: The ID of the target node
            prev_gaze_pos: Previous gaze position (x,y)
            curr_gaze_pos: Current gaze position (x,y)
            num_bins: Number of angle bins for discretization
        """
        self.source_id = source_id
        self.target_id = target_id
        self.prev_gaze_pos = prev_gaze_pos
        self.curr_gaze_pos = curr_gaze_pos
        self.num_bins = num_bins
    
    @property
    def angle(self) -> float:
        """
        Angle between source and target gaze positions.
        Works with normalized [0,1] gaze coordinates.
        """
        prev_x, prev_y = self.prev_gaze_pos
        curr_x, curr_y = self.curr_gaze_pos
        
        # Calculate vector components in the normalized space
        dx, dy = curr_x - prev_x, curr_y - prev_y
        
        return AngleUtils.get_angle_bin(dx, dy, self.num_bins)
    
    @property
    def distance(self) -> float:
        """
        Euclidean distance between source and target gaze positions.
        Works with normalized [0,1] gaze coordinates, returns a relative distance.
        """
        prev_x, prev_y = self.prev_gaze_pos
        curr_x, curr_y = self.curr_gaze_pos
        
        # Calculate Euclidean distance in normalized space
        dx, dy = curr_x - prev_x, curr_y - prev_y
        
        return np.sqrt(dx**2 + dy**2)
    
    @staticmethod
    def create_bidirectional_edges(
        source_id: NodeId,
        target_id: NodeId,
        is_root: bool,
        prev_gaze_pos: GazePosition,
        curr_gaze_pos: GazePosition,
        num_bins: int = 8
    ) -> Tuple['Edge', Optional['Edge']]:
        """
        Create bidirectional edges between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            is_root: Whether the source node is the root node
            prev_gaze_pos: Previous gaze position (x,y)
            curr_gaze_pos: Current gaze position (x,y)
            num_bins: Number of angle bins
            
        Returns:
            Tuple of (forward_edge, backward_edge)
            backward_edge is None if source is the root node
        """
        # Create forward edge
        forward_edge = Edge(source_id, target_id, prev_gaze_pos, curr_gaze_pos, num_bins)
        
        # Create backward edge only if not connecting to root
        backward_edge = None
        if not is_root:
            # Create backward edge - pass the same positions but swapped
            backward_edge = Edge(target_id, source_id, curr_gaze_pos, prev_gaze_pos, num_bins)
        
        return forward_edge, backward_edge
    
    def get_features(self) -> Dict[str, Any]:
        """
        Get a dictionary of human-readable features for this edge.
        
        Returns:
            Dictionary with basic edge information
        """
        return {
            "angle": self.angle,
            "angle_degrees": AngleUtils.to_degrees(self.angle),
            "distance": self.distance,
            "prev_pos": self.prev_gaze_pos,
            "curr_pos": self.curr_gaze_pos
        }
    
    def get_features_tensor(self) -> EdgeFeature:
        """
        Get features for this edge as a tensor, with values ready for machine learning.
        
        Returns:
            Feature tensor containing:
            - Previous gaze position (x, y)
            - Current gaze position (x, y)
            - Angle bin (discretized direction)
            - Euclidean distance
        """
        # Get basic features first
        features = self.get_features()
        
        # Extract coordinates from positions
        prev_x, prev_y = features["prev_pos"]
        curr_x, curr_y = features["curr_pos"]
        
        # Combine all features into a single tensor
        return torch.tensor([
            prev_x, prev_y,          # Previous gaze position
            curr_x, curr_y,          # Current gaze position
            features["angle"],       # Discretized angle bin
            features["distance"]     # Euclidean distance
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