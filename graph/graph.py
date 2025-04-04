from typing import Dict, List, Optional, Set, Tuple, Any
import torch
import numpy as np
from collections import deque, defaultdict
import math
from typing import TYPE_CHECKING

from graph.node import Node, VisitRecord
from graph.edge import Edge
from graph.utils import AngleUtils, GraphTraversal
from graph.visualizer import GraphVisualizer
from egtea_gaze.utils import resolution
from logger import get_logger

if TYPE_CHECKING:
    from graph.checkpoint_manager import GraphCheckpoint

GazePosition = Tuple[int, int]
EdgeFeature = torch.Tensor
EdgeIndex = List[List[int]]
NodeId = int
EdgeId = Tuple[NodeId, NodeId]

logger = get_logger(__name__)

class Graph:
    """A scene graph representing objects and their spatial relationships."""
    
    def __init__(self, labels_to_int: Dict[str, int] = None, num_object_classes: int = 0, video_length: int = 0):
        """Initialize an empty graph with a root node.
        
        Args:
            labels_to_int: Mapping from object labels to class indices
            num_object_classes: Number of object classes
            video_length: Total length of the video
        """
        self.root = Node(id=-1, object_label='root')
        self.current_node = self.root
        self.nodes: Dict[NodeId, Node] = {-1: self.root}
        self.num_nodes = 0
        self.edges: List[Edge] = []
        self.adjacency = defaultdict(list)
        self.tracer = None
        self.checkpoints: List["GraphCheckpoint"] = []
        
        self.labels_to_int = labels_to_int or {}
        self.num_object_classes = num_object_classes
        self.video_length = video_length
        
    def get_all_nodes(self) -> List[Node]:
        """Get all nodes in the graph including the root.
        
        Returns:
            List of all nodes in the graph
        """
        return list(self.nodes.values())
        
    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """Find a node by its ID.
        
        Args:
            node_id: The ID of the node to find
            
        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(node_id)
    
    def add_node(self, label: str, visit: VisitRecord) -> Node:
        """Add a new node to the graph.
        
        Args:
            label: The object label
            visit: The visit period [start_frame, end_frame]
            
        Returns:
            The newly created node
        """
        node = Node.create(
            node_id=self.num_nodes,
            label=label,
            visit=visit
        )
        self.nodes[node.id] = node
        self.num_nodes += 1
        return node
    
    def has_neighbor(self, source_id: NodeId, target_id: NodeId) -> bool:
        """Check if two nodes are connected.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            True if the nodes are connected, False otherwise
        """
        return target_id in self.adjacency[source_id]
    
    def get_node_neighbors(self, node_id: NodeId) -> List[NodeId]:
        """Get all neighbors of a node.
        
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
        """Add an edge between two nodes.
        
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
            is_root = source_node.id == self.root.id
            
            forward_edge, backward_edge = Edge.create_bidirectional_edges(
                source_id=source_node.id,
                target_id=target_node.id,
                is_root=is_root,
                prev_gaze_pos=prev_gaze_pos,
                curr_gaze_pos=curr_gaze_pos,
                num_bins=num_bins
            )
            
            self.edges.append(forward_edge)
            self.adjacency[source_node.id].append(target_node.id)
            
            if backward_edge:
                self.edges.append(backward_edge)
                self.adjacency[target_node.id].append(source_node.id)
                
            return forward_edge, backward_edge
        
        return None, None
    
    def update_graph(
        self, 
        label_counts: Dict[str, int], 
        visit: VisitRecord, 
        prev_gaze_pos: GazePosition, 
        curr_gaze_pos: GazePosition, 
        num_bins: int = 8
    ) -> Node:
        """Update the graph with a new observation.
        
        Args:
            label_counts: Dictionary of object labels and their counts
            visit: First and last frame of the object fixation
            prev_gaze_pos: Previous gaze position (x,y)
            curr_gaze_pos: Current gaze position (x,y)
            num_bins: Number of angle bins
            
        Returns:
            The next node (either existing or newly created)
        """
        most_likely_label = max(label_counts, key=label_counts.get)
        
        matching_node = self._find_matching_node(most_likely_label)
        next_node = Node.merge(visit, matching_node)
        
        if next_node:
            frame_number = visit[1]
            self.tracer.log_node_updated(
                frame_number,
                next_node.id,
                next_node.object_label,
                next_node.get_features(),
                visit
            )
            logger.info(f"Node {next_node.id} updated with new visit at frames {visit}")
        else:
            next_node = self.add_node(most_likely_label, visit)
        
        if (next_node != self.current_node and 
            not self.has_neighbor(self.current_node.id, next_node.id)):
            self.add_edge(
                self.current_node,
                next_node,
                prev_gaze_pos,
                curr_gaze_pos,
                num_bins
            )
        
        self.current_node = next_node
        return next_node
    
    def _find_matching_node(self, label: str) -> Optional[Node]:
        """Find a matching node by searching from current node.
        
        Args:
            label: Object label to match
            
        Returns:
            Matching node if found, None otherwise
        """
        return Node.find_matching(
            graph=self,
            curr_node=self.current_node, 
            label=label
        )
    
    def print_graph(self, use_degrees: bool = True) -> None:
        """Print the graph structure.
        
        Args:
            use_degrees: Whether to display angles in degrees (True) or radians (False)
        """
        if self.num_nodes == 0:
            logger.info("Graph is empty.")
            return
            
        logger.info(f"Graph with {self.num_nodes} nodes:")
        GraphVisualizer.print_levels(self.root, use_degrees, self.edges, self)
    
    def _get_node_features(
        self,
        current_frame: int,
        non_black_frame_count: int,
        timestamps: List[int],
        timestamp_ratios: List[float],
        gaze_data_length: int
    ) -> torch.Tensor:
        """Extract node features as a tensor.
        
        Args:
            current_frame: Current frame number
            non_black_frame_count: Number of non-black frames processed
            timestamps: List of predefined checkpoint frame numbers
            timestamp_ratios: Corresponding ratios for each timestamp
            gaze_data_length: Length of gaze data
            
        Returns:
            Tensor of node features
        """
        timestamp_fraction = self._calculate_timestamp_fraction(
            current_frame, 
            gaze_data_length, 
            timestamps, 
            timestamp_ratios, 
            self.video_length
        )
        
        nodes = []
        for node in self.nodes.values():
            if node.id >= 0:
                features_tensor = node.get_feature_tensor(
                    self.video_length,
                    current_frame,
                    non_black_frame_count,
                    timestamp_fraction,
                    self.labels_to_int,
                    self.num_object_classes
                )
                nodes.append(features_tensor)
        
        if not nodes:
            return torch.tensor([])
        
        node_features = torch.stack(nodes)
        
        node_features[:, 0] /= non_black_frame_count
        
        if node_features[:, 1].max() > 0:
            node_features[:, 1] /= node_features[:, 1].max()
            
        return node_features
    
    def _get_edge_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract edge indices and attributes as tensors.
        
        Returns:
            Tuple of (edge_indices, edge_attributes)
        """
        edge_index = [[], []]
        edge_attrs = []
        
        for edge in self.edges:
            if edge.source_id < 0 or edge.target_id < 0:
                continue
                
            edge_index[0].append(edge.source_id)
            edge_index[1].append(edge.target_id)
            
            edge_attrs.append(edge.get_feature_tensor())
        
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) if edge_index[0] else torch.tensor([[],[]], dtype=torch.long)
        edge_attr_tensor = torch.stack(edge_attrs) if edge_attrs else torch.tensor([])
            
        return edge_index_tensor, edge_attr_tensor
    
    def get_feature_tensor(
        self,
        video_length: int,
        current_frame: int,
        non_black_frame_count: int,
        timestamps: List[int],
        timestamp_ratios: List[float],
        gaze_data_length: int,
        labels_to_int: Dict[str, int],
        num_object_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get graph features as tensors for model input.
        
        Args:
            video_length: Total length of the video
            current_frame: Current frame number
            non_black_frame_count: Number of non-black frames processed
            timestamps: List of predefined checkpoint frame numbers
            timestamp_ratios: Corresponding ratios for each timestamp
            gaze_data_length: Length of gaze data
            labels_to_int: Mapping from object labels to class indices
            num_object_classes: Number of object classes
            
        Returns:
            Tuple of (node_features, edge_indices, edge_features)
        """
        node_features = self._get_node_features(
            current_frame, 
            non_black_frame_count,
            timestamps,
            timestamp_ratios,
            gaze_data_length
        )
        edge_index, edge_attr = self._get_edge_features()
            
        return node_features, edge_index, edge_attr
    
    def _calculate_timestamp_fraction(
        self, 
        frame_num: int, 
        gaze_data_length: int, 
        timestamps: List[int], 
        timestamp_ratios: List[float], 
        video_length: int
    ) -> float:
        """Calculate the timestamp fraction for the current frame.
        
        Args:
            frame_num: Current frame number
            gaze_data_length: Length of gaze data
            timestamps: List of predefined checkpoint frame numbers
            timestamp_ratios: Corresponding ratios for each timestamp
            video_length: Total video length
            
        Returns:
            Fraction of video completed at current timestamp
        """
        if frame_num < gaze_data_length:
            timestamp_idx = timestamps.index(frame_num) if frame_num in timestamps else -1
            return timestamp_ratios[timestamp_idx] if timestamp_idx >= 0 else frame_num / video_length
        return frame_num / video_length
    
    def get_edge(self, source_id: NodeId, target_id: NodeId) -> Optional[Edge]:
        """Get an edge between two nodes if it exists.
        
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
        
    def get_checkpoints(self) -> List["GraphCheckpoint"]:
        """Get all saved checkpoints.
        
        Returns:
            List of GraphCheckpoint objects
        """
        return self.checkpoints