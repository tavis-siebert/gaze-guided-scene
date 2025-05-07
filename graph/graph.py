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
from datasets.egtea_gaze.utils import resolution
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
    
    def update(
        self, 
        frame_number: int,
        fixated_object: str,
        visit: VisitRecord, 
        prev_gaze_pos: GazePosition, 
        curr_gaze_pos: GazePosition, 
        num_bins: int = 8
    ) -> Node:
        """Update the graph with a new observation.
        
        Args:
            fixated_object: The fixated object label
            visit: First and last frame of the object fixation
            prev_gaze_pos: Previous gaze position (x,y)
            curr_gaze_pos: Current gaze position (x,y)
            num_bins: Number of angle bins
            
        Returns:
            The node representing the fixated object (either existing or newly created)
        """
        prev_node_id = self.current_node.id
        
        # Find matching node or create a new one
        matching_node = self._find_matching_node(fixated_object)
        
        if matching_node:
            # Update existing node
            matching_node.add_new_visit(visit)
            next_node = matching_node
            
            # Log node update
            self.tracer.log_node_updated(
                frame_number,
                next_node.id,
                next_node.object_label,
                next_node.get_features(),
                visit
            )
            logger.info(f"Node {next_node.id} updated with new visit at frames {visit}")
        else:
            # Create new node
            next_node = self.add_node(fixated_object, visit)
            
            # Log node addition
            self.tracer.log_node_added(
                frame_number, 
                next_node.id, 
                next_node.object_label, 
                next_node.get_features()
            )
            logger.info(f"New node {next_node.id} created for object '{fixated_object}'")
        
        # Add edge if needed (node changed and no existing connection)
        if next_node != self.current_node and not self.has_neighbor(self.current_node.id, next_node.id):
            forward_edge, backward_edge = self.add_edge(
                self.current_node,
                next_node,
                prev_gaze_pos,
                curr_gaze_pos,
                num_bins
            )
            
            # Log edge addition
            if forward_edge and prev_node_id >= 0:
                self.tracer.log_edge_added(
                    frame_number, 
                    prev_node_id, 
                    next_node.id, 
                    "saccade", 
                    forward_edge.get_features()
                )
                logger.info(f"Edge added from node {prev_node_id} to node {next_node.id}")
        
        # Update current node reference
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