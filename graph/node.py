from typing import List, Optional, Any, Tuple, Union, Set, Dict
from collections import deque
import torch

# Type aliases for better readability
VisitRecord = List[int]
NodeSet = Set['Node']
NodeList = List['Node']
FeatureList = List[Any]
EdgeList = List['Edge']  # Forward reference to Edge class

class Node:
    """
    Represents a node in the scene graph.
    
    Each node corresponds to an object in the scene and maintains information about
    when it was visited, its visual features, and connections to other nodes.
    
    Attributes:
        id: Unique identifier for the node
        object_label: The object class/label (e.g., "cup")
        visits: List of visit periods, each containing [start_frame, end_frame]
        keypoints: List of keypoints per frame returned by feature detector (e.g., SIFT)
        descriptors: List of descriptors per frame returned by feature detector
        outgoing_edges: List of outgoing Edge objects from this node
    """
    def __init__(
        self, 
        id: int,
        object_label: str = '', 
        visits: Optional[List[VisitRecord]] = None, 
        keypoints: Optional[FeatureList] = None, 
        descriptors: Optional[FeatureList] = None
    ):
        """
        Initialize a new Node.
        
        Args:
            id: Unique identifier for the node
            object_label: The object class/label
            visits: List of visit periods
            keypoints: List of keypoints from feature detector
            descriptors: List of descriptors from feature detector
        """
        self.id = id
        self.object_label = object_label
        self.visits = [] if visits is None else visits
        self.keypoints = [] if keypoints is None else keypoints
        self.descriptors = [] if descriptors is None else descriptors
        self.outgoing_edges: EdgeList = []

    def set_object_label(self, label: str) -> None:
        """Set the object label for this node."""
        self.object_label = label

    def add_new_visit(self, visit: VisitRecord) -> None:
        """
        Add a new visit period to this node.
        
        Args:
            visit: A list containing [start_frame, end_frame]
        """
        self.visits.append(visit)
    
    def add_new_feature(self, keypoint: Any, descriptor: Any) -> None:
        """
        Add a new feature (keypoint and descriptor) to this node.
        
        Args:
            keypoint: Keypoint from feature detector
            descriptor: Descriptor from feature detector
        """
        self.keypoints.append(keypoint)
        self.descriptors.append(descriptor)

    def add_edge(self, edge: 'Edge') -> None:
        """
        Add an outgoing edge from this node.
        
        Args:
            edge: The Edge object to add
        """
        self.outgoing_edges.append(edge)

    def has_neighbor(self, node: 'Node') -> bool:
        """
        Check if a node is already a neighbor of this node.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node is a neighbor, False otherwise
        """
        return any(edge.target == node for edge in self.outgoing_edges)
    
    def get_neighbor_by_id(self, node_id: int) -> Optional['Node']:
        """
        Find a neighbor by its ID.
        
        Args:
            node_id: The ID of the neighbor to find
            
        Returns:
            The neighbor node if found, None otherwise
        """
        for edge in self.outgoing_edges:
            if edge.target.id == node_id:
                return edge.target
        return None
    
    def get_edge_to_neighbor(self, node: 'Node') -> Optional['Edge']:
        """
        Find the edge connecting to a specific neighbor.
        
        Args:
            node: The neighbor node
            
        Returns:
            The edge if found, None otherwise
        """
        for edge in self.outgoing_edges:
            if edge.target == node:
                return edge
        return None
    
    def get_neighbors(self) -> List['Node']:
        """
        Get all neighbor nodes.
        
        Returns:
            List of all neighbor nodes
        """
        return [edge.target for edge in self.outgoing_edges]
    
    def get_visit_duration(self) -> int:
        """
        Calculate the total number of frames this node was visited.
        
        Returns:
            Total number of frames visited
        """
        if not self.visits:
            return 0
        return sum(visit[1] - visit[0] + 1 for visit in self.visits)
    
    def get_first_visit_frame(self) -> Optional[int]:
        """
        Get the first frame this node was visited.
        
        Returns:
            First frame number or None if never visited
        """
        if not self.visits:
            return None
        return self.visits[0][0]
    
    def get_last_visit_frame(self) -> Optional[int]:
        """
        Get the last frame this node was visited.
        
        Returns:
            Last frame number or None if never visited
        """
        if not self.visits:
            return None
        return self.visits[-1][1]
    
    def get_features(
        self,
        video_length: int,
        current_frame: int,
        relative_frame: int,
        timestamp_fraction: float,
        labels_to_int: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Get a dictionary of human-readable features for this node.
        
        Args:
            video_length: Total length of the video
            current_frame: Current frame number
            relative_frame: Relative frame number (accounting for black frames)
            timestamp_fraction: Fraction of video at current timestamp
            labels_to_int: Mapping from object labels to class indices
            
        Returns:
            Dictionary with feature keys and values
        """
        # Calculate temporal features
        total_frames_visited = self.get_visit_duration()
        num_visits = len(self.visits)
        
        # Normalize frame positions
        normalization_factor = video_length - current_frame + relative_frame
        first_frame_normalized = self.get_first_visit_frame() / normalization_factor if self.get_first_visit_frame() else 0
        last_frame_normalized = self.get_last_visit_frame() / normalization_factor if self.get_last_visit_frame() else 0
        
        # Get class index for object label
        class_idx = labels_to_int.get(self.object_label, 0)
        
        return {
            "total_frames_visited": total_frames_visited,
            "num_visits": num_visits,
            "first_frame_normalized": first_frame_normalized,
            "last_frame_normalized": last_frame_normalized,
            "timestamp_fraction": timestamp_fraction,
            "object_class_idx": class_idx,
            "object_label": self.object_label
        }
    
    def get_features_tensor(
        self,
        video_length: int,
        current_frame: int,
        relative_frame: int,
        timestamp_fraction: float,
        labels_to_int: Dict[str, int],
        num_object_classes: int
    ) -> torch.Tensor:
        """
        Get features for this node as a tensor.
        
        Args:
            video_length: Total length of the video
            current_frame: Current frame number
            relative_frame: Relative frame number (accounting for black frames)
            timestamp_fraction: Fraction of video at current timestamp
            labels_to_int: Mapping from object labels to class indices
            num_object_classes: Number of object classes
            
        Returns:
            Feature tensor for the node
        """
        features = self.get_features(video_length, current_frame, relative_frame, timestamp_fraction, labels_to_int)
        
        # Create one-hot encoding for object label
        one_hot = torch.zeros(num_object_classes)
        one_hot[features["object_class_idx"]] = 1
        
        # Create temporal features tensor
        temporal_features = torch.tensor([
            features["total_frames_visited"], 
            features["num_visits"],
            features["first_frame_normalized"], 
            features["last_frame_normalized"],
            features["timestamp_fraction"]
        ])
        
        # Combine features
        return torch.cat([temporal_features, one_hot])
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two nodes are equal based on their ID.
        
        Args:
            other: The other node to compare with
            
        Returns:
            True if the nodes have the same ID, False otherwise
        """
        if not isinstance(other, Node):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """
        Hash function for Node based on ID.
        
        Returns:
            Hash value
        """
        return hash(self.id)
    
    def __repr__(self) -> str:
        """
        String representation of the node.
        
        Returns:
            String representation
        """
        return f"Node(id={self.id}, label='{self.object_label}', visits={len(self.visits)})"


class NodeManager:
    """Utilities for managing nodes in the scene graph."""
    
    @staticmethod
    def find_matching_node(
        curr_node: Node, 
        keypoints: FeatureList, 
        descriptors: FeatureList, 
        label: str, 
        inlier_thresh: float, 
        one_label_assumption: bool = True
    ) -> Optional[Node]:
        """
        Find a matching node in the graph for the given keypoints and label.
        
        Args:
            curr_node: Current node to start search from
            keypoints: List of keypoints for the potential new node
            descriptors: List of descriptors for the potential new node
            label: Object label to match
            inlier_thresh: Minimum inlier ratio threshold for feature matching
            one_label_assumption: If True, assumes only one instance of each object class exists
            
        Returns:
            Matching node if found, None otherwise
        """
        from graph.utils import FeatureMatcher
        
        if curr_node.object_label == 'root':
            return None

        visited = set([curr_node])
        queue = deque([curr_node])
        
        # Use first frame's features for matching
        kp1, des1 = keypoints[0], descriptors[0]
        
        while queue:
            node = queue.popleft()

            if node.object_label == label:
                # If we assume only one instance of each object class, return first match
                if one_label_assumption:
                    return node
                    
                # Otherwise, verify match with feature comparison
                kp2, des2 = node.keypoints[0], node.descriptors[0]
                
                matches = FeatureMatcher.match_features(des1, des2)
                if not matches:
                    continue
                    
                H, inliers = FeatureMatcher.compute_homography(kp1, kp2, matches)
                inlier_ratio = FeatureMatcher.calculate_inlier_ratio(inliers, matches)
                
                if H is not None and inlier_ratio > inlier_thresh:
                    return node

            # Continue BFS
            NodeManager._add_unvisited_neighbors_to_queue(node, visited, queue)

        return None
    
    @staticmethod
    def _add_unvisited_neighbors_to_queue(node: Node, visited: NodeSet, queue: deque) -> None:
        """Add unvisited neighbors to the BFS queue."""
        for edge in node.outgoing_edges:
            neighbor = edge.target
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    @staticmethod
    def merge_node(visit: VisitRecord, matching_node: Optional[Node]) -> Optional[Node]:
        """Add a new visit to an existing node if a match is found."""
        if matching_node is not None:
            matching_node.add_new_visit(visit)
            return matching_node
        return None

    @staticmethod
    def create_node(
        node_id: int, 
        label: str, 
        visit: VisitRecord, 
        keypoints: FeatureList, 
        descriptors: FeatureList
    ) -> Node:
        """Create a new node with the given properties."""
        return Node(
            id=node_id,
            object_label=label,
            visits=[visit],
            keypoints=keypoints,
            descriptors=descriptors
        )