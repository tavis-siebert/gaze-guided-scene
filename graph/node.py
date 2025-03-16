from typing import List, Optional, Any, Tuple, Union, Set, Dict
from collections import deque
import torch

# Type aliases for better readability
VisitRecord = List[int]
NeighborInfo = List[Any]  # [Node, float, float]
NodeSet = Set['Node']
NodeList = List['Node']
FeatureList = List[Any]

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
        neighbors: List of connected nodes with edge information [neighbor_node, angle, distance]
    """
    def __init__(
        self, 
        id: int,
        object_label: str = '', 
        visits: Optional[List[VisitRecord]] = None, 
        keypoints: Optional[List[Any]] = None, 
        descriptors: Optional[List[Any]] = None, 
        neighbors: Optional[List[NeighborInfo]] = None
    ):
        """
        Initialize a new Node.
        
        Args:
            id: Unique identifier for the node
            object_label: The object class/label
            visits: List of visit periods
            keypoints: List of keypoints from feature detector
            descriptors: List of descriptors from feature detector
            neighbors: List of connected nodes with edge information
        """
        self.id = id
        self.object_label = object_label
        self.visits = [] if visits is None else visits
        self.keypoints = [] if keypoints is None else keypoints
        self.descriptors = [] if descriptors is None else descriptors
        self.neighbors = [] if neighbors is None else neighbors

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

    def add_neighbor(self, neighbor: 'Node', angle: float, distance: float) -> None:
        """
        Add a neighbor node with edge information.
        
        Args:
            neighbor: The neighboring node
            angle: The angle between this node and the neighbor
            distance: The distance between this node and the neighbor
        """
        self.neighbors.append([neighbor, angle, distance])

    def has_neighbor(self, node: 'Node') -> bool:
        """
        Check if a node is already a neighbor of this node.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node is a neighbor, False otherwise
        """
        return any(n[0] == node for n in self.neighbors)
    
    def get_neighbor_by_id(self, node_id: int) -> Optional['Node']:
        """
        Find a neighbor by its ID.
        
        Args:
            node_id: The ID of the neighbor to find
            
        Returns:
            The neighbor node if found, None otherwise
        """
        for neighbor, _, _ in self.neighbors:
            if neighbor.id == node_id:
                return neighbor
        return None
    
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
    
    def create_features(
        self,
        video_length: int,
        current_frame: int,
        relative_frame: int,
        timestamp_fraction: float,
        labels_to_int: Dict[str, int],
        num_object_classes: int
    ) -> torch.Tensor:
        """
        Create feature tensor for this node.
        
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
        # Calculate temporal features
        num_visits = len(self.visits)
        total_frames_visited = self.get_visit_duration()
        first_frame = self.get_first_visit_frame() / (video_length - current_frame + relative_frame)
        last_frame = self.get_last_visit_frame() / (video_length - current_frame + relative_frame)
        
        # Create one-hot encoding for object label
        one_hot = torch.zeros(num_object_classes)
        class_idx = labels_to_int.get(self.object_label, 0)
        one_hot[class_idx] = 1
        
        # Combine features
        return torch.cat([
            torch.tensor([total_frames_visited, num_visits, first_frame, last_frame, timestamp_fraction]),
            one_hot
        ])
    
    def update_features(
        self,
        node_data: Dict[int, torch.Tensor],
        video_length: int,
        current_frame: int,
        relative_frame: int,
        timestamp_fraction: float,
        labels_to_int: Dict[str, int],
        num_object_classes: int
    ) -> None:
        """
        Update features for this node in the node_data dictionary.
        
        Args:
            node_data: Dictionary mapping node IDs to feature tensors
            video_length: Total length of the video
            current_frame: Current frame number
            relative_frame: Relative frame number (accounting for black frames)
            timestamp_fraction: Fraction of video at current timestamp
            labels_to_int: Mapping from object labels to class indices
            num_object_classes: Number of object classes
        """
        num_visits = len(self.visits)
        total_frames_visited = self.get_visit_duration()
        first_frame = self.get_first_visit_frame() / (video_length - current_frame + relative_frame)
        last_frame = self.get_last_visit_frame() / (video_length - current_frame + relative_frame)
        
        if self.id in node_data:
            # Update existing features
            node_data[self.id][:4] = torch.tensor([
                total_frames_visited,
                num_visits,
                first_frame,
                last_frame
            ])
            node_data[self.id][4] = timestamp_fraction
        else:
            # Create new features
            one_hot = torch.zeros(num_object_classes)
            class_idx = labels_to_int.get(self.object_label, 0)
            one_hot[class_idx] = 1
            
            node_data[self.id] = torch.cat([
                torch.tensor([total_frames_visited, num_visits, first_frame, last_frame, timestamp_fraction]),
                one_hot
            ])
    
    @staticmethod
    def normalize_features(node_features: torch.Tensor, relative_frame: int, timestamp_fraction: float) -> torch.Tensor:
        """
        Normalize node features.
        
        Args:
            node_features: Tensor of node features
            relative_frame: Relative frame number for normalization
            timestamp_fraction: Fraction of video at current timestamp
            
        Returns:
            Normalized node features
        """
        normalized = node_features.clone()
        
        # Normalize visit duration by relative frame number
        normalized[:, 0] /= relative_frame
        
        # Normalize number of visits by maximum value
        if normalized[:, 1].max() > 0:
            normalized[:, 1] /= normalized[:, 1].max()
        
        # Set timestamp fraction
        normalized[:, 4] = timestamp_fraction
        
        return normalized
    
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
        for neighbor, _, _ in node.neighbors:
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