from typing import List, Optional, Any, Tuple, Union

# Type aliases for better readability
VisitRecord = List[int]
NeighborInfo = List[Any]  # [Node, float, float]

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