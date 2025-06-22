from typing import List, Optional, Any, Set, Dict
from collections import deque
import torch

VisitRecord = List[int]
NodeSet = Set["Node"]
NodeList = List["Node"]


class Node:
    """
    Represents a node in the scene graph.

    Each node corresponds to an object in the scene and maintains information about
    when it was visited.

    Attributes:
        id: Unique identifier for the node
        object_label: The object class/label (e.g., "cup")
        visits: List of visit periods, each containing [start_frame, end_frame]
    """

    def __init__(
        self,
        id: int,
        object_label: str = "",
        visits: Optional[List[VisitRecord]] = None,
    ):
        """
        Initialize a new Node.

        Args:
            id: Unique identifier for the node
            object_label: The object class/label
            visits: List of visit periods
        """
        self.id = id
        self.object_label = object_label
        self.visits = [] if visits is None else visits

    @staticmethod
    def create(node_id: int, label: str, visit: VisitRecord) -> "Node":
        """
        Create a new node with the given properties.

        Args:
            node_id: Unique identifier for the node
            label: The object class/label
            visit: The visit period [start_frame, end_frame]

        Returns:
            The newly created node
        """
        return Node(id=node_id, object_label=label, visits=[visit])

    @staticmethod
    def find_matching(
        graph: "Graph", curr_node: "Node", label: str
    ) -> Optional["Node"]:
        """
        Find a matching node in the graph based on object label.

        Args:
            graph: The Graph instance to search in
            curr_node: Current node to start search from
            label: Object label to match

        Returns:
            Matching node if found, None otherwise
        """
        if curr_node.object_label == "root":
            return None

        visited = set([curr_node.id])
        queue = deque([curr_node.id])

        while queue:
            node_id = queue.popleft()
            node = graph.get_node_by_id(node_id)

            if not node:
                continue

            if node.object_label == label:
                return node

            for neighbor_id in graph.get_node_neighbors(node_id):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)

        return None

    def add_new_visit(self, visit: VisitRecord) -> None:
        """
        Add a new visit period to this node.

        Args:
            visit: A list containing [start_frame, end_frame]
        """
        self.visits.append(visit)

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

    def get_features(self) -> Dict[str, Any]:
        """
        Get a dictionary of human-readable features for this node.

        Returns:
            Dictionary with basic node information
        """
        return {
            "total_frames_visited": self.get_visit_duration(),
            "num_visits": len(self.visits),
            "first_visit_frame": self.get_first_visit_frame(),
            "last_visit_frame": self.get_last_visit_frame(),
            "object_label": self.object_label,
        }

    def get_feature_tensor(
        self,
        video_length: int,
        current_frame: int,
        non_black_frame_count: int,
        timestamp_fraction: float,
        object_label_to_id: Dict[str, int],
    ) -> torch.Tensor:
        """
        Get features for this node as a tensor, with normalization for machine learning.

        Args:
            video_length: Total length of the video
            current_frame: Current frame number
            non_black_frame_count: Number of non-black frames processed
            timestamp_fraction: Fraction of video at current timestamp
            object_label_to_id: Mapping from object labels to class indices
            num_object_classes: Number of object classes

        Returns:
            Feature tensor for the node
        """
        features = self.get_features()

        first_frame = features["first_visit_frame"]
        last_frame = features["last_visit_frame"]
        first_frame_normalized = (
            first_frame / non_black_frame_count if first_frame else 0
        )
        last_frame_normalized = last_frame / non_black_frame_count if last_frame else 0

        temporal_features = torch.tensor(
            [
                features["total_frames_visited"],
                features["num_visits"],
                first_frame_normalized,
                last_frame_normalized,
                timestamp_fraction,
            ]
        )

        class_idx = object_label_to_id.get(features["object_label"], 0)
        one_hot = torch.zeros(len(object_label_to_id))
        one_hot[class_idx] = 1

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
