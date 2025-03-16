import cv2
import torch
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Set, Dict, Any
from egtea_gaze.utils import resolution

from graph.node import Node

# Type aliases for better readability
NodeSet = Set[Node]
NodeList = List[Node]
FeatureList = List[Any]
VisitRecord = List[int]
Position = Tuple[int, int]
EdgeFeature = torch.Tensor
EdgeIndex = List[List[int]]

class GraphTraversal:
    """Utilities for traversing and exploring graph structures."""
    
    @staticmethod
    def dfs(start_node: Node) -> NodeSet:
        """Depth-first search traversal of a graph starting from a given node."""
        visited = set()
        stack = [start_node]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor, _, _ in node.neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)

        return visited

    @staticmethod
    def get_all_nodes(start_node: Node, mode: str = 'dfs') -> NodeList:
        """Returns all nodes in the graph using the specified traversal method."""
        if mode == 'dfs':
            return list(GraphTraversal.dfs(start_node))
        elif mode == 'bfs':
            raise NotImplementedError("BFS traversal not yet implemented")
        else:
            raise ValueError(f"Unknown traversal mode: {mode}")


class AngleUtils:
    """Utilities for angle calculations and conversions."""
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to be in range [0, 2π)."""
        return angle + 2*np.pi if angle < 0 else angle

    @staticmethod
    def get_angle_bin(x: float, y: float, num_bins: int = 8) -> float:
        """
        Convert a vector (x,y) to an angle and assign it to a discrete bin.
        
        Args:
            x: X component of the vector
            y: Y component of the vector
            num_bins: Number of angular bins (use -1 for raw angle)
            
        Returns:
            Binned angle in radians or raw angle if num_bins is -1
        """
        theta = AngleUtils.normalize_angle(np.arctan2(y, x))
        
        if num_bins == -1:  # Return raw angle
            return theta

        bin_width = 2 * np.pi / num_bins
        bin_index = round(theta / bin_width)
        bins = [i * bin_width for i in range(num_bins)]
        
        return bins[bin_index % num_bins]
    
    @staticmethod
    def get_opposite_angle(angle: float) -> float:
        """Calculate the opposite angle (180° rotation)."""
        return angle + np.pi if angle < np.pi else angle - np.pi
    
    @staticmethod
    def to_degrees(angle: float) -> float:
        """Convert angle from radians to degrees."""
        return angle * 180 / np.pi


class FeatureMatcher:
    """Utilities for matching features between images."""
    
    @staticmethod
    def match_features(des1: np.ndarray, des2: np.ndarray) -> List:
        """Match descriptors between two sets of features using brute force matching."""
        if des1 is None or des2 is None:
            return []
            
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)

    @staticmethod
    def compute_homography(
        kp1: List, 
        kp2: List, 
        matches: List, 
        ransac_threshold: float = 7.0
    ) -> Tuple[Optional[np.ndarray], List]:
        """
        Compute homography between two sets of keypoints using RANSAC.
        
        Args:
            kp1: First set of keypoints
            kp2: Second set of keypoints
            matches: List of matches between keypoints
            ransac_threshold: Threshold for RANSAC algorithm
            
        Returns:
            Tuple containing homography matrix and list of inlier matches
        """
        if len(matches) < 10:
            return None, []

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]

        return H, inliers
    
    @staticmethod
    def calculate_inlier_ratio(inliers: List, matches: List) -> float:
        """Calculate the ratio of inliers to total matches."""
        if not matches:
            return 0.0
        return len(inliers) / len(matches)


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


class EdgeManager:
    """Utilities for managing edges in the scene graph."""
    
    @staticmethod
    def calculate_edge_features(
        prev_pos: Position, 
        curr_pos: Position, 
        num_bins: int
    ) -> Tuple[float, float]:
        """
        Calculate edge features between two positions.
        
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
    def normalize_positions(
        prev_pos: Position, 
        curr_pos: Position
    ) -> Tuple[float, float, float, float]:
        """Normalize positions by resolution for edge features."""
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos
        
        return (
            prev_x / resolution[0], 
            prev_y / resolution[1], 
            curr_x / resolution[0], 
            curr_y / resolution[1]
        )
    
    @staticmethod
    def create_edge_feature(
        prev_x: float, 
        prev_y: float, 
        curr_x: float, 
        curr_y: float
    ) -> EdgeFeature:
        """Create an edge feature tensor from normalized positions."""
        return torch.tensor([prev_x, prev_y, curr_x, curr_y])
    
    @staticmethod
    def update_edge_indices(
        edge_index: EdgeIndex, 
        source_id: int, 
        target_id: int
    ) -> None:
        """Update edge indices for a bidirectional edge."""
        edge_index[0].extend([source_id, target_id])
        edge_index[1].extend([target_id, source_id])

    @staticmethod
    def add_bidirectional_edge(
        curr_node: Node, 
        next_node: Node, 
        angle: float, 
        distance: float,
        edge_data: List[EdgeFeature],
        edge_index: EdgeIndex,
        prev_pos: Position,
        curr_pos: Position
    ) -> None:
        """
        Add bidirectional edges between two nodes with appropriate features.
        
        Args:
            curr_node: Source node
            next_node: Target node
            angle: Angle between nodes
            distance: Distance between nodes
            edge_data: List to store edge features
            edge_index: List to store edge indices
            prev_pos: Previous position (x,y)
            curr_pos: Current position (x,y)
        """
        # Add forward edge
        curr_node.add_neighbor(next_node, angle, distance)
        
        # Add backward edge only if not connecting to root
        if curr_node.object_label != 'root':
            # Calculate opposite angle
            opposite_angle = AngleUtils.get_opposite_angle(angle)
            next_node.add_neighbor(curr_node, opposite_angle, distance)
            
            # Normalize positions and create edge feature
            norm_prev_x, norm_prev_y, norm_curr_x, norm_curr_y = EdgeManager.normalize_positions(prev_pos, curr_pos)
            edge_feature = EdgeManager.create_edge_feature(norm_prev_x, norm_prev_y, norm_curr_x, norm_curr_y)
            
            # Add edge features for both directions (same feature for both)
            edge_data.append(edge_feature)
            edge_data.append(edge_feature)
            
            # Update edge indices
            EdgeManager.update_edge_indices(edge_index, curr_node.id, next_node.id)


class GraphBuilder:
    """Main utilities for building and updating the scene graph."""
    
    @staticmethod
    def update_graph(
        curr_node: Node, 
        label_counts: Dict[str, int], 
        visit: VisitRecord, 
        keypoints: FeatureList, 
        descriptors: FeatureList, 
        prev_gaze_pos: Position, 
        curr_gaze_pos: Position, 
        edge_data: List[EdgeFeature],
        edge_index: EdgeIndex,
        num_nodes: List[int], 
        num_bins: int = 8, 
        inlier_thresh: float = 0.3
    ) -> Node:
        """
        Update the scene graph with a new observation.
        
        This function either finds a matching existing node or creates a new one,
        then connects it to the current node in the graph.
        
        Args:
            curr_node: Current node in the graph
            label_counts: Dictionary of object labels and their counts
            visit: First and last frame of the object fixation
            keypoints: SIFT keypoints for the object
            descriptors: SIFT descriptors for the object
            prev_gaze_pos: Previous gaze position (x,y)
            curr_gaze_pos: Current gaze position (x,y)
            edge_data: List to store edge features
            edge_index: List to store edge indices
            num_nodes: List containing the current node count
            num_bins: Number of angle bins
            inlier_thresh: Inlier threshold for RANSAC
            
        Returns:
            The next node (either existing or newly created)
        """
        # Calculate edge features
        angle, distance = EdgeManager.calculate_edge_features(prev_gaze_pos, curr_gaze_pos, num_bins)
        
        # Find most likely object label
        most_likely_label = max(label_counts, key=label_counts.get)
        
        # Try to find matching node
        matching_node = NodeManager.find_matching_node(
            curr_node, keypoints, descriptors, most_likely_label, inlier_thresh
        )
        next_node = NodeManager.merge_node(visit, matching_node)
        
        # Create new node if no match found
        if next_node is None:
            next_node = NodeManager.create_node(num_nodes[0], most_likely_label, visit, keypoints, descriptors)
            num_nodes[0] += 1
        
        # Connect nodes if not already connected and not self-loop
        if next_node != curr_node and not curr_node.has_neighbor(next_node):
            EdgeManager.add_bidirectional_edge(
                curr_node, 
                next_node, 
                angle, 
                distance, 
                edge_data, 
                edge_index,
                prev_gaze_pos,
                curr_gaze_pos
            )
        
        return next_node


class GraphVisualizer:
    """Utilities for visualizing graph structures."""
    
    @staticmethod
    def format_node_info(node: Node, prev_obj: str, theta: Any, use_degrees: bool) -> Dict[str, Any]:
        """Format node information for display."""
        angle = theta
        if isinstance(theta, float) and use_degrees:
            angle = AngleUtils.to_degrees(theta)
            
        return {
            'object': node.object_label,
            'visits': node.visits,
            'from': prev_obj,
            'angle': angle
        }
    
    @staticmethod
    def print_node_info(node_info: Dict[str, Any]) -> None:
        """Print formatted node information."""
        print('-----------------')
        print(f'Object: {node_info["object"]}')
        print(f'Visited at: {node_info["visits"]}')
        print(f'Visited from: {node_info["from"]}')
        print(f'Angle from prev: {node_info["angle"]}')
    
    @staticmethod
    def print_levels(start_node: Node, use_degrees: bool = True) -> None:
        """
        Print graph structure by levels, showing node relationships.
        
        Args:
            start_node: Root node to start traversal from
            use_degrees: Whether to display angles in degrees (True) or radians (False)
        """
        visited = set([start_node])
        queue = deque([(start_node, 'none', 'none')])
        
        curr_depth = 0
        while queue:
            level_size = len(queue)
            print(f'Depth: {curr_depth}')
            
            for _ in range(level_size):
                node, prev_obj, theta = queue.popleft()
                node_info = GraphVisualizer.format_node_info(node, prev_obj, theta, use_degrees)
                GraphVisualizer.print_node_info(node_info)
                
                GraphVisualizer._queue_unvisited_neighbors(node, visited, queue)
                    
            print('================')
            curr_depth += 1
    
    @staticmethod
    def _queue_unvisited_neighbors(node: Node, visited: NodeSet, queue: deque) -> None:
        """Add unvisited neighbors to the visualization queue."""
        for neighbor, angle, _ in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, node.object_label, angle))


# Expose main functions at module level for backward compatibility
def get_all_nodes(start_node: Node, mode: str = 'dfs') -> NodeList:
    """Returns all nodes in the graph using the specified traversal method."""
    return GraphTraversal.get_all_nodes(start_node, mode)

def get_angle_bin(x: float, y: float, num_bins: int = 8) -> float:
    """Convert a vector (x,y) to an angle and assign it to a discrete bin."""
    return AngleUtils.get_angle_bin(x, y, num_bins)

def update_graph(
    curr_node: Node, 
    label_counts: Dict[str, int], 
    visit: VisitRecord, 
    keypoints: FeatureList, 
    descriptors: FeatureList, 
    prev_gaze_pos: Position, 
    curr_gaze_pos: Position, 
    edge_data: List[EdgeFeature],
    edge_index: EdgeIndex,
    num_nodes: List[int], 
    num_bins: int = 8, 
    inlier_thresh: float = 0.3
) -> Node:
    """Update the scene graph with a new observation."""
    return GraphBuilder.update_graph(
        curr_node, label_counts, visit, keypoints, descriptors,
        prev_gaze_pos, curr_gaze_pos, edge_data, edge_index,
        num_nodes, num_bins, inlier_thresh
    )

def print_levels(start_node: Node, use_degrees: bool = True) -> None:
    """Print graph structure by levels, showing node relationships."""
    GraphVisualizer.print_levels(start_node, use_degrees)