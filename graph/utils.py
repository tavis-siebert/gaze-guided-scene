import cv2
import torch
import numpy as np
from collections import deque
from egtea_gaze.utils import resolution

from graph.node import Node

""" 
Helper functions for all things graph-related
"""

# Graph traversal algorithms
def dfs(start_node: Node) -> set[Node]:
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

def get_all_nodes(start_node: Node, mode: str='dfs') -> list[Node]:
    """Returns all nodes in the graph using the specified traversal method."""
    if mode == 'dfs':
        return list(dfs(start_node))
    elif mode == 'bfs':
        raise NotImplementedError("BFS traversal not yet implemented")
    else:
        raise ValueError(f"Unknown traversal mode: {mode}")

# Angle and position utilities
def normalize_angle(angle: float) -> float:
    """Normalize angle to be in range [0, 2π)."""
    return angle + 2*np.pi if angle < 0 else angle

def get_angle_bin(x: float, y: float, num_bins: int=8) -> float:
    """
    Convert a vector (x,y) to an angle and assign it to a discrete bin.
    
    Args:
        x: X component of the vector
        y: Y component of the vector
        num_bins: Number of angular bins (use -1 for raw angle)
        
    Returns:
        Binned angle in radians or raw angle if num_bins is -1
    """
    theta = normalize_angle(np.arctan2(y, x))
    
    if num_bins == -1:  # Return raw angle
        return theta

    bin_width = 2 * np.pi / num_bins
    bin_index = round(theta / bin_width)
    bins = [i * bin_width for i in range(num_bins)]
    
    return bins[bin_index % num_bins]

# Feature matching utilities
def match_features(des1, des2):
    """Match descriptors between two sets of features using brute force matching."""
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)
    return sorted(matches, key=lambda x: x.distance)

def compute_homography(kp1, kp2, matches, ransac_threshold=7.0):
    """Compute homography between two sets of keypoints using RANSAC."""
    if len(matches) < 10:
        return None, []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    inliers = [matches[i] for i in range(len(matches)) if mask[i]]

    return H, inliers

# Node merging and graph construction
def find_matching_node(
    curr_node: Node, 
    keypoints: list, 
    descriptors: list, 
    label: str, 
    inlier_thresh: float, 
    one_label_assumption: bool=True
) -> Node:
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
    kp1, des1 = keypoints[0], descriptors[0]
    
    while queue:
        node = queue.popleft()

        if node.object_label == label:
            # If we assume only one instance of each object class, return first match
            if one_label_assumption:
                return node
                
            # Otherwise, verify match with feature comparison
            kp2, des2 = node.keypoints[0], node.descriptors[0]
            if des1 is not None and des2 is not None:
                matches = match_features(des1, des2)
                H, inliers = compute_homography(kp1, kp2, matches)

                if H is not None and len(inliers) / len(matches) > inlier_thresh:
                    return node

        # Continue BFS
        for neighbor, _, _ in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return None

def merge_node(node: Node, visit: list[int], matching_node: Node) -> Node:
    """Add a new visit to an existing node if a match is found."""
    if matching_node is not None:
        matching_node.add_new_visit(visit)
        return matching_node
    return None

def create_node(node_id: int, label: str, visit: list[int], keypoints: list, descriptors: list) -> Node:
    """Create a new node with the given properties."""
    return Node(
        id=node_id,
        object_label=label,
        visits=[visit],
        keypoints=keypoints,
        descriptors=descriptors
    )

def calculate_edge_features(prev_pos: tuple[int, int], curr_pos: tuple[int, int], num_bins: int) -> tuple:
    """Calculate edge features between two positions."""
    prev_x, prev_y = prev_pos
    curr_x, curr_y = curr_pos
    dx, dy = curr_x - prev_x, curr_y - prev_y
    
    angle = get_angle_bin(dx, dy, num_bins)
    distance = np.sqrt(dx**2 + dy**2)
    
    return angle, distance

def add_bidirectional_edge(
    curr_node: Node, 
    next_node: Node, 
    angle: float, 
    distance: float,
    edge_data: list[torch.Tensor],
    edge_index: list[list[int]],
    prev_pos: tuple[float, float],
    curr_pos: tuple[float, float]
):
    """Add bidirectional edges between two nodes with appropriate features."""
    # Add forward edge
    curr_node.add_neighbor(next_node, angle, distance)
    
    # Add backward edge only if not connecting to root
    if curr_node.object_label != 'root':
        # Calculate opposite angle
        opposite_angle = angle + np.pi if angle < np.pi else angle - np.pi
        next_node.add_neighbor(curr_node, opposite_angle, distance)
        
        # Normalize positions
        prev_x, prev_y = prev_pos[0] / resolution[0], prev_pos[1] / resolution[1]
        curr_x, curr_y = curr_pos[0] / resolution[0], curr_pos[1] / resolution[1]
        
        # Add edge features for both directions
        edge_feature = torch.tensor([prev_x, prev_y, curr_x, curr_y])
        edge_data.append(edge_feature)
        edge_data.append(edge_feature)  # Same feature for both directions
        
        # Update edge indices
        edge_index[0].extend([curr_node.id, next_node.id])
        edge_index[1].extend([next_node.id, curr_node.id])

def update_graph(
    curr_node: Node, 
    label_counts: dict[str, int], 
    visit: list[int], 
    keypoints: list, 
    descriptors: list, 
    prev_gaze_pos: tuple[int, int], 
    curr_gaze_pos: tuple[int, int], 
    edge_data: list[torch.Tensor],
    edge_index: list[list[int]],
    num_nodes: list[int], 
    num_bins: int=8, 
    inlier_thresh: float=0.3
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
    angle, distance = calculate_edge_features(prev_gaze_pos, curr_gaze_pos, num_bins)
    
    # Find most likely object label
    most_likely_label = max(label_counts, key=label_counts.get)
    
    # Try to find matching node
    matching_node = find_matching_node(curr_node, keypoints, descriptors, most_likely_label, inlier_thresh)
    next_node = merge_node(curr_node, visit, matching_node)
    
    # Create new node if no match found
    if next_node is None:
        next_node = create_node(num_nodes[0], most_likely_label, visit, keypoints, descriptors)
        num_nodes[0] += 1
    
    # Connect nodes if not already connected and not self-loop
    if next_node != curr_node and not curr_node.has_neighbor(next_node):
        add_bidirectional_edge(
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

# Visualization
def format_angle_display(theta: float, use_degrees: bool) -> float:
    """Format angle for display, optionally converting to degrees."""
    if isinstance(theta, float) and use_degrees:
        return theta * 180 / np.pi
    return theta

def print_levels(start_node: Node, use_degrees: bool=True):
    """Print graph structure by levels, showing node relationships."""
    visited = set([start_node])
    queue = deque([(start_node, 'none', 'none')])
    
    curr_depth = 0
    while queue:
        level_size = len(queue)
        print(f'Depth: {curr_depth}')
        
        for _ in range(level_size):
            node, prev_obj, theta = queue.popleft()
            print('-----------------')
            print(f'Object: {node.object_label}')
            print(f'Visited at: {node.visits}')
            print(f'Visited from: {prev_obj}')
            print(f'Angle from prev: {format_angle_display(theta, use_degrees)}')
            
            for neighbor, t, _ in node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, node.object_label, t))
                    
        print('================')
        curr_depth += 1