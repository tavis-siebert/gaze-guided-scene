import torch
import numpy as np
import cv2
from typing import Tuple, List, Optional, Set, Any, Dict
from collections import deque
from itertools import islice

from gaze_guided_scene.graph.node import Node

# Type aliases for better readability
Position = Tuple[int, int]

def get_roi(image: torch.Tensor, roi_center: Tuple[int, int], roi_size: int) -> Tuple[torch.Tensor, List[int]]:
    """
    Extract a region of interest (ROI) from an image.
    
    Args:
        image: Input image tensor (C, H, W)
        roi_center: Center coordinates (x, y) of the ROI
        roi_size: Size of the ROI (square)
        
    Returns:
        Tuple containing:
            - ROI tensor
            - Bounding box coordinates [x, y, width, height]
    """
    _, H, W = image.shape
    x, y = roi_center

    # Define the ROI bounds
    roi_half = roi_size // 2
    roi_y1 = max(0, y - roi_half)
    roi_y2 = min(H, y + roi_half)
    roi_x1 = max(0, x - roi_half)
    roi_x2 = min(W, x + roi_half)

    # Calculate width and height
    width = roi_x2 - roi_x1
    height = roi_y2 - roi_y1

    # Extract the ROI
    roi = image[:, roi_y1:roi_y2, roi_x1:roi_x2]
    bbox = [roi_x1, roi_y1, width, height]

    return roi, bbox

def split_list(lst: List[Any], n: int) -> List[List[Any]]:
    """Splits a list into n roughly equal parts.
    
    Args:
        lst: The list to split
        n: Number of parts to split into
        
    Returns:
        List of n sublists
    """
    avg = len(lst) // n
    remainder = len(lst) % n
    split_sizes = [avg + (1 if i < remainder else 0) for i in range(n)]
    it = iter(lst)
    return [list(islice(it, size)) for size in split_sizes]

def filter_videos(video_list: List[str], filter_names: Optional[List[str]], logger) -> List[str]:
    """Filter video list based on specified video names.
    
    Args:
        video_list: List of all available videos
        filter_names: List of video names to keep, or None to keep all
        logger: Logger instance for logging messages
        
    Returns:
        Filtered list of videos (maintains original order from video_list)
    """
    if not filter_names:
        return video_list
    
    filtered_videos = [vid for vid in video_list if any(name in vid for name in filter_names)]
    
    if not filtered_videos:
        logger.warning(f"No videos matched the specified filters: {filter_names} in this split")
        return []
    
    # Show which filters matched which videos for better debugging
    matches_by_filter = {}
    for filter_name in filter_names:
        matches = [vid for vid in filtered_videos if filter_name in vid]
        if matches:
            matches_by_filter[filter_name] = matches
    
    for filter_name, matches in matches_by_filter.items():
        match_count = len(matches)
        logger.info(f"Filter '{filter_name}' matched {match_count} video(s) in this split")
        if match_count <= 5:  # Show specific matches only if there aren't too many
            logger.info(f"  Matched: {', '.join(matches)}")
    
    logger.info(f"Filtered {len(video_list)} videos down to {len(filtered_videos)} based on specified names")
    return filtered_videos


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


class GraphTraversal:
    """Utilities for traversing and exploring graph structures."""
    
    @staticmethod
    def dfs(
        graph: 'Graph',
        start_node_id: int, 
        visited: Optional[Set[int]] = None
    ) -> Set[Node]:
        """
        Depth-first search traversal of a graph starting from a given node.
        
        Args:
            graph: The Graph instance to traverse
            start_node_id: ID of the node to start traversal from
            visited: Set of already visited node IDs
            
        Returns:
            Set of all visited nodes
        """
        if visited is None:
            visited = set()
        
        # Get the node from the graph
        start_node = graph.get_node_by_id(start_node_id)
        if start_node is None:
            return set()
            
        # Add node to visited set
        result_nodes = {start_node}
        visited.add(start_node_id)
        
        # Traverse neighbors using graph's adjacency list
        for neighbor_id in graph.get_node_neighbors(start_node_id):
            if neighbor_id not in visited:
                # Recursively traverse unvisited neighbors
                neighbor_nodes = GraphTraversal.dfs(graph, neighbor_id, visited)
                result_nodes.update(neighbor_nodes)
                
        return result_nodes

    @staticmethod
    def get_all_nodes(
        graph: 'Graph', 
        start_node_id: int = -1,
        mode: str = 'dfs'
    ) -> List[Node]:
        """
        Returns all nodes in the graph using the specified traversal method.
        
        Args:
            graph: The Graph instance to traverse
            start_node_id: ID of the node to start traversal from (default: root node)
            mode: Traversal mode ('dfs' for depth-first search)
            
        Returns:
            List of all nodes in the graph
        """
        if mode == 'dfs':
            return list(GraphTraversal.dfs(graph, start_node_id))
        else:
            raise ValueError(f"Unknown traversal mode: {mode}")


class FeatureMatcher:
    """Utilities for matching features between images."""
    
    @staticmethod
    def match_features(des1: Any, des2: Any) -> List:
        """
        Match descriptors between two sets of features using brute force matching.
        
        Args:
            des1: First set of descriptors
            des2: Second set of descriptors
            
        Returns:
            List of matches sorted by distance
        """
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
    ) -> Tuple[Optional[Any], List]:
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

        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        
        # Extract inliers
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]

        return H, inliers
    
    @staticmethod
    def calculate_inlier_ratio(inliers: List, matches: List) -> float:
        """
        Calculate the ratio of inliers to total matches.
        
        Args:
            inliers: List of inlier matches
            matches: List of all matches
            
        Returns:
            Ratio of inliers to total matches
        """
        if not matches:
            return 0.0
        return len(inliers) / len(matches)