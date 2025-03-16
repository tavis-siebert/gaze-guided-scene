import torch
import numpy as np
import cv2
from typing import Tuple, List, Optional, Set, Any, Dict
from collections import deque

from graph.node import Node

# Type aliases for better readability
Position = Tuple[int, int]

def get_roi(image: torch.Tensor, roi_center: Tuple[int, int], roi_size: int) -> Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Extract a region of interest (ROI) from an image.
    
    Args:
        image: Input image tensor (C, H, W)
        roi_center: Center coordinates (x, y) of the ROI
        roi_size: Size of the ROI (square)
        
    Returns:
        Tuple containing the ROI tensor and the bounding box coordinates
    """
    _, H, W = image.shape
    x, y = roi_center

    # Define the ROI bounds
    roi_half = roi_size // 2
    roi_y1 = max(0, y - roi_half)
    roi_y2 = min(H, y + roi_half)
    roi_x1 = max(0, x - roi_half)
    roi_x2 = min(W, x + roi_half)

    # Extract the ROI
    bbox = ((roi_x1, roi_y1), (roi_x2, roi_y2))
    roi = image[:, roi_y1:roi_y2, roi_x1:roi_x2]

    return roi, bbox


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
    def dfs(start_node: Node, visited: Optional[Set[Node]] = None) -> Set[Node]:
        """
        Depth-first search traversal of a graph starting from a given node.
        
        Args:
            start_node: Node to start traversal from
            visited: Set of already visited nodes
            
        Returns:
            Set of all visited nodes
        """
        if visited is None:
            visited = set()
        
        visited.add(start_node)
        for neighbor, _, _ in start_node.neighbors:
            if neighbor not in visited:
                GraphTraversal.dfs(neighbor, visited)
                
        return visited

    @staticmethod
    def get_all_nodes(start_node: Node, mode: str = 'dfs') -> List[Node]:
        """
        Returns all nodes in the graph using the specified traversal method.
        
        Args:
            start_node: Node to start traversal from
            mode: Traversal mode ('dfs' for depth-first search)
            
        Returns:
            List of all nodes in the graph
        """
        if mode == 'dfs':
            return list(GraphTraversal.dfs(start_node))
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