"""
Graph visualization utilities.
"""

from typing import Dict, List, Any, Set
from collections import deque

from graph.node import Node
from graph.utils import AngleUtils
from logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

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
        logger.info('-----------------')
        logger.info(f'Object: {node_info["object"]}')
        logger.info(f'Visited at: {node_info["visits"]}')
        logger.info(f'Visited from: {node_info["from"]}')
        logger.info(f'Angle from prev: {node_info["angle"]}')
    
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
            logger.info(f'Depth: {curr_depth}')
            
            for _ in range(level_size):
                node, prev_obj, theta = queue.popleft()
                node_info = GraphVisualizer.format_node_info(node, prev_obj, theta, use_degrees)
                GraphVisualizer.print_node_info(node_info)
                
                GraphVisualizer._queue_unvisited_neighbors(node, visited, queue)
                    
            logger.info('================')
            curr_depth += 1
    
    @staticmethod
    def _queue_unvisited_neighbors(node: Node, visited: Set[Node], queue: deque) -> None:
        """Add unvisited neighbors to the visualization queue."""
        for neighbor, angle, _ in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, node.object_label, angle)) 