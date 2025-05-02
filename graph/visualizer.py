"""Graph visualizer for displaying scene graph construction."""
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import numpy as np

# Import the Dashboard component that integrates all visualization components
from graph.dashboard.dashboard import Dashboard


class GraphVisualizer:
    """Utility class for static graph visualization and printing.
    
    This class provides methods for formatting and printing graph
    structures in a readable format, primarily for debugging and
    static analysis purposes.
    """
    
    @staticmethod
    def format_node_info(node: Any, prev_obj: str, theta: Any) -> Dict[str, Any]:
        """Format node information for display.
        
        Args:
            node: The node object
            prev_obj: The previous object label
            theta: The angle from the previous node
            
        Returns:
            Dictionary with formatted node information
        """
        return {
            'object': getattr(node, 'object_label', str(node)),
            'from': prev_obj,
            'angle': theta
        }
    
    @staticmethod
    def print_node_info(node_info: Dict[str, Any]) -> None:
        """Print node information to the console.
        
        Args:
            node_info: Dictionary containing node information
        """
        print('-----------------')
        print(f'Object: {node_info["object"]}')
        print(f'Visited from: {node_info["from"]}')
        print(f'Angle from prev: {node_info["angle"]}')
    
    @staticmethod
    def print_levels(start_node: Any, use_degrees: bool = True, edges: List = None, graph: 'Graph' = None) -> None:
        """Print the graph structure by levels (BFS traversal).
        
        Args:
            start_node: The node to start BFS from
            use_degrees: Whether to display angles in degrees
            edges: List of edges in the graph
            graph: Optional Graph instance (preferred over edges list)
        """
        # Create adjacency map for edge lookup
        adjacency = defaultdict(list)
        visited_nodes = set([start_node])
        nodes_by_id = {start_node.id: start_node}
        
        # Initialize with the given edges
        if edges:
            for edge in edges:
                target_node = None
                
                # If graph is available, use it to look up nodes
                if graph:
                    target_node = graph.get_node_by_id(edge.target_id)
                else:
                    # This is a fallback when graph instance isn't provided
                    # Note: this might not find all target nodes if they haven't been visited yet
                    if edge.target_id in nodes_by_id:
                        target_node = nodes_by_id[edge.target_id]
                
                if target_node:
                    adjacency[edge.source_id].append((target_node, edge.angle, edge.distance))
                    nodes_by_id[target_node.id] = target_node
                
        queue = deque([(start_node, 'none', 'none')])
        
        curr_depth = 0
        while queue:
            level_size = len(queue)
            print(f'Depth: {curr_depth}')
            
            for _ in range(level_size):
                node, prev_obj, theta = queue.popleft()
                
                # Convert angle to degrees if requested
                if theta != 'none' and use_degrees:
                    theta = f"{(theta * 180.0 / np.pi):.2f}°"
                
                node_info = GraphVisualizer.format_node_info(node, prev_obj, theta)
                GraphVisualizer.print_node_info(node_info)
                
                # Use adjacency map to get neighbors
                for neighbor, angle, distance in adjacency.get(node.id, []):
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        nodes_by_id[neighbor.id] = neighbor
                        queue.append((neighbor, getattr(node, 'object_label', str(node)), angle))
            
            print('================')
            curr_depth += 1


def visualize_graph_construction(
    trace_file: str,
    video_path: Optional[str] = None,
    action_mapping_path: Optional[str] = None,
    port: int = 8050,
    debug: bool = False,
    verb_idx_file: Optional[str] = None,
    noun_idx_file: Optional[str] = None,
    train_split_file: Optional[str] = None,
    val_split_file: Optional[str] = None
) -> None:
    """Run the interactive graph visualization dashboard.
    
    This function initializes the Dashboard component with the specified
    trace file and video path, then runs the Dash server.
    
    Args:
        trace_file: Path to the trace file with graph events
        video_path: Optional path to the video file
        action_mapping_path: Optional path to the action mapping CSV file
        port: Port number to run the server on
        debug: Whether to run in debug mode
        verb_idx_file: Path to the verb index mapping file
        noun_idx_file: Path to the noun index mapping file
        train_split_file: Path to the training data split file
        val_split_file: Path to the validation data split file
    """
    dashboard = Dashboard(
        trace_file, 
        video_path, 
        action_mapping_path=action_mapping_path,
        verb_idx_file=verb_idx_file,
        noun_idx_file=noun_idx_file,
        train_split_file=train_split_file,
        val_split_file=val_split_file
    )
    dashboard.run(port=port, debug=debug)