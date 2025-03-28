"""Utility functions for graph layout computation."""
from typing import Dict, Any, Tuple
import numpy as np
import networkx as nx
from logger import get_logger

from graph.dashboard.utils.constants import (
    MAX_ANGLE_NODES, LAYOUT_RADIUS_STEP, LAYOUT_JITTER_SCALE,
    LAYOUT_START_JITTER_SCALE
)

logger = get_logger(__name__)

def get_deterministic_jitter(node_id: Any, scale: float = LAYOUT_JITTER_SCALE) -> Tuple[float, float]:
    """Generate deterministic jitter based on node ID.
    
    Args:
        node_id: Node identifier
        scale: Scale factor for jitter
        
    Returns:
        Tuple of (x, y) jitter values
    """
    node_hash = hash(str(node_id))
    jitter_x = ((node_hash % 1000) / 1000.0 - 0.5) * scale
    jitter_y = ((node_hash // 1000 % 1000) / 1000.0 - 0.5) * scale
    return jitter_x, jitter_y

def initialize_positions_from_angles(G: nx.DiGraph) -> Dict:
    """Initialize node positions based on edge angle features.
    
    This creates an initial layout where connected nodes are positioned
    according to their angular relationships, which serves as a starting
    point for the Kamada-Kawai algorithm.
    
    Args:
        G: NetworkX directed graph to lay out
        
    Returns:
        Dictionary mapping node IDs to (x,y) positions
    """
    pos = {}
    processed_nodes = set()
    
    # Start with a node that has outgoing edges with angle information
    start_node = None
    for node in G.nodes():
        if any('angle_degrees' in G[node][target].get('features', {}) 
                for target in G.successors(node)):
            start_node = node
            break
    
    # If no suitable start node found, fall back to first node
    if start_node is None and G.nodes:
        start_node = list(G.nodes())[0]
    
    if start_node is None:
        return {}
    
    # Create deterministic jitter based on node hash
    def get_jitter(node_id, scale=0.05):
        """Generate deterministic jitter based on node ID."""
        # Use hash of node ID to create consistent jitter values
        node_hash = hash(str(node_id))
        # Use modulo to constrain values within desired range
        jitter_x = ((node_hash % 1000) / 1000.0 - 0.5) * scale
        jitter_y = ((node_hash // 1000 % 1000) / 1000.0 - 0.5) * scale
        return jitter_x, jitter_y
        
    # Place the start node at the center with small deterministic jitter
    jitter_x, jitter_y = get_jitter(start_node, scale=0.02)
    pos[start_node] = (jitter_x, jitter_y)
    processed_nodes.add(start_node)
    
    # Process nodes in breadth-first order to propagate positions
    nodes_to_process = [start_node]
    radius = 1.0  # Distance from center for first layer
    
    while nodes_to_process:
        current_nodes = nodes_to_process.copy()
        nodes_to_process = []
        
        for node in current_nodes:
            # Process outgoing edges with angle information
            for target in G.successors(node):
                if target in processed_nodes:
                    continue
                
                edge_data = G[node][target]
                angle_degrees = edge_data.get('features', {}).get('angle_degrees')
                
                if angle_degrees is not None:
                    # Convert angle to radians (adjust as needed for your angle convention)
                    angle_rad = np.radians(angle_degrees)
                    # Get deterministic jitter for this target node
                    jitter_x, jitter_y = get_jitter(target)
                    # Position target based on angle and radius with deterministic jitter
                    x = pos[node][0] + radius * np.cos(angle_rad) + jitter_x
                    y = pos[node][1] + radius * np.sin(angle_rad) + jitter_y
                    pos[target] = (x, y)
                    processed_nodes.add(target)
                    nodes_to_process.append(target)
            
            # Process incoming edges with angle information
            for source in G.predecessors(node):
                if source in processed_nodes:
                    continue
                
                edge_data = G[source][node]
                angle_degrees = edge_data.get('features', {}).get('angle_degrees')
                
                if angle_degrees is not None:
                    # For incoming edges, use opposite angle
                    angle_rad = np.radians((angle_degrees + 180) % 360)
                    # Get deterministic jitter for this source node
                    jitter_x, jitter_y = get_jitter(source)
                    # Position source based on angle and radius with deterministic jitter
                    x = pos[node][0] + radius * np.cos(angle_rad) + jitter_x
                    y = pos[node][1] + radius * np.sin(angle_rad) + jitter_y
                    pos[source] = (x, y)
                    processed_nodes.add(source)
                    nodes_to_process.append(source)
        
        # Increase radius for next layer to avoid overlaps
        radius += 0.5
    
    # For any remaining nodes without angle information, place them using deterministic positions
    for node in G.nodes():
        if node not in pos:
            # Use deterministic values based on node hash
            node_hash = hash(str(node))
            x = -2 + 4 * ((node_hash % 1000) / 1000.0)
            y = -2 + 4 * ((node_hash // 1000 % 1000) / 1000.0)
            pos[node] = (x, y)
    
    return pos

def compute_graph_layout(G: nx.DiGraph) -> Dict[Any, Tuple[float, float]]:
    """Compute the final graph layout using Kamada-Kawai or spring layout.
    
    Args:
        G: NetworkX directed graph to lay out
        
    Returns:
        Dictionary mapping node IDs to (x,y) positions
    """
    if not G.nodes:
        return {}
        
    # Initialize positions based on edge angles for small graphs
    initial_pos = initialize_positions_from_angles(G) if len(G.nodes) < MAX_ANGLE_NODES else None
    
    try:
        return nx.kamada_kawai_layout(G, pos=initial_pos)
    except Exception as e:
        logger.warning(f"Kamada-Kawai layout failed: {e}. Falling back to spring layout.")
        return nx.spring_layout(G, pos=initial_pos, iterations=50, seed=42) 