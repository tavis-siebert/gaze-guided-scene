"""Utility functions for graph layout computation."""
from typing import Dict, Any, Tuple
import numpy as np
import networkx as nx
from logger import get_logger

from graph.dashboard.graph_constants import (
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

def initialize_positions_from_angles(G: nx.DiGraph) -> Dict[Any, Tuple[float, float]]:
    """Initialize node positions based on edge angle features.
    
    Args:
        G: NetworkX directed graph to lay out
        
    Returns:
        Dictionary mapping node IDs to (x,y) positions
    """
    pos = {}
    processed_nodes = set()
    
    # Find start node with angle information
    start_node = next((node for node in G.nodes() 
                      if any('angle_degrees' in G[node][target].get('features', {}) 
                            for target in G.successors(node))), 
                     list(G.nodes())[0] if G.nodes else None)
    
    if not start_node:
        return {}
    
    # Place start node at center with small jitter
    jitter_x, jitter_y = get_deterministic_jitter(start_node, LAYOUT_START_JITTER_SCALE)
    pos[start_node] = (jitter_x, jitter_y)
    processed_nodes.add(start_node)
    
    # Process nodes in breadth-first order
    nodes_to_process = [start_node]
    radius = 1.0
    
    while nodes_to_process:
        current_nodes = nodes_to_process.copy()
        nodes_to_process = []
        
        for node in current_nodes:
            # Process outgoing edges
            for target in G.successors(node):
                if target in processed_nodes:
                    continue
                
                edge_data = G[node][target]
                angle_degrees = edge_data.get('features', {}).get('angle_degrees')
                
                if angle_degrees is not None:
                    angle_rad = np.radians(angle_degrees)
                    jitter_x, jitter_y = get_deterministic_jitter(target)
                    x = pos[node][0] + radius * np.cos(angle_rad) + jitter_x
                    y = pos[node][1] + radius * np.sin(angle_rad) + jitter_y
                    pos[target] = (x, y)
                    processed_nodes.add(target)
                    nodes_to_process.append(target)
            
            # Process incoming edges
            for source in G.predecessors(node):
                if source in processed_nodes:
                    continue
                
                edge_data = G[source][node]
                angle_degrees = edge_data.get('features', {}).get('angle_degrees')
                
                if angle_degrees is not None:
                    angle_rad = np.radians((angle_degrees + 180) % 360)
                    jitter_x, jitter_y = get_deterministic_jitter(source)
                    x = pos[node][0] + radius * np.cos(angle_rad) + jitter_x
                    y = pos[node][1] + radius * np.sin(angle_rad) + jitter_y
                    pos[source] = (x, y)
                    processed_nodes.add(source)
                    nodes_to_process.append(source)
        
        radius += LAYOUT_RADIUS_STEP
    
    # Place remaining nodes using deterministic positions
    for node in G.nodes():
        if node not in pos:
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
    
    # Ensure all nodes have valid positions
    if initial_pos is None:
        initial_pos = nx.spring_layout(G, iterations=50, seed=42)
    elif not all(node in initial_pos for node in G.nodes):
        # If some nodes are missing positions, initialize them
        missing_nodes = set(G.nodes) - set(initial_pos.keys())
        for node in missing_nodes:
            node_hash = hash(str(node))
            x = -2 + 4 * ((node_hash % 1000) / 1000.0)
            y = -2 + 4 * ((node_hash // 1000 % 1000) / 1000.0)
            initial_pos[node] = (x, y)
    
    try:
        return nx.kamada_kawai_layout(G, pos=initial_pos)
    except Exception as e:
        logger.warning(f"Kamada-Kawai layout failed: {e}. Falling back to spring layout.")
        return nx.spring_layout(G, pos=initial_pos, iterations=50, seed=42) 