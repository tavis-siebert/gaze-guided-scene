import random
import torch
from typing import List, Optional, Set
from torch_geometric.data import Data
from itertools import combinations

from gazegraph.training.dataset.graph_utils import edge_idx_to_adj_list, tarjans, is_connected_after_dropping


def find_valid_dropped_nodes(
    adj_lists: List[List[int]], 
    drop_candidates: List[int], 
    num_to_drop: int,
    max_attempts: int = 100
) -> Optional[List[int]]:
    """Find a valid set of nodes that can be dropped while maintaining connectivity.
    
    Args:
        adj_lists: Adjacency list representation of graph
        drop_candidates: List of candidate nodes that can be dropped
        num_to_drop: Number of nodes to drop
        max_attempts: Maximum number of random combinations to try
        
    Returns:
        List of nodes to drop or None if no valid combination found
    """
    if not drop_candidates or num_to_drop <= 0 or num_to_drop > len(drop_candidates):
        return None
        
    # Get all possible combinations and shuffle
    combos = list(combinations(drop_candidates, num_to_drop))
    if not combos:
        return None
        
    # Try random combinations up to max_attempts
    attempts = min(max_attempts, len(combos))
    indices = random.sample(range(len(combos)), attempts)
    
    for idx in indices:
        dropped_nodes = set(combos[idx])
        if is_connected_after_dropping(adj_lists, len(adj_lists), dropped_nodes):
            return list(dropped_nodes)
            
    return None


def node_dropping(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, max_droppable: int) -> Optional[Data]:
    """Apply node dropping augmentation while preserving graph connectivity.
    
    Args:
        x: Node features tensor
        edge_index: Edge indices tensor
        edge_attr: Edge attributes tensor
        max_droppable: Maximum number of nodes that can be dropped
        
    Returns:
        Augmented PyG Data object or None if augmentation not possible
    """
    num_nodes = x.size(0)
    
    # Handle edge cases
    if num_nodes <= 2 or max_droppable <= 0:
        return None
        
    # Special case: no edges
    if edge_index.numel() == 0:
        # Keep at least 1 node
        keep_n = max(1, num_nodes - min(max_droppable, num_nodes - 1))
        keep_indices = torch.randperm(num_nodes)[:keep_n]
        keep_mask = torch.zeros(num_nodes, dtype=torch.bool)
        keep_mask[keep_indices] = True
        return Data(x=x[keep_mask], edge_index=edge_index, edge_attr=edge_attr)
        
    # Build adjacency list and find articulation points
    adj_lists = edge_idx_to_adj_list(edge_index, num_nodes)
    articulation_points = tarjans(adj_lists)
    
    # Get droppable nodes (excluding articulation points)
    drop_candidates = [i for i in range(num_nodes) if i not in articulation_points]
    
    # Determine number of nodes to drop
    max_possible = min(max_droppable, len(drop_candidates))
    if max_possible == 0:
        return None
        
    num_to_drop = random.randint(1, max_possible)
    
    # Find valid nodes to drop
    dropped_nodes = find_valid_dropped_nodes(adj_lists, drop_candidates, num_to_drop)
    if dropped_nodes is None:
        return None
        
    # Create node mapping and masks
    keep_nodes = list(set(range(num_nodes)) - set(dropped_nodes))
    node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_nodes)}
    
    # Filter nodes
    new_x = x[keep_nodes]
    
    # Filter and remap edges
    edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in dropped_nodes or dst in dropped_nodes:
            edge_mask[i] = False
            
    new_edge_index = edge_index[:, edge_mask].clone()  # Clone to avoid modifying original
    new_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    
    # Remap node indices
    for i in range(new_edge_index.size(1)):
        new_edge_index[0, i] = node_map[new_edge_index[0, i].item()]
        new_edge_index[1, i] = node_map[new_edge_index[1, i].item()]
        
    return Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr) 