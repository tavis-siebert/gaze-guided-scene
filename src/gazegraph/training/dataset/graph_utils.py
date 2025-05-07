import torch
from typing import List, Set


def edge_idx_to_adj_list(edge_index, num_nodes=None):
    """Convert edge index to adjacency list.
    
    Args:
        edge_index: PyG edge index
        num_nodes: Number of nodes in graph
        
    Returns:
        Adjacency list
    """
    if num_nodes is None:
        if edge_index.numel() == 0:
            return []
        num_nodes = edge_index.max().item() + 1
        
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(dst)
        
    return adj_list


def tarjans(adj_lists: list[list[int]]):
    """Tarjan's algorithm for finding articulation points (critical nodes).
    
    Args:
        adj_lists: Adjacency list representation of graph
        
    Returns:
        Set of articulation points (node IDs)
    """
    n = len(adj_lists)
    discovered = [False] * n
    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    artPoints = set()
    time = 0
    
    def dfs_AP(u, parent_u):
        nonlocal time
        children = 0
        discovered[u] = True
        disc[u] = low[u] = time
        time += 1
        
        for v in adj_lists[u]:
            if not discovered[v]:
                children += 1
                parent[v] = u
                dfs_AP(v, u)
                
                low[u] = min(low[u], low[v])
                
                # Root with at least two children
                if parent[u] == -1 and children > 1:
                    artPoints.add(u)
                    
                # Non-root node with back edge condition
                if parent[u] != -1 and low[v] >= disc[u]:
                    artPoints.add(u)
            
            elif v != parent_u:
                low[u] = min(low[u], disc[v])
                
    for i in range(n):
        if not discovered[i]:
            dfs_AP(i, -1)
            
    return artPoints


def is_connected_after_dropping(adj_lists: List[List[int]], num_nodes: int, dropped_nodes: Set[int]) -> bool:
    """Check if graph remains connected after dropping specified nodes using BFS.
    
    Args:
        adj_lists: Adjacency list representation of graph
        num_nodes: Total number of nodes in graph
        dropped_nodes: Set of node indices to be dropped
        
    Returns:
        bool: True if graph remains connected after dropping nodes
    """
    if num_nodes <= len(dropped_nodes):
        return False
        
    # Get remaining nodes
    remaining_nodes = set(range(num_nodes)) - dropped_nodes
    if not remaining_nodes:
        return False
        
    # Start BFS from first remaining node
    start = next(iter(remaining_nodes))
    visited = {start}
    queue = [start]
    
    while queue:
        node = queue.pop(0)
        for neighbor in adj_lists[node]:
            if neighbor not in dropped_nodes and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    # Graph is connected if we visited all remaining nodes
    return visited == remaining_nodes 