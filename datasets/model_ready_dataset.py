import random
import torch
from itertools import combinations
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np

from graph.checkpoint_manager import GraphCheckpoint

class GraphDataset(Dataset):
    """Dataset for loading graph checkpoints and creating PyG data objects."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_timestamps: List[float] = [0.25, 0.5, 0.75],
        task_mode: str = "future_actions",
        node_drop_p: float = 0.0,
        max_droppable: int = 0,
        transform=None,
        pre_transform=None,
        pre_filter=None
    ):
        """Initialize the dataset.
        
        Args:
            root_dir: Root directory containing graph checkpoints
            split: Dataset split ("train" or "val")
            val_timestamps: Timestamps to sample for validation set (as fractions of video length)
            task_mode: Task mode ("future_actions", "future_actions_ordered", or "next_action")
            node_drop_p: Probability of node dropping augmentation
            max_droppable: Maximum number of nodes to drop
            transform: PyG transform to apply to each data object
            pre_transform: PyG pre-transform to apply to each data object
            pre_filter: PyG pre-filter to apply to each data object
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.val_timestamps = val_timestamps
        self.task_mode = task_mode
        self.node_drop_p = node_drop_p
        self.max_droppable = max_droppable
        
        # Find all graph checkpoint files
        self.checkpoint_files = list(self.root_dir.glob("*_graph.pth"))
        
        # Load each video's checkpoints and select only the frames we want
        self.processed_checkpoints = []
        for file_path in tqdm(self.checkpoint_files, desc=f"Loading {split} checkpoints"):
            checkpoints = self._load_and_filter_checkpoints(file_path)
            self.processed_checkpoints.extend(checkpoints)
            
        # Initialize PyG Dataset
        super().__init__(root=str(self.root_dir), transform=transform, 
                         pre_transform=pre_transform, pre_filter=pre_filter)
    
    def _load_and_filter_checkpoints(self, file_path: Path) -> List[GraphCheckpoint]:
        """Load checkpoints from file and filter based on split.
        
        Args:
            file_path: Path to checkpoint file
            
        Returns:
            List of filtered checkpoints
        """
        # Load all checkpoints for this video
        all_checkpoints = torch.load(file_path)
        
        # In train mode, keep all checkpoints
        if self.split == "train":
            return all_checkpoints
        
        # In val mode, sample checkpoints at specific timestamps
        elif self.split == "val":
            # Group by video
            if not all_checkpoints:
                return []
                
            # Get video info from first checkpoint
            video_name = all_checkpoints[0].video_name
            video_length = all_checkpoints[0].video_length
            
            # Calculate frame numbers from timestamp ratios
            timestamp_frames = [int(ratio * video_length) for ratio in self.val_timestamps]
            
            # Find the closest checkpoint to each target frame
            selected_checkpoints = []
            for target_frame in timestamp_frames:
                closest = min(all_checkpoints, 
                              key=lambda cp: abs(cp.frame_number - target_frame))
                selected_checkpoints.append(closest)
                
            return selected_checkpoints
    
    def len(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.processed_checkpoints)
    
    def get(self, idx: int) -> Data:
        """Get a single graph data object.
        
        Args:
            idx: Index of the sample
            
        Returns:
            PyG Data object
        """
        checkpoint = self.processed_checkpoints[idx]
        
        # Get node features
        node_features = self._extract_node_features(checkpoint)
        
        # Get edge indices and attributes
        edge_index, edge_attr = self._extract_edge_features(checkpoint)
        
        # Get label for the specified task mode
        y = checkpoint.action_labels[self.task_mode]
        
        # Create PyG data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        # Apply node dropping augmentation if enabled
        if self.node_drop_p > 0 and random.random() < self.node_drop_p:
            augmented_data = self._apply_node_dropping(data)
            if augmented_data is not None:
                data = augmented_data
                
        return data
    
    def _extract_node_features(self, checkpoint: GraphCheckpoint) -> torch.Tensor:
        """Extract node features from a checkpoint.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tensor of node features
        """
        features_list = []
        for node_id, node_data in checkpoint.nodes.items():
            # Extract basic node information
            total_frames_visited = sum(end - start for start, end in node_data["visits"])
            num_visits = len(node_data["visits"])
            
            first_visit_frame = node_data["visits"][0][0] if node_data["visits"] else 0
            last_visit_frame = node_data["visits"][-1][1] if node_data["visits"] else 0
            
            # Normalize temporal features
            first_frame_normalized = first_visit_frame / checkpoint.non_black_frame_count
            last_frame_normalized = last_visit_frame / checkpoint.non_black_frame_count
            frame_fraction = checkpoint.frame_number / checkpoint.video_length
            
            # Create temporal features tensor
            temporal_features = torch.tensor([
                total_frames_visited,
                num_visits,
                first_frame_normalized,
                last_frame_normalized,
                frame_fraction
            ])
            
            # Normalize first feature (total frames visited)
            if checkpoint.non_black_frame_count > 0:
                temporal_features[0] /= checkpoint.non_black_frame_count
                
            # Create one-hot encoding for object class
            class_idx = checkpoint.labels_to_int.get(node_data["object_label"], 0)
            one_hot = torch.zeros(checkpoint.num_object_classes)
            one_hot[class_idx] = 1
            
            # Combine features
            node_features = torch.cat([temporal_features, one_hot])
            features_list.append(node_features)
        
        if not features_list:
            return torch.tensor([])
            
        # Stack all node features
        node_features_tensor = torch.stack(features_list)
        
        # Further normalization for visit count if needed
        if node_features_tensor[:, 1].max() > 0:
            node_features_tensor[:, 1] /= node_features_tensor[:, 1].max()
            
        return node_features_tensor
    
    def _extract_edge_features(self, checkpoint: GraphCheckpoint) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract edge indices and attributes from a checkpoint.
        
        Args:
            checkpoint: GraphCheckpoint object
            
        Returns:
            Tuple of (edge_indices, edge_attributes)
        """
        edge_index = [[], []]
        edge_attrs = []
        
        for edge in checkpoint.edges:
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            
            # Skip edges connected to root node
            if source_id < 0 or target_id < 0:
                continue
                
            edge_index[0].append(source_id)
            edge_index[1].append(target_id)
            
            # Extract edge features
            prev_x, prev_y = edge["prev_gaze_pos"]
            curr_x, curr_y = edge["curr_gaze_pos"]
            
            edge_attrs.append(torch.tensor([prev_x, prev_y, curr_x, curr_y]))
        
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) if edge_index[0] else torch.tensor([[],[]], dtype=torch.long)
        edge_attr_tensor = torch.stack(edge_attrs) if edge_attrs else torch.tensor([])
            
        return edge_index_tensor, edge_attr_tensor
    
    def _apply_node_dropping(self, data: Data) -> Optional[Data]:
        """Apply node dropping augmentation.
        
        Args:
            data: PyG Data object
            
        Returns:
            Augmented Data object or None if augmentation failed
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        aug_result = node_dropping(x, edge_index, edge_attr, self.max_droppable)
        
        if aug_result is None:
            return None
            
        x_aug, edge_index_aug, edge_attr_aug = aug_result
        return Data(x=x_aug, edge_index=edge_index_aug, edge_attr=edge_attr_aug, y=data.y)


### Node dropping augmentation ###
def edge_idx_to_adj_list(edge_index, num_nodes=None):
    """
    Remaps the edge_index tensor to an adjacency list 
    where each node index maps to a list of node indices it shares an edge with

    Args:
        edge_index: a tensor of shape (2), num_nodes) following torch geometric standard
        num_nodes: the number of nodes in the graph (if known ahead of time)
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    adj_lists = [[] for _ in range(num_nodes)]
    for u,v in edge_index.transpose(0,1):
        u, v = u.item(), v.item()
        adj_lists[u].append(v)
    return adj_lists

def tarjans(adj_lists: list[list[int]]):
    """
    Tarjan's algorithm 

    Args:
        adj_list: adjacency list of graph (see format of edge_idx_to_adj_list() function)
    """
    n = len(adj_lists)
    disc_time = [0] * n
    low = [0] * n
    ap = [False] * n
    time = [0]

    def dfs_AP(u, parent):
        children = 0
        time[0] += 1
        disc_time[u] = low[u] = time[0]

        for v in adj_lists[u]:
            if v == parent:
                continue
            if disc_time[v] == 0:
                children += 1
                dfs_AP(v, u)
                low[u] = min(low[u], low[v])

                # articulation point condition (excluding root)
                if parent != u and disc_time[u] <= low[v]:
                    ap[u] = True
            else:
                # update low[u] if v was already visited
                low[u] = min(low[u], disc_time[v])

        # special case for root node
        if parent == u and children > 1:
            ap[u] = True

    for u in range(n):
        if disc_time[u] == 0:
            dfs_AP(u, u)

    return {i for i, is_ap in enumerate(ap) if is_ap}

def is_connected_after_dropping(adj_lists, num_nodes, dropped_nodes_set):
    """Simple BFS to check connectivity after dropping nodes."""
    visited = set()
    all_nodes = set(range(num_nodes)) - dropped_nodes_set
    if not all_nodes:
        return False

    start = next(iter(all_nodes))
    queue = [start]
    visited.add(start)

    while queue:
        u = queue.pop()
        for v in adj_lists[u]:
            if v in dropped_nodes_set:
                continue
            if v not in visited:
                visited.add(v)
                queue.append(v)

    return visited == all_nodes

def find_valid_dropped_nodes(adj_lists, drop_candidates, num_dropped):
    combos = list(combinations(drop_candidates, num_dropped))
    random.shuffle(combos)
    for dropped_nodes in combos:
        if is_connected_after_dropping(adj_lists, len(adj_lists), set(dropped_nodes)):
            return dropped_nodes
    return None

def node_dropping(x, edge_index, edge_attr, max_droppable):
    """Apply node dropping augmentation.
    
    Args:
        x: Node features tensor
        edge_index: Edge indices tensor
        edge_attr: Edge attributes tensor
        max_droppable: Maximum number of nodes to drop
        
    Returns:
        Tuple of (augmented_x, augmented_edge_index, augmented_edge_attr) or None if augmentation failed
    """
    # select random number of nodes to drop
    num_nodes = x.shape[0]
    adj_lists = edge_idx_to_adj_list(edge_index, num_nodes)
    not_droppable = tarjans(adj_lists)  # find articulation points which would disconnect the graph
    drop_candidates = [u for u in range(num_nodes) if u not in not_droppable]

    max_droppable = min(max_droppable, len(drop_candidates))
    if max_droppable == 0:
        return None
    num_dropped = random.randint(1, max_droppable)

    # Try to find a good sample of dropped nodes
    dropped_nodes = find_valid_dropped_nodes(adj_lists, drop_candidates, num_dropped)
    if dropped_nodes is None:
        return None
        
    # drop nodes
    node_mask = torch.ones(num_nodes, dtype=torch.bool)
    node_mask[dropped_nodes] = False
    x_aug = x[node_mask]

    if x_aug.size(0) == 0:
        return None

    # remap indices since otherwise we would relabel the nodes but not account for edge relabeling
    # (-1 means dropped)
    new_idx_map = torch.full((num_nodes,), -1, dtype=torch.long)
    new_idx_map[node_mask] = torch.arange(node_mask.sum())

    # keep edges where both src and dest are in the node mask (non-dropped)
    src = edge_index[0]
    dst = edge_index[1]
    valid_edge_mask = node_mask[src] & node_mask[dst]

    # filter edges and attributes
    edge_index_aug = edge_index[:, valid_edge_mask]
    edge_attr_aug = edge_attr[valid_edge_mask]

    # remap to new node indices using pytorch black magic
    edge_index_aug = new_idx_map[edge_index_aug]

    return x_aug, edge_index_aug, edge_attr_aug

def create_dataloader(
    root_dir: str,
    split: str = "train",
    val_timestamps: List[float] = [0.25, 0.5, 0.75],
    task_mode: str = "future_actions",
    batch_size: int = 64,
    node_drop_p: float = 0.0,
    max_droppable: int = 0,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a PyG dataloader from graph checkpoints.
    
    Args:
        root_dir: Root directory containing graph checkpoints
        split: Dataset split ("train" or "val")
        val_timestamps: Timestamps to sample for validation set
        task_mode: Task mode ("future_actions", "future_actions_ordered", or "next_action")
        batch_size: Batch size for DataLoader
        node_drop_p: Probability of node dropping augmentation
        max_droppable: Maximum number of nodes to drop
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for loading
        
    Returns:
        PyG DataLoader
    """
    dataset = GraphDataset(
        root_dir=root_dir,
        split=split,
        val_timestamps=val_timestamps,
        task_mode=task_mode,
        node_drop_p=node_drop_p,
        max_droppable=max_droppable
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )