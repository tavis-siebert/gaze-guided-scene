import random
import torch
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
from itertools import combinations

from graph.checkpoint_manager import GraphCheckpoint, CheckpointManager
from datasets.egtea_gaze.action_record import ActionRecord
from datasets.egtea_gaze.video_metadata import VideoMetadata

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
        pre_filter=None,
        config=None
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
            config: Configuration object to pass to VideoMetadata
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.val_timestamps = val_timestamps
        self.task_mode = task_mode
        self.node_drop_p = node_drop_p
        self.max_droppable = max_droppable
        
        # Initialize video metadata
        self.metadata = VideoMetadata(config)
        
        # Find all graph checkpoint files
        self.checkpoint_files = list(self.root_dir.glob("*_graph.pth"))
        
        # Load each video's checkpoints and select only the frames we want
        self.processed_checkpoints = []
        for file_path in tqdm(self.checkpoint_files, desc=f"Loading {self.split} checkpoints"):
            checkpoints = self._load_and_filter_checkpoints(file_path)
            self.processed_checkpoints.extend(checkpoints)
            
        # Apply sampling based on configuration
        if config and getattr(config.dataset, 'sampling', None) and self.split == 'train':
            sampling_cfg = config.dataset.sampling
            strategy = sampling_cfg.strategy
            k = sampling_cfg.samples_per_video
            allow_dup = sampling_cfg.allow_duplicates
            seed = sampling_cfg.random_seed
            # Set seed for reproducibility if provided
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            from collections import defaultdict
            grouped = defaultdict(list)
            for cp in self.processed_checkpoints:
                grouped[cp.video_name].append(cp)
            sampled = []
            # 'all' strategy or no sampling requested
            if strategy == 'all' or k <= 0:
                sampled = self.processed_checkpoints
            # 'uniform' or 'random' sampling
            elif strategy in ('uniform', 'random'):
                for cps in grouped.values():
                    cps_sorted = sorted(cps, key=lambda x: x.frame_number)
                    n = len(cps_sorted)
                    # Uniform sampling: pick evenly spaced frames
                    if strategy == 'uniform':
                        if k >= n:
                            sampled.extend(random.choices(cps_sorted, k=k) if allow_dup else cps_sorted)
                        else:
                            indices = np.linspace(0, n - 1, k, dtype=int).tolist()
                            sampled.extend([cps_sorted[i] for i in indices])
                    # Random sampling: pick random frames
                    else:  # strategy == 'random'
                        if k >= n:
                            sampled.extend(random.choices(cps_sorted, k=k) if allow_dup else random.sample(cps_sorted, n))
                        else:
                            sampled.extend(random.choices(cps_sorted, k=k) if allow_dup else random.sample(cps_sorted, k))
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
            self.processed_checkpoints = sampled
        
        # Initialize PyG Dataset
        super().__init__(root=str(self.root_dir), transform=transform, 
                         pre_transform=pre_transform, pre_filter=pre_filter)
    
    def _load_and_filter_checkpoints(self, file_path: Path) -> List[GraphCheckpoint]:
        """Load checkpoints from file and filter based on split.
        
        Args:
            file_path: Path to checkpoint file
            
        Returns:
            List of filtered checkpoints with added action labels
        """
        # Extract video name for looking up records
        video_name = Path(file_path).stem.split('_')[0]  # Assuming format: video_name_graph.pth
        
        # Use the CheckpointManager to load checkpoints
        all_checkpoints = CheckpointManager.load_checkpoints(str(file_path))
        
        # Filter and add action labels
        processed_checkpoints = []
        for checkpoint in all_checkpoints:
            # Add action labels to checkpoint
            action_labels = self.metadata.get_future_action_labels(
                video_name, checkpoint.frame_number
            )
            
            # Skip if no action labels available
            if action_labels is None:
                continue
                
            # Add action labels to checkpoint
            setattr(checkpoint, 'action_labels', action_labels)
            processed_checkpoints.append(checkpoint)
        
        # In train mode, keep all checkpoints
        if self.split == "train":
            return processed_checkpoints
        
        # In val mode, sample checkpoints at specific timestamps
        elif self.split == "val":
            # Group by video
            if not processed_checkpoints:
                return []
                
            # Get video info from first checkpoint
            video_length = processed_checkpoints[0].video_length
            
            # Calculate frame numbers from timestamp ratios
            timestamp_frames = [int(ratio * video_length) for ratio in self.val_timestamps]
            
            # Find the closest checkpoint to each target frame
            selected_checkpoints = []
            for target_frame in timestamp_frames:
                closest = min(processed_checkpoints, 
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
        if not checkpoint.edges:
            # No edges, return empty tensors
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
            
        # Collect edge data
        edge_list = []
        edge_attrs = []
        
        for edge in checkpoint.edges:
            edge_list.append((edge["source_id"], edge["target_id"]))
            
            # Edge attribute is angle
            edge_attrs.append([edge.get("angle", 0.0)])
            
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Normalize edge attributes if needed
        if edge_attr.shape[0] > 0 and edge_attr.max() > 0:
            edge_attr = edge_attr / (edge_attr.max() + 1e-8)
            
        return edge_index, edge_attr
        
    def _apply_node_dropping(self, data: Data) -> Optional[Data]:
        """Apply node dropping augmentation to a graph.
        
        Args:
            data: PyG Data object
            
        Returns:
            Augmented PyG Data object or None if augmentation failed
        """
        # Don't augment very small graphs
        if data.x.size(0) <= 2:
            return None
            
        # Apply node dropping
        augmented_data = self._apply_node_dropping(data)
        
        # Make sure to preserve the label
        if augmented_data is not None and hasattr(data, 'y'):
            augmented_data.y = data.y
            
        return augmented_data


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
    
    def dfs_AP(u, parent):
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
            
            elif v != parent[u]:
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


def create_dataloader(
    root_dir: str,
    split: str = "train",
    val_timestamps: List[float] = [0.25, 0.5, 0.75],
    task_mode: str = "future_actions",
    batch_size: int = 64,
    node_drop_p: float = 0.0,
    max_droppable: int = 0,
    shuffle: bool = True,
    num_workers: int = 4,
    config=None
) -> DataLoader:
    """Create a PyG DataLoader for graph data.
    
    Args:
        root_dir: Root directory containing graph checkpoints
        split: Dataset split ("train" or "val")
        val_timestamps: Timestamps to sample for validation set (as fractions of video length)
        task_mode: Task mode ("future_actions", "future_actions_ordered", or "next_action")
        batch_size: Batch size for DataLoader
        node_drop_p: Probability of node dropping augmentation
        max_droppable: Maximum number of nodes that can be dropped
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for DataLoader
        config: Configuration object to pass to VideoMetadata
        
    Returns:
        PyG DataLoader
    """
    dataset = GraphDataset(
        root_dir=root_dir,
        split=split,
        val_timestamps=val_timestamps,
        task_mode=task_mode,
        node_drop_p=node_drop_p if split == "train" else 0.0,  # Only apply augmentations to training set
        max_droppable=max_droppable,
        config=config
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    ) 