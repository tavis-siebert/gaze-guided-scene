from typing import List
from torch_geometric.loader import DataLoader

from training.dataset.graph_dataset import GraphDataset


def create_dataloader(
    root_dir: str,
    split: str = "train",
    val_timestamps: List[float] = None,
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
                        If None, will use config.dataset.timestamps[split]
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
        val_timestamps=val_timestamps,  # Will default to config if None
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