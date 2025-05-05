from typing import List
from torch_geometric.loader import DataLoader

from training.dataset.graph_dataset import GraphDataset


def create_dataloader(
    root_dir: str,
    split: str = "train",
    task_mode: str = "future_actions",
    config=None
) -> DataLoader:
    """Create a PyG DataLoader for graph data.
    
    Args:
        root_dir: Root directory containing graph checkpoints
        split: Dataset split ("train" or "val")
        task_mode: Task mode ("future_actions", "future_actions_ordered", or "next_action")
        config: Configuration object containing dataset and training parameters
        
    Returns:
        PyG DataLoader
    """
    batch_size = config.training.batch_size if split == "train" else 1
    shuffle = True if split == "train" else False
    num_workers = config.processing.dataloader_workers
    node_drop_p = config.training.node_drop_p if split == "train" else 0.0
    max_droppable = config.training.max_nodes_droppable if split == "train" else 0
    
    dataset = GraphDataset(
        root_dir=root_dir,
        split=split,
        task_mode=task_mode,
        node_drop_p=node_drop_p,
        max_droppable=max_droppable,
        config=config
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    ) 