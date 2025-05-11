from typing import List
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
from gazegraph.training.dataset.graph_dataset import GraphDataset
from gazegraph.training.dataset.node_features import get_node_feature_extractor
from gazegraph.logger import get_logger

logger = get_logger(__name__)


def create_dataloader(
    root_dir: str,
    split: str = "train",
    task_mode: str = "future_actions",
    config=None,
    node_feature_type: str = "one-hot",
    device: str = "cuda",
    load_cached: bool = False
) -> DataLoader:
    """Create a PyG DataLoader for graph data.
    
    Args:
        root_dir: Root directory containing graph checkpoints
        split: Dataset split ("train" or "val")
        task_mode: Task mode ("future_actions", "future_actions_ordered", or "next_action")
        config: Configuration object containing dataset and training parameters
        node_feature_type: Type of node features to use
        device: Device to use for processing
        load_cached: Whether to load cached dataset from file
        
    Returns:
        PyG DataLoader
    """
    batch_size = config.training.batch_size if split == "train" else 1
    shuffle = True if split == "train" else False
    num_workers = config.processing.dataloader_workers
    node_drop_p = config.training.node_drop_p if split == "train" else 0.0
    max_droppable = config.training.max_nodes_droppable if split == "train" else 0
    
    # Define cache file path
    cache_dir = Path(config.directories.data) / "datasets" if hasattr(config.directories, "data") else Path("data/datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"graph-dataset-{split}.pth"
    
    # Try to load cached dataset or create a new one
    if load_cached and cache_file.exists():
        logger.info(f"Loading cached dataset from {cache_file}")
        try:
            dataset = torch.load(cache_file)
            
            # Check for configuration differences and warn if needed
            if dataset.node_feature_type != node_feature_type:
                logger.warning(f"Cached dataset uses '{dataset.node_feature_type}' features, but '{node_feature_type}' was requested")
            if dataset.task_mode != task_mode:
                logger.warning(f"Cached dataset uses '{dataset.task_mode}' task mode, but '{task_mode}' was requested")
            if dataset.node_drop_p != node_drop_p:
                logger.warning(f"Cached dataset uses {dataset.node_drop_p} node_drop_p, but {node_drop_p} was requested")
            if dataset.max_droppable != max_droppable:
                logger.warning(f"Cached dataset uses {dataset.max_droppable} max_droppable, but {max_droppable} was requested")
                
        except Exception as e:
            logger.error(f"Failed to load cached dataset: {e}")
            dataset = create_new_dataset(root_dir, split, task_mode, node_drop_p, max_droppable, config, node_feature_type, device, cache_file)
    else:
        if load_cached:
            logger.info(f"No cached dataset found at {cache_file}, creating new dataset")
        dataset = create_new_dataset(root_dir, split, task_mode, node_drop_p, max_droppable, config, node_feature_type, device, cache_file)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )


def create_new_dataset(root_dir, split, task_mode, node_drop_p, max_droppable, config, node_feature_type, device, cache_file=None):
    """Create a new GraphDataset and optionally save it to cache.
    
    Args:
        root_dir: Root directory containing graph checkpoints
        split: Dataset split ("train" or "val")
        task_mode: Task mode ("future_actions", "future_actions_ordered", or "next_action")
        node_drop_p: Probability of dropping nodes during training
        max_droppable: Maximum number of nodes to drop during augmentation
        config: Configuration object containing dataset and training parameters
        node_feature_type: Type of node features to use
        device: Device to use for processing
        cache_file: Path to save the dataset cache
        
    Returns:
        GraphDataset: The newly created dataset
    """
    dataset = GraphDataset(
        root_dir=root_dir,
        split=split,
        task_mode=task_mode,
        node_drop_p=node_drop_p,
        max_droppable=max_droppable,
        config=config,
        node_feature_type=node_feature_type,
        device=device
    )
    
    # Save the dataset to cache if a cache file is provided
    if cache_file is not None:
        try:
            logger.info(f"Saving dataset to cache: {cache_file}")
            torch.save(dataset, cache_file)
        except Exception as e:
            logger.error(f"Failed to save dataset to cache: {e}")
    
    return dataset 