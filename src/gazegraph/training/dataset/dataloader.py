from typing import Literal
import torch
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from gazegraph.training.dataset.graph_dataset import GraphDataset
from gazegraph.logger import get_logger

logger = get_logger(__name__)


def create_dataloader(
    root_dir: str,
    split: str = "train",
    task_mode: str = "future_actions",
    config=None,
    object_node_feature: str = "roi-embeddings",
    action_node_feature: str = "action-label-embedding",
    device: str = "cuda",
    load_cached: bool = False,
    graph_type: str = "object-graph",
) -> DataLoader:
    """Create a PyG DataLoader for graph data.

    Args:
        root_dir: Root directory containing graph checkpoints
        split: Dataset split ("train" or "val")
        task_mode: Task mode ("future_actions", "future_actions_ordered", or "next_action")
        config: Configuration object containing dataset and training parameters
        object_node_feature: Type of object node features to use
        action_node_feature: Type of action node features to use
        device: Device to use for processing
        load_cached: Whether to load cached dataset from file
        graph_type: Type of graph dataset to use ("object-graph", "action-graph", or "action-object-graph")
    Returns:
        PyG DataLoader
    """
    batch_size = config.training.batch_size if split == "train" else 1
    shuffle = True if split == "train" else False
    node_drop_p = config.training.node_drop_p if split == "train" else 0.0
    max_droppable = config.training.max_nodes_droppable if split == "train" else 0

    # Define cache file path
    cache_dir = Path(config.directories.data_dir) / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if graph_type == "action-graph":
        feature_type = action_node_feature
    elif graph_type == "object-graph":
        feature_type = object_node_feature
    else: 
        feature_type = action_node_feature + '_' + object_node_feature

    cache_file = cache_dir / f"graph-dataset-{split}-{graph_type}-{feature_type}.pth"

    # Try to load cached dataset or create a new one
    if load_cached and cache_file.exists():
        logger.info(f"Loading cached dataset from {cache_file}")
        try:
            dataset = torch.load(cache_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load cached dataset: {e}")

        fail = False

        # Check for configuration differences and warn and fail if needed
        if "object" in graph_type and dataset.object_node_feature != object_node_feature:
            logger.warning(f"Cached dataset uses '{dataset.object_node_feature}' object features, but '{object_node_feature}' was requested")
            fail = True
        if "action" in graph_type and hasattr(dataset, 'action_node_feature') and dataset.action_node_feature != action_node_feature:
            logger.warning(f"Cached dataset uses '{dataset.action_node_feature}' action features, but '{action_node_feature}' was requested")
            fail = True
        if dataset.task_mode != task_mode:
            logger.warning(
                f"Cached dataset uses '{dataset.task_mode}' task mode, but '{task_mode}' was requested"
            )
            fail = True
        if dataset.node_drop_p != node_drop_p:
            logger.warning(
                f"Cached dataset uses {dataset.node_drop_p} node_drop_p, but {node_drop_p} was requested"
            )
            fail = True
        if dataset.max_droppable != max_droppable:
            logger.warning(
                f"Cached dataset uses {dataset.max_droppable} max_droppable, but {max_droppable} was requested"
            )
            fail = True
        if dataset.graph_type != graph_type:
            logger.warning(
                f"Cached dataset uses '{dataset.graph_type}' graph type, but '{graph_type}' was requested"
            )
            fail = True

        if fail:
            logger.warning(
                "Cached dataset does not match requested parameters. Run again without --load_cached to create a new dataset."
            )
            raise RuntimeError(
                "Cached dataset does not match requested parameters. Rerun without --load_cached to create a new dataset."
            )
    else:
        if load_cached:
            logger.info(f"No cached dataset found at {cache_file}, creating new dataset")
            if graph_type == "action-object-graph":
                logger.info(f"NOTE: if action-graph or object-graph caches exist, they will be loaded during dataset creation")

        dataset = create_new_dataset(root_dir, split, task_mode, node_drop_p, max_droppable, config, object_node_feature, action_node_feature, device, cache_file, graph_type)
    
    num_workers = config.processing.dataloader_workers
    if object_node_feature == "roi-embeddings":
        # Multiprocessing currently unsupported due to unpicklable AV instances
        logger.warning("Multiprocessing currently unsupported for roi-embeddings due to unpicklable AV instances")
        num_workers = 0

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def create_new_dataset(
    root_dir, 
    split, 
    task_mode, 
    node_drop_p, 
    max_droppable, 
    config, 
    object_node_feature, 
    action_node_feature, 
    device,
    cache_file=None,
    graph_type: Literal["object-graph", "action-graph", "action-object-graph"] = "object-graph"
):
    """Create a new GraphDataset and optionally save it to cache.

    Args:
        root_dir: Root directory containing graph checkpoints
        split: Dataset split ("train" or "val")
        task_mode: Task mode ("future_actions", "future_actions_ordered", or "next_action")
        node_drop_p: Probability of dropping nodes during training
        max_droppable: Maximum number of nodes to drop during augmentation
        config: Configuration object containing dataset and training parameters
        object_node_feature: Type of object node features to use
        action_node_feature: Type of action node features to use
        device: Device to use for processing
        cache_file: Path to save the dataset cache
        graph_type: Type of graph dataset to use ("object-graph", "action-graph", or "action-object-graph")
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
        object_node_feature=object_node_feature,
        action_node_feature=action_node_feature,
        device=device,
        graph_type=graph_type,
    )

    # Populate dataset's cache
    logger.info(f"Populating data cache for {split} dataset")
    for i in tqdm(range(len(dataset))):
        _ = dataset[i]
    
    # Save the dataset to cache if a cache file is provided
    # TODO this doesn't work as of now due to some issues with pickling in torch.save
    # future cleanups should use InMemoryDataset or store just the .cache attribute
    # currently left as is for compatability reasons

    if cache_file is not None:
        try:
            logger.info(f"Saving dataset to cache: {cache_file}")
            torch.save(dataset, cache_file)
        except Exception as e:
            logger.error(f"Failed to save dataset to cache: {e}")

    return dataset
