"""
Graph building module for creating scene graphs from video data.
"""

import json
import os
import multiprocessing as mp
import torch
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any

from graph.build_graphs import build_graph
from egtea_gaze.constants import NUM_ACTION_CLASSES
from config.config_utils import DotDict
from logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

def split_list(lst: List[Any], n: int) -> List[List[Any]]:
    """Splits a list into n roughly equal parts.
    
    Args:
        lst: The list to split
        n: Number of parts to split into
        
    Returns:
        List of n sublists
    """
    avg = len(lst) // n
    remainder = len(lst) % n
    split_sizes = [avg + (1 if i < remainder else 0) for i in range(n)]
    it = iter(lst)
    return [list(islice(it, size)) for size in split_sizes]

def build_graphs_subset(
    train_vids: List[str], 
    val_vids: List[str], 
    device_id: int, 
    config: DotDict, 
    result_queue: mp.Queue, 
    use_gpu: bool = False, 
    enable_tracing: bool = False
) -> None:
    """Build graph checkpoints using specified device (GPU or CPU).
    
    Args:
        train_vids: List of training videos to process
        val_vids: List of validation videos to process
        device_id: Device ID for processing
        config: Configuration dictionary
        result_queue: Queue for returning results
        use_gpu: Whether to use GPU for processing
        enable_tracing: Whether to enable graph construction tracing
    """
    # Get a logger for the subprocess
    subprocess_logger = get_logger(f"{__name__}.device{device_id}")
    
    if use_gpu:
        torch.cuda.set_device(device_id)
    
    device_name = f"GPU {device_id}" if use_gpu else f"CPU {device_id}"
    subprocess_logger.info(f"Starting graph building on {device_name}")
    
    # Setup graphs output directory
    graphs_dir = Path(config.directories.repo.datasets) / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    # Process each split
    train_paths = build_graph(
        video_list=train_vids,
        config=config,
        split='train',
        desc=f"{device_name} - Training",
        enable_tracing=enable_tracing,
        output_dir=str(graphs_dir)
    )
    
    val_paths = build_graph(
        video_list=val_vids,
        config=config,
        split='val',
        desc=f"{device_name} - Validation",
        enable_tracing=enable_tracing,
        output_dir=str(graphs_dir)
    )
    
    # Return list of processed videos
    result = {
        'train': train_paths,
        'val': val_paths
    }
    
    result_queue.put(result)

def filter_videos(video_list: List[str], filter_names: Optional[List[str]]) -> List[str]:
    """Filter video list based on specified video names.
    
    Args:
        video_list: List of all available videos
        filter_names: List of video names to keep, or None to keep all
        
    Returns:
        Filtered list of videos
    """
    if not filter_names:
        return video_list
    
    filtered_videos = [vid for vid in video_list if any(name in vid for name in filter_names)]
    
    if not filtered_videos:
        logger.warning(f"No videos matched the specified filters: {filter_names}")
        return []
    
    logger.info(f"Filtered {len(video_list)} videos down to {len(filtered_videos)} based on specified names")
    return filtered_videos

def initialize_multiprocessing() -> None:
    """Set the multiprocessing start method to 'spawn'.
    
    This is required for using CUDA with multiprocessing to avoid issues with
    CUDA context initialization in forked processes.
    """
    mp.set_start_method('spawn', force=True)
    logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")

def build_graphs(
    config: DotDict, 
    use_gpu: bool = True, 
    videos: Optional[List[str]] = None, 
    enable_tracing: bool = False
) -> Dict[str, List[str]]:
    """Build graph checkpoints for videos using specified device type and optional filtering.
    
    Args:
        config: Configuration object
        use_gpu: Whether to use GPU for processing (if available)
        videos: Optional list of video names to process. If None, all videos will be processed.
        enable_tracing: Whether to enable graph construction tracing
        
    Returns:
        Dictionary containing lists of saved checkpoint paths for each split
    """
    logger.info("Starting graph checkpoint building process...")
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    initialize_multiprocessing()
    
    # Configure tracing if enabled
    if enable_tracing:
        trace_dir = config.directories.repo.traces
        logger.info(f"Graph tracing enabled. Traces will be saved to {trace_dir}")
        
    # Load video splits
    with open(config.dataset.ego_topo.splits.train_test) as f:
        split = json.load(f)

    # Filter videos if specific ones are requested
    train_videos = filter_videos(split['train_vids'], videos)
    val_videos = filter_videos(split['val_vids'], videos)
    
    if not train_videos and not val_videos:
        logger.error("No videos to process after filtering. Aborting.")
        return None
    
    # Determine device configuration
    if use_gpu and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device_type = "GPU"
    else:
        num_devices = config.processing.n_cores
        use_gpu = False
        device_type = "CPU"

    logger.info(f"Using {num_devices} {device_type}(s) for graph building")
    logger.info(f"Total videos to process - Train: {len(train_videos)}, Val: {len(val_videos)}")
    
    # Split videos across devices
    train_splits = split_list(train_videos, num_devices)
    val_splits = split_list(val_videos, num_devices)

    # Create and start processes
    processes, result_queue = [], mp.Queue()
    
    with tqdm(total=num_devices, desc="Launching processes") as pbar:
        for device_id in range(num_devices):
            train_subset = train_splits[device_id]
            val_subset = val_splits[device_id]
            
            p = mp.Process(
                target=build_graphs_subset, 
                args=(train_subset, val_subset, device_id, config, result_queue, use_gpu, enable_tracing)
            )
            p.start()
            processes.append(p)
            pbar.update(1)

    # Collect results from all processes
    all_paths = {
        'train': [],
        'val': []
    }
    
    with tqdm(total=num_devices, desc="Collecting results") as pbar:
        for _ in range(num_devices):
            result = result_queue.get()
            all_paths['train'].extend(result['train'])
            all_paths['val'].extend(result['val'])
            pbar.update(1)

    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Log summary
    logger.info(f"Graph building completed successfully!")
    logger.info(f"Created {len(all_paths['train'])} train checkpoints and {len(all_paths['val'])} val checkpoints")
    logger.info(f"Checkpoints saved under {Path(config.directories.repo.datasets) / 'graphs'}")
    
    return all_paths 