"""
Graph processing module for building multiple scene graphs with optional multiprocessing support.
"""

import json
import torch
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from graph.graph_builder import GraphBuilder
from graph.utils import split_list, filter_videos
from config.config_utils import DotDict
from logger import get_logger

logger = get_logger(__name__)

def build_graph(
    video_list: List[str], 
    config: DotDict, 
    split: str, 
    print_graph: bool = False, 
    desc: Optional[str] = None, 
    enable_tracing: bool = False,
    output_dir: Optional[str] = None
) -> List[str]:
    """Build graph checkpoints for a list of videos.
    
    Args:
        video_list: List of video names to process
        config: Configuration dictionary
        split: Dataset split ('train' or 'val')
        print_graph: Whether to print final graph structure
        desc: Description for progress bar
        enable_tracing: Whether to enable detailed tracing
        output_dir: Directory to save graph checkpoints to
        
    Returns:
        List of paths to saved checkpoint files
    """
    logger.info(f"Building graphs for {len(video_list)} videos in {split} split")
    builder = GraphBuilder(config, split, enable_tracing=enable_tracing, output_dir=output_dir)
    saved_paths = []
    progress_desc = desc or f"Processing {split} videos"
    
    for video_name in tqdm(video_list, desc=progress_desc):
        if enable_tracing:
            logger.info(f"Building graph with tracing for {video_name}")
        
        saved_path = builder.process_video(video_name, print_graph)
        if saved_path:
            saved_paths.append(saved_path)
    
    logger.info(f"Created graph checkpoints for {len(saved_paths)} videos in {split} split")
    return saved_paths 

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
    graphs_dir = Path(config.directories.repo.graphs)
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
    train_videos = filter_videos(split['train_vids'], videos, logger)
    val_videos = filter_videos(split['val_vids'], videos, logger)
    
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
    logger.info(f"Checkpoints saved under {Path(config.directories.repo.graphs)}")
    
    return all_paths 