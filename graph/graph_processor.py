"""
Graph processing module for building multiple scene graphs with optional multiprocessing support.
"""

import json
import torch
import os
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from graph.graph_builder import GraphBuilder
from graph.utils import split_list, filter_videos
from config.config_utils import DotDict
from logger import get_logger

logger = get_logger(__name__)

def process_video(
    video_name: str, 
    config: DotDict, 
    split: str, 
    print_graph: bool = False, 
    enable_tracing: bool = False,
    output_dir: Optional[str] = None,
    overwrite: bool = False
) -> Optional[str]:
    """Process a single video to build its scene graph.
    
    Args:
        video_name: Name of the video to process
        config: Configuration dictionary
        split: Dataset split ('train' or 'val')
        print_graph: Whether to print final graph structure
        enable_tracing: Whether to enable detailed tracing
        output_dir: Directory to save graph checkpoints to
        overwrite: Whether to overwrite existing checkpoints
        
    Returns:
        Path to saved checkpoint file or None if processing was skipped or failed
    """
    # Skip if checkpoint exists and we're not overwriting
    if not overwrite:
        graphs_dir = Path(output_dir) / split
        checkpoint_path = graphs_dir / f"{video_name}_graph.pth"
        if checkpoint_path.exists():
            logger.info(f"Skipping {video_name} - checkpoint already exists")
            return str(checkpoint_path)
    
    # Process video
    builder = GraphBuilder(config, split, enable_tracing=enable_tracing, output_dir=output_dir)
    saved_path = builder.process_video(video_name, print_graph)
    return saved_path

def build_graph(
    video_list: List[str], 
    config: DotDict, 
    split: str, 
    print_graph: bool = False, 
    desc: Optional[str] = None, 
    enable_tracing: bool = False,
    output_dir: Optional[str] = None,
    overwrite: bool = False
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
        overwrite: Whether to overwrite existing checkpoints
        
    Returns:
        List of paths to saved checkpoint files
    """
    logger.info(f"Building graphs for {len(video_list)} videos in {split} split")
    saved_paths = []
    progress_desc = desc or f"Processing {split} videos"
    
    for video_name in tqdm(video_list, desc=progress_desc):
        if enable_tracing:
            logger.info(f"Building graph with tracing for {video_name}")
        
        saved_path = process_video(
            video_name=video_name,
            config=config, 
            split=split,
            print_graph=print_graph,
            enable_tracing=enable_tracing,
            output_dir=output_dir,
            overwrite=overwrite
        )
        
        if saved_path:
            saved_paths.append(saved_path)
    
    logger.info(f"Created graph checkpoints for {len(saved_paths)} videos in {split} split")
    return saved_paths 

def initialize_multiprocessing() -> None:
    """Set the multiprocessing start method to 'spawn'.
    
    This is required for using CUDA with multiprocessing to avoid issues with
    CUDA context initialization in forked processes.
    """
    try:
        mp.set_start_method('spawn', force=True)
        logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
    except RuntimeError:
        logger.warning("Multiprocessing start method already set, could not change to 'spawn'")

def process_video_worker(
    args: Tuple[str, DotDict, str, bool, bool, str, bool, int]
) -> Tuple[str, Optional[str], str]:
    """Worker function for processing a video in parallel.
    
    Args:
        args: Tuple containing:
            - video_name: Name of the video to process
            - config: Configuration dictionary
            - split: Dataset split ('train' or 'val')
            - print_graph: Whether to print final graph structure
            - enable_tracing: Whether to enable detailed tracing
            - output_dir: Directory to save graph checkpoints to
            - overwrite: Whether to overwrite existing checkpoints
            - worker_id: Worker ID for device assignment
    
    Returns:
        Tuple containing:
            - video_name: Name of the processed video
            - saved_path: Path to saved checkpoint file or None if processing failed
            - split: Dataset split processed
    """
    video_name, config, split, print_graph, enable_tracing, output_dir, overwrite, worker_id = args
    
    # Set CUDA device if using GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(worker_id % torch.cuda.device_count())
    
    try:
        saved_path = process_video(
            video_name=video_name,
            config=config,
            split=split,
            print_graph=print_graph,
            enable_tracing=enable_tracing,
            output_dir=output_dir,
            overwrite=overwrite
        )
    except Exception as e:
        logger.error(f"Error processing {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        saved_path = None
    
    return video_name, saved_path, split

def build_graphs(
    config: DotDict, 
    use_gpu: bool = True, 
    videos: Optional[List[str]] = None, 
    enable_tracing: bool = False,
    overwrite: bool = False
) -> Dict[str, List[str]]:
    """Build graph checkpoints for videos using multiprocessing.
    
    Args:
        config: Configuration object
        use_gpu: Whether to use GPU for processing (if available)
        videos: Optional list of video names to process. If None, all videos will be processed.
        enable_tracing: Whether to enable graph construction tracing
        overwrite: Whether to overwrite existing checkpoints
        
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
    
    # Only show error if no videos matched across both splits
    if not train_videos and not val_videos:
        logger.error(f"No videos matched the specified filters: {videos} in any split. Aborting.")
        return None
    
    # Summarize which videos will be processed
    logger.info("-" * 50)
    logger.info("Video processing summary:")
    if train_videos:
        if videos:
            logger.info(f"Train split: Processing {len(train_videos)} matched videos")
        else:
            logger.info(f"Train split: Processing all {len(train_videos)} videos")
    else:
        logger.info("Train split: No matching videos")
        
    if val_videos:
        if videos:
            logger.info(f"Val split: Processing {len(val_videos)} matched videos")
        else:
            logger.info(f"Val split: Processing all {len(val_videos)} videos")
    else:
        logger.info("Val split: No matching videos")
    logger.info("-" * 50)
    
    # Determine number of workers
    num_workers = config.processing.n_cores
    if use_gpu and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Using {device_count} GPU(s) for processing")
        if device_count < num_workers:
            logger.info(f"Number of workers ({num_workers}) exceeds available GPUs ({device_count})")
            logger.info(f"Multiple workers will share GPUs")
    else:
        use_gpu = False
        logger.info(f"Using {num_workers} CPU workers for processing")
    
    # Setup graphs output directory
    graphs_dir = Path(config.directories.repo.graphs)
    graphs_dir.mkdir(exist_ok=True)
    
    # Prepare work items
    work_items = []
    for i, video_name in enumerate(train_videos):
        work_items.append((
            video_name, 
            config, 
            'train', 
            False, 
            enable_tracing, 
            str(graphs_dir),
            overwrite,
            i % num_workers
        ))
    
    for i, video_name in enumerate(val_videos):
        work_items.append((
            video_name, 
            config, 
            'val', 
            False, 
            enable_tracing, 
            str(graphs_dir),
            overwrite,
            i % num_workers
        ))
    
    # Process videos in parallel
    all_paths = {
        'train': [],
        'val': []
    }
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_video_worker, item) for item in work_items]
        
        with tqdm(total=len(futures), desc="Processing videos") as pbar:
            for future in as_completed(futures):
                try:
                    video_name, saved_path, split = future.result()
                    if saved_path:
                        all_paths[split].append(saved_path)
                except Exception as e:
                    logger.error(f"Worker failed: {str(e)}")
                pbar.update(1)
    
    # Log summary
    logger.info(f"Graph building completed successfully!")
    logger.info(f"Created {len(all_paths['train'])} train checkpoints and {len(all_paths['val'])} val checkpoints")
    logger.info(f"Checkpoints saved under {Path(config.directories.repo.graphs)}")
    
    return all_paths 