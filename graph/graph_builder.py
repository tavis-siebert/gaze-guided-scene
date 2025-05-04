"""
Graph building module for creating scene graphs from video data.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from collections import defaultdict
from tqdm import tqdm
import itertools
import multiprocessing as mp

from graph.graph import Graph
from graph.node import Node
from graph.io import DataLoader, VideoProcessor
from graph.record import Record
from graph.graph_tracer import GraphTracer
from graph.checkpoint_manager import CheckpointManager, GraphCheckpoint
from graph.gaze import GazeProcessor, GazePoint, GazeType
from graph.object_detection import ObjectDetector
from models.sift import SIFT
from datasets.egtea_gaze.gaze_data.gaze_io_sample import parse_gtea_gaze
from datasets.egtea_gaze.constants import GAZE_TYPE_FIXATION, GAZE_TYPE_SACCADE, ACTION_TUPLE_IDX, NUM_ACTION_CLASSES
from config.config_utils import DotDict
from logger import get_logger
from graph.utils import split_list, filter_videos

logger = get_logger(__name__)

class StopProcessingException(Exception):
    """Exception to indicate that frame processing should stop."""
    def __init__(self, reason: str = ""):
        self.reason = reason
        super().__init__(self.reason)

class GraphBuilder:
    """Builds scene graphs from video data and gaze information."""
    
    def __init__(self, config: DotDict, split: str, enable_tracing: bool = False, output_dir: Optional[str] = None):
        """Initialize the graph builder.
        
        Args:
            config: Configuration dictionary
            split: Dataset split ('train' or 'val')
            enable_tracing: Whether to enable graph construction tracing
            output_dir: Directory to save graph checkpoints to
        """
        self.config = config
        self.split = split
        self.enable_tracing = enable_tracing
        self.output_dir = output_dir
        self.tracer = GraphTracer(self.config.directories.repo.traces, "", enabled=False)
        
        # Initialize object labels
        noun_idx_path = Path(self.config.dataset.egtea.noun_idx_file)
        self.obj_labels, self.labels_to_int = DataLoader.load_object_labels(noun_idx_path)
        
        # Initialize YOLO-World model path
        yolo_model_file = self.config.models.yolo_world.model_file
        yolo_model_dir = Path(self.config.models.yolo_world.model_dir)
        yolo_model_path = yolo_model_dir / yolo_model_file
        
        # Create object detector (will be re-initialized for each video with proper tracer)
        self.object_detector = None
        self.yolo_model_path = yolo_model_path
        
        # Load video metadata
        ann_file = (self.config.dataset.ego_topo.splits.train if self.split == 'train' 
                   else self.config.dataset.ego_topo.splits.val)
        self.vid_lengths = DataLoader.load_video_lengths(ann_file)
        _, self.records_by_vid = DataLoader.load_records(ann_file)

        # Initialize action mapping using training records
        train_records, _ = DataLoader.load_records(self.config.dataset.ego_topo.splits.train)
        Record.set_action_mapping(train_records)
        
        # Tracking state variables
        self.prev_gaze_point = None
        self.visit_start = -1
        self.visit_end = -1
        self.frame_num = 0
        self.non_black_frame_count = 0
        
        # Processing context
        self.scene_graph = None
        self.checkpoint_manager = None
        self.video_name = ""
        self.raw_gaze_data = None
        self.gaze_processor = None
        self.records_current = None
        self.vid_length = 0
        
        # Frame range to process
        self.first_frame = 0
        self.last_frame = 0
    
    def _get_action_frame_range(self, video_name: str) -> Tuple[int, int]:
        """
        Get the first and last frame from action annotations for a video.
        
        Args:
            video_name: Name of the video
            
        Returns:
            Tuple of (first_frame, last_frame)
        """
        if video_name not in self.records_by_vid or not self.records_by_vid[video_name]:
            return 0, self.vid_lengths.get(video_name, 0)
            
        records = self.records_by_vid[video_name]
        first_frame = min(r.start_frame for r in records)
        last_frame = max(r.end_frame for r in records)
        
        return first_frame, last_frame
    
    def process_video(self, video_name: str, print_graph: bool = False) -> Optional[str]:
        """Process a video to build its scene graph.
        
        Args:
            video_name: Name of the video to process
            print_graph: Whether to print the final graph structure
            
        Returns:
            Path to saved checkpoint file or None if saving failed
        """
        logger.info(f"\nProcessing video: {video_name}")
        self.video_name = video_name
        self.tracer = GraphTracer(self.config.directories.repo.traces, video_name, enabled=self.enable_tracing)
        if self.enable_tracing:
            logger.info(f"Tracing enabled for {video_name}")
        
        # Initialize object detector with video-specific tracer
        self.object_detector = ObjectDetector(
            model_path=self.yolo_model_path,
            obj_labels=self.obj_labels,
            labels_to_int=self.labels_to_int,
            config=self.config,
            tracer=self.tracer
        )
        
        # Load raw gaze data and initialize gaze processor
        self.raw_gaze_data = parse_gtea_gaze(str(Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt"))
        self.gaze_processor = GazeProcessor(self.config, self.raw_gaze_data)
        
        video_processor = VideoProcessor(Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4")
        self.records_current = self.records_by_vid[video_name]
        self.vid_length = self.vid_lengths[video_name]
        
        self.first_frame, self.last_frame = self._get_action_frame_range(video_name)
        logger.info(f"Processing frames from {self.first_frame} to {self.last_frame} based on action annotations")
        
        self.scene_graph = Graph(
            labels_to_int=self.labels_to_int,
            num_object_classes=len(self.obj_labels),
            video_length=self.vid_length
        )
        self.scene_graph.tracer = self.tracer
        
        self.checkpoint_manager = CheckpointManager(
            graph=self.scene_graph,
            records=self.records_current,
            gaze_data_length=len(self.raw_gaze_data),
            video_name=video_name,
            output_dir=self.output_dir,
            split=self.split
        )
        
        self._reset_tracking_state()
        
        try:
            for (frame, _, is_black_frame), gaze_point in zip(video_processor, self.gaze_processor):
                self._process_frame(frame, gaze_point, is_black_frame)
                if not is_black_frame:
                    self.non_black_frame_count += 1
                self.frame_num += 1
        except StopProcessingException as e:
            logger.info(f"[Frame {self.frame_num}] {e.reason}")
        except StopIteration:
            logger.info(f"[Frame {self.frame_num}] End of data reached")

        self._finish_final_fixation()
        
        if print_graph and self.scene_graph.num_nodes > 0:
            logger.info("\nFinal graph structure:")
            self.scene_graph.print_graph()
        
        # Save checkpoints to disk
        saved_path = self.checkpoint_manager.save_checkpoints()
        if saved_path:
            logger.info(f"Saved {len(self.checkpoint_manager.checkpoints)} checkpoints to {saved_path}")
            return saved_path
        
        return None

    def _reset_tracking_state(self):
        """Reset all tracking state variables."""
        self.prev_gaze_point = None
        self.visit_start = -1
        self.visit_end = -1
        self.frame_num = 0
        self.non_black_frame_count = 0
        if self.object_detector:
            self.object_detector.reset()

    def _process_frame(self, frame: torch.Tensor, gaze_point: GazePoint, is_black_frame: bool) -> None:
        """
        Process a single frame and update the scene graph.
        
        Args:
            frame: Video frame tensor
            gaze_point: Processed gaze point for this frame
            is_black_frame: Whether the frame is a black frame
            
        Raises:
            StopProcessingException: If processing should stop
        """
        # Skip frames outside the action annotation range
        if self.frame_num < self.first_frame:
            return
            
        if self.frame_num > self.last_frame:
            raise StopProcessingException(reason=f"Reached end of annotated frames (frame {self.last_frame})")
        
        node_id = self.scene_graph.current_node.id if self.scene_graph.current_node.id >= 0 else None
        self.tracer.log_frame(self.frame_num, gaze_point.position, int(gaze_point.type), node_id)

        if is_black_frame:
            return

        # Create checkpoint for every frame
        self.checkpoint_manager.checkpoint_if_needed(
            frame_num=self.frame_num,
            non_black_frame_count=self.non_black_frame_count
        )

        if gaze_point.type == GazeType.FIXATION:
            self._handle_fixation(frame, gaze_point)
        elif gaze_point.type == GazeType.SACCADE:
            self._handle_saccade(gaze_point)

    def _handle_fixation(self, frame: torch.Tensor, gaze_point: GazePoint) -> None:
        """Handle a fixation frame."""
        if self.visit_start == -1:
            self.visit_start = self.non_black_frame_count
            logger.info(f"\n[Frame {self.frame_num}] New fixation started at ({gaze_point.x:.1f}, {gaze_point.y:.1f})")
        
        # Run object detection
        self.object_detector.detect_objects(frame, gaze_point, self.frame_num)
        
        # Store current gaze point for future reference
        self.prev_gaze_point = gaze_point
    
    def _handle_saccade(self, gaze_point: GazePoint) -> None:
        """Handle a saccade between fixations."""
        self.visit_end = self.non_black_frame_count - 1
        fixation_duration = self.visit_end - self.visit_start + 1
        
        logger.info(f"\n[Frame {self.frame_num}] Saccade detected:")
        logger.debug(f"- Current gaze position (normalized): ({gaze_point.position[0]:.2f}, {gaze_point.position[1]:.2f})")
        
        # Only proceed if we found fixated objects during this fixation period
        if not self.object_detector.has_fixated_objects():
            logger.info(f"- No fixated objects found during this fixation period, skipping node creation")
            self._reset_fixation_state(gaze_point)
            return
        
        fixated_object, confidence = self.object_detector.get_fixated_object()
        
        logger.info(f"- Fixated object: {fixated_object} (confidence: {confidence:.2f})")
        logger.info(f"- Visit duration: {fixation_duration} frames")
        
        visit_record = [self.visit_start, self.visit_end]
        
        # Get the previous gaze position or default to (0,0) if not available
        prev_position = self.prev_gaze_point.position if self.prev_gaze_point else (0, 0)
        
        # Update the graph with the fixated object
        self.scene_graph.update(
            self.frame_num,
            fixated_object,
            visit_record,
            prev_position,
            gaze_point.position
        )
        
        self._reset_fixation_state(gaze_point)
    
    def _reset_fixation_state(self, gaze_point: GazePoint):
        """Reset the state related to the current fixation."""
        self.visit_start = -1
        self.visit_end = -1
        self.prev_gaze_point = gaze_point
        self.object_detector.reset()
    
    def _finish_final_fixation(self) -> None:
        """Process the final fixation if video ends during one.
        
        Creates a dummy saccade to end the current fixation and adds the node to the graph.
        """
        # Skip if no ongoing fixation or no objects detected
        if self.visit_start == -1 or not self.object_detector.has_fixated_objects():
            return
        
        logger.info("- Final fixation detected, creating dummy saccade")
        self.visit_end = self.non_black_frame_count - 1
        
        # Create a dummy saccade point at a slightly different position from the last fixation
        last_position = self.prev_gaze_point.position if self.prev_gaze_point else (0.5, 0.5)
        dummy_saccade = GazePoint(
            x=min(max(last_position[0] + 0.05, 0.0), 1.0),
            y=min(max(last_position[1] + 0.05, 0.0), 1.0),
            raw_type=GazeType.SACCADE,
            type=GazeType.SACCADE,
            frame_idx=self.frame_num
        )
        
        # Use standard saccade handling logic to process the final fixation
        self._handle_saccade(dummy_saccade)

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