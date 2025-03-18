"""
Graph building module for creating scene graphs from video data.
"""

import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from tqdm import tqdm

from graph.graph import Graph
from graph.node import Node
from graph.io import Record, DataLoader, get_future_action_labels, VideoProcessor
from graph.utils import get_roi
from graph.graph_tracer import GraphTracer
from models.clip import ClipModel
from models.sift import SIFT
from egtea_gaze.gaze_data.gaze_io_sample import parse_gtea_gaze
from config.config_utils import DotDict
from logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

class GraphBuilder:
    """
    Builds scene graphs from video data and gaze information.
    """
    
    def __init__(self, config: DotDict, split: str, enable_tracing: bool = False):
        """
        Initialize the graph builder.
        
        Args:
            config: Configuration dictionary
            split: Dataset split ('train' or 'val')
            enable_tracing: Whether to enable graph construction tracing
        """
        self.config = config
        self.split = split
        self.clip_model = self._initialize_clip_model()
        self.sift = SIFT()
        self.enable_tracing = enable_tracing
        self.tracer = None
        
        # Load object labels and dataset information
        self._load_object_labels()
        self._load_dataset_info()
    
    def _initialize_clip_model(self) -> ClipModel:
        """Initialize and load the CLIP model."""
        clip_model = ClipModel(self.config.models.clip.model_id)
        clip_model.load_model(Path(self.config.models.clip.model_dir))
        
        return clip_model
    
    def _load_object_labels(self) -> None:
        """Load object labels and create CLIP-compatible label formats."""
        noun_idx_path = Path(self.config.dataset.egtea.noun_idx_file)
        self.obj_labels, self.labels_to_int = DataLoader.load_object_labels(noun_idx_path)
        self.clip_labels = [f"a picture of a {obj}" for obj in self.obj_labels.values()]
    
    def _load_dataset_info(self) -> None:
        """Load dataset information including video lengths, records, and action indices."""
        # Use the appropriate split file based on the split name
        if self.split == 'train':
            ann_file = self.config.dataset.ego_topo.splits.train
        else:  # val
            ann_file = self.config.dataset.ego_topo.splits.val
            
        self.vid_lengths = DataLoader.load_video_lengths(ann_file)
        self.records, self.records_by_vid = DataLoader.load_records(ann_file)
        self.action_to_idx = DataLoader.create_action_index(self.records)
    
    def process_fixation(
        self,
        frame: torch.Tensor,
        gaze_position: Tuple[float, float],
        roi_size: int = 256
    ) -> str:
        """
        Process a fixation point in a frame.
        
        Args:
            frame: The current video frame
            gaze_position: The (x, y) gaze position
            roi_size: Size of the region of interest
            
        Returns:
            Predicted object label
        """
        # Extract region of interest around gaze point
        roi, _ = get_roi(frame, (int(gaze_position[0]), int(gaze_position[1])), roi_size)
        
        # Run CLIP inference
        label = self.clip_model.run_inference(roi, self.clip_labels, self.obj_labels)
        
        return label
    
    def process_video(self, video_name: str, print_graph: bool = False) -> Dict[str, List]:
        """
        Process a single video to build a scene graph.
        
        Args:
            video_name: Name of the video to process
            print_graph: Whether to print the graph structure
            
        Returns:
            Dictionary with node data, edge indices, edge attributes, and labels
        """
        logger.info(f"\nProcessing video: {video_name}")
        
        # Initialize tracer if tracing is enabled
        if self.enable_tracing:
            # Use the traces directory from config
            trace_dir = self.config.directories.repo.traces
            # GraphTracer handles directory creation internally
            self.tracer = GraphTracer(trace_dir, video_name, enabled=True)
            logger.info(f"Tracing enabled for {video_name}")
        else:
            self.tracer = None
        
        # Get video-specific data and setup
        records_for_vid = self.records_by_vid[video_name]
        vid_length = self.vid_lengths[video_name]
        timestamps = self._calculate_timestamps(vid_length)
        gaze = self._load_gaze_data(video_name)
        
        # Initialize scene graph and video processor
        scene_graph = Graph()
        video_processor = self._initialize_video_processor(video_name)
        
        # Initialize tracking variables
        tracking_data = self._initialize_tracking_data()
        tracking_data['video_name'] = video_name
        
        # Results storage
        results = {
            'x': [],
            'edge_index': [],
            'edge_attr': [],
            'y': []
        }
        
        # Process video frames
        self._process_frames(
            video_processor, 
            scene_graph, 
            tracking_data, 
            timestamps, 
            gaze, 
            records_for_vid, 
            vid_length, 
            results
        )
        
        # Print final graph structure if requested
        if print_graph:
            self._print_final_graph(scene_graph)
        
        return results
    
    def _calculate_timestamps(self, vid_length: int) -> List[int]:
        """Calculate frame timestamps based on configuration ratios."""
        timestamp_ratios = self.config.dataset.timestamps[self.split]
        timestamps = [int(frac*vid_length) for frac in sorted(timestamp_ratios)]
        logger.info(f"Total frames: {vid_length}, Timestamps: {timestamps}")
        return timestamps
    
    def _load_gaze_data(self, video_name: str) -> Any:
        """Load gaze data for the specified video."""
        gaze_path = Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt"
        return parse_gtea_gaze(str(gaze_path))
    
    def _initialize_video_processor(self, video_name: str) -> VideoProcessor:
        """Initialize video processor for the specified video."""
        video_path = Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4"
        return VideoProcessor(video_path)
    
    def _initialize_tracking_data(self) -> Dict[str, Any]:
        """Initialize tracking variables for video processing."""
        return {
            'prev_gaze_pos': (-1, -1),
            'potential_labels': defaultdict(int),
            'keypoints': [],
            'descriptors': [],
            'visit': [],  # [start_frame, end_frame]
            'frame_num': 0,
            'relative_frame_num': 0,
            'node_data': {},
            'video_name': ""
        }
    
    def _process_frames(
        self, 
        video_processor: VideoProcessor,
        scene_graph: Graph,
        tracking_data: Dict[str, Any],
        timestamps: List[int],
        gaze: Any,
        records_for_vid: List[Record],
        vid_length: int,
        results: Dict[str, List]
    ) -> None:
        """Process all frames in the video to build the scene graph."""
        for frame, _, is_black_frame in video_processor:
            frame_num = tracking_data['frame_num']
            
            # Take periodic graph snapshots for visualization if tracing is enabled
            if self.tracer and frame_num % 30 == 0:  # Every 30 frames
                self._take_graph_snapshot(scene_graph, frame_num)
                
            # Skip black frames but still log them
            if is_black_frame:
                if self.tracer and frame_num < len(gaze):
                    # Log black frame with gaze type -1 (special value for black frames)
                    gaze_pos = gaze[frame_num, :2] if frame_num < len(gaze) else (-1, -1)
                    self.tracer.log_frame_processed(
                        frame_num,
                        gaze_pos if isinstance(gaze_pos, list) else list(gaze_pos),
                        -1,  # Special value for black frames
                        None,  # No ROI
                        scene_graph.current_node.id if scene_graph.current_node.id >= 0 else None
                    )
                tracking_data['frame_num'] += 1
                continue
                
            # Check if we need to save graph state at this timestamp
            if self._should_save_graph_state(tracking_data['frame_num'], timestamps, gaze, scene_graph):
                self._save_graph_state(
                    scene_graph, 
                    tracking_data,
                    records_for_vid,
                    vid_length,
                    timestamps,
                    gaze,
                    results
                )
                
                # Exit if we've reached the final timestamp
                if self._reached_end_condition(tracking_data['frame_num'], timestamps, gaze):
                    logger.info(f"[Frame {tracking_data['frame_num']}] Reached final timestamp or end of gaze data")
                    break
            
            # Process gaze data
            if frame_num < len(gaze):
                self._process_gaze_frame(frame, gaze, tracking_data, scene_graph)
            else:
                # Log frames beyond gaze data with type 0 (no gaze data)
                if self.tracer:
                    self.tracer.log_frame_processed(
                        frame_num,
                        (-1, -1),  # No valid gaze position
                        0,  # Type 0 = no gaze data
                        None,  # No ROI
                        scene_graph.current_node.id if scene_graph.current_node.id >= 0 else None
                    )
            
            tracking_data['relative_frame_num'] += 1
            tracking_data['frame_num'] += 1
        
        # Handle final fixation if video ends during fixation
        if tracking_data['potential_labels']:
            self._process_final_fixation(scene_graph, tracking_data, gaze)
            
        # Take final graph snapshot if tracing is enabled
        if self.tracer:
            self._take_graph_snapshot(scene_graph, tracking_data['frame_num'])
    
    def _should_save_graph_state(self, frame_num: int, timestamps: List[int], gaze: Any, scene_graph: Graph) -> bool:
        """Determine if the graph state should be saved at the current frame."""
        return (frame_num in timestamps or frame_num >= len(gaze)) and scene_graph.edge_data
    
    def _reached_end_condition(self, frame_num: int, timestamps: List[int], gaze: Any) -> bool:
        """Check if we've reached the end condition for processing."""
        return frame_num == timestamps[-1] or frame_num >= len(gaze)
    
    def _process_gaze_frame(
        self,
        frame: torch.Tensor,
        gaze: Any,
        tracking_data: Dict[str, Any],
        scene_graph: Graph
    ) -> None:
        """Process a single frame of gaze data."""
        frame_num = tracking_data['frame_num']
        gaze_type = int(gaze[frame_num, 2])
        gaze_pos = gaze[frame_num, :2]
        
        if gaze_type == 1:  # Fixation
            self._handle_fixation(
                frame,
                gaze_pos,
                gaze_type,
                tracking_data
            )
            
        elif gaze_type == 2:  # Saccade
            if tracking_data['potential_labels']:
                self._handle_saccade(
                    scene_graph,
                    tracking_data,
                    gaze_pos,
                    gaze_type
                )
                
                # Reset tracking variables
                tracking_data['visit'] = []
                tracking_data['keypoints'], tracking_data['descriptors'] = [], []
                tracking_data['prev_gaze_pos'] = gaze_pos
                tracking_data['potential_labels'] = defaultdict(int)
            else:
                # Log saccade frame without fixation data
                if self.tracer:
                    self.tracer.log_frame_processed(
                        frame_num,
                        gaze_pos if isinstance(gaze_pos, list) else list(gaze_pos),
                        gaze_type,
                        None,  # No ROI for saccade without previous fixation
                        scene_graph.current_node.id if scene_graph.current_node.id >= 0 else None
                    )
        else:
            # Any other gaze type (0, 3, etc.)
            if self.tracer:
                self.tracer.log_frame_processed(
                    frame_num,
                    gaze_pos if isinstance(gaze_pos, list) else list(gaze_pos),
                    gaze_type,
                    None,  # No ROI
                    scene_graph.current_node.id if scene_graph.current_node.id >= 0 else None
                )
    
    def _handle_fixation(
        self,
        frame: torch.Tensor,
        gaze_pos: Tuple[float, float],
        gaze_type: int,
        tracking_data: Dict[str, Any]
    ) -> None:
        """Handle a fixation frame."""
        frame_num = tracking_data['frame_num']
        
        # Record start of visit if this is the first fixation
        if not tracking_data['visit']:
            tracking_data['visit'].append(tracking_data['relative_frame_num'])
            x, y = gaze_pos
            logger.info(f"\n[Frame {frame_num}] New fixation started at ({x:.1f}, {y:.1f})")
        
        # Get object label from CLIP
        roi, roi_coords = get_roi(frame, (int(gaze_pos[0]), int(gaze_pos[1])), 256)
        label = self.process_fixation(frame, gaze_pos)
        tracking_data['potential_labels'][label] += 1
        logger.info(f"[Frame {frame_num}] CLIP detected: {label} (count: {tracking_data['potential_labels'][label]})")
        
        # Extract features using SIFT
        kp, desc = self.sift.extract_features(frame)
        tracking_data['keypoints'].append(kp)
        tracking_data['descriptors'].append(desc)
        
        # Log frame processing with fixation info
        if self.tracer:
            self.tracer.log_frame_processed(
                frame_num,
                gaze_pos if isinstance(gaze_pos, list) else list(gaze_pos),
                gaze_type,
                roi_coords,
                None  # Node ID not assigned until saccade
            )
    
    def _handle_saccade(
        self,
        scene_graph: Graph,
        tracking_data: Dict[str, Any],
        curr_pos: Tuple[float, float],
        gaze_type: int
    ) -> None:
        """Handle a saccade (eye movement between fixations)."""
        frame_num = tracking_data['frame_num']
        
        # Record end of visit
        tracking_data['visit'].append(tracking_data['relative_frame_num'] - 1)
        
        # Calculate fixation duration
        fixation_duration = tracking_data['visit'][-1] - tracking_data['visit'][0] + 1
        
        # Get most likely object label
        most_likely_label = max(tracking_data['potential_labels'].items(), key=lambda x: x[1])[0]
        logger.info(f"\n[Frame {frame_num}] Saccade detected:")
        logger.info(f"- Most likely object: {most_likely_label}")
        logger.info(f"- Visit duration: {fixation_duration} frames")
        
        prev_pos = tracking_data['prev_gaze_pos']
        
        # Trace saccade before updating graph
        if self.tracer:
            self.tracer.log_saccade(
                frame_num,
                prev_pos if isinstance(prev_pos, list) else list(prev_pos),
                curr_pos if isinstance(curr_pos, list) else list(curr_pos),
                scene_graph.current_node.id if scene_graph.current_node.id >= 0 else None,
                None  # Target node not known yet
            )
        
        # Update graph
        prev_node_id = scene_graph.current_node.id
        next_node = scene_graph.update_graph(
            tracking_data['potential_labels'],
            tracking_data['visit'],
            tracking_data['keypoints'],
            tracking_data['descriptors'],
            prev_pos,
            curr_pos
        )
        
        if next_node.id != prev_node_id:
            logger.info(f"- New node created: {next_node.id}")
            # Trace node addition
            if self.tracer:
                features = {}  # Placeholder for node features
                self.tracer.log_node_added(
                    frame_num,
                    next_node.id,
                    next_node.object_label,
                    list(curr_pos) if not isinstance(curr_pos, list) else curr_pos,
                    features
                )
        else:
            logger.info(f"- Merged with existing node: {next_node.id}")
        
        # Trace edge addition if it's a new node
        if self.tracer and prev_node_id >= 0 and next_node.id != prev_node_id:
            self.tracer.log_edge_added(
                frame_num,
                prev_node_id,
                next_node.id,
                "saccade",
                {"angle": None}  # Placeholder for edge properties
            )
        
        # Update node features
        next_node.update_features(
            tracking_data['node_data'],
            self.vid_lengths[tracking_data['video_name']],
            frame_num,
            tracking_data['relative_frame_num'],
            -1,  # Placeholder for timestamp fraction
            self.labels_to_int,
            len(self.obj_labels)
        )
        
        if next_node.id in tracking_data['node_data']:
            logger.info(f"- Updated node features: visits={len(next_node.visits)}, "
                  f"total_frames={next_node.get_visit_duration()}")
        else:
            logger.info(f"- Created new node features")
        
        # Log frame processing with saccade info and current node
        if self.tracer:
            self.tracer.log_frame_processed(
                frame_num,
                curr_pos if isinstance(curr_pos, list) else list(curr_pos),
                gaze_type,
                None,  # No ROI for saccade frame
                next_node.id
            )
    
    def _process_final_fixation(
        self,
        scene_graph: Graph,
        tracking_data: Dict[str, Any],
        gaze: Any
    ) -> None:
        """Process the final fixation if the video ends during a fixation."""
        logger.info(f"- Final fixation detected, updating graph...")
        tracking_data['visit'].append(tracking_data['relative_frame_num'] - 1)
        
        # Get last valid gaze position
        last_gaze_pos = gaze[min(tracking_data['frame_num'], len(gaze)-1), :2]
        
        scene_graph.update_graph(
            tracking_data['potential_labels'],
            tracking_data['visit'], 
            tracking_data['keypoints'],
            tracking_data['descriptors'],
            tracking_data['prev_gaze_pos'],
            last_gaze_pos
        )
        
        # Log final frame
        if self.tracer:
            self.tracer.log_frame_processed(
                tracking_data['frame_num'],
                last_gaze_pos if isinstance(last_gaze_pos, list) else list(last_gaze_pos),
                1,  # Assume it's a fixation
                None,  # No ROI
                scene_graph.current_node.id if scene_graph.current_node.id >= 0 else None
            )
    
    def _save_graph_state(
        self,
        scene_graph: Graph,
        tracking_data: Dict[str, Any],
        records_for_vid: List[Record],
        vid_length: int,
        timestamps: List[int],
        gaze: Any,
        results: Dict[str, List]
    ) -> None:
        """Save the current state of the graph at a timestamp."""
        # Get future action labels
        frame_num = tracking_data['frame_num']
        action_labels = get_future_action_labels(records_for_vid, frame_num, self.action_to_idx)
        if action_labels.numel() == 0:
            logger.info(f"[Frame {frame_num}] Skipping timestamp - insufficient action data")
            return
        
        logger.info(f"\n[Frame {frame_num}] Saving graph state:")
        logger.info(f"- Current nodes: {scene_graph.num_nodes}")
        logger.info(f"- Edge count: {len(scene_graph.edge_data)}")
        
        # Calculate timestamp fraction
        timestamp_ratios = self.config.dataset.timestamps[self.split]
        timestamp_fraction = timestamp_ratios[timestamps.index(frame_num)] if frame_num < len(gaze) else frame_num / vid_length
        
        # Update node features for all nodes
        self._update_all_node_features(
            scene_graph, 
            tracking_data['node_data'], 
            frame_num, 
            tracking_data['relative_frame_num'], 
            timestamp_fraction
        )
        
        # Extract and normalize features
        node_features, edge_indices, edge_features = scene_graph.extract_features(
            tracking_data['node_data'], tracking_data['relative_frame_num'], timestamp_fraction
        )
        
        # Store results
        results['x'].append(node_features)
        results['edge_index'].append(edge_indices)
        results['edge_attr'].append(edge_features)
        results['y'].append(action_labels)
    
    def _update_all_node_features(
        self,
        scene_graph: Graph,
        node_data: Dict[int, torch.Tensor],
        frame_num: int,
        relative_frame_num: int,
        timestamp_fraction: float
    ) -> None:
        """Update features for all nodes in the graph."""
        for node in scene_graph.get_all_nodes():
            if node.id >= 0:  # Skip root
                node.update_features(
                    node_data,
                    self.vid_lengths[tracking_data['video_name']],
                    frame_num,
                    relative_frame_num,
                    timestamp_fraction,
                    self.labels_to_int,
                    len(self.obj_labels)
                )
    
    def _print_final_graph(self, scene_graph: Graph) -> None:
        """Print the final graph structure if requested."""
        if scene_graph.num_nodes > 0:
            logger.info("\nFinal graph structure:")
            scene_graph.print_graph()
        else:
            logger.info('\nError: No nodes were added to the graph. Video may be empty or no fixations occurred.')

    def _take_graph_snapshot(self, graph: Graph, frame_num: int) -> None:
        """
        Take a snapshot of the current graph state.
        
        Args:
            graph: Graph to snapshot
            frame_num: Current frame number
        """
        if not self.tracer:
            return
            
        # Create a serializable representation of the graph
        nodes = []
        for node in graph.get_all_nodes():
            nodes.append({
                "id": node.id,
                "label": node.object_label,
                "visits": node.visits if hasattr(node, 'visits') else []
            })
        
        edges = []
        for source in graph.get_all_nodes():
            if not hasattr(source, 'neighbors'):
                continue
                
            for target, angle, edge_type in source.neighbors:
                edges.append({
                    "source": source.id,
                    "target": target.id,
                    "angle": float(angle) if isinstance(angle, (int, float)) else None,
                    "type": edge_type
                })
        
        self.tracer.log_graph_snapshot(frame_num, {
            "nodes": nodes,
            "edges": edges,
            "current_node": graph.current_node.id if graph.current_node != graph.root else -1
        })


def build_graph(video_list: List[str], config: DotDict, split: str, print_graph: bool = False, desc: Optional[str] = None, enable_tracing: bool = False) -> Dict:
    """
    Build graph representations for a list of videos.
    
    Args:
        video_list: List of video names to process
        config: Configuration dictionary
        split: Dataset split ('train' or 'val')
        print_graph: Whether to print graph visualization
        desc: Description for the progress bar
        enable_tracing: Whether to enable tracing for graph construction visualization
        
    Returns:
        Dictionary with node features, edge indices, edge attributes, and labels
    """
    logger.info(f"Building graph for {len(video_list)} videos in {split} split")
    
    # Initialize graph builder
    builder = GraphBuilder(config, split, enable_tracing=enable_tracing)
    
    # Process each video
    all_node_data = []
    all_edge_data = []
    all_edge_indices = []
    all_labels = []
    
    for video_name in tqdm(video_list, desc=desc or f"Processing {split} videos"):
        if enable_tracing:
            logger.info(f"Building graph with tracing for {video_name}")
        
        # Process the video and collect features
        result = builder.process_video(video_name, print_graph)
        
        # Always collect features
        all_node_data.extend(result['x'])
        all_edge_indices.extend(result['edge_index'])
        all_edge_data.extend(result['edge_attr'])
        all_labels.extend(result['y'])
        
    return {
        'x': all_node_data,
        'edge_index': all_edge_indices,
        'edge_attr': all_edge_data,
        'y': all_labels
    }