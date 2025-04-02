"""
Graph building module for creating scene graphs from video data.
"""

import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from collections import defaultdict
from tqdm import tqdm

from graph.graph import Graph, GraphCheckpoint
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
        self.enable_tracing = enable_tracing
        self.tracer = GraphTracer(self.config.directories.repo.traces, "", enabled=False)
        
        # Initialize models
        self.clip_model = ClipModel(self.config.models.clip.model_id)
        self.clip_model.load_model(Path(self.config.models.clip.model_dir))
        
        # Load object labels
        noun_idx_path = Path(self.config.dataset.egtea.noun_idx_file)
        self.obj_labels, self.labels_to_int = DataLoader.load_object_labels(noun_idx_path)
        self.clip_labels = [f"a picture of a {obj}" for obj in self.obj_labels.values()]
        
        # Load dataset information
        ann_file = (self.config.dataset.ego_topo.splits.train if self.split == 'train' 
                   else self.config.dataset.ego_topo.splits.val)
        self.vid_lengths = DataLoader.load_video_lengths(ann_file)
        self.records, self.records_by_vid = DataLoader.load_records(ann_file)
        self.action_to_idx = DataLoader.create_action_index(self.records)
        
        # Tracking state variables
        self.prev_gaze_pos = (-1, -1)
        self.potential_labels = defaultdict(int)
        self.visit_start = -1
        self.visit_end = -1
        self.frame_num = 0
        self.relative_frame_num = 0
        
        # Video processing variables
        self.scene_graph = None
        self.video_name = ""
        self.gaze_data = None
        self.records_current = None
        self.vid_length = 0
        self.timestamps = []
    
    def process_video(self, video_name: str, print_graph: bool = False) -> List[GraphCheckpoint]:
        """
        Process a video to build its scene graph.
        
        Args:
            video_name: Name of the video to process
            print_graph: Whether to print the final graph structure
            
        Returns:
            List of GraphCheckpoint objects representing graph states at different timestamps
        """
        logger.info(f"\nProcessing video: {video_name}")
        
        # Store video name
        self.video_name = video_name
        
        # Update tracer with new video name
        self.tracer = GraphTracer(self.config.directories.repo.traces, video_name, enabled=self.enable_tracing)
        if self.enable_tracing:
            logger.info(f"Tracing enabled for {video_name}")
        
        # Load video data
        self.gaze_data = parse_gtea_gaze(str(Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt"))
        video_processor = VideoProcessor(Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4")
        self.records_current = self.records_by_vid[video_name]
        self.vid_length = self.vid_lengths[video_name]
        self.timestamps = [int(ratio * self.vid_length) for ratio in sorted(self.config.dataset.timestamps[self.split])]
        
        # Initialize scene graph
        self.scene_graph = Graph()
        self.scene_graph.tracer = self.tracer
        
        # Reset tracking state
        self._reset_tracking_state()

        # Process video frames
        for frame, _, is_black_frame in video_processor:
            # Process current frame
            should_continue = self._process_frame(frame, is_black_frame)
            
            if not should_continue:
                break
            
            # Update frame counters
            self.relative_frame_num += 1
            self.frame_num += 1

        # Handle final fixation if video ends during fixation
        if self.potential_labels:
            self._finish_final_fixation()
        
        # Print final graph if requested
        if print_graph and self.scene_graph.num_nodes > 0:
            logger.info("\nFinal graph structure:")
            self.scene_graph.print_graph()
        
        return self.scene_graph.get_checkpoints()

    def _reset_tracking_state(self):
        """Reset all tracking state variables."""
        self.prev_gaze_pos = (-1, -1)
        self.potential_labels = defaultdict(int)
        self.visit_start = -1
        self.visit_end = -1
        self.frame_num = 0
        self.relative_frame_num = 0

    def _process_frame(
        self, 
        frame: torch.Tensor,
        is_black_frame: bool
    ) -> bool:
        """Process a single frame and update the scene graph accordingly.
        
        Returns:
            bool: False if processing should stop, True to continue
        """
        # Get current gaze data if available
        gaze_pos = None
        gaze_type = None
        if self.frame_num < len(self.gaze_data):
            gaze_pos = self.gaze_data[self.frame_num, :2].tolist()
            gaze_type = int(self.gaze_data[self.frame_num, 2])

        # Log frame processing
        node_id = self.scene_graph.current_node.id if self.scene_graph.current_node.id >= 0 else None
        self.tracer.log_frame(self.frame_num, gaze_pos, gaze_type, node_id)

        # Skip black frames without tracing
        if is_black_frame:
            return True

        # Check if we need to save graph state at this timestamp
        if self.scene_graph.edges and (self.frame_num in self.timestamps or self.frame_num >= len(self.gaze_data)):
            self._save_graph_state()
            
            # Exit if we've reached the final condition
            if self.frame_num == self.timestamps[-1] or self.frame_num >= len(self.gaze_data):
                logger.info(f"[Frame {self.frame_num}] Reached final timestamp or end of gaze data")
                return False

        # Only process and trace frames with valid gaze data
        if self.frame_num < len(self.gaze_data):
            self._process_frame_with_gaze(frame)
            
        return True

    def _process_frame_with_gaze(
        self,
        frame: torch.Tensor
    ) -> None:
        """Process a frame using available gaze data."""
        gaze_type = int(self.gaze_data[self.frame_num, 2])
        gaze_pos = self.gaze_data[self.frame_num, :2]
        
        if gaze_type == 1:  # Fixation
            self._handle_fixation(frame, gaze_pos)
        elif gaze_type == 2 and self.potential_labels:  # Saccade after fixation
            self._handle_saccade(gaze_pos)

    def _handle_fixation(
        self,
        frame: torch.Tensor,
        gaze_pos: Tuple[float, float]
    ) -> None:
        """Handle a fixation frame."""
        # Record start of visit if this is the first fixation
        if self.visit_start == -1:
            self.visit_start = self.relative_frame_num
            logger.info(f"\n[Frame {self.frame_num}] New fixation started at ({gaze_pos[0]:.1f}, {gaze_pos[1]:.1f})")
        
        # Convert normalized gaze coordinates (0-1) to pixel coordinates
        _, H, W = frame.shape
        gaze_x, gaze_y = int(gaze_pos[0] * W), int(gaze_pos[1] * H)
        
        # Extract region of interest
        roi, roi_bbox = get_roi(frame, (gaze_x, gaze_y), 256)
        logger.debug(f"[Frame {self.frame_num}] ROI bounding box: {roi_bbox}")

        # Run object detection on the region of interest
        current_label = self.clip_model.run_inference(roi, self.clip_labels, self.obj_labels)
        self.potential_labels[current_label] += 1
        
        logger.info(f"[Frame {self.frame_num}] CLIP detected: {current_label} (count: {self.potential_labels[current_label]})")
        
        # Get the current most likely label (the one with the highest count)
        most_likely_label = max(self.potential_labels.items(), key=lambda x: x[1])[0]
        
        self.tracer.log_gaze_object_detected(
            self.frame_num,
            most_likely_label,
            current_label,
            roi_bbox,
            dict(self.potential_labels)  # Convert defaultdict to regular dict
        )
    
    def _handle_saccade(
        self,
        curr_gaze_pos: Tuple[float, float]
    ) -> None:
        """Handle a saccade between fixations."""
        # Record end of fixation visit
        self.visit_end = self.relative_frame_num - 1
        fixation_duration = self.visit_end - self.visit_start + 1
        
        # Get most likely object label
        most_likely_label = max(self.potential_labels.items(), key=lambda x: x[1])[0]
        
        logger.info(f"\n[Frame {self.frame_num}] Saccade detected:")
        logger.info(f"- Most likely object: {most_likely_label}")
        logger.info(f"- Visit duration: {fixation_duration} frames")
        logger.debug(f"- Current gaze position (normalized): ({curr_gaze_pos[0]:.2f}, {curr_gaze_pos[1]:.2f})")
        
        # Update graph with new observation using normalized gaze coordinates
        prev_node_id = self.scene_graph.current_node.id
        visit_record = [self.visit_start, self.visit_end]
        next_node = self.scene_graph.update_graph(
            self.potential_labels,
            visit_record,
            self.prev_gaze_pos,
            curr_gaze_pos
        )
        
        # Log node creation/update
        if next_node.id != prev_node_id:
            logger.info(f"- New node created: {next_node.id}")
            
            # Log node addition
            self.tracer.log_node_added(self.frame_num, next_node.id, next_node.object_label, next_node.get_features())
            
            # Log edge addition if applicable
            if prev_node_id >= 0:
                # Find the edge connecting previous and current node
                edge = self.scene_graph.get_edge(prev_node_id, next_node.id)
                if edge:
                    # Get edge features
                    edge_features = edge.get_features()
                    self.tracer.log_edge_added(self.frame_num, prev_node_id, next_node.id, "saccade", edge_features)
        else:
            logger.info(f"- Merged with existing node: {next_node.id}")
            logger.info(f"- Updated node features: visits={len(next_node.visits)}, "
                      f"total_frames={next_node.get_visit_duration()}")
        
        # Reset fixation tracking data for next fixation
        self._reset_fixation_state(curr_gaze_pos)
    
    def _reset_fixation_state(self, curr_gaze_pos: Tuple[float, float]):
        """Reset the state related to the current fixation."""
        self.visit_start = -1
        self.visit_end = -1
        self.prev_gaze_pos = curr_gaze_pos
        self.potential_labels = defaultdict(int)
    
    def _finish_final_fixation(self) -> None:
        """Process the final fixation if video ends during one."""
        logger.info("- Final fixation detected, updating graph...")
        
        # Record end of visit
        self.visit_end = self.relative_frame_num - 1
        visit_record = [self.visit_start, self.visit_end]
        
        # Get last valid gaze position (already normalized to [0,1])
        last_frame = min(self.frame_num, len(self.gaze_data)-1)
        last_gaze_pos = self.gaze_data[last_frame, :2]
        
        logger.debug(f"- Final gaze position (normalized): ({last_gaze_pos[0]:.2f}, {last_gaze_pos[1]:.2f})")
        
        # Update graph with final observation using normalized gaze coordinates
        self.scene_graph.update_graph(
            self.potential_labels,
            visit_record, 
            self.prev_gaze_pos,
            last_gaze_pos
        )
    
    def _save_graph_state(self) -> None:
        """
        Save the current state of the graph at a timestamp.
        """
        # Get future action labels
        action_labels = get_future_action_labels(self.records_current, self.frame_num, self.action_to_idx)
        
        # Calculate timestamp fraction
        timestamp_ratios = self.config.dataset.timestamps[self.split]
        if self.frame_num < len(self.gaze_data):
            timestamp_idx = self.timestamps.index(self.frame_num) 
            timestamp_fraction = timestamp_ratios[timestamp_idx]
        else:
            timestamp_fraction = self.frame_num / self.vid_length
        
        # Save checkpoint directly in the graph
        self.scene_graph.save_checkpoint(
            self.vid_length,
            self.frame_num,
            self.relative_frame_num,
            timestamp_fraction,
            self.labels_to_int,
            len(self.obj_labels),
            action_labels
        )

def build_graph(
    video_list: List[str], 
    config: DotDict, 
    split: str, 
    print_graph: bool = False, 
    desc: Optional[str] = None, 
    enable_tracing: bool = False
) -> Dict:
    """
    Build graph representations for a list of videos.
    
    Args:
        video_list: List of video names to process
        config: Configuration dictionary
        split: Dataset split ('train' or 'val')
        print_graph: Whether to print final graph structure
        desc: Description for progress bar
        enable_tracing: Whether to enable detailed tracing
        
    Returns:
        Dictionary with x, edge_index, edge_attr, and y keys for all videos
    """
    logger.info(f"Building graph for {len(video_list)} videos in {split} split")
    
    # Initialize graph builder
    builder = GraphBuilder(config, split, enable_tracing=enable_tracing)
    
    # Process each video and collect checkpoints
    all_checkpoints = []
    progress_desc = desc or f"Processing {split} videos"
    
    for video_name in tqdm(video_list, desc=progress_desc):
        if enable_tracing:
            logger.info(f"Building graph with tracing for {video_name}")
        
        # Process video and collect checkpoints
        video_checkpoints = builder.process_video(video_name, print_graph)
        
        # Add video checkpoints to overall list
        all_checkpoints.extend(video_checkpoints)
    
    # Convert all checkpoints to dataset format
    dataset = {
        'x': [],
        'edge_index': [],
        'edge_attr': [],
        'y': []
    }
    
    for checkpoint in all_checkpoints:
        x, edge_index, edge_attr, y = checkpoint.dataset_format
        dataset['x'].append(x)
        dataset['edge_index'].append(edge_index)
        dataset['edge_attr'].append(edge_attr)
        dataset['y'].append(y)
    
    return dataset