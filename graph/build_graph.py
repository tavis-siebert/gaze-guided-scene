"""
Graph building module for creating scene graphs from video data.
"""

import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
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

class GraphCheckpoint:
    """
    Encapsulates graph state at a specific timestamp.
    
    Stores features, edges, and labels for a graph at a given frame.
    """
    
    def __init__(
        self, 
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        action_labels: Dict[str, torch.Tensor]
    ):
        """
        Initialize a new checkpoint.
        
        Args:
            node_features: Tensor of node features
            edge_index: Tensor of edge indices
            edge_attr: Tensor of edge attributes
            action_labels: Dictionary of action labels
        """
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.action_labels = action_labels
    
    @property
    def dataset_format(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get data in dataset-compatible format.
        
        Returns:
            Tuple of (x, edge_index, edge_attr, y) where y is the future_actions tensor
        """
        return (
            self.node_features,
            self.edge_index,
            self.edge_attr,
            self.action_labels['future_actions'].unsqueeze(0)
        )

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
        self.sift = SIFT()
        
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
    
    def process_video(self, video_name: str, print_graph: bool = False) -> Dict[str, List]:
        """
        Process a video to build its scene graph.
        
        Args:
            video_name: Name of the video to process
            print_graph: Whether to print the final graph structure
            
        Returns:
            Dictionary with x, edge_index, edge_attr, and y lists for the video
        """
        logger.info(f"\nProcessing video: {video_name}")
        
        # Update tracer with new video name
        self.tracer = GraphTracer(self.config.directories.repo.traces, video_name, enabled=self.enable_tracing)
        if self.enable_tracing:
            logger.info(f"Tracing enabled for {video_name}")
        
        # Load video data
        gaze_data = parse_gtea_gaze(str(Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt"))
        video_processor = VideoProcessor(Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4")
        records = self.records_by_vid[video_name]
        vid_length = self.vid_lengths[video_name]
        timestamps = [int(ratio * vid_length) for ratio in sorted(self.config.dataset.timestamps[self.split])]
        
        # Initialize structures
        scene_graph = Graph()
        # Set the tracer in the scene graph
        scene_graph.tracer = self.tracer
        
        tracking = {
            'video_name': video_name,
            'prev_gaze_pos': (-1, -1),
            'potential_labels': defaultdict(int),
            'keypoints': [],
            'descriptors': [],
            'visit': [],
            'frame_num': 0,
            'relative_frame_num': 0,
            'node_data': {}
        }
        results = {'x': [], 'edge_index': [], 'edge_attr': [], 'y': []}

        # Process video frames
        for frame, _, is_black_frame in video_processor:
            frame_num = tracking['frame_num']
            
            # Process current frame
            should_continue = self._process_frame(
                frame, is_black_frame, frame_num, scene_graph, tracking,
                timestamps, gaze_data, records, vid_length, results
            )
            
            if not should_continue:
                break
            
            # Update frame counters
            tracking['relative_frame_num'] += 1
            tracking['frame_num'] += 1

        # Handle final fixation if video ends during fixation
        if tracking['potential_labels']:
            self._finish_final_fixation(scene_graph, tracking, gaze_data)
        
        # Print final graph if requested
        if print_graph and scene_graph.num_nodes > 0:
            logger.info("\nFinal graph structure:")
            scene_graph.print_graph()
        
        return results

    def _process_frame(
        self, 
        frame: torch.Tensor,
        is_black_frame: bool,
        frame_num: int,
        scene_graph: Graph,
        tracking: Dict[str, Any],
        timestamps: List[int],
        gaze_data: Any,
        records: List[Record],
        vid_length: int,
        results: Dict[str, List]
    ) -> bool:
        """Process a single frame and update the scene graph accordingly.
        
        Returns:
            bool: False if processing should stop, True to continue
        """
        # Get current gaze data if available
        gaze_pos = None
        gaze_type = None
        if frame_num < len(gaze_data):
            gaze_pos = gaze_data[frame_num, :2].tolist()
            gaze_type = int(gaze_data[frame_num, 2])

        # Log frame processing
        node_id = scene_graph.current_node.id if scene_graph.current_node.id >= 0 else None
        self.tracer.log_frame(frame_num, gaze_pos, gaze_type, node_id)

        # Skip black frames without tracing
        if is_black_frame:
            return True

        # Check if we need to save graph state at this timestamp
        if scene_graph.edges and (frame_num in timestamps or frame_num >= len(gaze_data)):
            self._save_graph_state(scene_graph, tracking, records, frame_num, 
                                 timestamps, gaze_data, vid_length, results)
            
            # Exit if we've reached the final condition
            if frame_num == timestamps[-1] or frame_num >= len(gaze_data):
                logger.info(f"[Frame {frame_num}] Reached final timestamp or end of gaze data")
                return False

        # Only process and trace frames with valid gaze data
        if frame_num < len(gaze_data):
            self._process_frame_with_gaze(frame, frame_num, gaze_data, tracking, scene_graph)
            
        return True

    def _process_frame_with_gaze(
        self,
        frame: torch.Tensor,
        frame_num: int,
        gaze_data: Any,
        tracking: Dict[str, Any],
        scene_graph: Graph
    ) -> None:
        """Process a frame using available gaze data."""
        gaze_type = int(gaze_data[frame_num, 2])
        gaze_pos = gaze_data[frame_num, :2]
        
        if gaze_type == 1:  # Fixation
            self._handle_fixation(frame, frame_num, gaze_pos, tracking)
        elif gaze_type == 2 and tracking['potential_labels']:  # Saccade after fixation
            self._handle_saccade(frame_num, tracking, curr_gaze_pos=gaze_pos, scene_graph=scene_graph)

    def _handle_fixation(
        self,
        frame: torch.Tensor,
        frame_num: int,
        gaze_pos: Tuple[float, float],
        tracking: Dict[str, Any]
    ) -> None:
        """Handle a fixation frame."""
        # Record start of visit if this is the first fixation
        if not tracking['visit']:
            tracking['visit'].append(tracking['relative_frame_num'])
            logger.info(f"\n[Frame {frame_num}] New fixation started at ({gaze_pos[0]:.1f}, {gaze_pos[1]:.1f})")
        
        # Convert normalized gaze coordinates (0-1) to pixel coordinates
        _, H, W = frame.shape
        gaze_x, gaze_y = int(gaze_pos[0] * W), int(gaze_pos[1] * H)
        
        # Extract region of interest
        roi, roi_bbox = get_roi(frame, (gaze_x, gaze_y), 256)
        logger.debug(f"[Frame {frame_num}] ROI bounding box: {roi_bbox}")

        # Run object detection on the region of interest
        current_label = self.clip_model.run_inference(roi, self.clip_labels, self.obj_labels)
        tracking['potential_labels'][current_label] += 1
        
        # Extract SIFT features
        kp, desc = self.sift.extract_features(frame)
        tracking['keypoints'].append(kp)
        tracking['descriptors'].append(desc)
        
        logger.info(f"[Frame {frame_num}] CLIP detected: {current_label} (count: {tracking['potential_labels'][current_label]})")
        
        # Get the current most likely label (the one with the highest count)
        most_likely_label = max(tracking['potential_labels'].items(), key=lambda x: x[1])[0]
        
        self.tracer.log_gaze_object_detected(
            frame_num,
            most_likely_label,
            current_label,
            roi_bbox,
            dict(tracking['potential_labels'])  # Convert defaultdict to regular dict
        )
    
    def _handle_saccade(
        self,
        frame_num: int,
        tracking: Dict[str, Any],
        curr_gaze_pos: Tuple[float, float],
        scene_graph: Graph
    ) -> None:
        """Handle a saccade between fixations."""
        # Record end of fixation visit
        tracking['visit'].append(tracking['relative_frame_num'] - 1)
        fixation_duration = tracking['visit'][1] - tracking['visit'][0] + 1
        
        # Get most likely object label
        most_likely_label = max(tracking['potential_labels'].items(), key=lambda x: x[1])[0]
        
        logger.info(f"\n[Frame {frame_num}] Saccade detected:")
        logger.info(f"- Most likely object: {most_likely_label}")
        logger.info(f"- Visit duration: {fixation_duration} frames")
        logger.debug(f"- Current gaze position (normalized): ({curr_gaze_pos[0]:.2f}, {curr_gaze_pos[1]:.2f})")
        
        # Update graph with new observation using normalized gaze coordinates
        prev_node_id = scene_graph.current_node.id
        next_node = scene_graph.update_graph(
            tracking['potential_labels'],
            tracking['visit'],
            tracking['keypoints'],
            tracking['descriptors'],
            tracking['prev_gaze_pos'],
            curr_gaze_pos
        )
        
        # Log node creation/update
        if next_node.id != prev_node_id:
            logger.info(f"- New node created: {next_node.id}")
            
            # Log node addition
            self.tracer.log_node_added(frame_num, next_node.id, next_node.object_label, next_node.get_features())
            
            # Log edge addition if applicable
            if prev_node_id >= 0:
                # Find the edge connecting previous and current node
                edge = scene_graph.get_edge(prev_node_id, next_node.id)
                if edge:
                    # Get edge features
                    edge_features = edge.get_features()
                    self.tracer.log_edge_added(frame_num, prev_node_id, next_node.id, "saccade", edge_features)
        else:
            logger.info(f"- Merged with existing node: {next_node.id}")
        
        # Update node features
        tracking['node_data'][next_node.id] = next_node.get_features_tensor(
            self.vid_lengths[tracking['video_name']],
            frame_num,
            tracking['relative_frame_num'],
            -1,  # Placeholder for timestamp fraction
            self.labels_to_int,
            len(self.obj_labels)
        )
        
        # Log feature update
        if next_node.id in tracking['node_data']:
            logger.info(f"- Updated node features: visits={len(next_node.visits)}, "
                      f"total_frames={next_node.get_visit_duration()}")
        else:
            logger.info("- Created new node features")
        
        # Reset tracking data for next fixation
        tracking['visit'] = []
        tracking['keypoints'], tracking['descriptors'] = [], []
        tracking['prev_gaze_pos'] = curr_gaze_pos
        tracking['potential_labels'] = defaultdict(int)
    
    def _finish_final_fixation(
        self,
        scene_graph: Graph,
        tracking: Dict[str, Any],
        gaze_data: Any
    ) -> None:
        """Process the final fixation if video ends during one."""
        logger.info("- Final fixation detected, updating graph...")
        
        # Record end of visit
        tracking['visit'].append(tracking['relative_frame_num'] - 1)
        
        # Get last valid gaze position (already normalized to [0,1])
        last_frame = min(tracking['frame_num'], len(gaze_data)-1)
        last_gaze_pos = gaze_data[last_frame, :2]
        
        logger.debug(f"- Final gaze position (normalized): ({last_gaze_pos[0]:.2f}, {last_gaze_pos[1]:.2f})")
        
        # Update graph with final observation using normalized gaze coordinates
        scene_graph.update_graph(
            tracking['potential_labels'],
            tracking['visit'], 
            tracking['keypoints'],
            tracking['descriptors'],
            tracking['prev_gaze_pos'],
            last_gaze_pos
        )
    
    def _save_graph_state(
        self,
        scene_graph: Graph,
        tracking: Dict[str, Any],
        records: List[Record],
        frame_num: int,
        timestamps: List[int],
        gaze_data: Any,
        vid_length: int,
        results: Dict[str, List]
    ) -> None:
        """Save the current state of the graph at a timestamp."""
        # Get future action labels
        action_labels = get_future_action_labels(records, frame_num, self.action_to_idx)
        if action_labels is None:
            logger.info(f"[Frame {frame_num}] Skipping timestamp - insufficient action data")
            return
        
        logger.info(f"\n[Frame {frame_num}] Saving graph state:")
        logger.info(f"- Current nodes: {scene_graph.num_nodes}")
        logger.info(f"- Edge count: {len(scene_graph.edges)}")
        
        # Calculate timestamp fraction
        timestamp_ratios = self.config.dataset.timestamps[self.split]
        if frame_num < len(gaze_data):
            timestamp_idx = timestamps.index(frame_num) 
            timestamp_fraction = timestamp_ratios[timestamp_idx]
        else:
            timestamp_fraction = frame_num / vid_length
        
        # Get graph features using the scene graph's method
        node_features, edge_indices, edge_features = scene_graph.get_features_tensor(
            vid_length,
            frame_num,
            tracking['relative_frame_num'],
            timestamp_fraction,
            self.labels_to_int,
            len(self.obj_labels),
            tracking['node_data']
        )
        
        # Create a checkpoint
        checkpoint = GraphCheckpoint(
            node_features=node_features,
            edge_index=edge_indices,
            edge_attr=edge_features,
            action_labels=action_labels
        )
        
        # Store results in dataset format
        x, edge_index, edge_attr, y = checkpoint.dataset_format
        results['x'].append(x)
        results['edge_index'].append(edge_index)
        results['edge_attr'].append(edge_attr)
        results['y'].append(y)

def dataset_from_checkpoints(checkpoints: List[GraphCheckpoint]) -> Dict[str, List]:
    """
    Convert a list of checkpoints to a dataset dictionary.
    
    Args:
        checkpoints: List of GraphCheckpoint objects
        
    Returns:
        Dictionary with x, edge_index, edge_attr, and y keys
    """
    dataset = {
        'x': [],
        'edge_index': [],
        'edge_attr': [],
        'y': []
    }
    
    for checkpoint in checkpoints:
        x, edge_index, edge_attr, y = checkpoint.dataset_format
        dataset['x'].append(x)
        dataset['edge_index'].append(edge_index)
        dataset['edge_attr'].append(edge_attr)
        dataset['y'].append(y)
    
    return dataset

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
        
        # Process video and collect individual checkpoints
        video_results = builder.process_video(video_name, print_graph)
        
        # Convert individual results into checkpoints
        for i in range(len(video_results['x'])):
            checkpoint = GraphCheckpoint(
                node_features=video_results['x'][i],
                edge_index=video_results['edge_index'][i],
                edge_attr=video_results['edge_attr'][i],
                # Create a dummy action_labels dict since we already have the future_actions tensor
                action_labels={'future_actions': video_results['y'][i].squeeze(0)}
            )
            all_checkpoints.append(checkpoint)
    
    # Convert checkpoints to final dataset format
    return dataset_from_checkpoints(all_checkpoints)