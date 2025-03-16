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
from models.clip import ClipModel
from models.sift import SIFT
from egtea_gaze.gaze_data.gaze_io_sample import parse_gtea_gaze
from config.config_utils import DotDict


class GraphBuilder:
    """
    Builds scene graphs from video data and gaze information.
    """
    
    def __init__(self, config: DotDict, split: str):
        """
        Initialize the graph builder.
        
        Args:
            config: Configuration dictionary
            split: Dataset split ('train' or 'val')
        """
        self.config = config
        self.split = split
        self.clip_model = self._initialize_clip_model()
        self.sift = SIFT()
        
        # Load object labels and dataset information
        self._load_object_labels()
        self._load_dataset_info()
    
    def _initialize_clip_model(self) -> ClipModel:
        """Initialize and load the CLIP model."""
        model_id = "openai/clip-vit-base-patch16"
        model_dir = Path(self.config.paths.scratch_dir) / "egtea_gaze/clip_model"
        
        clip_model = ClipModel(model_id)
        clip_model.load_model(model_dir)
        
        return clip_model
    
    def _load_object_labels(self) -> None:
        """Load object labels and create CLIP-compatible label formats."""
        noun_idx_path = Path(self.config.paths.egtea_dir) / "action_annotation/noun_idx.txt"
        self.obj_labels, self.labels_to_int = DataLoader.load_object_labels(noun_idx_path)
        self.clip_labels = [f"a picture of a {obj}" for obj in self.obj_labels.values()]
    
    def _load_dataset_info(self) -> None:
        """Load dataset information including video lengths, records, and action indices."""
        ann_file = self.config.dataset.splits[self.split]
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
        print(f"\nProcessing video: {video_name}")
        
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
        timestamp_ratios = self.config.dataset[f"{self.split}_timestamps"]
        timestamps = [int(frac*vid_length) for frac in sorted(timestamp_ratios)]
        print(f"Total frames: {vid_length}, Timestamps: {timestamps}")
        return timestamps
    
    def _load_gaze_data(self, video_name: str) -> Any:
        """Load gaze data for the specified video."""
        gaze_path = Path(self.config.paths.egtea_dir) / "gaze_data/gaze_data" / f"{video_name}.txt"
        return parse_gtea_gaze(str(gaze_path))
    
    def _initialize_video_processor(self, video_name: str) -> VideoProcessor:
        """Initialize video processor for the specified video."""
        video_path = Path(self.config.paths.scratch_dir) / "egtea_gaze/raw_videos" / f"{video_name}.mp4"
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
            'node_data': {}
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
            # Skip black frames
            if is_black_frame:
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
                    print(f"[Frame {tracking_data['frame_num']}] Reached final timestamp or end of gaze data")
                    break
            
            # Process gaze data
            if tracking_data['frame_num'] < len(gaze):
                self._process_gaze_frame(frame, gaze, tracking_data, scene_graph)
            
            tracking_data['relative_frame_num'] += 1
            tracking_data['frame_num'] += 1
        
        # Handle final fixation if video ends during fixation
        if tracking_data['potential_labels']:
            self._process_final_fixation(scene_graph, tracking_data, gaze)
    
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
        gaze_type = gaze[frame_num, 2]
        
        if gaze_type == 1:  # Fixation
            self._handle_fixation(
                frame,
                gaze[frame_num, :2],
                tracking_data
            )
            
        elif gaze_type == 2:  # Saccade
            if tracking_data['potential_labels']:
                self._handle_saccade(
                    scene_graph,
                    tracking_data,
                    gaze[frame_num, :2]
                )
                
                # Reset tracking variables
                tracking_data['visit'] = []
                tracking_data['keypoints'], tracking_data['descriptors'] = [], []
                tracking_data['prev_gaze_pos'] = gaze[frame_num, :2]
                tracking_data['potential_labels'] = defaultdict(int)
    
    def _handle_fixation(
        self,
        frame: torch.Tensor,
        gaze_pos: Tuple[float, float],
        tracking_data: Dict[str, Any]
    ) -> None:
        """Handle a fixation frame."""
        # Record start of visit if this is the first fixation
        if not tracking_data['visit']:
            tracking_data['visit'].append(tracking_data['relative_frame_num'])
            x, y = gaze_pos
            print(f"\n[Frame {tracking_data['frame_num']}] New fixation started at ({x:.1f}, {y:.1f})")
        
        # Get object label from CLIP
        label = self.process_fixation(frame, gaze_pos)
        tracking_data['potential_labels'][label] += 1
        print(f"[Frame {tracking_data['frame_num']}] CLIP detected: {label} (count: {tracking_data['potential_labels'][label]})")
        
        # Extract features using SIFT
        kp, desc = self.sift.extract_features(frame)
        tracking_data['keypoints'].append(kp)
        tracking_data['descriptors'].append(desc)
    
    def _handle_saccade(
        self,
        scene_graph: Graph,
        tracking_data: Dict[str, Any],
        curr_pos: Tuple[float, float]
    ) -> None:
        """Handle a saccade (eye movement between fixations)."""
        # Record end of visit
        tracking_data['visit'].append(tracking_data['relative_frame_num'] - 1)
        
        # Get most likely object label
        most_likely_label = max(tracking_data['potential_labels'].items(), key=lambda x: x[1])[0]
        print(f"\n[Frame {tracking_data['frame_num']}] Saccade detected:")
        print(f"- Most likely object: {most_likely_label}")
        print(f"- Visit duration: {tracking_data['visit'][-1] - tracking_data['visit'][0] + 1} frames")
        
        # Update graph
        prev_node_id = scene_graph.current_node.id
        next_node = scene_graph.update_graph(
            tracking_data['potential_labels'],
            tracking_data['visit'],
            tracking_data['keypoints'],
            tracking_data['descriptors'],
            tracking_data['prev_gaze_pos'],
            curr_pos
        )
        
        if next_node.id != prev_node_id:
            print(f"- New node created: {next_node.id}")
        else:
            print(f"- Merged with existing node: {next_node.id}")
        
        # Update node features
        next_node.update_features(
            tracking_data['node_data'],
            self.vid_lengths[self._get_current_video_name(tracking_data)],
            tracking_data['frame_num'],
            tracking_data['relative_frame_num'],
            -1,  # Placeholder for timestamp fraction
            self.labels_to_int,
            len(self.obj_labels)
        )
        
        if next_node.id in tracking_data['node_data']:
            print(f"- Updated node features: visits={len(next_node.visits)}, "
                  f"total_frames={next_node.get_visit_duration()}")
        else:
            print(f"- Created new node features")
    
    def _get_current_video_name(self, tracking_data: Dict[str, Any]) -> str:
        """Get the current video name from tracking data (placeholder implementation)."""
        # This is a placeholder - in a real implementation, you'd need to track the current video name
        # or pass it as a parameter to the relevant methods
        for video_name, length in self.vid_lengths.items():
            if tracking_data['frame_num'] < length:
                return video_name
        return list(self.vid_lengths.keys())[0]
    
    def _process_final_fixation(
        self,
        scene_graph: Graph,
        tracking_data: Dict[str, Any],
        gaze: Any
    ) -> None:
        """Process the final fixation if the video ends during a fixation."""
        print(f"- Final fixation detected, updating graph...")
        tracking_data['visit'].append(tracking_data['relative_frame_num'] - 1)
        scene_graph.update_graph(
            tracking_data['potential_labels'],
            tracking_data['visit'], 
            tracking_data['keypoints'],
            tracking_data['descriptors'],
            tracking_data['prev_gaze_pos'],
            gaze[min(tracking_data['frame_num'], len(gaze)-1), :2]
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
            print(f"[Frame {frame_num}] Skipping timestamp - insufficient action data")
            return
        
        print(f"\n[Frame {frame_num}] Saving graph state:")
        print(f"- Current nodes: {scene_graph.num_nodes}")
        print(f"- Edge count: {len(scene_graph.edge_data)}")
        
        # Calculate timestamp fraction
        timestamp_ratios = self.config.dataset[f"{self.split}_timestamps"]
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
                    self.vid_lengths[self._get_current_video_name({'frame_num': frame_num})],
                    frame_num,
                    relative_frame_num,
                    timestamp_fraction,
                    self.labels_to_int,
                    len(self.obj_labels)
                )
    
    def _print_final_graph(self, scene_graph: Graph) -> None:
        """Print the final graph structure if requested."""
        if scene_graph.num_nodes > 0:
            print("\nFinal graph structure:")
            scene_graph.print_graph()
        else:
            print('\nError: No nodes were added to the graph. Video may be empty or no fixations occurred.')


def build_graph(video_list: List[str], config: DotDict, split: str, print_graph: bool = False, desc: Optional[str] = None) -> Dict:
    """
    Build graph representations for a list of videos.
    
    Args:
        video_list: List of video names to process
        config: Configuration dictionary
        split: Dataset split ('train' or 'val')
        print_graph: Whether to print graph visualization
        desc: Description for the progress bar
        
    Returns:
        Dictionary with node features, edge indices, edge attributes, and labels
    """
    print(f"Building graph for {len(video_list)} videos in {split} split")
    
    # Initialize graph builder
    builder = GraphBuilder(config, split)
    
    # Process each video
    all_node_data = []
    all_edge_data = []
    all_edge_indices = []
    all_labels = []
    
    for video_name in tqdm(video_list, desc=desc or f"Processing {split} videos"):
        result = builder.process_video(video_name, print_graph)
        
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