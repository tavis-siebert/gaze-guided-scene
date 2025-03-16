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
        
        # Load object labels
        noun_idx_path = Path(config.paths.egtea_dir) / "action_annotation/noun_idx.txt"
        self.obj_labels, self.labels_to_int = DataLoader.load_object_labels(noun_idx_path)
        self.clip_labels = [f"a picture of a {obj}" for obj in self.obj_labels.values()]
        
        # Load dataset information
        ann_file = config.dataset.splits[split]
        self.vid_lengths = DataLoader.load_video_lengths(ann_file)
        self.records, self.records_by_vid = DataLoader.load_records(ann_file)
        
        # Create action index
        self.action_to_idx = DataLoader.create_action_index(self.records)
    
    def _initialize_clip_model(self) -> ClipModel:
        """Initialize and load the CLIP model."""
        model_id = "openai/clip-vit-base-patch16"
        model_dir = Path(self.config.paths.scratch_dir) / "egtea_gaze/clip_model"
        
        clip_model = ClipModel(model_id)
        clip_model.load_model(model_dir)
        
        return clip_model
    
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
        
        # Get video-specific data
        records_for_vid = self.records_by_vid[video_name]
        vid_length = self.vid_lengths[video_name]
        
        # Calculate timestamps
        timestamp_ratios = self.config.dataset[f"{self.split}_timestamps"]
        timestamps = [int(frac*vid_length) for frac in sorted(timestamp_ratios)]
        print(f"Total frames: {vid_length}, Timestamps: {timestamps}")
        
        # Load gaze data
        gaze_path = Path(self.config.paths.egtea_dir) / "gaze_data/gaze_data" / f"{video_name}.txt"
        gaze = parse_gtea_gaze(str(gaze_path))
        
        # Initialize scene graph and video processor
        scene_graph = Graph()
        video_path = Path(self.config.paths.scratch_dir) / "egtea_gaze/raw_videos" / f"{video_name}.mp4"
        video_processor = VideoProcessor(video_path)
        
        # Initialize tracking variables
        prev_gaze_pos = (-1, -1)
        potential_labels = defaultdict(int)
        keypoints, descriptors = [], []
        visit = []  # [start_frame, end_frame]
        frame_num, relative_frame_num = 0, 0
        node_data = {}
        
        # Results storage
        node_features_list = []
        edge_indices_list = []
        edge_features_list = []
        labels_list = []
        
        # Process video frames
        for frame, _, is_black_frame in video_processor:
            # Skip black frames
            if is_black_frame:
                frame_num += 1
                continue
                
            # Check if we need to save graph state at this timestamp
            if (frame_num in timestamps or frame_num >= len(gaze)) and scene_graph.edge_data:
                self._save_graph_state(
                    scene_graph, 
                    node_data,
                    records_for_vid,
                    frame_num,
                    relative_frame_num,
                    vid_length,
                    timestamp_ratios,
                    timestamps,
                    gaze,
                    node_features_list,
                    edge_indices_list,
                    edge_features_list,
                    labels_list
                )
                
                # Exit if we've reached the final timestamp
                if frame_num == timestamps[-1] or frame_num >= len(gaze):
                    print(f"[Frame {frame_num}] Reached final timestamp or end of gaze data")
                    break
            
            # Process gaze data
            if frame_num < len(gaze):
                gaze_type = gaze[frame_num, 2]
                
                if gaze_type == 1:  # Fixation
                    self._process_fixation(
                        frame,
                        gaze[frame_num, :2],
                        visit,
                        potential_labels,
                        keypoints,
                        descriptors,
                        relative_frame_num,
                        frame_num
                    )
                    
                elif gaze_type == 2:  # Saccade
                    if potential_labels:
                        self._process_saccade(
                            scene_graph,
                            potential_labels,
                            visit,
                            keypoints,
                            descriptors,
                            prev_gaze_pos,
                            gaze[frame_num, :2],
                            node_data,
                            vid_length,
                            frame_num,
                            relative_frame_num,
                            frame_num
                        )
                        
                        # Reset tracking variables
                        visit = []
                        keypoints, descriptors = [], []
                        prev_gaze_pos = gaze[frame_num, :2]
                        potential_labels = defaultdict(int)
            
            relative_frame_num += 1
            frame_num += 1
        
        # Handle final fixation if video ends during fixation
        if potential_labels:
            print(f"- Final fixation detected, updating graph...")
            visit.append(relative_frame_num - 1)
            scene_graph.update_graph(
                potential_labels,
                visit, 
                keypoints,
                descriptors,
                prev_gaze_pos,
                gaze[min(frame_num, len(gaze)-1), :2]
            )
        
        # Print final graph structure if requested
        if print_graph and scene_graph.num_nodes > 0:
            print("\nFinal graph structure:")
            scene_graph.print_graph()
        elif print_graph:
            print('\nError: No nodes were added to the graph. Video may be empty or no fixations occurred.')
        
        return {
            'x': node_features_list,
            'edge_index': edge_indices_list,
            'edge_attr': edge_features_list,
            'y': labels_list
        }
    
    def _process_fixation(
        self,
        frame: torch.Tensor,
        gaze_pos: Tuple[float, float],
        visit: List[int],
        potential_labels: Dict[str, int],
        keypoints: List,
        descriptors: List,
        relative_frame_num: int,
        frame_num: int
    ) -> None:
        """Process a fixation frame."""
        # Record start of visit if this is the first fixation
        if not visit:
            visit.append(relative_frame_num)
            x, y = gaze_pos
            print(f"\n[Frame {frame_num}] New fixation started at ({x:.1f}, {y:.1f})")
        
        # Get object label from CLIP
        label = self.process_fixation(frame, gaze_pos)
        potential_labels[label] += 1
        print(f"[Frame {frame_num}] CLIP detected: {label} (count: {potential_labels[label]})")
        
        # Extract features using SIFT
        kp, desc = self.sift.extract_features(frame)
        keypoints.append(kp)
        descriptors.append(desc)
    
    def _process_saccade(
        self,
        scene_graph: Graph,
        potential_labels: Dict[str, int],
        visit: List[int],
        keypoints: List,
        descriptors: List,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        node_data: Dict[int, torch.Tensor],
        vid_length: int,
        frame_num: int,
        relative_frame_num: int,
        current_timestamp: int
    ) -> None:
        """Process a saccade (eye movement between fixations)."""
        # Record end of visit
        visit.append(relative_frame_num - 1)
        
        # Get most likely object label
        most_likely_label = max(potential_labels.items(), key=lambda x: x[1])[0]
        print(f"\n[Frame {frame_num}] Saccade detected:")
        print(f"- Most likely object: {most_likely_label}")
        print(f"- Visit duration: {visit[-1] - visit[0] + 1} frames")
        
        # Update graph
        prev_node_id = scene_graph.current_node.id
        next_node = scene_graph.update_graph(
            potential_labels,
            visit,
            keypoints,
            descriptors,
            prev_pos,
            curr_pos
        )
        
        if next_node.id != prev_node_id:
            print(f"- New node created: {next_node.id}")
        else:
            print(f"- Merged with existing node: {next_node.id}")
        
        # Update node features
        next_node.update_features(
            node_data,
            vid_length,
            frame_num,
            relative_frame_num,
            -1,  # Placeholder for timestamp fraction
            self.labels_to_int,
            len(self.obj_labels)
        )
        
        if next_node.id in node_data:
            print(f"- Updated node features: visits={len(next_node.visits)}, "
                  f"total_frames={next_node.get_visit_duration()}")
        else:
            print(f"- Created new node features")
    
    def _save_graph_state(
        self,
        scene_graph: Graph,
        node_data: Dict[int, torch.Tensor],
        records_for_vid: List[Record],
        frame_num: int,
        relative_frame_num: int,
        vid_length: int,
        timestamp_ratios: List[float],
        timestamps: List[int],
        gaze: Any,
        node_features_list: List[torch.Tensor],
        edge_indices_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
        labels_list: List[torch.Tensor]
    ) -> None:
        """Save the current state of the graph at a timestamp."""
        # Get future action labels
        action_labels = get_future_action_labels(records_for_vid, frame_num, self.action_to_idx)
        if action_labels.numel() == 0:
            print(f"[Frame {frame_num}] Skipping timestamp - insufficient action data")
            return
        
        print(f"\n[Frame {frame_num}] Saving graph state:")
        print(f"- Current nodes: {scene_graph.num_nodes}")
        print(f"- Edge count: {len(scene_graph.edge_data)}")
        
        # Calculate timestamp fraction
        timestamp_fraction = timestamp_ratios[timestamps.index(frame_num)] if frame_num < len(gaze) else frame_num / vid_length
        
        # Update node features for all nodes
        for node in scene_graph.get_all_nodes():
            if node.id >= 0:  # Skip root
                node.update_features(
                    node_data,
                    vid_length,
                    frame_num,
                    relative_frame_num,
                    timestamp_fraction,
                    self.labels_to_int,
                    len(self.obj_labels)
                )
        
        # Extract and normalize features
        node_features, edge_indices, edge_features = scene_graph.extract_features(
            node_data, relative_frame_num, timestamp_fraction
        )
        
        # Store results
        node_features_list.append(node_features)
        edge_indices_list.append(edge_indices)
        edge_features_list.append(edge_features)
        labels_list.append(action_labels)


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