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
from graph.io import Record, DataLoader, VideoProcessor
from graph.utils import get_roi
from graph.graph_tracer import GraphTracer
from graph.checkpoint_manager import CheckpointManager, GraphCheckpoint
from graph.action_utils import ActionUtils
from models.clip import ClipModel
from models.yolo_world import YOLOWorldModel
from models.sift import SIFT
from egtea_gaze.gaze_data.gaze_io_sample import parse_gtea_gaze
from config.config_utils import DotDict
from logger import get_logger

logger = get_logger(__name__)

class GraphBuilder:
    """Builds scene graphs from video data and gaze information."""
    
    def __init__(self, config: DotDict, split: str, enable_tracing: bool = False):
        """Initialize the graph builder.
        
        Args:
            config: Configuration dictionary
            split: Dataset split ('train' or 'val')
            enable_tracing: Whether to enable graph construction tracing
        """
        self.config = config
        self.split = split
        self.enable_tracing = enable_tracing
        self.tracer = GraphTracer(self.config.directories.repo.traces, "", enabled=False)
        
        self.clip_model = ClipModel(self.config.models.clip.model_id)
        self.clip_model.load_model(Path(self.config.models.clip.model_dir))
        
        # Initialize YOLO-World model
        yolo_model_file = self.config.models.yolo_world.model_file
        yolo_model_dir = Path(self.config.models.yolo_world.model_dir)
        yolo_model_path = yolo_model_dir / yolo_model_file
        
        logger.info(f"Initializing YOLO-World model: {yolo_model_file} (conf_threshold={self.config.models.yolo_world.conf_threshold}, "
                   f"iou_threshold={self.config.models.yolo_world.iou_threshold})")
        
        self.yolo_model = YOLOWorldModel(
            conf_threshold=self.config.models.yolo_world.conf_threshold,
            iou_threshold=self.config.models.yolo_world.iou_threshold
        )
        self.yolo_model.load_model(yolo_model_path)
        
        noun_idx_path = Path(self.config.dataset.egtea.noun_idx_file)
        self.obj_labels, self.labels_to_int = DataLoader.load_object_labels(noun_idx_path)
        self.clip_labels = [f"a picture of a {obj}" for obj in self.obj_labels.values()]
        
        self.yolo_model.set_classes(list(self.obj_labels.values()))
        
        ann_file = (self.config.dataset.ego_topo.splits.train if self.split == 'train' 
                   else self.config.dataset.ego_topo.splits.val)
        self.vid_lengths = DataLoader.load_video_lengths(ann_file)
        self.records, self.records_by_vid = DataLoader.load_records(ann_file)
        self.action_to_idx = DataLoader.create_action_index(self.records)
        
        self.prev_gaze_pos = (-1, -1)
        self.potential_labels = defaultdict(int)
        self.visit_start = -1
        self.visit_end = -1
        self.frame_num = 0
        self.relative_frame_num = 0
        
        self.scene_graph = None
        self.checkpoint_manager = None
        self.video_name = ""
        self.gaze_data = None
        self.records_current = None
        self.vid_length = 0
        self.timestamps = []
    
    def process_video(self, video_name: str, print_graph: bool = False) -> Graph:
        """Process a video to build its scene graph.
        
        Args:
            video_name: Name of the video to process
            print_graph: Whether to print the final graph structure
            
        Returns:
            Graph instance representing the scene graph
        """
        logger.info(f"\nProcessing video: {video_name}")
        self.video_name = video_name
        self.tracer = GraphTracer(self.config.directories.repo.traces, video_name, enabled=self.enable_tracing)
        if self.enable_tracing:
            logger.info(f"Tracing enabled for {video_name}")
        
        self.gaze_data = parse_gtea_gaze(str(Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt"))
        video_processor = VideoProcessor(Path(self.config.dataset.egtea.raw_videos) / f"{video_name}.mp4")
        self.records_current = self.records_by_vid[video_name]
        self.vid_length = self.vid_lengths[video_name]
        self.timestamps = [int(ratio * self.vid_length) for ratio in sorted(self.config.dataset.timestamps[self.split])]
        
        self.scene_graph = Graph(
            labels_to_int=self.labels_to_int,
            num_object_classes=len(self.obj_labels),
            video_length=self.vid_length
        )
        self.scene_graph.tracer = self.tracer
        
        self.checkpoint_manager = CheckpointManager(
            graph=self.scene_graph,
            checkpoint_frames=self.timestamps,
            timestamps=self.timestamps,
            timestamp_ratios=self.config.dataset.timestamps[self.split],
            records=self.records_current,
            action_to_idx=self.action_to_idx,
            gaze_data_length=len(self.gaze_data)
        )
        
        self._reset_tracking_state()

        for frame, _, is_black_frame in video_processor:
            should_continue = self._process_frame(frame, is_black_frame)
            if not should_continue:
                break
            self.relative_frame_num += 1
            self.frame_num += 1

        if self.potential_labels:
            self._finish_final_fixation()
        
        if print_graph and self.scene_graph.num_nodes > 0:
            logger.info("\nFinal graph structure:")
            self.scene_graph.print_graph()
        
        return self.scene_graph

    def _reset_tracking_state(self):
        """Reset all tracking state variables."""
        self.prev_gaze_pos = (-1, -1)
        self.potential_labels = defaultdict(int)
        self.visit_start = -1
        self.visit_end = -1
        self.frame_num = 0
        self.relative_frame_num = 0

    def _process_frame(self, frame: torch.Tensor, is_black_frame: bool) -> bool:
        """Process a single frame and update the scene graph.
        
        Returns:
            bool: False if processing should stop, True to continue
        """
        gaze_pos = None
        gaze_type = None
        if self.frame_num < len(self.gaze_data):
            gaze_pos = self.gaze_data[self.frame_num, :2].tolist()
            gaze_type = int(self.gaze_data[self.frame_num, 2])

        node_id = self.scene_graph.current_node.id if self.scene_graph.current_node.id >= 0 else None
        self.tracer.log_frame(self.frame_num, gaze_pos, gaze_type, node_id)

        if is_black_frame:
            return True

        checkpoint = self.checkpoint_manager.checkpoint_if_needed(
            frame_num=self.frame_num,
            relative_frame=self.relative_frame_num
        )
        
        should_stop = (
            self.frame_num >= len(self.gaze_data) or 
            (self.frame_num in self.timestamps and self.frame_num == self.timestamps[-1])
        )
        
        if should_stop:
            logger.info(f"[Frame {self.frame_num}] Reached final timestamp or end of gaze data")
            return False

        if self.frame_num < len(self.gaze_data):
            self._process_frame_with_gaze(frame)
            
        return True

    def _process_frame_with_gaze(self, frame: torch.Tensor) -> None:
        """Process a frame using available gaze data."""
        gaze_type = int(self.gaze_data[self.frame_num, 2])
        gaze_pos = self.gaze_data[self.frame_num, :2]
        
        if gaze_type == 1:
            self._handle_fixation(frame, gaze_pos)
        elif gaze_type == 2 and self.potential_labels:
            self._handle_saccade(gaze_pos)

    def _handle_fixation(self, frame: torch.Tensor, gaze_pos: Tuple[float, float]) -> None:
        """Handle a fixation frame."""
        if self.visit_start == -1:
            self.visit_start = self.relative_frame_num
            logger.info(f"\n[Frame {self.frame_num}] New fixation started at ({gaze_pos[0]:.1f}, {gaze_pos[1]:.1f})")
        
        _, H, W = frame.shape
        gaze_x, gaze_y = int(gaze_pos[0] * W), int(gaze_pos[1] * H)
        
        roi, roi_bbox = get_roi(frame, (gaze_x, gaze_y), 256)
        logger.debug(f"[Frame {self.frame_num}] ROI bounding box: {roi_bbox}")

        # Run CLIP inference for object detection (used for graph construction)
        clip_label = self.clip_model.run_inference(roi, self.clip_labels, self.obj_labels)
        self.potential_labels[clip_label] += 1
        
        logger.info(f"[Frame {self.frame_num}] CLIP detected: {clip_label} (count: {self.potential_labels[clip_label]})")
        
        # Run YOLO-World inference but only log results (don't use for graph construction yet)
        try:
            yolo_detections = self.yolo_model.run_inference(roi, self.clip_labels, self.obj_labels)
            
            if yolo_detections:
                # Sort by confidence score (highest first)
                sorted_detections = sorted(yolo_detections, key=lambda x: x['score'], reverse=True)
                
                # Log detections
                logger.info(f"[Frame {self.frame_num}] YOLO-World detections:")
                for i, detection in enumerate(sorted_detections[:3]):  # Log top 3 detections
                    bbox = detection['bbox']
                    logger.info(f"  - {detection['class_name']} (conf: {detection['score']:.2f}, "
                              f"bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}])")
        except Exception as e:
            logger.warning(f"[Frame {self.frame_num}] YOLO-World inference failed: {e}")
        
        most_likely_label = max(self.potential_labels.items(), key=lambda x: x[1])[0]
        
        self.tracer.log_gaze_object_detected(
            self.frame_num,
            most_likely_label,
            clip_label,
            roi_bbox,
            dict(self.potential_labels)
        )
    
    def _handle_saccade(self, curr_gaze_pos: Tuple[float, float]) -> None:
        """Handle a saccade between fixations."""
        self.visit_end = self.relative_frame_num - 1
        fixation_duration = self.visit_end - self.visit_start + 1
        
        most_likely_label = max(self.potential_labels.items(), key=lambda x: x[1])[0]
        
        logger.info(f"\n[Frame {self.frame_num}] Saccade detected:")
        logger.info(f"- Most likely object: {most_likely_label}")
        logger.info(f"- Visit duration: {fixation_duration} frames")
        logger.debug(f"- Current gaze position (normalized): ({curr_gaze_pos[0]:.2f}, {curr_gaze_pos[1]:.2f})")
        
        prev_node_id = self.scene_graph.current_node.id
        visit_record = [self.visit_start, self.visit_end]
        next_node = self.scene_graph.update_graph(
            self.potential_labels,
            visit_record,
            self.prev_gaze_pos,
            curr_gaze_pos
        )
        
        if next_node.id != prev_node_id:
            logger.info(f"- New node created: {next_node.id}")
            self.tracer.log_node_added(self.frame_num, next_node.id, next_node.object_label, next_node.get_features())
            
            if prev_node_id >= 0:
                edge = self.scene_graph.get_edge(prev_node_id, next_node.id)
                if edge:
                    edge_features = edge.get_features()
                    self.tracer.log_edge_added(self.frame_num, prev_node_id, next_node.id, "saccade", edge_features)
        else:
            logger.info(f"- Merged with existing node: {next_node.id}")
            logger.info(f"- Updated node features: visits={len(next_node.visits)}, "
                      f"total_frames={next_node.get_visit_duration()}")
        
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
        self.visit_end = self.relative_frame_num - 1
        visit_record = [self.visit_start, self.visit_end]
        
        last_frame = min(self.frame_num, len(self.gaze_data)-1)
        last_gaze_pos = self.gaze_data[last_frame, :2]
        
        logger.debug(f"- Final gaze position (normalized): ({last_gaze_pos[0]:.2f}, {last_gaze_pos[1]:.2f})")
        
        self.scene_graph.update_graph(
            self.potential_labels,
            visit_record, 
            self.prev_gaze_pos,
            last_gaze_pos
        )

def build_graph(
    video_list: List[str], 
    config: DotDict, 
    split: str, 
    print_graph: bool = False, 
    desc: Optional[str] = None, 
    enable_tracing: bool = False
) -> Dict:
    """Build graph representations for a list of videos.
    
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
    builder = GraphBuilder(config, split, enable_tracing=enable_tracing)
    all_graphs = []
    progress_desc = desc or f"Processing {split} videos"
    
    for video_name in tqdm(video_list, desc=progress_desc):
        if enable_tracing:
            logger.info(f"Building graph with tracing for {video_name}")
        graph = builder.process_video(video_name, print_graph)
        all_graphs.append(graph)
    
    return CheckpointManager.build_dataset_from_graphs(all_graphs)