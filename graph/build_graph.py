"""
Graph building module for creating scene graphs from video data.
"""

import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from collections import defaultdict
from tqdm import tqdm
import itertools

from graph.graph import Graph
from graph.node import Node
from graph.io import Record, DataLoader, VideoProcessor
from graph.graph_tracer import GraphTracer
from graph.checkpoint_manager import CheckpointManager, GraphCheckpoint
from graph.action_utils import ActionUtils
from graph.gaze import GazeProcessor, GazePoint, GazeType
from models.yolo_world import YOLOWorldModel
from models.sift import SIFT
from egtea_gaze.gaze_data.gaze_io_sample import parse_gtea_gaze
from egtea_gaze.constants import GAZE_TYPE_FIXATION, GAZE_TYPE_SACCADE
from config.config_utils import DotDict
from logger import get_logger

logger = get_logger(__name__)

class StopProcessingException(Exception):
    """Exception to indicate that frame processing should stop."""
    def __init__(self, reason: str = ""):
        self.reason = reason
        super().__init__(self.reason)

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
        
        self.prev_gaze_point = None
        self.potential_labels = defaultdict(float)  # Changed to float for accumulating confidence scores
        self.visit_start = -1
        self.visit_end = -1
        self.frame_num = 0
        # Keeps track of frames excluding black frames, used for visit tracking and feature normalization
        self.non_black_frame_count = 0
        self.fixated_objects_found = False  # Track if any objects were fixated during a fixation
        
        self.scene_graph = None
        self.checkpoint_manager = None
        self.video_name = ""
        self.raw_gaze_data = None
        self.gaze_processor = None
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
        
        # Load raw gaze data and initialize gaze processor
        self.raw_gaze_data = parse_gtea_gaze(str(Path(self.config.dataset.egtea.gaze_data) / f"{video_name}.txt"))
        self.gaze_processor = GazeProcessor(self.config, self.raw_gaze_data)
        
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
            gaze_data_length=len(self.raw_gaze_data)
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

        if self.potential_labels and self.fixated_objects_found:
            self._finish_final_fixation()
        
        if print_graph and self.scene_graph.num_nodes > 0:
            logger.info("\nFinal graph structure:")
            self.scene_graph.print_graph()
        
        return self.scene_graph

    def _reset_tracking_state(self):
        """Reset all tracking state variables."""
        self.prev_gaze_point = None
        self.potential_labels = defaultdict(float)
        self.visit_start = -1
        self.visit_end = -1
        self.frame_num = 0
        self.non_black_frame_count = 0
        self.fixated_objects_found = False

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
        node_id = self.scene_graph.current_node.id if self.scene_graph.current_node.id >= 0 else None
        self.tracer.log_frame(self.frame_num, gaze_point.position, int(gaze_point.type), node_id)

        if is_black_frame:
            return

        self.checkpoint_manager.checkpoint_if_needed(
            frame_num=self.frame_num,
            non_black_frame_count=self.non_black_frame_count
        )
        
        if self.frame_num in self.timestamps and self.frame_num == self.timestamps[-1]:
            raise StopProcessingException(reason="Reached final timestamp")

        self._process_frame_with_gaze(frame, gaze_point)

    def _process_frame_with_gaze(self, frame: torch.Tensor, gaze_point: GazePoint) -> None:
        """Process a frame using available gaze data."""
        if gaze_point.type == GazeType.FIXATION:
            self._handle_fixation(frame, gaze_point)
        elif gaze_point.type == GazeType.SACCADE and self.potential_labels:
            self._handle_saccade(gaze_point)

    def _handle_fixation(self, frame: torch.Tensor, gaze_point: GazePoint) -> None:
        """Handle a fixation frame."""
        if self.visit_start == -1:
            self.visit_start = self.non_black_frame_count
            logger.info(f"\n[Frame {self.frame_num}] New fixation started at ({gaze_point.x:.1f}, {gaze_point.y:.1f})")
        
        _, H, W = frame.shape
        gaze_x, gaze_y = int(gaze_point.x * W), int(gaze_point.y * H)

        # Run YOLO-World inference
        yolo_detections = []
        found_fixated_objects = False
        
        try:
            # Get raw detections from YOLO World
            detections = self.yolo_model.run_inference(frame, self.clip_labels, self.obj_labels)
            
            if detections:
                # Sort by confidence score (highest first)
                sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
                
                # Process detections and check which ones are fixated by gaze
                for detection in sorted_detections:
                    left, top, width, height = detection['bbox']
                    
                    # Check if gaze intersects with this object
                    is_fixated = (
                        left <= gaze_x <= left + width and
                        top <= gaze_y <= top + height
                    )
                    
                    # Create a new detection object with fixation info
                    yolo_detections.append({
                        'bbox': (left, top, width, height),
                        'class_name': detection['class_name'],
                        'score': detection['score'],
                        'class_id': detection['class_id'],
                        'is_fixated': is_fixated
                    })
                    
                    # Accumulate confidence scores for fixated objects
                    if is_fixated:
                        found_fixated_objects = True
                        self.fixated_objects_found = True
                        # Accumulate the confidence score (weighted by the score itself)
                        self.potential_labels[detection['class_name']] += detection['score']
                
                # Log top 3 detections with fixation info
                logger.info(f"[Frame {self.frame_num}] Top 3 of {len(yolo_detections)} YOLO-World detections:")
                for i, detection in enumerate(sorted_detections[:min(3, len(sorted_detections))]):
                    bbox = detection['bbox']
                    # Get the corresponding detection with fixation info
                    fixation_info = yolo_detections[i]['is_fixated'] if i < len(yolo_detections) else False
                    logger.info(f"  - {detection['class_name']} (conf: {detection['score']:.2f}, "
                              f"bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}], "
                              f"fixated: {fixation_info})")

                # Log all fixated detections
                fixated_detections = [d for d in yolo_detections if d['is_fixated']]
                logger.info(f"[Frame {self.frame_num}] {len(fixated_detections)} fixated YOLO-World detections:")
                for detection in fixated_detections:
                    logger.info(f"  - {detection['class_name']} (conf: {detection['score']:.2f}, "
                              f"bbox: [{detection['bbox'][0]}, {detection['bbox'][1]}, {detection['bbox'][2]}, {detection['bbox'][3]}])")
                
                # Log all YOLO detections to the tracer
                self.tracer.log_yolo_objects_detected(
                    self.frame_num,
                    yolo_detections
                )
                
                # Log current accumulated object confidences
                if found_fixated_objects:
                    logger.info(f"[Frame {self.frame_num}] Current accumulated object confidences:")
                    for label, confidence in sorted(self.potential_labels.items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"  - {label}: {confidence:.2f}")
                
        except Exception as e:
            logger.warning(f"[Frame {self.frame_num}] YOLO-World inference failed: {str(e)}")
            import traceback
            logger.debug(f"Error details: {traceback.format_exc()}")
        
        # Store current gaze point for future reference
        self.prev_gaze_point = gaze_point
    
    def _handle_saccade(self, gaze_point: GazePoint) -> None:
        """Handle a saccade between fixations."""
        self.visit_end = self.non_black_frame_count - 1
        fixation_duration = self.visit_end - self.visit_start + 1
        
        logger.info(f"\n[Frame {self.frame_num}] Saccade detected:")
        logger.debug(f"- Current gaze position (normalized): ({gaze_point.position[0]:.2f}, {gaze_point.position[1]:.2f})")
        
        # Only proceed if we found fixated objects during this fixation period
        if not self.fixated_objects_found or not self.potential_labels:
            logger.info(f"- No fixated objects found during this fixation period, skipping node creation")
            self._reset_fixation_state(gaze_point)
            return
        
        most_likely_label = max(self.potential_labels.items(), key=lambda x: x[1])[0]
        accumulated_confidence = self.potential_labels[most_likely_label]
        
        logger.info(f"- Most likely object: {most_likely_label} (accumulated confidence: {accumulated_confidence:.2f})")
        logger.info(f"- Visit duration: {fixation_duration} frames")
        
        prev_node_id = self.scene_graph.current_node.id
        visit_record = [self.visit_start, self.visit_end]
        
        # Get the previous gaze position or default to (0,0) if not available
        prev_position = self.prev_gaze_point.position if self.prev_gaze_point else (0, 0)
        
        next_node = self.scene_graph.update_graph(
            self.potential_labels,
            visit_record,
            prev_position,
            gaze_point.position
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
        
        self._reset_fixation_state(gaze_point)
    
    def _reset_fixation_state(self, gaze_point: GazePoint):
        """Reset the state related to the current fixation."""
        self.visit_start = -1
        self.visit_end = -1
        self.prev_gaze_point = gaze_point
        self.potential_labels = defaultdict(float)
        self.fixated_objects_found = False
    
    def _finish_final_fixation(self) -> None:
        """Process the final fixation if video ends during one."""
        logger.info("- Final fixation detected, updating graph...")
        self.visit_end = self.non_black_frame_count - 1
        visit_record = [self.visit_start, self.visit_end]
        
        # Get the last gaze position
        self.gaze_processor.reset()
        last_gaze_point = None
        for gp in self.gaze_processor:
            if gp.frame_idx == self.frame_num - 1:
                last_gaze_point = gp
                break
                
        if last_gaze_point is None:
            # Fallback if we can't get the last gaze point
            last_gaze_point = GazePoint(
                x=self.prev_gaze_point.x if self.prev_gaze_point else 0.5, 
                y=self.prev_gaze_point.y if self.prev_gaze_point else 0.5, 
                raw_type=GazeType.FIXATION,
                type=GazeType.FIXATION,
                frame_idx=self.frame_num - 1
            )
            
        logger.debug(f"- Final gaze position (normalized): ({last_gaze_point.position[0]:.2f}, {last_gaze_point.position[1]:.2f})")
        
        # Get the previous gaze position or default to (0,0) if not available
        prev_position = self.prev_gaze_point.position if self.prev_gaze_point else (0, 0)
        
        self.scene_graph.update_graph(
            self.potential_labels,
            visit_record, 
            prev_position,
            last_gaze_point.position
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