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
from graph.object_detection import ObjectDetector
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
        self.records, self.records_by_vid = DataLoader.load_records(ann_file)
        self.action_to_idx = DataLoader.create_action_index(self.records)
        
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

        if self.object_detector.get_potential_labels() and self.object_detector.has_fixated_objects():
            self._finish_final_fixation()
        
        if print_graph and self.scene_graph.num_nodes > 0:
            logger.info("\nFinal graph structure:")
            self.scene_graph.print_graph()
        
        return self.scene_graph

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
        
        prev_node_id = self.scene_graph.current_node.id
        visit_record = [self.visit_start, self.visit_end]
        
        # Get the previous gaze position or default to (0,0) if not available
        prev_position = self.prev_gaze_point.position if self.prev_gaze_point else (0, 0)
        
        next_node = self.scene_graph.update(
            fixated_object,
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
        self.object_detector.reset()
    
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

        # Get the most likely fixated object from the detector
        fixated_object, confidence = self.object_detector.get_fixated_object()
        logger.info(f"- Final fixated object: {fixated_object} (accumulated confidence: {confidence:.2f})")
        
        self.scene_graph.update(
            fixated_object,
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