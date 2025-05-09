"""
Graph building module for creating scene graphs from video data.
"""

import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from gazegraph.graph.graph import Graph
from gazegraph.graph.graph_tracer import GraphTracer
from gazegraph.graph.checkpoint_manager import CheckpointManager
from gazegraph.graph.gaze import GazePoint, GazeType
from gazegraph.graph.object_detection import ObjectDetector
from gazegraph.datasets.egtea_gaze.video_metadata import VideoMetadata
from gazegraph.datasets.egtea_gaze.video_processor import Video
from gazegraph.config.config_utils import DotDict
from gazegraph.logger import get_logger

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
        self.tracer = GraphTracer(self.config.directories.traces, "", enabled=False)
        
        self.metadata = VideoMetadata(self.config)
        
        # Initialize YOLO-World model path
        backend = self.config.models.yolo_world.backend
        self.yolo_model_path = Path(self.config.models.yolo_world.paths[backend])
        
        # Create object detector (will be re-initialized for each video with proper tracer)
        self.object_detector = None
        
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
        self.video = None
        
        # Frame range to process
        self.first_frame = 0
        self.last_frame = 0
    
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
        self.tracer = GraphTracer(self.config.directories.traces, video_name, enabled=self.enable_tracing)
        if self.enable_tracing:
            logger.info(f"Tracing enabled for {video_name}")
        
        # Initialize video abstraction
        self.video = Video(video_name, self.config)
        
        # Create and initialize components
        self._initialize_components()
        
        # Process video frames
        self._process_video_frames()
        
        # Finalize processing
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
    
    def _initialize_components(self):
        """Initialize all components needed for processing."""
        # Initialize object detector with video-specific tracer
        self.object_detector = ObjectDetector(
            model_path=self.yolo_model_path,
            obj_labels=self.metadata.obj_labels,
            labels_to_int=self.metadata.labels_to_int,
            config=self.config,
            tracer=self.tracer
        )

        # Get frame range
        self.first_frame, self.last_frame = self.video.first_frame, self.video.last_frame
        logger.info(f"Processing frames from {self.first_frame} to {self.last_frame} based on action annotations")
        
        # Create scene graph
        self.scene_graph = Graph(
            labels_to_int=self.metadata.labels_to_int,
            num_object_classes=self.metadata.num_object_classes,
            video_length=self.video.length
        )
        self.scene_graph.tracer = self.tracer
        
        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            graph=self.scene_graph,
            video_name=self.video_name,
            output_dir=self.output_dir,
            split=self.split
        )
        
        # Reset tracking state
        self._reset_tracking_state()
    
    def _process_video_frames(self):
        """Process all frames in the video."""
        try:
            for frame, _, is_black_frame, gaze_point in self.video:
                self._process_frame(frame, gaze_point, is_black_frame)
                if not is_black_frame:
                    self.non_black_frame_count += 1
                self.frame_num += 1
        except StopProcessingException as e:
            logger.info(f"[Frame {self.frame_num}] {e.reason}")
        except StopIteration:
            logger.info(f"[Frame {self.frame_num}] End of data reached")

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
        
        if gaze_point is not None:
            self.tracer.log_frame(self.frame_num, gaze_point.position, int(gaze_point.type), node_id)
        else:
            # Log frame with None values for gaze position and type
            self.tracer.log_frame(self.frame_num, None, -1, node_id)

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
            logger.debug(f"- No fixated objects found during this fixation period, skipping node creation")
            self._reset_fixation_state(gaze_point)
            return
        
        fixated_object, confidence = self.object_detector.get_fixated_object()
        
        logger.debug(f"- Fixated object: {fixated_object} (confidence: {confidence:.2f})")
        logger.debug(f"- Visit duration: {fixation_duration} frames")
        
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
        
        logger.info("- Final fixation detected, updating graph...")
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