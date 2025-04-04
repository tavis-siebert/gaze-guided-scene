"""
Object detection module for detecting and tracking objects in video frames.
"""

import torch
from typing import Dict, List, Tuple, Any, Optional, DefaultDict
from collections import defaultdict
import logging
from dataclasses import dataclass
from pathlib import Path

from models.yolo_world import YOLOWorldModel
from graph.graph_tracer import GraphTracer
from graph.gaze import GazePoint

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Represents a detected object."""
    bbox: Tuple[int, int, int, int]  # left, top, width, height
    class_name: str
    score: float
    class_id: int
    is_fixated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to a dictionary representation.
        
        Returns:
            Dictionary representation of the detection
        """
        return {
            'bbox': self.bbox,
            'class_name': self.class_name,
            'score': self.score,
            'class_id': self.class_id,
            'is_fixated': self.is_fixated
        }


class ObjectDetector:
    """Handles object detection and tracking for scene graph construction."""
    
    def __init__(
        self, 
        model_path: Path,
        conf_threshold: float,
        iou_threshold: float,
        obj_labels: Dict[int, str],
        labels_to_int: Dict[str, int],
        tracer: Optional[GraphTracer] = None
    ):
        """Initialize the object detector.
        
        Args:
            model_path: Path to the detection model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            obj_labels: Mapping of class IDs to object labels
            labels_to_int: Mapping of object labels to class IDs
            tracer: Optional GraphTracer for logging detection info
        """
        self.obj_labels = obj_labels
        self.labels_to_int = labels_to_int
        self.clip_labels = [f"a picture of a {obj}" for obj in self.obj_labels.values()]
        self.tracer = tracer
        
        logger.info(f"Initializing YOLO-World model: {model_path.name} "
                    f"(conf_threshold={conf_threshold}, iou_threshold={iou_threshold})")
        
        self.model = YOLOWorldModel(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        self.model.load_model(model_path)
        self.model.set_classes(list(self.obj_labels.values()))
        
        # State tracking
        self.potential_labels = defaultdict(float)
        self.fixated_objects_found = False
    
    def detect_objects(
        self, 
        frame: torch.Tensor, 
        gaze_point: GazePoint, 
        frame_idx: int
    ) -> List[Detection]:
        """Detect objects in a frame and track fixated objects.
        
        Args:
            frame: Video frame tensor
            gaze_point: Current gaze point
            frame_idx: Current frame index
            
        Returns:
            List of Detection objects
        """
        _, H, W = frame.shape
        gaze_x, gaze_y = int(gaze_point.x * W), int(gaze_point.y * H)
        
        detections = []
        
        try:
            detections = self._perform_detection(frame, gaze_x, gaze_y)
            found_fixated_objects = any(d.is_fixated for d in detections)
            
            if found_fixated_objects:
                self.fixated_objects_found = True
                self._log_fixated_objects(frame_idx, detections)
                
                # Log to tracer if available
                if self.tracer and detections:
                    detection_dicts = [d.to_dict() for d in detections]
                    self.tracer.log_yolo_objects_detected(frame_idx, detection_dicts)
            
        except Exception as e:
            logger.warning(f"[Frame {frame_idx}] Object detection failed: {str(e)}")
            import traceback
            logger.debug(f"Error details: {traceback.format_exc()}")
        
        return detections
    
    def _perform_detection(
        self, 
        frame: torch.Tensor, 
        gaze_x: int, 
        gaze_y: int
    ) -> List[Detection]:
        """Perform object detection and identify fixated objects.
        
        Args:
            frame: Video frame tensor
            gaze_x: Gaze x-coordinate in frame pixels
            gaze_y: Gaze y-coordinate in frame pixels
            
        Returns:
            List of Detection objects
        """
        # Get raw detections from the model
        raw_detections = self.model.run_inference(frame, self.clip_labels, self.obj_labels)
        if not raw_detections:
            return []
            
        detections = []
        for detection in raw_detections:
            left, top, width, height = detection['bbox']
            
            # Check if gaze intersects with this object
            is_fixated = (left <= gaze_x <= left + width and top <= gaze_y <= top + height)
            
            # Create a new detection object
            detection_obj = Detection(
                bbox=(left, top, width, height),
                class_name=detection['class_name'],
                score=detection['score'],
                class_id=detection['class_id'],
                is_fixated=is_fixated
            )
            detections.append(detection_obj)
            
            # Accumulate confidence scores for fixated objects
            if is_fixated:
                self.potential_labels[detection['class_name']] += detection['score']
                
        return detections
    
    def _log_fixated_objects(self, frame_idx: int, detections: List[Detection]) -> None:
        """Log only fixated objects.
        
        Args:
            frame_idx: Current frame index
            detections: List of detections
        """
        # Log only fixated detections
        fixated_detections = [d for d in detections if d.is_fixated]
        if fixated_detections:
            logger.info(f"[Frame {frame_idx}] {len(fixated_detections)} fixated object detections:")
            for detection in fixated_detections:
                bbox = detection.bbox
                logger.info(f"  - {detection.class_name} (conf: {detection.score:.2f}, "
                          f"bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}])")
            
            # Log accumulated object confidences
            logger.info(f"[Frame {frame_idx}] Current accumulated object confidences:")
            for label, confidence in sorted(self.potential_labels.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {label}: {confidence:.2f}")
    
    def reset(self) -> None:
        """Reset detection state."""
        self.potential_labels = defaultdict(float)
        self.fixated_objects_found = False
    
    def get_potential_labels(self) -> DefaultDict[str, float]:
        """Get accumulated potential object labels with confidence scores.
        
        Returns:
            Dictionary of object labels to confidence scores
        """
        return self.potential_labels
    
    def has_fixated_objects(self) -> bool:
        """Check if any objects were fixated during detection.
        
        Returns:
            True if any objects were fixated
        """
        return self.fixated_objects_found
        
    def get_fixated_object(self) -> Tuple[str, float]:
        """Get the most likely fixated object based on accumulated confidence scores.
        
        Returns:
            Tuple of (object_label, confidence_score)
        """
        if not self.potential_labels:
            raise ValueError("No potential labels found")
            
        fixated_object, confidence = max(self.potential_labels.items(), key=lambda x: x[1])
        return fixated_object, confidence