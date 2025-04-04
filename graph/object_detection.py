"""
Object detection module for detecting and tracking objects in video frames.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional, DefaultDict
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
from pathlib import Path

from models.yolo_world import YOLOWorldModel
from graph.graph_tracer import GraphTracer
from graph.gaze import GazePoint
from config.config_utils import DotDict

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Represents a detected object."""
    bbox: Tuple[int, int, int, int]  # left, top, width, height
    class_name: str
    score: float
    class_id: int
    is_fixated: bool = False
    frame_idx: int = -1
    
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
            'is_fixated': self.is_fixated,
            'frame_idx': self.frame_idx
        }


class ObjectDetector:
    """Handles object detection and tracking for scene graph construction."""
    
    def __init__(
        self, 
        model_path: Path,
        obj_labels: Dict[int, str],
        labels_to_int: Dict[str, int],
        config: DotDict,
        tracer: Optional[GraphTracer] = None
    ):
        """Initialize the object detector.
        
        Args:
            model_path: Path to the detection model
            obj_labels: Mapping of class IDs to object labels
            labels_to_int: Mapping of object labels to class IDs
            config: Configuration dictionary containing detection settings
            tracer: Optional GraphTracer for logging detection info
        """
        self.obj_labels = obj_labels
        self.labels_to_int = labels_to_int
        self.clip_labels = [f"a picture of a {obj}" for obj in self.obj_labels.values()]
        self.tracer = tracer
        self.config = config
        
        # Extract settings from config
        self.conf_threshold = config.models.yolo_world.conf_threshold
        self.iou_threshold = config.models.yolo_world.iou_threshold
        
        # Advanced fixation parameters
        self.min_fixation_frame_ratio = config.graph.min_fixation_frame_ratio
        self.enable_advanced_fixation = config.graph.advanced_fixation.enabled
        self.bbox_stability_weight = config.graph.advanced_fixation.weights.bbox_stability
        self.gaze_proximity_weight = config.graph.advanced_fixation.weights.gaze_proximity
        self.confidence_weight = config.graph.advanced_fixation.weights.confidence
        self.duration_weight = config.graph.advanced_fixation.weights.duration
        
        logger.info(f"Initializing YOLO-World model: {model_path.name} "
                    f"(conf_threshold={self.conf_threshold}, iou_threshold={self.iou_threshold})")
        
        self.model = YOLOWorldModel(
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        self.model.load_model(model_path)
        self.model.set_classes(list(self.obj_labels.values()))
        
        # State tracking
        self.reset()
    
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
            detections = self._perform_detection(frame, gaze_x, gaze_y, frame_idx)
            found_fixated_objects = any(d.is_fixated for d in detections)
            
            if found_fixated_objects:
                self.fixated_objects_found = True
                self._log_fixated_objects(frame_idx, detections)
                
                # Store gaze points for advanced fixation analysis
                self.gaze_points.append((gaze_x, gaze_y, frame_idx))
                
                # Log to tracer if available
                if self.tracer and detections:
                    detection_dicts = [d.to_dict() for d in detections]
                    self.tracer.log_yolo_objects_detected(frame_idx, detection_dicts)
            
            # Store detections for advanced fixation calculation
            self.all_detections.extend(detections)
            self.total_frames += 1
            
        except Exception as e:
            logger.warning(f"[Frame {frame_idx}] Object detection failed: {str(e)}")
            import traceback
            logger.debug(f"Error details: {traceback.format_exc()}")
        
        return detections
    
    def _perform_detection(
        self, 
        frame: torch.Tensor, 
        gaze_x: int, 
        gaze_y: int,
        frame_idx: int
    ) -> List[Detection]:
        """Perform object detection and identify fixated objects.
        
        Args:
            frame: Video frame tensor
            gaze_x: Gaze x-coordinate in frame pixels
            gaze_y: Gaze y-coordinate in frame pixels
            frame_idx: Current frame index
            
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
                is_fixated=is_fixated,
                frame_idx=frame_idx
            )
            detections.append(detection_obj)
            
            # Accumulate confidence scores for fixated objects (for backwards compatibility)
            if is_fixated:
                self.potential_labels[detection['class_name']] += detection['score']
                
                # Update object presence tracking
                if detection['class_name'] not in self.object_frame_counts:
                    self.object_frame_counts[detection['class_name']] = 0
                self.object_frame_counts[detection['class_name']] += 1
                
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
        self.all_detections = []
        self.gaze_points = []
        self.total_frames = 0
        self.object_frame_counts = {}
        self.fixation_scores = {}
    
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
    
    def _compute_bbox_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two bounding boxes.
        
        Args:
            box1: First bounding box (left, top, width, height)
            box2: Second bounding box (left, top, width, height)
            
        Returns:
            IoU score between 0 and 1
        """
        # Convert to x1, y1, x2, y2 format
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / max(union_area, 1e-6)
    
    def _compute_mean_iou(self, bboxes: List[Tuple[int, int, int, int]]) -> float:
        """Compute mean IoU across sequential pairs of bounding boxes.
        
        Args:
            bboxes: List of bounding boxes
            
        Returns:
            Mean IoU stability score
        """
        if len(bboxes) <= 1:
            return 1.0  # Perfect stability for single detection
            
        iou_sum = 0.0
        for i in range(len(bboxes) - 1):
            iou_sum += self._compute_bbox_iou(bboxes[i], bboxes[i + 1])
            
        return iou_sum / (len(bboxes) - 1)
    
    def _compute_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Compute the center of a bounding box.
        
        Args:
            bbox: Bounding box (left, top, width, height)
            
        Returns:
            (center_x, center_y)
        """
        left, top, width, height = bbox
        return left + width / 2, top + height / 2
    
    def _compute_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _compute_gaze_distance(self, bbox: Tuple[int, int, int, int], gaze: Tuple[int, int]) -> float:
        """Compute distance from gaze point to the center of the bounding box.
        
        Args:
            bbox: Bounding box (left, top, width, height)
            gaze: Gaze point (x, y)
            
        Returns:
            Distance from gaze point to bbox center
        """
        left, top, width, height = bbox
        gaze_x, gaze_y = gaze
        
        # Calculate bbox center
        center_x = left + width / 2
        center_y = top + height / 2
        
        # Compute distance to center
        return self._compute_distance((gaze_x, gaze_y), (center_x, center_y))
    
    def _compute_mean_gaze_distance(
        self, 
        bboxes: List[Tuple[int, int, int, int]], 
        gaze_points: List[Tuple[int, int, int]]
    ) -> float:
        """Compute mean distance between gaze points and bounding boxes.
        
        Args:
            bboxes: List of bounding boxes
            gaze_points: List of gaze points and frame indices (x, y, frame_idx)
            
        Returns:
            Mean distance score
        """
        if not bboxes or not gaze_points:
            return float('inf')
            
        # Create a frame index to bbox mapping
        frame_to_bbox = {frame_idx: bbox for bbox, (_, _, frame_idx) in zip(bboxes, gaze_points)}
        
        total_distance = 0.0
        count = 0
        
        for gaze_x, gaze_y, frame_idx in gaze_points:
            if frame_idx in frame_to_bbox:
                distance = self._compute_gaze_distance(frame_to_bbox[frame_idx], (gaze_x, gaze_y))
                total_distance += distance
                count += 1
                
        return total_distance / max(count, 1)
    
    def _compute_geometric_mean(self, values: List[float]) -> float:
        """Compute geometric mean of a list of values.
        
        Args:
            values: List of values
            
        Returns:
            Geometric mean
        """
        if not values:
            return 0.0
            
        # Use log sum to avoid numerical issues
        log_sum = sum(math.log(max(v, 1e-10)) for v in values)
        return math.exp(log_sum / len(values))
    
    def _compute_fixation_scores(self) -> Dict[str, float]:
        """Compute fixation scores for all detected objects.
        
        Returns:
            Dictionary mapping object names to fixation scores
        """
        fixation_scores = {}
        
        # Extract unique objects from detections
        unique_objects = set(d.class_name for d in self.all_detections if d.is_fixated)
        
        for obj_name in unique_objects:
            # Get all fixated detections for this object
            obj_detections = [d for d in self.all_detections if d.class_name == obj_name and d.is_fixated]
            
            # Skip if not enough detections
            if not obj_detections:
                continue
                
            # Compute fixation ratio
            fixation_frames = len(set(d.frame_idx for d in obj_detections))
            fixation_ratio = fixation_frames / max(self.total_frames, 1)
            
            # Apply minimum fixation threshold
            if fixation_ratio < self.min_fixation_frame_ratio:
                logger.info(f"Object {obj_name} filtered out: fixation ratio {fixation_ratio:.2f} < threshold {self.min_fixation_frame_ratio}")
                continue
                
            # Get confidence scores and bounding boxes
            confidence_scores = [d.score for d in obj_detections]
            bboxes = [d.bbox for d in obj_detections]
            
            # Get relevant gaze points (those in frames where object was detected)
            obj_frame_indices = set(d.frame_idx for d in obj_detections)
            relevant_gaze_points = [g for g in self.gaze_points if g[2] in obj_frame_indices]
            
            # 1. Compute geometric mean of confidence scores
            mean_confidence = self._compute_geometric_mean(confidence_scores)
            
            # 2. Weight by fixation duration
            duration_weighted_score = mean_confidence * fixation_ratio
            
            # 3. Compute bounding box stability
            stability_score = self._compute_mean_iou(bboxes)
            
            # 4. Compute gaze proximity weighting
            gaze_distance = self._compute_mean_gaze_distance(bboxes, relevant_gaze_points)
            gaze_weight = 1.0 / (1.0 + gaze_distance)
            
            # 5. Compute final weighted fixation score
            final_score = (
                pow(duration_weighted_score, self.duration_weight) * 
                pow(stability_score, self.bbox_stability_weight) * 
                pow(gaze_weight, self.gaze_proximity_weight)
            )
            
            # Store score
            fixation_scores[obj_name] = final_score
            
            logger.info(f"Fixation score for {obj_name}: {final_score:.4f}")
            logger.info(f"  - Duration score: {duration_weighted_score:.4f} (ratio: {fixation_ratio:.2f})")
            logger.info(f"  - Stability score: {stability_score:.4f}")
            logger.info(f"  - Gaze weight: {gaze_weight:.4f} (distance to center: {gaze_distance:.2f})")
            
        return fixation_scores
    
    def get_fixated_object(self) -> Tuple[str, float]:
        """Get the most likely fixated object based on fixation scores.
        
        Returns:
            Tuple of (object_label, confidence_score)
        """
        # If no detections or advanced fixation is disabled, fall back to basic method
        if not self.all_detections or not self.enable_advanced_fixation:
            if not self.potential_labels:
                raise ValueError("No potential labels found")
                
            fixated_object, confidence = max(self.potential_labels.items(), key=lambda x: x[1])
            return fixated_object, confidence
            
        # Compute fixation scores if not already done
        if not self.fixation_scores:
            self.fixation_scores = self._compute_fixation_scores()
            
        # If no fixation scores found, fall back to basic method
        if not self.fixation_scores:
            if not self.potential_labels:
                raise ValueError("No potential labels found")
                
            fixated_object, confidence = max(self.potential_labels.items(), key=lambda x: x[1])
            return fixated_object, confidence
            
        # Get object with highest fixation score
        fixated_object, score = max(self.fixation_scores.items(), key=lambda x: x[1])
        
        # Also log all scores for comparison
        logger.info("Final fixation scores:")
        for obj, s in sorted(self.fixation_scores.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {obj}: {s:.4f}")
            
        return fixated_object, score
    
    # Alias for backward compatibility
    get_most_likely_object = get_fixated_object