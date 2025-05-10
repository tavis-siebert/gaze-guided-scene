"""
Object detection module for detecting and tracking objects in video frames.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional, DefaultDict, Union, TYPE_CHECKING
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image

from gazegraph.models.yolo_world import YOLOWorldModel

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from gazegraph.graph.graph_tracer import GraphTracer
from gazegraph.graph.gaze import GazePoint
from gazegraph.config.config_utils import DotDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class ScoreComponents:
    """Contains component scores for object fixation evaluation."""
    confidence: float = 0.0
    stability: float = 0.0
    gaze_proximity: float = 0.0
    fixation_ratio: float = 0.0
    duration_weighted: float = 0.0
    gaze_distance: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert component scores to a dictionary.
        
        Returns:
            Dictionary of score components
        """
        return {
            'confidence': self.confidence,
            'stability': self.stability,
            'gaze_proximity': self.gaze_proximity,
            'fixation_ratio': self.fixation_ratio,
            'duration_weighted': self.duration_weighted,
            'gaze_distance': self.gaze_distance
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ScoreComponents':
        """Create ScoreComponents from a dictionary.
        
        Args:
            data: Dictionary containing score components
            
        Returns:
            New ScoreComponents instance
        """
        return cls(
            confidence=data.get('confidence', 0.0),
            stability=data.get('stability', 0.0),
            gaze_proximity=data.get('gaze_proximity', 0.0),
            fixation_ratio=data.get('fixation_ratio', 0.0),
            duration_weighted=data.get('duration_weighted', 0.0),
            gaze_distance=data.get('gaze_distance', 0.0)
        )

@dataclass
class Detection:
    """Represents a detected object."""
    # Basic detection properties
    bbox: Tuple[int, int, int, int]  # left, top, width, height
    class_name: str
    score: float
    class_id: int
    frame_idx: int = -1
    
    # Fixation properties
    is_fixated: bool = False
    is_top_scoring: bool = False
    fixation_score: float = 0.0
    
    # Component scores as a nested structure
    components: ScoreComponents = field(default_factory=ScoreComponents)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to a dictionary representation.
        
        Returns:
            Dictionary representation of the detection
        """
        return {
            'detection': {
                'bbox': self.bbox,
                'class_name': self.class_name,
                'score': self.score,
                'class_id': self.class_id,
                'frame_idx': self.frame_idx
            },
            'fixation': {
                'is_fixated': self.is_fixated,
                'is_top_scoring': self.is_top_scoring,
                'score': self.fixation_score,
                'components': self.components.to_dict()
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], frame_idx: Optional[int] = None) -> 'Detection':
        """Create Detection from a dictionary.
        
        Args:
            data: Dictionary containing detection data
            frame_idx: Optional frame index to override the one in data
            
        Returns:
            New Detection instance
        """
        det_data = data['detection']
        fix_data = data['fixation']
        
        # Convert bbox from list to tuple if needed
        bbox = tuple(det_data['bbox']) if isinstance(det_data['bbox'], list) else det_data['bbox']
        
        # Use provided frame_idx if available, otherwise from data
        actual_frame_idx = frame_idx if frame_idx is not None else det_data.get('frame_idx', -1)
        
        return cls(
            bbox=bbox,
            class_name=det_data['class_name'],
            score=det_data['score'],
            class_id=det_data['class_id'],
            frame_idx=actual_frame_idx,
            is_fixated=fix_data['is_fixated'],
            is_top_scoring=fix_data['is_top_scoring'],
            fixation_score=fix_data['score'],
            components=ScoreComponents.from_dict(fix_data['components'])
        )


class ObjectDetector:
    """Handles object detection and tracking for scene graph construction."""
    
    def __init__(
        self, 
        model_path: Path,
        obj_labels: Dict[int, str],
        labels_to_int: Dict[str, int],
        config: DotDict,
        tracer: Optional['GraphTracer'] = None
    ):
        """Initialize the object detector."""
        self.obj_labels = obj_labels
        self.labels_to_int = labels_to_int
        self.class_names = list(self.obj_labels.values())
        self.tracer = tracer
        self.config = config
        
        # Backend selection
        backend = config.models.yolo_world.backend
        
        # Fixation parameters
        self.min_fixation_frame_threshold = config.graph.fixated_object_detection.min_fixation_frame_threshold
        self.min_fixation_frame_ratio = config.graph.fixated_object_detection.min_fixation_frame_ratio
        self.bbox_stability_weight = config.graph.fixated_object_detection.weights.bbox_stability
        self.gaze_proximity_weight = config.graph.fixated_object_detection.weights.gaze_proximity
        self.confidence_weight = config.graph.fixated_object_detection.weights.confidence
        self.duration_weight = config.graph.fixated_object_detection.weights.duration
        
        # Thresholds for component scores
        self.bbox_stability_threshold = config.graph.fixated_object_detection.thresholds.bbox_stability
        self.gaze_proximity_threshold = config.graph.fixated_object_detection.thresholds.gaze_proximity
        self.confidence_threshold = config.graph.fixated_object_detection.thresholds.confidence
        
        logger.info(f"Initializing YOLO-World model: {model_path.name} (backend={backend})")
        
        logger.info(f"Fixation detection with thresholds: "
                  f"stability={self.bbox_stability_threshold}, "
                  f"gaze_proximity={self.gaze_proximity_threshold}, "
                  f"confidence={self.confidence_threshold}, "
                  f"min_fixation_ratio={self.min_fixation_frame_ratio}")
        
        # Set up model
        self.model = YOLOWorldModel(
            conf_threshold=self.config.models.yolo_world.onnx.conf_threshold,
            iou_threshold=self.config.models.yolo_world.onnx.iou_threshold
        )
        num_workers = getattr(config.processing, "n_cores", None)
        self.model.load_model(model_path, num_workers)
        self.model.set_classes(self.class_names)
        
        # Set class names
        self.model.set_classes(self.class_names)
        
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
                
                # Store detections for fixation calculation
                self.all_detections.extend(detections)
                
                # Store gaze points for advanced fixation analysis
                self.gaze_points.append((gaze_x, gaze_y, frame_idx))
                
                # Calculate current fixation scores and component scores
                if len(self.all_detections) > 0:
                    # Compute scores
                    scores = self._compute_fixation_scores()
                    
                    # Find top scoring object
                    top_score_class = max(scores.keys(), key=lambda x: scores[x]['final_score']) if scores else None
                    
                    # Update each detection with its scores and status
                    for detection in detections:
                        if detection.is_fixated and detection.class_name in scores:
                            obj_scores = scores[detection.class_name]
                            # Set overall fixation score
                            detection.fixation_score = obj_scores['final_score']
                            detection.is_top_scoring = (detection.class_name == top_score_class)
                            
                            # Set component scores
                            detection.components.confidence = obj_scores['confidence']
                            detection.components.stability = obj_scores['stability']
                            detection.components.gaze_proximity = obj_scores['gaze_proximity']
                            detection.components.fixation_ratio = obj_scores['fixation_ratio']
                            detection.components.duration_weighted = obj_scores['duration_weighted']
                            detection.components.gaze_distance = obj_scores['gaze_distance']
                
                self._log_fixated_objects(frame_idx, detections)
                
                # Log to tracer if available
                if self.tracer and detections:
                    detection_dicts = [d.to_dict() for d in detections]
                    self.tracer.log_yolo_objects_detected(frame_idx, detection_dicts)
            
            # Store new detections after logging
            if not found_fixated_objects:
                self.all_detections.extend(detections)
            
            self.total_frames += 1
            
        except Exception as e:
            logger.warning(f"[Frame {frame_idx}] Object detection failed: {str(e)}")
            import traceback
            logger.debug(f"Error details: {traceback.format_exc()}")
        
        return detections
    
    def _compute_fixation_scores(self) -> Dict[str, Dict[str, float]]:
        """Compute fixation scores for all detected objects.
        
        Returns:
            Dictionary mapping object names to score components and final score
        """
        scores = {}
        
        # Extract unique objects from detections
        unique_objects = set(d.class_name for d in self.all_detections if d.is_fixated)
        
        # Reset filter statistics for this computation
        self.filtered_stats = {
            'fixation_ratio': 0,
            'confidence': 0,
            'stability': 0,
            'gaze_proximity': 0,
            'total_considered': 0,
            'passed_all': 0
        }
        
        for obj_name in unique_objects:
            # Get all fixated detections for this object
            obj_detections = [d for d in self.all_detections if d.class_name == obj_name and d.is_fixated]
            
            # Skip if not enough detections
            if not obj_detections:
                continue
                
            # Track total objects considered
            self.filtered_stats['total_considered'] += 1
            
            # Initialize component scores dict
            components = {}
                
            # Apply absolute minimum fixation threshold
            fixation_frames = len(set(d.frame_idx for d in obj_detections))
            if fixation_frames < self.min_fixation_frame_threshold:
                logger.debug(f"Object {obj_name} filtered out: fixation frames {fixation_frames} < threshold {self.min_fixation_frame_threshold}")
                continue

            # Compute fixation ratio
            fixation_ratio = fixation_frames / max(self.total_frames, 1)
            components['fixation_ratio'] = fixation_ratio
            
            # Apply minimum fixation threshold
            if fixation_ratio < self.min_fixation_frame_ratio:
                logger.debug(f"Object {obj_name} filtered out: fixation ratio {fixation_ratio:.2f} < threshold {self.min_fixation_frame_ratio}")
                self.filtered_stats['fixation_ratio'] += 1
                continue
                
            # Get confidence scores and bounding boxes
            confidence_scores = [d.score for d in obj_detections]
            bboxes = [d.bbox for d in obj_detections]
            
            # Get relevant gaze points (those in frames where object was detected)
            obj_frame_indices = set(d.frame_idx for d in obj_detections)
            relevant_gaze_points = [g for g in self.gaze_points if g[2] in obj_frame_indices]
            
            # 1. Compute geometric mean of confidence scores
            mean_confidence = self._compute_geometric_mean(confidence_scores)
            components['confidence'] = mean_confidence
            
            # Apply confidence threshold
            if mean_confidence < self.confidence_threshold:
                logger.debug(f"Object {obj_name} filtered out: confidence score {mean_confidence:.2f} < threshold {self.confidence_threshold}")
                self.filtered_stats['confidence'] += 1
                continue
            
            # 2. Compute bounding box stability
            stability_score = self._compute_mean_iou(bboxes)
            components['stability'] = stability_score
            
            # Apply stability threshold
            if stability_score < self.bbox_stability_threshold:
                logger.debug(f"Object {obj_name} filtered out: stability score {stability_score:.2f} < threshold {self.bbox_stability_threshold}")
                self.filtered_stats['stability'] += 1
                continue
            
            # 3. Compute gaze proximity weighting
            gaze_distance = self._compute_mean_gaze_distance(bboxes, relevant_gaze_points)
            gaze_weight = 1.0 / (1.0 + gaze_distance)
            components['gaze_proximity'] = gaze_weight
            components['gaze_distance'] = gaze_distance
            
            # Apply gaze proximity threshold
            if gaze_weight < self.gaze_proximity_threshold:
                logger.debug(f"Object {obj_name} filtered out: gaze proximity {gaze_weight:.2f} < threshold {self.gaze_proximity_threshold}")
                self.filtered_stats['gaze_proximity'] += 1
                continue
            
            # 4. Weight by fixation duration
            duration_weighted_score = mean_confidence * fixation_ratio
            components['duration_weighted'] = duration_weighted_score
            
            # 5. Compute final weighted fixation score
            final_score = (
                pow(duration_weighted_score, self.duration_weight) * 
                pow(stability_score, self.bbox_stability_weight) * 
                pow(gaze_weight, self.gaze_proximity_weight)
            )
            
            # Store final score with components
            components['final_score'] = final_score
            scores[obj_name] = components
                
            self.filtered_stats['passed_all'] += 1
            
            logger.debug(f"Fixation score for {obj_name}: {final_score:.4f}")
            logger.debug(f"  - Duration score: {duration_weighted_score:.4f} (ratio: {fixation_ratio:.2f})")
            logger.debug(f"  - Confidence score: {mean_confidence:.4f} (threshold: {self.confidence_threshold})")
            logger.debug(f"  - Stability score: {stability_score:.4f} (threshold: {self.bbox_stability_threshold})")
            logger.debug(f"  - Gaze weight: {gaze_weight:.4f} (distance to center: {gaze_distance:.2f}, threshold: {self.gaze_proximity_threshold})")
            
        # Log filter statistics
        if self.filtered_stats['total_considered'] > 0:
            logger.debug(f"Fixation filtering stats: {self.filtered_stats['passed_all']}/{self.filtered_stats['total_considered']} passed all filters")
            logger.debug(f"  - Filtered by fixation ratio: {self.filtered_stats['fixation_ratio']}")
            logger.debug(f"  - Filtered by confidence: {self.filtered_stats['confidence']}")
            logger.debug(f"  - Filtered by stability: {self.filtered_stats['stability']}")
            logger.debug(f"  - Filtered by gaze proximity: {self.filtered_stats['gaze_proximity']}")

        # Store final scores in the instance variable
        self.fixation_scores = {obj: scores[obj]['final_score'] for obj in scores}
            
        return scores
    
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
        raw_detections = self.model.run_inference(frame, self.class_names, self.obj_labels)
        if not raw_detections:
            return []
            
        detections = []
        for detection in raw_detections:
            left, top, width, height = detection['bbox']
            
            # Check if gaze intersects with this object
            margin = self.config.graph.fixated_object_detection.bbox_margin
            is_fixated = (left - margin <= gaze_x <= left + width + margin and top - margin <= gaze_y <= top + height + margin)
            
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
            fixated_count = len(fixated_detections)
            
            top_object = next((d for d in fixated_detections if d.is_top_scoring), None)
            if top_object:
                logger.info(f"[Frame {frame_idx}] Top fixated object: {top_object.class_name} (score: {top_object.fixation_score:.2f})")
            else:
                logger.info(f"[Frame {frame_idx}] Found {fixated_count} fixated objects")
            
            logger.debug(f"[Frame {frame_idx}] {fixated_count} fixated object detections:")
            for detection in fixated_detections:
                bbox = detection.bbox
                top_indicator = " (TOP)" if detection.is_top_scoring else ""
                
                score_info = f"detection score: {detection.score:.2f}"
                if detection.fixation_score > 0:
                    score_info += f", fixation score: {detection.fixation_score:.2f}{top_indicator}"
                    
                logger.debug(f"  - {detection.class_name} ({score_info}, "
                          f"bbox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}])")
                
                # Log component scores if available and significant
                if detection.fixation_score > 0:
                    comp = detection.components
                    logger.debug(f"    Components: conf={comp.confidence:.2f}, "
                               f"stab={comp.stability:.2f}, "
                               f"gaze={comp.gaze_proximity:.2f}, "
                               f"ratio={comp.fixation_ratio:.2f}")
    
    def reset(self) -> None:
        """Reset detection state."""
        self.fixated_objects_found = False
        self.all_detections = []
        self.gaze_points = []
        self.total_frames = 0
        self.fixation_scores = {}
        
        # Statistics for filtered objects
        self.filtered_stats = {
            'fixation_ratio': 0,
            'confidence': 0,
            'stability': 0,
            'gaze_proximity': 0,
            'total_considered': 0,
            'passed_all': 0
        }
    
    def has_fixated_objects(self) -> bool:
        """Check if any objects were fixated during detection.
        
        Returns:
            True if any objects were fixated
        """
        if not self.fixated_objects_found:
            return False
            
        # Compute fixation scores if not already done
        if not self.fixation_scores:
            scores = self._compute_fixation_scores()
            # fixation_scores is now populated by _compute_fixation_scores

        # Only return true if any object passes all thresholds
        return len(self.fixation_scores) > 0
    
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
    
    def get_fixated_object(self) -> Tuple[str, float]:
        """Get the most likely fixated object based on fixation scores.
        
        Returns:
            Tuple of (object_label, confidence_score)
            
        Raises:
            ValueError: If no fixated object is found
        """
        # Compute fixation scores if not already done
        if not self.fixation_scores:
            self._compute_fixation_scores()
            # fixation_scores is now populated by _compute_fixation_scores
            
        # If no fixation scores found, no object passed our thresholds
        if not self.fixation_scores:
            logger.warning("No objects passed all thresholds. Filter statistics:")
            logger.warning(f"  - Total objects considered: {self.filtered_stats['total_considered']}")
            logger.warning(f"  - Filtered by fixation ratio: {self.filtered_stats['fixation_ratio']}")
            logger.warning(f"  - Filtered by confidence: {self.filtered_stats['confidence']}")
            logger.warning(f"  - Filtered by stability: {self.filtered_stats['stability']}")
            logger.warning(f"  - Filtered by gaze proximity: {self.filtered_stats['gaze_proximity']}")
            
            raise ValueError("No fixated objects found that pass all thresholds")
            
        # Get object with highest fixation score
        fixated_object, score = max(self.fixation_scores.items(), key=lambda x: x[1])
        
        # Log all scores for comparison (at INFO level as this is final output)
        logger.debug("Final fixation scores:")
        for obj, s in sorted(self.fixation_scores.items(), key=lambda x: x[1], reverse=True):
            logger.debug(f"  - {obj}: {s:.4f}")
            
        return fixated_object, score