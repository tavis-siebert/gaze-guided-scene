"""
Graph tracing module for recording graph construction events.

This module provides functionality to trace and log the graph construction process,
enabling playback and visualization for debugging and analysis.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import numpy as np

from logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

class GraphTracer:
    """
    Records and logs graph construction events for later visualization and analysis.
    
    This class captures significant events during graph construction such as node creation,
    edge addition, and saccades, storing them in a structured format for
    later playback and visualization.
    """
    
    def __init__(self, output_path: Union[str, Path], video_name: str, enabled: bool = True):
        """
        Initialize the graph tracer.
        
        Args:
            output_path: Directory where trace files will be saved
            video_name: Name of the video being processed (used in filename)
            enabled: Whether tracing is enabled
        """
        self.enabled = enabled
        if not self.enabled:
            logger.info("Graph tracing disabled")
            return
            
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.trace_file = self.output_path / f"{video_name}_trace.jsonl"
        # Clear any existing trace file
        with open(self.trace_file, 'w') as f:
            pass
            
        self.event_count = 0
        logger.info(f"Graph tracer initialized. Logging to {self.trace_file}")
    
    def log_event(self, event_type: str, frame_number: int, data: Dict[str, Any]) -> None:
        """
        Log a graph construction event.
        
        Args:
            event_type: Type of event (e.g., 'node_added', 'edge_added', 'frame_processed')
            frame_number: Video frame number when the event occurred
            data: Event-specific data
        """
        if not self.enabled:
            return
            
        # Sanitize data to ensure it's JSON serializable
        data = self._sanitize_data(data)
        
        event = {
            "event_type": event_type,
            "frame_number": frame_number,
            "timestamp": time.time(),
            "event_id": self.event_count,
            "data": data
        }
        
        try:
            self._write_event(event)
            self.event_count += 1
        except Exception as e:
            logger.error(f"Error writing event: {e}")
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize data to ensure it's JSON serializable.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, tuple):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, (np.ndarray, np.generic)):
            return data.tolist()
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            # Convert to string if not a basic type
            return str(data)
    
    def log_node_added(self, frame_number: int, node_id: int, label: str, 
                      features: Dict[str, Any]) -> None:
        """
        Log a node addition event.
        
        Args:
            frame_number: Video frame number
            node_id: ID of the added node
            label: Object label of the node
            features: Node features from Node.get_features()
        """
        data = {
            "node_id": node_id,
            "label": label,
            "features": features
        }
            
        self.log_event("node_added", frame_number, data)
    
    def log_node_updated(self, frame_number: int, node_id: int, label: str,
                        features: Dict[str, Any], visit: List[int]) -> None:
        """
        Log a node update event when a new visit is added to an existing node.
        
        Args:
            frame_number: Video frame number
            node_id: ID of the updated node
            label: Object label of the node
            features: Updated node features from Node.get_features()
            visit: The new visit period [start_frame, end_frame] that was added
        """
        data = {
            "node_id": node_id,
            "label": label,
            "features": features,
            "new_visit": visit
        }
            
        self.log_event("node_updated", frame_number, data)
    
    def log_edge_added(self, frame_number: int, source_id: int, target_id: int, 
                      edge_type: str, features: Dict[str, Any]) -> None:
        """
        Log an edge addition event.
        
        Args:
            frame_number: Video frame number
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Type of edge (e.g., "saccade")
            features: Edge features from Edge.get_features()
        """
        data = {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "features": features
        }
            
        self.log_event("edge_added", frame_number, data)
    
    def log_checkpoint_created(self, frame_number: int, node_count: int, edge_count: int, 
                              action_labels: Dict[str, Any]) -> None:
        """
        Log a checkpoint creation event.
        
        Args:
            frame_number: Video frame number
            node_count: Number of nodes in the graph
            edge_count: Number of edges in the graph
            action_labels: Dictionary containing the future action labels
        """
        data = {
            "node_count": node_count,
            "edge_count": edge_count,
            "action_labels": action_labels
        }
        
        self.log_event("checkpoint_created", frame_number, data)
    
    def log_frame(self, frame_number: int, gaze_position: Optional[List[float]], 
                 gaze_type: int, node_id: Optional[int] = None) -> None:
        """
        Log a frame event.
        
        Args:
            frame_number: Video frame number
            gaze_position: Optional [x, y] gaze position
            gaze_type: Type of gaze (1 for fixation, 2 for saccade, etc.)
            node_id: Optional ID of the associated node
        """
        data = {
            "gaze_position": gaze_position,
            "gaze_type": gaze_type
        }
            
        if node_id is not None:
            data["node_id"] = node_id
            
        self.log_event("frame_processed", frame_number, data)
    
    def log_gaze_object_detected(self, frame_number: int, 
                               detected_object: str,
                               current_detected_label: str,
                               bounding_box: List[int],
                               potential_labels: Dict[str, int]) -> None:
        """
        Log a gaze object detection event.
        
        Args:
            frame_number: Video frame number
            detected_object: Fixated object label based on counts
            current_detected_label: Current frame's detected object label
            bounding_box: Region of interest coordinates [x, y, width, height]
            potential_labels: Dictionary of potential object labels and their counts
        """
        data = {
            "detected_object": detected_object,
            "current_detected_label": current_detected_label,
            "bounding_box": bounding_box,
            "potential_labels": potential_labels
        }
        
        self.log_event("gaze_object_detected", frame_number, data)
    
    def log_yolo_objects_detected(self, frame_number: int, detections: List[Dict[str, Any]]) -> None:
        """
        Log YOLO-World object detection results.
        
        Args:
            frame_number: Video frame number
            detections: List of detection objects with is_fixated field
        """
        data = {
            "detections": detections
        }
        
        self.log_event("yolo_objects_detected", frame_number, data)
    
    def _write_event(self, event: Dict[str, Any]) -> None:
        """
        Write an event to the trace file.
        
        Args:
            event: Event data to write
        """
        with open(self.trace_file, 'a') as f:
            try:
                f.write(json.dumps(event) + '\n')
            except (TypeError, ValueError, OverflowError) as e:
                logger.error(f"Error serializing event: {e}")
                # Try to write a simplified version of the event
                simplified_event = {
                    "event_type": event["event_type"],
                    "frame_number": event["frame_number"],
                    "timestamp": event["timestamp"],
                    "event_id": event["event_id"],
                    "data": {"error": "Failed to serialize original data"}
                }
                f.write(json.dumps(simplified_event) + '\n') 