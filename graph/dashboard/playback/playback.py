"""Graph playback component for managing trace events and graph building."""
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
import networkx as nx
import math

from graph.dashboard.playback.event import GraphEvent


class Playback:
    """Manages graph construction playback from a trace file.
    
    This class loads events from a trace file and builds the graph incrementally.
    It provides methods to access events for specific frames and controls the 
    graph state as playback progresses.
    
    Attributes:
        trace_file_path: Path to the trace file
        graph: NetworkX directed graph instance
        last_built_frame: Last frame number that was built
        last_added_node: Last node ID that was added
        last_added_edge: Last edge that was added (source_id, target_id)
        object_detections: Dictionary mapping frame numbers to detection events
        events: List of all events loaded from the trace file
        frame_to_events: Dictionary mapping frame numbers to events
        min_frame: Minimum frame number in the trace
        max_frame: Maximum frame number in the trace
    """
    
    def __init__(self, trace_file_path: str):
        """Initialize the playback system with a trace file.
        
        Args:
            trace_file_path: Path to the trace file containing graph events
        """
        self.trace_file_path = Path(trace_file_path)
        self.graph = nx.DiGraph()
        self.last_built_frame = -1
        self.last_added_node = None
        self.last_added_edge = None
        self.object_detections = {}  # Frame number -> detection event
        self.yolo_detections = {}  # Frame number -> YOLO detection event
        self._load_events()
    
    def _load_events(self) -> None:
        """Load events from the trace file and organize them by frame number."""
        self.events = []
        self.frame_to_events = defaultdict(list)
        
        with open(self.trace_file_path, 'r') as f:
            for line in f:
                event_data = json.loads(line.strip())
                event = GraphEvent(event_data)
                self._rotate_event_angles(event)
                self._normalize_detection_format(event)
                self.events.append(event)
                self.frame_to_events[event.frame_number].append(event)
                
                # Track object detection events separately for quick access
                if event.event_type == "gaze_object_detected":
                    self.object_detections[event.frame_number] = event
                elif event.event_type == "yolo_objects_detected":
                    self.yolo_detections[event.frame_number] = event
        
        frames = list(self.frame_to_events.keys())
        self.min_frame = min(frames) if frames else 0
        self.max_frame = max(frames) if frames else 0

    def _rotate_event_angles(self, event: GraphEvent) -> None:
        # we need to rotate all data.features.angle (radians) and data.features.angle_degrees (degrees) to match our intuition
        # check if this property exists in the event data
        if "features" in event.data and "angle" in event.data["features"]:
            event.data["features"]["angle"] = self._rotate_angle(event.data["features"]["angle"], True)
            if "angle_degrees" in event.data["features"]:
                event.data["features"]["angle_degrees"] = self._rotate_angle(event.data["features"]["angle_degrees"], False)

    def _rotate_angle(self, angle: float, is_radian: bool) -> float:
        # Angle calculation assumes origin of gaze vectors is at top left
        # corner of image. We need to flip the angles so that they coincide 
        # with our intuition of angles when looking at the rendered image
        # with an origin in the bottom left
        if is_radian:
            modulo = 2 * math.pi
        else:
            modulo = 360
        return (modulo - angle) % (modulo)
    
    def get_events_for_frame(self, frame_number: int) -> List[GraphEvent]:
        """Get all events for a specific frame number.
        
        Args:
            frame_number: The frame number to get events for
            
        Returns:
            List of GraphEvent objects for the specified frame
        """
        return self.frame_to_events.get(frame_number, [])
    
    def get_object_detection(self, frame_number: int) -> Optional[GraphEvent]:
        """Get the object detection event for a specific frame, if any.
        
        Args:
            frame_number: The frame number to get object detection for
            
        Returns:
            GraphEvent containing object detection data, or None if not available
        """
        return self.object_detections.get(frame_number)
    
    def get_yolo_detections(self, frame_number: int) -> Optional[GraphEvent]:
        """Get the YOLO-World detections for a specific frame, if any.
        
        Args:
            frame_number: The frame number to get YOLO detections for
            
        Returns:
            GraphEvent containing YOLO detection data, or None if not available
        """
        return self.yolo_detections.get(frame_number)
    
    def _process_event(self, event: GraphEvent) -> None:
        """Process a single event and update the graph state accordingly.
        
        Args:
            event: The event to process
        """
        if event.event_type == "node_added":
            node_id = event.data["node_id"]
            label = event.data["label"]
            features = event.data.get("features", {})
            
            # Position is calculated by the layout algorithm instead of stored
            self.graph.add_node(
                node_id,
                label=label,
                features=features
            )
            self.last_added_node = node_id
            
        elif event.event_type == "node_updated":
            node_id = event.data["node_id"]
            label = event.data["label"]
            features = event.data.get("features", {})
            
            # Update existing node with new features
            if node_id in self.graph:
                self.graph.nodes[node_id].update(
                    label=label,
                    features=features
                )
            
        elif event.event_type == "edge_added":
            source_id = event.data["source_id"]
            target_id = event.data["target_id"]
            edge_type = event.data["edge_type"]
            features = event.data.get("features", {})
            
            self.graph.add_edge(
                source_id,
                target_id,
                edge_type=edge_type,
                features=features
            )
            self.last_added_edge = (source_id, target_id)
        
        elif event.event_type == "gaze_object_detected":
            # Store the latest object detection for this frame
            self.object_detections[event.frame_number] = event
            
        elif event.event_type == "yolo_objects_detected":
            # Store the latest YOLO detections for this frame
            self.yolo_detections[event.frame_number] = event
    
    def build_graph_until_frame(self, frame_number: int) -> nx.DiGraph:
        """Build the graph incrementally up to the specified frame.
        
        If the requested frame is earlier than the last built frame,
        the graph is reset and rebuilt from the beginning.
        
        Args:
            frame_number: The frame number to build the graph up to
            
        Returns:
            The built NetworkX directed graph
        """
        if frame_number < self.last_built_frame:
            self.graph = nx.DiGraph()
            self.last_built_frame = -1
        
        if frame_number > self.last_built_frame:
            # Process events for each frame between last_built_frame+1 and frame_number
            for current_frame in range(self.last_built_frame + 1, frame_number + 1):
                for event in self.frame_to_events.get(current_frame, []):
                    self._process_event(event)
            
            self.last_built_frame = frame_number
        
        return self.graph

    def _normalize_detection_format(self, event: GraphEvent) -> None:
        """Ensure detection format consistency between old and new formats.
        
        This handles the transition from flat detection objects to the new nested structure
        with proper component scores and fixation information.
        
        Args:
            event: GraphEvent to normalize
        """
        if event.event_type != "yolo_objects_detected":
            return
            
        # Process each detection to ensure consistent structure
        if "detections" in event.data:
            for i, detection in enumerate(event.data["detections"]):
                # Skip if already in new format
                if "detection" in detection and "fixation" in detection:
                    continue
                    
                # Convert from flat to nested structure
                new_detection = {
                    "detection": {
                        "bbox": detection.get("bbox", [0, 0, 0, 0]),
                        "class_name": detection.get("class_name", "unknown"),
                        "score": detection.get("score", 0.0),
                        "class_id": detection.get("class_id", -1),
                        "frame_idx": detection.get("frame_idx", -1)
                    },
                    "fixation": {
                        "is_fixated": detection.get("is_fixated", False),
                        "is_top_scoring": detection.get("is_top_scoring", False),
                        "score": detection.get("fixation_score", 0.0),
                        "components": {}
                    }
                }
                
                # Move component scores to nested structure
                components = {}
                # Check for old-style flat component scores
                if "confidence_score" in detection:
                    components["confidence"] = detection.get("confidence_score", 0.0)
                if "stability_score" in detection:
                    components["stability"] = detection.get("stability_score", 0.0)
                if "gaze_proximity_score" in detection:
                    components["gaze_proximity"] = detection.get("gaze_proximity_score", 0.0)
                if "fixation_ratio" in detection:
                    components["fixation_ratio"] = detection.get("fixation_ratio", 0.0)
                
                # If we have component scores in the new format structure
                if not components and "components" in detection:
                    components = detection.get("components", {})
                    
                new_detection["fixation"]["components"] = components
                
                # Replace with normalized structure
                event.data["detections"][i] = new_detection