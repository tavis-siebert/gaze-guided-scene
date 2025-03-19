"""Graph playback component for managing trace events and graph building."""
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
import networkx as nx

from graph.dashboard.graph_event import GraphEvent


class GraphPlayback:
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
        self._load_events()
    
    def _load_events(self) -> None:
        """Load events from the trace file and organize them by frame number."""
        self.events = []
        self.frame_to_events = defaultdict(list)
        
        with open(self.trace_file_path, 'r') as f:
            for line in f:
                event_data = json.loads(line.strip())
                event = GraphEvent(event_data)
                self.events.append(event)
                self.frame_to_events[event.frame_number].append(event)
                
                # Track object detection events separately for quick access
                if event.event_type == "gaze_object_detected":
                    self.object_detections[event.frame_number] = event
        
        frames = list(self.frame_to_events.keys())
        self.min_frame = min(frames) if frames else 0
        self.max_frame = max(frames) if frames else 0
    
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