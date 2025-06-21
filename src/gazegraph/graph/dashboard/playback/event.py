"""Graph event data class for parsing events from trace files."""

from typing import Dict, Any


class GraphEvent:
    """Represents a single event in the graph construction trace.

    Attributes:
        event_type: Type of the event (e.g., "node_added", "edge_added")
        frame_number: Frame number when the event occurred
        timestamp: Timestamp of the event
        event_id: Unique identifier for the event
        data: Event-specific data
    """

    def __init__(self, event_data: Dict[str, Any]):
        """Initialize a graph event from raw event data.

        Args:
            event_data: Dictionary containing the event data
        """
        self.event_type = event_data["event_type"]
        self.frame_number = event_data["frame_number"]
        self.timestamp = event_data["timestamp"]
        self.event_id = event_data["event_id"]
        self.data = event_data["data"]
