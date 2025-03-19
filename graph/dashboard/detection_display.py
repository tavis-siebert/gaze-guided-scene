"""Detection display component for showing object detection information."""
from typing import List, Dict, Any
import dash_bootstrap_components as dbc
from dash import html

from graph.dashboard.graph_playback import GraphPlayback
from graph.dashboard.graph_constants import GAZE_TYPE_INFO, GAZE_TYPE_FIXATION


class DetectionDisplay:
    """Component for displaying object detection statistics.
    
    This component creates a panel showing object detection information
    for the current frame, including the current and most likely labels,
    bounding box coordinates, and potential label distributions.
    """
    
    def create_layout(self, frame_number: int, playback: GraphPlayback) -> List:
        """Create a layout of components showing detection statistics.
        
        Args:
            frame_number: The current frame number
            playback: The GraphPlayback instance for event access
            
        Returns:
            List of Dash HTML components for the detection panel
        """
        events = playback.get_events_for_frame(frame_number)
        detection_events = [e for e in events if e.event_type == "gaze_object_detected"]
        
        if not detection_events:
            return [html.P("No object detections in this frame", 
                         className="text-muted text-center")]
        
        # Use the most recent detection event for this frame
        event = detection_events[-1]
        most_likely_label = event.data["detected_object"]
        current_label = event.data.get("current_detected_label", most_likely_label)
        bbox = event.data["bounding_box"]
        potential_labels = event.data.get("potential_labels", {})
        
        # Calculate confidence percentage for most likely label
        confidence_pct = self._calculate_confidence(potential_labels)
        
        # Create the detection layout
        return self._create_detection_layout(
            current_label, 
            most_likely_label, 
            confidence_pct,
            bbox, 
            potential_labels
        )
    
    def _calculate_confidence(self, potential_labels: Dict[str, int]) -> float:
        """Calculate confidence percentage for the most likely label.
        
        Args:
            potential_labels: Dictionary of potential labels and their counts
            
        Returns:
            Confidence percentage (0-100)
        """
        if not potential_labels:
            return 0
            
        sorted_labels = sorted(potential_labels.items(), key=lambda x: x[1], reverse=True)
        top_confidence = sorted_labels[0][1]
        total_votes = sum(count for _, count in sorted_labels)
        
        return (top_confidence / total_votes) * 100 if total_votes > 0 else 0
    
    def _create_detection_layout(
        self, 
        current_label: str, 
        most_likely_label: str,
        confidence_pct: float,
        bbox: List[float], 
        potential_labels: Dict[str, int]
    ) -> List:
        """Create the detection panel layout components.
        
        Args:
            current_label: The current object label
            most_likely_label: The most likely object label
            confidence_pct: Confidence percentage for the most likely label
            bbox: Bounding box coordinates [x, y, width, height]
            potential_labels: Dictionary of potential labels and their counts
            
        Returns:
            List of Dash HTML components for the detection panel
        """
        # Sort labels by count (descending)
        sorted_labels = sorted(potential_labels.items(), key=lambda x: x[1], reverse=True)
        max_count = max(count for _, count in sorted_labels) if sorted_labels else 1
        
        children = [
            # Current detection row
            dbc.Row([
                dbc.Col([
                    html.H5("Current Detection:", className="text-primary mb-0")
                ], width=5),
                dbc.Col([
                    html.H5(f"{current_label.capitalize()}", className="mb-0")
                ], width=7)
            ], className="mb-2"),
            
            # Most likely object row
            dbc.Row([
                dbc.Col([
                    html.H5("Most Likely Object:", className="text-primary mb-0")
                ], width=5),
                dbc.Col([
                    html.H5([
                        f"{most_likely_label.capitalize()} ",
                        html.Small(f"({confidence_pct:.1f}%)", className="text-muted")
                    ], className="mb-0")
                ], width=7)
            ], className="mb-3"),
            
            # Additional detection details
            html.Div([
                # Bounding box information
                html.Div([
                    html.Strong("Bounding Box [x, y, width, height]: "),
                    html.Span(f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                ], className="mb-2"),
                
                # Potential labels header
                html.Strong("Potential Labels:", className="d-block mb-2"),
                
                # Potential labels progress bars
                html.Div([
                    self._create_label_progress_bar(
                        label, 
                        count, 
                        max_count, 
                        most_likely_label, 
                        current_label
                    ) for label, count in sorted_labels[:10]  # Show top 10 labels
                ])
            ])
        ]
        
        return children
    
    def _create_label_progress_bar(
        self, 
        label: str, 
        count: int, 
        max_count: int,
        most_likely_label: str, 
        current_label: str
    ) -> dbc.Row:
        """Create a progress bar row for a label.
        
        Args:
            label: The object label
            count: The label count/votes
            max_count: The maximum count across all labels
            most_likely_label: The most likely object label
            current_label: The current object label
            
        Returns:
            Dash Bootstrap Row component with label and progress bar
        """
        # Determine progress bar color based on label status
        color = "info" if label == most_likely_label else \
                "success" if label == current_label else "secondary"
        
        # Calculate progress percentage
        progress_pct = (count / max_count) * 100
        
        return dbc.Row([
            dbc.Col([
                html.Span(label.capitalize(), className="align-middle")
            ], width=3),
            dbc.Col([
                dbc.Progress(
                    value=progress_pct,
                    label=f"{count}",
                    color=color,
                    className="mb-2"
                )
            ], width=9)
        ]) 