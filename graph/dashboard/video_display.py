"""Video display component for the graph visualization dashboard."""
from typing import Optional, List, Tuple
import threading
import cv2
import numpy as np
import plotly.graph_objects as go

from graph.dashboard.graph_playback import GraphPlayback
from graph.dashboard.graph_constants import GAZE_TYPE_INFO, GAZE_TYPE_FIXATION
from graph.dashboard.utils import format_label


class VideoDisplay:
    """Component for displaying video frames with gaze and object overlays.
    
    This component manages the video capture, caching, and creating figures
    for display in the dashboard.
    
    Attributes:
        video_path: Path to the video file
        video_capture: OpenCV video capture object
        frame_cache: Cache of video frames
        max_cache_size: Maximum number of frames to cache
        video_lock: Thread lock for video operations
    """
    
    def __init__(self, video_path: Optional[str], max_cache_size: int = 100):
        """Initialize the video display component.
        
        Args:
            video_path: Path to the video file or None if no video
            max_cache_size: Maximum number of frames to cache
        """
        self.video_path = video_path
        self.video_capture = None
        self.frame_cache = {}
        self.max_cache_size = max_cache_size
        self.video_lock = threading.Lock()
        
        self._setup_video_capture()
    
    def _setup_video_capture(self) -> None:
        """Initialize the video capture if a valid video path is provided."""
        if self.video_path and self.video_path != "":
            try:
                self.video_capture = cv2.VideoCapture(self.video_path)
                cv2.setNumThreads(1)  # Limit OpenCV threading
            except Exception as e:
                print(f"Error opening video: {e}")
                self.video_capture = None
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a video frame by frame number.
        
        Args:
            frame_number: The frame number to retrieve
            
        Returns:
            The frame as a numpy array or None if not available
        """
        if self.video_capture is None:
            return None
            
        # Check if frame is in cache
        if frame_number in self.frame_cache:
            return self.frame_cache[frame_number]
            
        # Acquire frame from video
        with self.video_lock:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = self.video_capture.read()
            
            if not success:
                return None
                
            # Convert from BGR to RGB for Plotly
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Manage cache size
            if len(self.frame_cache) >= self.max_cache_size:
                oldest_frame = min(self.frame_cache.keys())
                del self.frame_cache[oldest_frame]
            
            self.frame_cache[frame_number] = frame_rgb
            return frame_rgb
    
    def create_empty_figure(self, height: int = 400) -> go.Figure:
        """Create an empty figure with appropriate layout.
        
        Args:
            height: Height of the figure in pixels
            
        Returns:
            Empty Plotly figure
        """
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0),
            height=height
        )
        return fig
    
    def add_gaze_overlay(
        self, 
        fig: go.Figure, 
        frame_number: int, 
        playback: GraphPlayback, 
        frame_dimensions: Tuple[int, int]
    ) -> None:
        """Add gaze point and object detection overlays to the figure.
        
        Args:
            fig: The Plotly figure to add overlays to
            frame_number: The current frame number
            playback: The GraphPlayback instance for event access
            frame_dimensions: Tuple of (width, height) for the frame
        """
        frame_width, frame_height = frame_dimensions
        events = playback.get_events_for_frame(frame_number)
        
        self._add_gaze_points(fig, events, frame_width, frame_height)
        self._add_object_detection(fig, playback, frame_number, frame_dimensions)
    
    def _add_gaze_points(
        self, 
        fig: go.Figure, 
        events: List, 
        frame_width: int, 
        frame_height: int
    ) -> None:
        """Add gaze point markers to the figure.
        
        Args:
            fig: The Plotly figure to add gaze points to
            events: List of events for the current frame
            frame_width: Width of the video frame
            frame_height: Height of the video frame
        """
        for event in events:
            if event.event_type == "frame_processed":
                pos = event.data["gaze_position"]
                if pos is None or (pos[0] == 0.0 and pos[1] == 0.0):
                    continue
                    
                x, y = pos[0] * frame_width, pos[1] * frame_height
                gaze_type = event.data["gaze_type"]
                gaze_info = GAZE_TYPE_INFO.get(
                    gaze_type, 
                    {"color": "black", "label": f"Other ({gaze_type})"}
                )
                
                fig.add_trace(go.Scattergl(
                    x=[x], y=[y],
                    mode="markers",
                    marker=dict(size=15, color=gaze_info["color"]),
                    hovertext=gaze_info["label"],
                    hoverinfo='text',
                    showlegend=False
                ))
    
    def _add_object_detection(
        self, 
        fig: go.Figure, 
        playback: GraphPlayback, 
        frame_number: int,
        frame_dimensions: Tuple[int, int]
    ) -> None:
        """Add object detection bounding box and labels to the figure.
        
        Args:
            fig: The Plotly figure to add object detection to
            playback: The GraphPlayback instance for event access
            frame_number: The current frame number
            frame_dimensions: Tuple of (width, height) for the frame
        """
        detection_event = playback.get_object_detection(frame_number)
        if not detection_event:
            return
            
        bbox = detection_event.data["bounding_box"]
        most_likely_label = detection_event.data["detected_object"]
        current_label = detection_event.data.get("current_detected_label", most_likely_label)
        potential_labels = detection_event.data.get("potential_labels", {})
        
        # Format the most likely label for display
        label_text = format_label(most_likely_label)
        
        # Create hover text with label information
        hover_text = self._create_detection_hover_text(
            current_label, 
            most_likely_label, 
            potential_labels
        )
        
        # Extract bounding box coordinates [x, y, width, height]
        x, y, width, height = bbox
        x0, y0, x1, y1 = x, y, x + width, y + height
        
        frame_width, frame_height = frame_dimensions
        
        # Add bounding box and label with improved styling
        self._add_styled_detection(
            fig, x0, y0, x1, y1, 
            label_text, hover_text, 
            frame_width, frame_height,
            potential_labels
        )
    
    def _create_detection_hover_text(
        self, 
        current_label: str, 
        most_likely_label: str,
        potential_labels: dict
    ) -> str:
        """Create hover text for object detection.
        
        Args:
            current_label: The current object label
            most_likely_label: The most likely object label
            potential_labels: Dictionary of potential labels and their counts
            
        Returns:
            Formatted hover text
        """
        sorted_labels = sorted(potential_labels.items(), key=lambda x: x[1], reverse=True)
        potential_labels_text = "<br>".join(
            [f"{format_label(obj)}: {count}" for obj, count in sorted_labels[:5]]
        )
        
        return (
            f"Current: {format_label(current_label)}<br>"
            f"Most likely: {format_label(most_likely_label)}<br><br>"
            f"Potential labels:<br>{potential_labels_text}"
        )
    
    def _add_styled_detection(
        self,
        fig: go.Figure,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        label_text: str,
        hover_text: str,
        frame_width: float,
        frame_height: float,
        potential_labels: dict
    ) -> None:
        """Add styled object detection with bounding box and label box.
        
        Args:
            fig: The Plotly figure to add detection to
            x0, y0: Top-left coordinates of the bounding box
            x1, y1: Bottom-right coordinates of the bounding box
            label_text: Text to display as the label
            hover_text: Text to display on hover
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            potential_labels: Dictionary of potential labels and their counts
        """        
        # Define colors
        box_color = GAZE_TYPE_INFO[GAZE_TYPE_FIXATION]["color"]
        box_fill = 'rgba(0, 0, 255, 0.1)'  # Blue with 10% opacity
        
        # Add main bounding box using Scattergl for better performance
        fig.add_trace(go.Scattergl(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            fill="toself",
            fillcolor=box_fill,
            mode="lines",
            line=dict(width=2, color=box_color),
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False
        ))
        
        # Calculate label text width based on its length
        text_width = len(label_text) * 7  # Approximate width based on character count
        
        # Calculate label box position
        # If the label would extend beyond right frame edge, align right edge with frame
        padding = 5  # Padding around text
        label_width = text_width + (padding * 2)
        
        # Ensure label box stays within frame boundaries
        label_x0 = x0
        label_x1 = label_x0 + label_width
        
        # If label extends beyond right edge, adjust position
        if label_x1 > frame_width:
            label_x1 = min(frame_width, x1)
            label_x0 = max(0, label_x1 - label_width)
        
        # Ensure label box stays within left edge
        label_x0 = max(0, label_x0)
        
        # Position the label box above the bounding box
        # If the box is too close to the top, put the label inside the top of the bounding box
        if y0 < 25:
            label_y0 = y0
            label_y1 = y0 + 20
        else:
            label_y0 = y0 - 20
            label_y1 = y0
        
        # Add colored label background using regular Scatter for proper fill rendering
        fig.add_trace(go.Scatter(
            x=[label_x0, label_x1, label_x1, label_x0, label_x0],
            y=[label_y0, label_y0, label_y1, label_y1, label_y0],
            fill="toself",
            fillcolor=box_color,
            mode="lines",
            line=dict(width=0, color=box_color),
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False
        ))
        
        # Add label text using regular Scatter for proper text rendering
        fig.add_trace(go.Scatter(
            x=[(label_x0 + label_x1) / 2],
            y=[(label_y0 + label_y1) / 2],
            mode="text",
            text=[label_text],
            textposition="middle center",
            textfont=dict(
                size=12, 
                color="white",
                family="Arial Bold"
            ),
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False
        ))
    
    def _calculate_confidence(self, potential_labels: dict) -> Optional[float]:
        """Calculate confidence percentage from potential labels."""
        if not potential_labels:
            return None
            
        sorted_labels = sorted(potential_labels.items(), key=lambda x: x[1], reverse=True)
        top_confidence = sorted_labels[0][1]
        total_votes = sum(count for _, count in sorted_labels)
        
        if total_votes <= 0:
            return None
            
        return (top_confidence / total_votes) * 100
    
    def create_figure(self, frame_number: int, playback: GraphPlayback) -> go.Figure:
        """Create a complete figure with the video frame and overlays.
        
        Args:
            frame_number: The current frame number
            playback: The GraphPlayback instance for event access
            
        Returns:
            Plotly figure with video frame and overlays
        """
        fig = self.create_empty_figure()
        
        frame = self.get_frame(frame_number)
        if frame is None:
            return fig
            
        frame_height, frame_width = frame.shape[:2]
        fig.add_trace(go.Image(z=frame))
        
        self.add_gaze_overlay(fig, frame_number, playback, (frame_width, frame_height))
        
        fig.update_layout(
            xaxis=dict(
                range=[0, frame_width],
                showgrid=False, 
                zeroline=False, 
                visible=False,
                constrain="domain"
            ),
            yaxis=dict(
                range=[frame_height, 0],
                showgrid=False, 
                zeroline=False, 
                visible=False,
                scaleanchor="x", 
                scaleratio=1,
                constrain="domain"
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            height=None,
            width=None,
            template="plotly_white"
        )
        
        return fig 