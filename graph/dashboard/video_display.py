"""Video display component for the graph visualization dashboard."""
from typing import Optional, List, Tuple, Dict
import threading
import cv2
import numpy as np
import plotly.graph_objects as go
import base64

from graph.dashboard.graph_playback import GraphPlayback
from graph.dashboard.graph_constants import GAZE_TYPE_INFO, GAZE_TYPE_FIXATION
from graph.dashboard.utils import format_label
from egtea_gaze.constants import RESOLUTION


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
        batch_size: Number of frames to read at once
        empty_figure: Pre-configured empty figure with proper layout
        frame_width: Width of video frames
        frame_height: Height of video frames
    """
    
    def __init__(self, video_path: Optional[str], max_cache_size: int = 240, batch_size: int = 48):
        """Initialize the video display component.
        
        Args:
            video_path: Path to the video file or None if no video
            max_cache_size: Maximum number of frames to cache
            batch_size: Number of frames to read at once
        """
        self.video_path = video_path
        self.video_capture = None
        self.frame_cache = {}
        self.max_cache_size = max_cache_size
        self.video_lock = threading.Lock()
        self.batch_size = batch_size
        self.frame_width, self.frame_height = RESOLUTION
        
        self.empty_figure = self._create_empty_figure()
        self._setup_video_capture()
    
    def _create_empty_figure(self) -> go.Figure:
        """Create an empty figure with proper layout based on constant resolution.
        
        Returns:
            Plotly figure with proper layout settings
        """
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(
                range=[0, self.frame_width],
                showgrid=False, 
                zeroline=False, 
                visible=False,
                constrain="domain"
            ),
            yaxis=dict(
                range=[self.frame_height, 0],
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
            
        # Calculate batch range
        batch_start = frame_number - (frame_number % self.batch_size)
        batch_end = min(batch_start + self.batch_size, int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        # Acquire frames in batch
        with self.video_lock:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, batch_start)
            
            # Read all frames in the batch
            for current_frame in range(batch_start, batch_end):
                success, frame = self.video_capture.read()
                if not success:
                    continue
                    
                # Convert from BGR to RGB for Plotly
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Manage cache size
                if len(self.frame_cache) >= self.max_cache_size:
                    oldest_frame = min(self.frame_cache.keys())
                    del self.frame_cache[oldest_frame]
                
                self.frame_cache[current_frame] = frame_rgb
            
            # Return the requested frame if it was successfully read
            return self.frame_cache.get(frame_number)
    
    def _get_gaze_traces(self, events: List, traces: List[go.Trace]) -> None:
        """Add gaze point marker traces to the provided traces list.
        
        Args:
            events: List of events for the current frame
            traces: List to append gaze traces to
        """
        for event in events:
            if event.event_type == "frame_processed":
                pos = event.data["gaze_position"]
                if pos is None or (pos[0] == 0.0 and pos[1] == 0.0):
                    continue
                    
                x, y = pos[0] * self.frame_width, pos[1] * self.frame_height
                gaze_type = event.data["gaze_type"]
                gaze_info = GAZE_TYPE_INFO.get(
                    gaze_type, 
                    {"color": "black", "label": f"Other ({gaze_type})"}
                )
                
                traces.append(go.Scattergl(
                    x=[x], y=[y],
                    mode="markers",
                    marker=dict(size=15, color=gaze_info["color"]),
                    hovertext=gaze_info["label"],
                    hoverinfo='text',
                    showlegend=False
                ))
    
    def _get_detection_traces(
        self, 
        playback: GraphPlayback, 
        frame_number: int,
        traces: List[go.Trace]
    ) -> None:
        """Add object detection bounding box and label traces to the provided traces list.
        
        Args:
            playback: The GraphPlayback instance for event access
            frame_number: The current frame number
            traces: List to append detection traces to
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
        
        # Define colors
        box_color = GAZE_TYPE_INFO[GAZE_TYPE_FIXATION]["color"]
        box_fill = 'rgba(0, 0, 255, 0.1)'  # Blue with 10% opacity
        
        # Add main bounding box using Scattergl for better performance
        traces.append(go.Scattergl(
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
        if label_x1 > self.frame_width:
            label_x1 = min(self.frame_width, x1)
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
        traces.append(go.Scatter(
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
        traces.append(go.Scatter(
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
    
    def create_figure(self, frame_number: int, playback: GraphPlayback) -> go.Figure:
        """Create a complete figure with the video frame and overlays.
        
        Args:
            frame_number: The current frame number
            playback: The GraphPlayback instance for event access
            
        Returns:
            Plotly figure with video frame and overlays
        """
        frame = self.get_frame(frame_number)
        fig = go.Figure(self.empty_figure)
        
        # Add background image if frame exists
        if frame is not None:
            img_src = self.numpy_to_base64(frame)
            fig.update_layout(
                images=[dict(
                    source=img_src,
                    xref="x",
                    yref="y",
                    x=0,
                    y=0,
                    sizex=self.frame_width,
                    sizey=self.frame_height,
                    sizing="stretch",
                    layer="below"
                )]
            )
        
        # Collect overlay traces
        traces = []
        events = playback.get_events_for_frame(frame_number)
        self._get_gaze_traces(events, traces)
        self._get_detection_traces(playback, frame_number, traces)
        
        fig.add_traces(traces)
        return fig

    def numpy_to_base64(self, frame: np.ndarray) -> str:
        """Convert a numpy frame to base64 encoded PNG."""
        if frame is None:
            return ""
        # Convert RGB to BGR for OpenCV encoding
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.png', frame_bgr)
        if not success:
            return ""
        return f"data:image/png;base64,{base64.b64encode(buffer).decode()}" 