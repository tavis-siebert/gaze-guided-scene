from typing import Optional, List, Tuple, Dict, Any
import threading
import cv2
import numpy as np
import plotly.graph_objects as go
import base64
import dash_bootstrap_components as dbc
from dash import dcc, html
import os
import pandas as pd

from gazegraph.logger import get_logger
from gazegraph.graph.dashboard.components.base import BaseComponent
from gazegraph.graph.dashboard.playback import Playback
from gazegraph.graph.dashboard.utils.constants import GAZE_TYPE_INFO, GAZE_TYPE_FIXATION
from gazegraph.graph.dashboard.utils import format_label
from gazegraph.datasets.egtea_gaze.constants import RESOLUTION

logger = get_logger(__name__)

class VideoDisplay(BaseComponent):
    """Component for displaying video frames with gaze and object overlays.
    
    This component manages the video capture, caching, and creating figures
    for display in the dashboard.
    
    Attributes:
        video_path: Path to the video file
        video_capture: OpenCV video capture object
        frame_cache: Cache of video frame batches
        max_cache_size: Maximum number of batches to cache
        video_lock: Thread lock for video operations
        batch_size: Number of frames to read at once
        empty_figure: Pre-configured empty figure with proper layout
        frame_width: Width of video frames
        frame_height: Height of video frames
        batch_order: List of batch numbers in FIFO order
    """
    
    def __init__(
        self, 
        video_path: Optional[str], 
        max_cache_size: int = 240, 
        batch_size: int = 96,
        playback = None,
        verb_idx_file: Optional[str] = None,
        noun_idx_file: Optional[str] = None,
        train_split_file: Optional[str] = None,
        val_split_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize the video display component.
        
        Args:
            video_path: Path to the video file or None if no video
            max_cache_size: Maximum number of frames to cache
            batch_size: Number of frames to read at once
            playback: Playback instance for event access
            verb_idx_file: Path to the verb index mapping file
            noun_idx_file: Path to the noun index mapping file
            train_split_file: Path to the training data split file
            val_split_file: Path to the validation data split file
            **kwargs: Additional arguments to pass to BaseComponent
        """
        self.video_path = video_path
        self.video_capture = None
        self.frame_cache = {}
        self.max_cache_size = max_cache_size // batch_size  # Convert to number of batches
        self.video_lock = threading.Lock()
        self.batch_size = batch_size
        self.frame_width, self.frame_height = RESOLUTION
        self.batch_order = []  # Track batch order for FIFO
        self.playback = playback
        
        self.empty_figure = self._create_empty_figure()
        self._setup_video_capture()
        
        # Store file paths
        self.verb_idx_file = verb_idx_file
        self.noun_idx_file = noun_idx_file
        self.train_split_file = train_split_file
        self.val_split_file = val_split_file
        
        # Load action annotations and mappings
        self.action_annotations = self._load_action_annotations()
        self.verbs = self._load_mapping_file(self.verb_idx_file) if self.verb_idx_file else {}
        self.nouns = self._load_mapping_file(self.noun_idx_file) if self.noun_idx_file else {}
        
        super().__init__(component_id="video-display", **kwargs)
    
    def _load_mapping_file(self, file_path: str) -> Dict[int, str]:
        """Load mapping from ID to label from file.
        
        Args:
            file_path: Path to the mapping file
            
        Returns:
            Dictionary mapping IDs to labels
        """
        mapping = {}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        label = " ".join(parts[:-1])
                        id_num = int(parts[-1]) - 1 # make zero-indexed
                        mapping[id_num] = label
        except Exception as e:
            logger.error(f"Failed to load mapping file {file_path}: {e}")
        return mapping
    
    def _load_action_annotations(self) -> Dict[str, List[Dict]]:
        """Load action annotations from train and test splits.
        
        Returns:
            Dictionary mapping video names to lists of annotation dictionaries
        """
        annotations = {}
        
        try:
            # Process both training and validation split files
            for file_path in [self.train_split_file, self.val_split_file]:
                if file_path and os.path.exists(file_path):
                    df = pd.read_csv(file_path, header=None, names=['name', 'start', 'end', 'label1', 'label2'], sep='\t')
                    
                    # Group by video name
                    for name, group in df.groupby('name'):
                        if name not in annotations:
                            annotations[name] = []
                            
                        # Store each annotation as a dictionary
                        for _, row in group.iterrows():
                            annotations[name].append({
                                'start': int(row['start']),
                                'end': int(row['end']),
                                'verb_id': int(row['label1']),
                                'noun_id': int(row['label2'])
                            })
        except Exception as e:
            logger.error(f"Failed to load action annotations: {e}")
            
        return annotations
    
    def _get_current_action(self, frame_number: int) -> List[Dict]:
        """Get all current actions for a specific frame number.
        
        Args:
            frame_number: Current frame number
            
        Returns:
            List of dictionaries with action information or empty list if no actions
        """
        if not self.video_path:
            return []
            
        # Extract video name from path
        video_filename = os.path.basename(self.video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # Find annotations for this video
        video_annotations = self.action_annotations.get(video_name, [])
        
        # Find all annotations that contain this frame
        current_actions = []
        for annotation in video_annotations:
            if annotation['start'] <= frame_number <= annotation['end']:
                verb_id = annotation['verb_id']
                noun_id = annotation['noun_id']
                
                # Get verb and noun labels
                verb = self.verbs.get(verb_id, f"Unknown verb ({verb_id})")
                noun = self.nouns.get(noun_id, f"Unknown noun ({noun_id})")
                
                current_actions.append({
                    'verb': verb,
                    'noun': noun,
                    'verb_id': verb_id,
                    'noun_id': noun_id,
                    'start': annotation['start'],
                    'end': annotation['end']
                })
                
        return current_actions
    
    def create_layout(self) -> dbc.Card:
        """Create the component's layout.
        
        Returns:
            Dash Bootstrap Card component with video display
        """
        return dbc.Card([
            dbc.CardHeader("Video Feed"),
            dbc.CardBody([
                dcc.Graph(
                    id=f"{self.component_id}-graph",
                    style={"height": "60vh"},
                    config={"responsive": True}
                )
            ], style={"padding": "0.5rem"})
        ], className="shadow-sm h-100 w-100")
    
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
            
        # Calculate batch number and frame offset within batch
        batch_number = frame_number // self.batch_size
        frame_offset = frame_number % self.batch_size
            
        # Check if frame's batch is in cache
        if batch_number in self.frame_cache:
            return self.frame_cache[batch_number][frame_offset]
            
        # Calculate batch range
        batch_start = batch_number * self.batch_size
        batch_end = min(batch_start + self.batch_size, int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        # Acquire frames in batch
        with self.video_lock:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, batch_start)
            
            # Read all frames in the batch
            batch_frames = []
            for _ in range(batch_start, batch_end):
                success, frame = self.video_capture.read()
                if not success:
                    logger.warning(f"Failed to read frame at position {batch_start + len(batch_frames)}")
                    continue

                # Convert from BGR to RGB for Plotly
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch_frames.append(frame_rgb)
            
            if not batch_frames:
                logger.warning(f"No frames read for batch {batch_number}")
                return None
                
            # Manage cache size using FIFO
            if len(self.frame_cache) >= self.max_cache_size:
                oldest_batch = self.batch_order.pop(0)
                if oldest_batch in self.frame_cache:
                    del self.frame_cache[oldest_batch]
            
            # Add new batch to cache
            self.frame_cache[batch_number] = batch_frames
            self.batch_order.append(batch_number)
            
            # Return the requested frame if it was successfully read
            if frame_offset < len(batch_frames):
                return batch_frames[frame_offset]
            return None
    
    def _get_gaze_traces(self, events: List, traces: List[go.Trace], fig: go.Figure) -> None:
        """Add gaze point marker traces and saccade arrows to the provided traces list.
        
        Args:
            events: List of events for the current frame
            traces: List to append gaze traces to
            fig: The figure object to update with annotations
        """
        annotations = []
        
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
                
                traces.append(go.Scatter(
                    x=[x], y=[y],
                    mode="markers",
                    marker=dict(size=15, color=gaze_info["color"]),
                    hovertext=gaze_info["label"],
                    hoverinfo='text',
                    showlegend=False
                ))
            
            elif event.event_type == "edge_added" and event.data["edge_type"] == "saccade":
                features = event.data["features"]
                prev_x = features["prev_pos"][0] * self.frame_width
                prev_y = features["prev_pos"][1] * self.frame_height
                curr_x = features["curr_pos"][0] * self.frame_width
                curr_y = features["curr_pos"][1] * self.frame_height
                
                # Add line trace
                traces.append(go.Scatter(
                    x=[prev_x, curr_x],
                    y=[prev_y, curr_y],
                    mode="lines",
                    line=dict(
                        width=4,
                        color="rgba(0, 0, 0, 0.8)"
                    ),
                    hovertext=(
                        f"Saccade<br>"
                        f"From node {event.data['source_id']} to {event.data['target_id']}<br>"
                        f"Angle: {features['angle_degrees']:.2f}°<br>"
                        f"Distance: {features['distance']:.2f}"
                    ),
                    hoverinfo='text',
                    showlegend=False
                ))
                
                # Add arrow annotation
                annotations.append(dict(
                    x=curr_x,
                    y=curr_y,
                    ax=prev_x,
                    ay=prev_y,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=4,
                    arrowsize=4,
                    arrowcolor="rgba(0, 0, 0, 0.8)"
                ))
        
        # Update the figure layout with annotations
        if annotations:
            fig.update_layout(annotations=annotations)
    
    def _get_detection_traces(
        self, 
        playback: Playback, 
        frame_number: int,
        traces: List[go.Trace]
    ) -> None:
        """Add object detection bounding box and label traces to the provided traces list.
        
        Args:
            playback: The Playback instance for event access
            frame_number: The current frame number
            traces: List to append detection traces to
        """
        detection_event = playback.get_object_detection(frame_number)
        if not detection_event:
            return
            
        bbox = detection_event.data["bounding_box"]
        fixated_object = detection_event.data["detected_object"]
        current_label = detection_event.data.get("current_detected_label", fixated_object)
        potential_labels = detection_event.data.get("potential_labels", {})
        
        # Format the most likely label for display
        label_text = format_label(fixated_object)
        
        # Create hover text with label information
        hover_text = self._create_detection_hover_text(
            current_label, 
            fixated_object, 
            potential_labels
        )
        
        # Extract bounding box coordinates [x, y, width, height]
        x, y, width, height = bbox
        x0, y0, x1, y1 = x, y, x + width, y + height
        
        # Define colors
        box_color = GAZE_TYPE_INFO[GAZE_TYPE_FIXATION]["color"]
        box_fill = 'rgba(0, 0, 255, 0.1)'  # Blue with 10% opacity
        
        traces.append(go.Scatter(
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
        fixated_object: str,
        potential_labels: dict
    ) -> str:
        """Create hover text for object detection.
        
        Args:
            current_label: The current object label
            fixated_object: The most likely object label
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
            f"Fixated: {format_label(fixated_object)}<br><br>"
            f"Potential labels:<br>{potential_labels_text}"
        )
    
    def _get_yolo_detection_traces(
        self, 
        playback: Playback, 
        frame_number: int,
        traces: List[go.Trace]
    ) -> None:
        """Add YOLO-World detection traces to the provided traces list.
        
        Args:
            playback: The Playback instance for event access
            frame_number: The current frame number
            traces: List to append detection traces to
        """
        yolo_event = playback.get_yolo_detections(frame_number)
        if not yolo_event:
            return
            
        detections = yolo_event.data.get("detections", [])
        if not detections:
            return
            
        for detection in detections:
            # Get detection data from nested structure
            detection_data = detection["detection"]
            fixation_data = detection["fixation"]
            
            bbox = detection_data["bbox"]
            class_name = detection_data["class_name"]
            score = detection_data["score"]
            
            # Get fixation info
            is_fixated = fixation_data["is_fixated"]
            is_top_scoring = fixation_data["is_top_scoring"]
            fixation_score = fixation_data["score"]
            components = fixation_data["components"]

            if not is_fixated:
                continue

            if not is_top_scoring and score < 0.3:
                continue

            # Format the label for display
            label_text = format_label(class_name)
            
            # Create detailed hover text
            hover_text = self._create_yolo_hover_text(class_name, score, is_fixated, is_top_scoring, fixation_score, components)
            
            # Extract bounding box coordinates [x, y, width, height]
            x, y, width, height = bbox
            x0, y0, x1, y1 = x, y, x + width, y + height
            
            # Define colors based on fixation status and top scoring status
            if is_top_scoring:
                # Top scoring object - Blue
                box_color = "rgba(0, 0, 255, 1)"
                box_fill = "rgba(0, 0, 255, 0.2)"
                line_width = 3
            elif is_fixated:
                # Fixated but not top scoring - Gray
                box_color = "rgba(128, 128, 128, 1)"
                box_fill = "rgba(128, 128, 128, 0.15)"
                line_width = 2
            
            traces.append(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                fillcolor=box_fill,
                mode="lines",
                line=dict(width=line_width, color=box_color),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ))
            
            # Calculate label text width based on its length
            conf_text = f"{score:.2f}"
            text_label = f"{label_text} ({conf_text})"
            text_width = len(text_label) * 7  # Approximate width based on character count
            
            # Calculate label box position
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
                text=[text_label],
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

    def _create_yolo_hover_text(
        self,
        class_name: str,
        score: float,
        is_fixated: bool,
        is_top_scoring: bool,
        fixation_score: float,
        components: Dict[str, float]
    ) -> str:
        """Create detailed hover text for YOLO detection.
        
        Args:
            class_name: Object class name
            score: Detection confidence score
            is_fixated: Whether object is fixated
            is_top_scoring: Whether object is the top scoring fixated object
            fixation_score: Overall fixation score
            components: Dictionary of component scores
            
        Returns:
            Formatted hover text
        """
        label_text = format_label(class_name)
        status = "Top Scoring" if is_top_scoring else "Fixated" if is_fixated else "Not Fixated"
        
        hover_text = (
            f"<b>{label_text}</b><br>"
            f"Status: <b>{status}</b><br>"
            f"Confidence: {score:.2f}<br>"
        )
        
        if is_fixated and fixation_score > 0:
            hover_text += f"<br><b>Fixation Score: {fixation_score:.4f}</b><br>"
            
            if components:
                hover_text += "<br><b>Component Scores:</b><br>"
                for name, value in components.items():
                    hover_text += f"{name.capitalize()}: {value:.2f}<br>"
        
        return hover_text
    
    def get_figure(self, frame_number: int) -> go.Figure:
        """Get a complete figure with the video frame and overlays.
        
        Args:
            frame_number: The current frame number
            
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
        events = self.playback.get_events_for_frame(frame_number) if self.playback else []
        self._get_gaze_traces(events, traces, fig)
        self._get_detection_traces(self.playback, frame_number, traces) if self.playback else None
        self._get_yolo_detection_traces(self.playback, frame_number, traces) if self.playback else None
        
        # Add current action overlay
        self._add_action_overlay(frame_number, traces)
        
        fig.add_traces(traces)
        return fig

    def _add_action_overlay(self, frame_number: int, traces: List[go.Trace]) -> None:
        """Add current action annotation overlay to the traces.
        
        Args:
            frame_number: Current frame number
            traces: List to append action overlay traces to
        """
        actions = self._get_current_action(frame_number)
        if not actions:
            return
            
        # Combine all actions into a single text string, separated by commas
        action_texts = []
        hover_texts = []
        
        for action in actions:
            action_text = f"{action['verb']} {action['noun']}"
            action_texts.append(action_text)
            
            hover_text = (
                f"Action: {action_text}<br>"
                f"Verb ID: {action['verb_id']}, Noun ID: {action['noun_id']}<br>"
                f"Frames: {action['start']} - {action['end']}"
            )
            hover_texts.append(hover_text)
        
        combined_action_text = ", ".join(action_texts)
        combined_hover_text = "<br><br>".join(hover_texts)
        
        # Set overlay position at the bottom center of the frame
        overlay_y = self.frame_height - 30  # 30 pixels from bottom
        
        # Add background rectangle for action text
        traces.append(go.Scatter(
            x=[0, self.frame_width, self.frame_width, 0, 0],
            y=[overlay_y - 15, overlay_y - 15, overlay_y + 15, overlay_y + 15, overlay_y - 15],
            fill="toself",
            fillcolor="rgba(0, 0, 0, 0.7)",
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False
        ))
        
        # Add action text
        traces.append(go.Scatter(
            x=[self.frame_width / 2],
            y=[overlay_y],
            mode="text",
            text=[combined_action_text],
            textposition="middle center",
            textfont=dict(
                size=16,
                color="white",
                family="Arial Bold"
            ),
            hoverinfo="text",
            hovertext=combined_hover_text,
            showlegend=False
        ))

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