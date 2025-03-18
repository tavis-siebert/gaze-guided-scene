"""
Graph visualization utilities.

This module provides utilities for visualizing graph structures, including
both static visualization and interactive visualization of the graph construction process.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple, Union, DefaultDict
from collections import deque, defaultdict
import numpy as np
import cv2
import networkx as nx
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache

from graph.node import Node
from graph.utils import AngleUtils
from graph.graph_tracer import GraphTracer
from config.config_utils import DotDict
from logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

class GraphVisualizer:
    """Utilities for visualizing graph structures."""
    
    @staticmethod
    def format_node_info(node: Node, prev_obj: str, theta: Any, use_degrees: bool) -> Dict[str, Any]:
        """Format node information for display."""
        angle = theta
        if isinstance(theta, float) and use_degrees:
            angle = AngleUtils.to_degrees(theta)
            
        return {
            'object': node.object_label,
            'visits': node.visits,
            'from': prev_obj,
            'angle': angle
        }
    
    @staticmethod
    def print_node_info(node_info: Dict[str, Any]) -> None:
        """Print formatted node information."""
        logger.info('-----------------')
        logger.info(f'Object: {node_info["object"]}')
        logger.info(f'Visited at: {node_info["visits"]}')
        logger.info(f'Visited from: {node_info["from"]}')
        logger.info(f'Angle from prev: {node_info["angle"]}')
    
    @staticmethod
    def print_levels(start_node: Node, use_degrees: bool = True) -> None:
        """
        Print graph structure by levels, showing node relationships.
        
        Args:
            start_node: Root node to start traversal from
            use_degrees: Whether to display angles in degrees (True) or radians (False)
        """
        visited = set([start_node])
        queue = deque([(start_node, 'none', 'none')])
        
        curr_depth = 0
        while queue:
            level_size = len(queue)
            logger.info(f'Depth: {curr_depth}')
            
            for _ in range(level_size):
                node, prev_obj, theta = queue.popleft()
                node_info = GraphVisualizer.format_node_info(node, prev_obj, theta, use_degrees)
                GraphVisualizer.print_node_info(node_info)
                
                GraphVisualizer._queue_unvisited_neighbors(node, visited, queue)
                    
            logger.info('================')
            curr_depth += 1
    
    @staticmethod
    def _queue_unvisited_neighbors(node: Node, visited: Set[Node], queue: deque) -> None:
        """Add unvisited neighbors to the visualization queue."""
        for neighbor, angle, _ in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, node.object_label, angle))


class GraphEvent:
    """Represents a single graph construction event from the trace file."""
    
    __slots__ = ['event_type', 'frame_number', 'timestamp', 'event_id', 'data']
    
    def __init__(self, event_data: Dict[str, Any]):
        """Initialize a graph event from trace data."""
        self.event_type = event_data["event_type"]
        self.frame_number = event_data["frame_number"]
        self.timestamp = event_data["timestamp"]
        self.event_id = event_data["event_id"]
        self.data = event_data["data"]
    
    def __repr__(self) -> str:
        return f"GraphEvent({self.event_type}, frame={self.frame_number}, id={self.event_id})"


class GraphPlayback:
    """Manages the playback of graph construction events from a trace file."""
    
    def __init__(self, trace_file_path: str):
        """Initialize the graph playback from a trace file."""
        self.trace_file_path = Path(trace_file_path)
        self._load_events()
        self.graph = nx.DiGraph()
        self.current_frame = 0
        self.last_built_frame = -1
        self._cached_layouts = {}
        
        logger.info(f"Loaded {len(self.events)} events from {trace_file_path}")
        logger.info(f"Frame range: {self.min_frame} to {self.max_frame}")
    
    def _load_events(self) -> None:
        """Load and index events from the trace file."""
        self.events = []
        self.frame_to_events = defaultdict(list)
        self.frame_to_event_indices = defaultdict(list)
        
        with open(self.trace_file_path, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    event_data = json.loads(line.strip())
                    event = GraphEvent(event_data)
                    self.events.append(event)
                    self.frame_to_events[event.frame_number].append(event)
                    self.frame_to_event_indices[event.frame_number].append(idx)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing event: {line}")
        
        # Extract frame range
        self.min_frame = min(self.frame_to_events.keys()) if self.frame_to_events else 0
        self.max_frame = max(self.frame_to_events.keys()) if self.frame_to_events else 0
        
        # Pre-sort events by frame number for faster access
        self.events.sort(key=lambda e: (e.frame_number, e.event_id))
    
    def get_events_for_frame(self, frame_number: int) -> List[GraphEvent]:
        """Get all events for a specific frame."""
        return self.frame_to_events.get(frame_number, [])
    
    def build_graph_until_frame(self, frame_number: int) -> nx.DiGraph:
        """Build the graph state up to a specific frame."""
        # If we're moving backward, reset the graph
        if frame_number < self.last_built_frame:
            self.graph = nx.DiGraph()
            self.last_built_frame = -1
            self._cached_layouts.clear()
        
        # Process only new events since last build
        if frame_number > self.last_built_frame:
            start_idx = 0
            if self.last_built_frame >= 0:
                # Find the starting index for the new frame's events
                for f in range(self.last_built_frame + 1, frame_number + 1):
                    if f in self.frame_to_event_indices:
                        start_idx = min(self.frame_to_event_indices[f])
                        break
            
            # Process events from start_idx
            for idx in range(start_idx, len(self.events)):
                event = self.events[idx]
                if event.frame_number > frame_number:
                    break
                self._apply_event_to_graph(event)
            
            self.last_built_frame = frame_number
            
            # Cache layout if graph size is reasonable
            if len(self.graph) <= 100:  # Only cache layouts for smaller graphs
                self._cached_layouts[frame_number] = self._compute_layout()
        
        return self.graph
    
    def _compute_layout(self) -> Dict[int, Tuple[float, float]]:
        """Compute graph layout with caching."""
        if len(self.graph) == 0:
            return {}
            
        # Try to reuse previous layout as initial positions
        prev_layout = None
        if self.last_built_frame - 1 in self._cached_layouts:
            prev_layout = self._cached_layouts[self.last_built_frame - 1]
            # Verify that prev_layout contains positions for current nodes
            if prev_layout and not all(node in prev_layout for node in self.graph.nodes()):
                prev_layout = None
        
        try:
            return nx.spring_layout(
                self.graph,
                pos=prev_layout if prev_layout else None,
                k=1/np.sqrt(len(self.graph))
            )
        except (ValueError, ZeroDivisionError):
            # Fallback to basic circular layout if spring layout fails
            return nx.circular_layout(self.graph)
    
    def _apply_event_to_graph(self, event: GraphEvent) -> None:
        """Apply an event to update the graph state."""
        if event.event_type == "node_added":
            node_id = event.data["node_id"]
            self.graph.add_node(
                node_id,
                label=event.data["label"],
                position=event.data["position"],
                frame_added=event.frame_number
            )
            
        elif event.event_type == "edge_added":
            source_id = event.data["source_id"]
            target_id = event.data["target_id"]
            edge_type = event.data["edge_type"]
            
            # Add edge with properties
            self.graph.add_edge(
                source_id, 
                target_id,
                edge_type=edge_type,
                frame_added=event.frame_number,
                **event.data.get("properties", {})
            )
    
    def get_graph_layout(self, frame_number: int) -> Dict[int, Tuple[float, float]]:
        """Get cached layout for the given frame or compute a new one."""
        if frame_number in self._cached_layouts:
            return self._cached_layouts[frame_number]
        return self._compute_layout()


class InteractiveGraphVisualizer:
    """Interactive dashboard for visualizing graph construction."""
    
    def __init__(self, trace_file_path: str, video_path: Optional[str] = None):
        """Initialize the interactive graph visualizer."""
        self.playback = GraphPlayback(trace_file_path)
        self.video_path = video_path
        self.video_capture = None
        self._frame_cache = {}
        self._max_frame_cache = 100  # Maximum number of frames to cache
        
        if video_path and os.path.exists(video_path):
            self.video_capture = cv2.VideoCapture(video_path)
            logger.info(f"Loaded video from {video_path}")
        
        self.app = self._create_dashboard()
    
    def _create_dashboard(self) -> dash.Dash:
        """Create the Dash application for the dashboard."""
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Create initial empty figures
        empty_video_fig = self._create_empty_figure("No video loaded")
        empty_graph_fig = self._create_empty_figure("No graph nodes yet")
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Graph Construction Visualizer", className="text-center my-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Video Frame"),
                        dcc.Graph(
                            id="video-display",
                            figure=empty_video_fig,
                            config={'displayModeBar': False}
                        )
                    ], className="border p-3 mb-3")
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.H4("Graph Visualization"),
                        dcc.Graph(
                            id="graph-display",
                            figure=empty_graph_fig,
                            config={'displayModeBar': True}
                        )
                    ], className="border p-3 mb-3")
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Playback Controls"),
                        dbc.Row([
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button("⏮️", id="btn-first-frame", n_clicks=0),
                                    dbc.Button("⏪", id="btn-prev-frame", n_clicks=0),
                                    dbc.Button("▶️", id="btn-play-pause", n_clicks=0),
                                    dbc.Button("⏩", id="btn-next-frame", n_clicks=0),
                                    dbc.Button("⏭️", id="btn-last-frame", n_clicks=0)
                                ])
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Playback Speed:"),
                                dcc.Slider(
                                    id="playback-speed",
                                    min=0.25,
                                    max=4,
                                    step=0.25,
                                    value=1,
                                    marks={i: f"{i}x" for i in [0.25, 1, 2, 3, 4]}
                                )
                            ], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Slider(
                                    id="frame-slider",
                                    min=self.playback.min_frame,
                                    max=self.playback.max_frame,
                                    step=1,
                                    value=self.playback.min_frame,
                                    marks={
                                        i: str(i) for i in range(
                                            self.playback.min_frame,
                                            self.playback.max_frame + 1,
                                            max(1, (self.playback.max_frame - self.playback.min_frame) // 10)
                                        )
                                    },
                                    updatemode='mouseup'  # Only trigger updates when mouse is released
                                )
                            ], width=12)
                        ], className="mt-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="frame-info", className="mt-2")
                            ], width=12)
                        ])
                    ], className="border p-3 mb-3")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Event Log"),
                        dcc.Loading(
                            id="event-log-loading",
                            children=[html.Div(id="event-log", className="event-log-container")],
                            type="circle"
                        )
                    ], className="border p-3")
                ], width=12)
            ]),
            
            # Hidden components for state management
            dcc.Store(id="playing-state", data={"playing": False}),
            dcc.Store(id="last-frame", data={"frame": self.playback.min_frame}),
            dcc.Interval(id="playback-interval", interval=1000, disabled=True)
        ], fluid=True)
        
        self._register_callbacks(app)
        return app
    
    @staticmethod
    def _create_empty_figure(text: str) -> go.Figure:
        """Create an empty figure with centered text."""
        fig = go.Figure()
        fig.add_annotation(
            text=text,
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        return fig
    
    def _register_callbacks(self, app: dash.Dash) -> None:
        """Register callbacks for the dashboard."""
        
        @app.callback(
            [
                Output("video-display", "figure", allow_duplicate=True),
                Output("graph-display", "figure", allow_duplicate=True),
                Output("event-log", "children", allow_duplicate=True),
                Output("frame-info", "children", allow_duplicate=True),
                Output("last-frame", "data", allow_duplicate=True)
            ],
            [Input("frame-slider", "value")],
            [State("last-frame", "data")],
            prevent_initial_call=True
        )
        def update_displays(frame_number, last_frame_data):
            """Update all displays based on the current frame."""
            if frame_number == last_frame_data.get("frame"):
                raise dash.exceptions.PreventUpdate
                
            video_fig = self._create_video_figure(frame_number)
            graph_fig = self._create_graph_figure(frame_number)
            event_log = self._create_event_log(frame_number)
            frame_info = self._create_frame_info(frame_number)
            
            return video_fig, graph_fig, event_log, frame_info, {"frame": frame_number}
        
        @app.callback(
            [
                Output("frame-slider", "value", allow_duplicate=True),
                Output("playing-state", "data", allow_duplicate=True),
                Output("playback-interval", "disabled", allow_duplicate=True),
                Output("playback-interval", "interval", allow_duplicate=True)
            ],
            [
                Input("btn-first-frame", "n_clicks"),
                Input("btn-prev-frame", "n_clicks"),
                Input("btn-play-pause", "n_clicks"),
                Input("btn-next-frame", "n_clicks"),
                Input("btn-last-frame", "n_clicks"),
                Input("playback-interval", "n_intervals"),
                Input("playback-speed", "value")
            ],
            [
                State("frame-slider", "value"),
                State("playing-state", "data")
            ],
            prevent_initial_call=True
        )
        def handle_controls(first_clicks, prev_clicks, play_clicks, next_clicks, 
                           last_clicks, interval, speed, current_frame, playing_state):
            """Handle playback control buttons and interval updates."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return current_frame, playing_state, True, 1000
                
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            # Calculate the interval based on speed
            interval_ms = int(1000 / speed) if speed > 0 else 1000
            
            if trigger_id == "btn-first-frame":
                return self.playback.min_frame, {"playing": False}, True, interval_ms
                
            elif trigger_id == "btn-prev-frame":
                prev_frame = max(self.playback.min_frame, current_frame - 1)
                return prev_frame, {"playing": False}, True, interval_ms
                
            elif trigger_id == "btn-play-pause":
                playing = not playing_state.get("playing", False)
                return current_frame, {"playing": playing}, not playing, interval_ms
                
            elif trigger_id == "btn-next-frame":
                next_frame = min(self.playback.max_frame, current_frame + 1)
                return next_frame, {"playing": False}, True, interval_ms
                
            elif trigger_id == "btn-last-frame":
                return self.playback.max_frame, {"playing": False}, True, interval_ms
                
            elif trigger_id == "playback-interval":
                if playing_state.get("playing", False):
                    next_frame = min(self.playback.max_frame, current_frame + 1)
                    if next_frame == self.playback.max_frame:
                        return next_frame, {"playing": False}, True, interval_ms
                    return next_frame, playing_state, False, interval_ms
                    
            elif trigger_id == "playback-speed":
                return current_frame, playing_state, not playing_state.get("playing", False), interval_ms
                
            return current_frame, playing_state, not playing_state.get("playing", False), interval_ms
    
    def _get_cached_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a cached video frame if available."""
        return self._frame_cache.get(frame_number)
    
    def _cache_frame(self, frame_number: int, frame: np.ndarray) -> None:
        """Cache a video frame, maintaining maximum cache size."""
        if len(self._frame_cache) >= self._max_frame_cache:
            # Remove oldest frame
            oldest_frame = min(self._frame_cache.keys())
            del self._frame_cache[oldest_frame]
        self._frame_cache[frame_number] = frame
    
    def _create_video_figure(self, frame_number: int) -> go.Figure:
        """Create a figure for displaying the video frame."""
        fig = go.Figure()
        
        if self.video_capture is not None:
            # Try to get frame from cache first
            frame = self._get_cached_frame(frame_number)
            
            if frame is None:
                # If not in cache, read from video
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = self.video_capture.read()
                
                if success:
                    # Cache the frame
                    self._cache_frame(frame_number, frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Get frame dimensions
                    frame_height, frame_width = frame_rgb.shape[:2]
                    
                    # Add the frame as an image
                    fig.add_trace(go.Image(z=frame_rgb))
                    
                    # Get events and add overlays efficiently
                    events = self.playback.get_events_for_frame(frame_number)
                    self._add_frame_overlays(fig, events, frame_width, frame_height)
                else:
                    return self._create_empty_figure("Video frame not available")
            else:
                # Use cached frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_height, frame_width = frame_rgb.shape[:2]
                fig.add_trace(go.Image(z=frame_rgb))
                
                # Add overlays
                events = self.playback.get_events_for_frame(frame_number)
                self._add_frame_overlays(fig, events, frame_width, frame_height)
        else:
            return self._create_empty_figure("No video loaded")
        
        # Update layout
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        
        return fig
    
    def _add_frame_overlays(self, fig: go.Figure, events: List[GraphEvent], 
                           frame_width: int, frame_height: int) -> None:
        """Add overlays to the video frame figure efficiently."""
        # Collect all points for batch processing
        gaze_points = []
        fixation_points = []
        roi_boxes = []
        
        for event in events:
            if event.event_type == "frame_processed" and "gaze_position" in event.data:
                pos = event.data["gaze_position"]
                if pos[0] != 0.0 or pos[1] != 0.0:
                    gaze_points.append((pos[0] * frame_width, pos[1] * frame_height))
            
            elif event.event_type == "fixation":
                pos = event.data["position"]
                fixation_points.append((pos[0] * frame_width, pos[1] * frame_height))
            
            elif event.event_type == "frame_processed" and "roi" in event.data:
                roi = event.data["roi"]
                roi_boxes.append(roi)
        
        # Add all points in batches
        if gaze_points:
            x, y = zip(*gaze_points)
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="markers",
                marker=dict(size=12, color="green", symbol="circle"),
                name="Gaze Position"
            ))
        
        if fixation_points:
            x, y = zip(*fixation_points)
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="markers",
                marker=dict(size=15, color="red", symbol="circle-open", line=dict(width=2)),
                name="Fixation"
            ))
        
        # Add ROI boxes
        for roi in roi_boxes:
            fig.add_shape(
                type="rect",
                x0=roi[0][0], y0=roi[0][1],
                x1=roi[1][0], y1=roi[1][1],
                line=dict(color="green", width=2),
                name="ROI"
            )
    
    def _create_graph_figure(self, frame_number: int) -> go.Figure:
        """Create a figure for displaying the graph state."""
        # Build graph up to this frame
        G = self.playback.build_graph_until_frame(frame_number)
        
        if len(G.nodes) == 0:
            return self._create_empty_figure("No graph nodes yet")
        
        # Get cached or computed layout
        pos = self.playback.get_graph_layout(frame_number)
        
        # Prepare node data
        node_x, node_y, node_text, node_color = [], [], [], []
        edge_x, edge_y, edge_colors = [], [], []
        edge_texts = []
        
        # Process nodes
        for node, attrs in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            label = attrs.get('label', 'Unknown')
            frame_added = attrs.get('frame_added', 'Unknown')
            node_text.append(f"ID: {node}<br>Label: {label}<br>Added: Frame {frame_added}")
            
            recency = 1.0 - min(1.0, max(0.0, (frame_number - frame_added) / 100))
            node_color.append(recency)
        
        # Process edges in a single pass
        for edge in G.edges(data=True):
            source, target = edge[0], edge[1]
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_type = edge[2].get('edge_type', 'Unknown')
            frame_added = edge[2].get('frame_added', 'Unknown')
            
            color = 'red' if edge_type == 'saccade' else 'blue'
            edge_colors.extend([color] * 3)  # One color per point including None
            edge_texts.append(f"Type: {edge_type}<br>Added: Frame {frame_added}")
        
        # Create figure with single traces for efficiency
        fig = go.Figure()
        
        # Add edges as a single trace
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color=edge_colors[0], width=1),
                hoverinfo='text',
                text=edge_texts,
                showlegend=False
            ))
        
        # Add nodes as a single trace
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=node_color,
                size=15,
                colorbar=dict(
                    title='Node Recency',
                    thickness=15,
                    tickvals=[0, 1],
                    ticktext=['Older', 'Newer']
                )
            ),
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=400
        )
        
        return fig
    
    def _create_event_log(self, frame_number: int) -> List[html.Div]:
        """Create event log display for the current frame."""
        events = self.playback.get_events_for_frame(frame_number)
        
        if not events:
            return [html.Div("No events for this frame", className="text-muted")]
        
        event_items = []
        for event in events:
            # Format event data based on type
            if event.event_type == "node_added":
                content = f"Added node {event.data['node_id']} ({event.data['label']})"
                color = "success"
            elif event.event_type == "edge_added":
                content = f"Added edge from {event.data['source_id']} to {event.data['target_id']} ({event.data['edge_type']})"
                color = "primary"
            elif event.event_type == "fixation":
                content = f"Fixation at {event.data['position']} for {event.data['duration']}ms"
                color = "warning"
            elif event.event_type == "saccade":
                content = f"Saccade from {event.data['start_position']} to {event.data['end_position']}"
                color = "info"
            else:
                content = f"{event.event_type}: {json.dumps(event.data)}"
                color = "secondary"
            
            event_items.append(
                html.Div(
                    dbc.Alert(content, color=color, className="py-1 mb-2")
                )
            )
        
        return event_items
    
    def _create_frame_info(self, frame_number: int) -> html.Div:
        """Create frame information display."""
        events = self.playback.get_events_for_frame(frame_number)
        event_count = len(events)
        
        G = self.playback.build_graph_until_frame(frame_number)
        node_count = len(G.nodes)
        edge_count = len(G.edges)
        
        return html.Div([
            html.Strong(f"Frame {frame_number}"),
            html.Span(f" | Events: {event_count} | Nodes: {node_count} | Edges: {edge_count}")
        ])
    
    def run_server(self, debug: bool = False, port: int = 8050) -> None:
        """Run the visualization server."""
        self.app.run(debug=debug, port=port)


def visualize_graph_construction(
    video_name: str,
    config: DotDict,
    video_path: Optional[str] = None,
    port: int = 8050,
    debug: bool = False
) -> None:
    """
    Launch the interactive visualization dashboard.
    
    Args:
        video_name: Name of the video to process
        config: Configuration dictionary
        video_path: Optional path to the video file
        port: Port to run the server on
        debug: Whether to run in debug mode
    """
    # Check if trace file exists - traces must be generated with build command
    trace_dir = config.directories.repo.traces
    trace_file = Path(trace_dir) / f"{video_name}_trace.jsonl"
    
    if not trace_file.exists():
        logger.error(f"No trace file found for {video_name}.")
        logger.error(f"Trace file should be at: {trace_file}")
        logger.error(f"To generate a trace file, run:")
        logger.error(f"    python main.py build --videos {video_name} --enable-tracing")
        logger.error("Exiting visualization process.")
        return
        
    logger.info(f"Using trace file: {trace_file}")
    
    # Find video if no path was provided
    if video_path is None:
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'egtea'):
            possible_video_path = Path(config.dataset.egtea.raw_videos) / f"{video_name}.mp4"
            if possible_video_path.exists():
                video_path = str(possible_video_path)
                logger.info(f"Found video file at {video_path}")
            else:
                logger.warning(f"Could not find video file at expected location: {possible_video_path}")
                logger.warning("Visualization will proceed without video display.")
        else:
            logger.warning("No video path provided and no default location configured.")
            logger.warning("Visualization will proceed without video display.")
    elif not Path(video_path).exists():
        logger.warning(f"Specified video path does not exist: {video_path}")
        logger.warning("Visualization will proceed without video display.")
        video_path = None
    else:
        logger.info(f"Using provided video path: {video_path}")
    
    # Launch the visualization dashboard
    logger.info(f"Launching interactive visualization dashboard on port {port}...")
    visualizer = InteractiveGraphVisualizer(str(trace_file), video_path)
    visualizer.run_server(debug=debug, port=port) 