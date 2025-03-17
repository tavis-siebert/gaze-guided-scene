"""
Graph visualization utilities.

This module provides utilities for visualizing graph structures, including
both static visualization and interactive visualization of the graph construction process.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple, Union
from collections import deque
import numpy as np
import cv2
import networkx as nx
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    def __init__(self, event_data: Dict[str, Any]):
        """
        Initialize a graph event from trace data.
        
        Args:
            event_data: Raw event data from the trace file
        """
        self.event_type = event_data["event_type"]
        self.frame_number = event_data["frame_number"]
        self.timestamp = event_data["timestamp"]
        self.event_id = event_data["event_id"]
        self.data = event_data["data"]
    
    def __repr__(self) -> str:
        return f"GraphEvent({self.event_type}, frame={self.frame_number}, id={self.event_id})"


class GraphPlayback:
    """
    Manages the playback of graph construction events from a trace file.
    """
    
    def __init__(self, trace_file_path: str):
        """
        Initialize the graph playback from a trace file.
        
        Args:
            trace_file_path: Path to the trace file
        """
        self.trace_file_path = Path(trace_file_path)
        self.events = self._load_events()
        self.graph = nx.DiGraph()
        self.current_event_index = 0
        self.frame_to_events = self._index_events_by_frame()
        
        # Extract frame range
        self.min_frame = min(event.frame_number for event in self.events) if self.events else 0
        self.max_frame = max(event.frame_number for event in self.events) if self.events else 0
        
        logger.info(f"Loaded {len(self.events)} events from {trace_file_path}")
        logger.info(f"Frame range: {self.min_frame} to {self.max_frame}")
    
    def _load_events(self) -> List[GraphEvent]:
        """Load events from the trace file."""
        events = []
        with open(self.trace_file_path, 'r') as f:
            for line in f:
                try:
                    event_data = json.loads(line.strip())
                    events.append(GraphEvent(event_data))
                except json.JSONDecodeError:
                    logger.error(f"Error parsing event: {line}")
        return sorted(events, key=lambda e: e.event_id)
    
    def _index_events_by_frame(self) -> Dict[int, List[GraphEvent]]:
        """Create an index of events by frame number."""
        frame_to_events = {}
        for event in self.events:
            if event.frame_number not in frame_to_events:
                frame_to_events[event.frame_number] = []
            frame_to_events[event.frame_number].append(event)
        return frame_to_events
    
    def get_events_for_frame(self, frame_number: int) -> List[GraphEvent]:
        """Get all events for a specific frame."""
        return self.frame_to_events.get(frame_number, [])
    
    def build_graph_until_frame(self, frame_number: int) -> nx.DiGraph:
        """
        Build the graph state up to a specific frame.
        
        Args:
            frame_number: Frame number to build the graph up to
            
        Returns:
            NetworkX DiGraph representing the graph state
        """
        # Reset graph
        self.graph = nx.DiGraph()
        
        # Process all events up to the specified frame
        for event in self.events:
            if event.frame_number > frame_number:
                break
                
            self._apply_event_to_graph(event)
            
        return self.graph
    
    def _apply_event_to_graph(self, event: GraphEvent) -> None:
        """
        Apply an event to update the graph state.
        
        Args:
            event: Event to apply
        """
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


class InteractiveGraphVisualizer:
    """
    Interactive dashboard for visualizing graph construction.
    """
    
    def __init__(self, trace_file_path: str, video_path: Optional[str] = None):
        """
        Initialize the interactive graph visualizer.
        
        Args:
            trace_file_path: Path to the trace file
            video_path: Optional path to the video file
        """
        self.playback = GraphPlayback(trace_file_path)
        self.video_path = video_path
        self.video_capture = None
        
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
                        dcc.Graph(id="video-display", figure=go.Figure())
                    ], className="border p-3 mb-3")
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.H4("Graph Visualization"),
                        dcc.Graph(id="graph-display", figure=go.Figure())
                    ], className="border p-3 mb-3")
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Playback Controls"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("⏮️", id="btn-first-frame", className="me-2"),
                                dbc.Button("⏪", id="btn-prev-frame", className="me-2"),
                                dbc.Button("▶️", id="btn-play-pause", className="me-2"),
                                dbc.Button("⏩", id="btn-next-frame", className="me-2"),
                                dbc.Button("⏭️", id="btn-last-frame")
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
                                    }
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
                        html.Div(id="event-log", className="event-log-container")
                    ], className="border p-3")
                ], width=12)
            ]),
            
            # Hidden components for state management
            dcc.Store(id="playing-state", data={"playing": False}),
            dcc.Interval(id="playback-interval", interval=1000, disabled=True)
        ], fluid=True)
        
        self._register_callbacks(app)
        return app
    
    def _register_callbacks(self, app: dash.Dash) -> None:
        """Register callbacks for the dashboard."""
        
        @app.callback(
            [Output("video-display", "figure"),
             Output("graph-display", "figure"),
             Output("event-log", "children"),
             Output("frame-info", "children")],
            [Input("frame-slider", "value")]
        )
        def update_displays(frame_number):
            """Update all displays based on the current frame."""
            video_fig = self._create_video_figure(frame_number)
            graph_fig = self._create_graph_figure(frame_number)
            event_log = self._create_event_log(frame_number)
            frame_info = self._create_frame_info(frame_number)
            
            return video_fig, graph_fig, event_log, frame_info
        
        @app.callback(
            [Output("frame-slider", "value"),
             Output("playing-state", "data"),
             Output("playback-interval", "disabled"),
             Output("playback-interval", "interval")],
            [Input("btn-first-frame", "n_clicks"),
             Input("btn-prev-frame", "n_clicks"),
             Input("btn-play-pause", "n_clicks"),
             Input("btn-next-frame", "n_clicks"),
             Input("btn-last-frame", "n_clicks"),
             Input("playback-interval", "n_intervals"),
             Input("playback-speed", "value")],
            [State("frame-slider", "value"),
             State("playing-state", "data")]
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
                # If playing, advance to next frame
                if playing_state.get("playing", False):
                    next_frame = min(self.playback.max_frame, current_frame + 1)
                    # Stop playing if we reached the end
                    if next_frame == self.playback.max_frame:
                        return next_frame, {"playing": False}, True, interval_ms
                    return next_frame, playing_state, False, interval_ms
                    
            elif trigger_id == "playback-speed":
                return current_frame, playing_state, not playing_state.get("playing", False), interval_ms
                
            return current_frame, playing_state, not playing_state.get("playing", False), interval_ms
    
    def _create_video_figure(self, frame_number: int) -> go.Figure:
        """
        Create a figure for displaying the video frame.
        
        Args:
            frame_number: Frame number to display
            
        Returns:
            Plotly figure with the video frame and overlays
        """
        fig = go.Figure()
        
        # If we have a video, display the frame
        if self.video_capture is not None:
            # Seek to the frame
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = self.video_capture.read()
            
            if success:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add the frame as an image
                fig.add_trace(go.Image(z=frame_rgb))
                
                # Get events for this frame to add overlays
                events = self.playback.get_events_for_frame(frame_number)
                
                # Add fixation points
                for event in events:
                    if event.event_type == "fixation":
                        pos = event.data["position"]
                        fig.add_trace(go.Scatter(
                            x=[pos[0]],
                            y=[pos[1]],
                            mode="markers",
                            marker=dict(
                                size=15,
                                color="red",
                                symbol="circle-open",
                                line=dict(width=2)
                            ),
                            name="Fixation"
                        ))
                    
                    # Add ROI boxes if available
                    if event.event_type == "frame_processed" and "roi" in event.data:
                        roi = event.data["roi"]
                        fig.add_shape(
                            type="rect",
                            x0=roi[0], y0=roi[1],
                            x1=roi[2], y1=roi[3],
                            line=dict(color="green", width=2),
                            name="ROI"
                        )
            else:
                # If frame read failed, show a blank frame
                fig.add_annotation(
                    text="Video frame not available",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
        else:
            # If no video, show a message
            fig.add_annotation(
                text="No video loaded",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        
        return fig
    
    def _create_graph_figure(self, frame_number: int) -> go.Figure:
        """
        Create a figure for displaying the graph state.
        
        Args:
            frame_number: Frame number to display
            
        Returns:
            Plotly figure with the graph visualization
        """
        # Build graph up to this frame
        G = self.playback.build_graph_until_frame(frame_number)
        
        # If graph is empty, show a message
        if len(G.nodes) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No graph nodes yet",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=400)
            return fig
        
        # Create a spring layout for the graph
        pos = nx.spring_layout(G)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node, attrs in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create node label with attributes
            label = attrs.get('label', 'Unknown')
            frame_added = attrs.get('frame_added', 'Unknown')
            node_text.append(f"ID: {node}<br>Label: {label}<br>Added: Frame {frame_added}")
            
            # Color nodes based on when they were added
            # Newer nodes are brighter
            recency = 1.0 - min(1.0, max(0.0, (frame_number - frame_added) / 100))
            node_color.append(recency)
        
        node_trace = go.Scatter(
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
            )
        )
        
        # Create edge traces
        edge_traces = []
        
        for edge in G.edges(data=True):
            source, target = edge[0], edge[1]
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_type = edge[2].get('edge_type', 'Unknown')
            frame_added = edge[2].get('frame_added', 'Unknown')
            
            # Different colors for different edge types
            color = 'red' if edge_type == 'saccade' else 'blue'
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color=color),
                hoverinfo='text',
                text=f"Type: {edge_type}<br>Added: Frame {frame_added}"
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
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
        """
        Create event log display for the current frame.
        
        Args:
            frame_number: Frame number to display events for
            
        Returns:
            List of HTML components for the event log
        """
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
        """
        Create frame information display.
        
        Args:
            frame_number: Current frame number
            
        Returns:
            HTML component with frame information
        """
        events = self.playback.get_events_for_frame(frame_number)
        event_count = len(events)
        
        # Build graph up to this frame
        G = self.playback.build_graph_until_frame(frame_number)
        node_count = len(G.nodes)
        edge_count = len(G.edges)
        
        return html.Div([
            html.Strong(f"Frame {frame_number}"),
            html.Span(f" | Events: {event_count} | Nodes: {node_count} | Edges: {edge_count}")
        ])
    
    def run_server(self, debug: bool = False, port: int = 8050) -> None:
        """
        Run the visualization server.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
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