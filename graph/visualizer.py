"""
Graph visualization utilities for efficient construction and interaction.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
import cv2
import networkx as nx
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import threading

class GraphVisualizer:
    """Static utilities for basic graph visualization."""
    
    @staticmethod
    def format_node_info(node: Any, prev_obj: str, theta: Any) -> Dict[str, Any]:
        """Format node information for display."""
        return {
            'object': getattr(node, 'object_label', str(node)),
            'from': prev_obj,
            'angle': theta
        }
    
    @staticmethod
    def print_node_info(node_info: Dict[str, Any]) -> None:
        """Print formatted node information."""
        print('-----------------')
        print(f'Object: {node_info["object"]}')
        print(f'Visited from: {node_info["from"]}')
        print(f'Angle from prev: {node_info["angle"]}')
    
    @staticmethod
    def print_levels(start_node: Any) -> None:
        """Print graph structure by levels."""
        visited = set([start_node])
        queue = deque([(start_node, 'none', 'none')])
        
        curr_depth = 0
        while queue:
            level_size = len(queue)
            print(f'Depth: {curr_depth}')
            
            for _ in range(level_size):
                node, prev_obj, theta = queue.popleft()
                node_info = GraphVisualizer.format_node_info(node, prev_obj, theta)
                GraphVisualizer.print_node_info(node_info)
                
                # Add unvisited neighbors
                if hasattr(node, 'neighbors'):
                    for neighbor, angle, _ in node.neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, getattr(node, 'object_label', str(node)), angle))
            
            print('================')
            curr_depth += 1

class GraphEvent:
    """Represents a single graph construction event from the trace file."""
    
    def __init__(self, event_data: Dict[str, Any]):
        self.event_type = event_data["event_type"]
        self.frame_number = event_data["frame_number"]
        self.timestamp = event_data["timestamp"]
        self.event_id = event_data["event_id"]
        self.data = event_data["data"]

class GraphPlayback:
    """Manages the playback of graph construction events."""
    
    def __init__(self, trace_file_path: str):
        self.trace_file_path = Path(trace_file_path)
        self.graph = nx.DiGraph()
        self.last_built_frame = -1
        self._load_events()
    
    def _load_events(self) -> None:
        """Load and index events from the trace file."""
        self.events = []
        self.frame_to_events = defaultdict(list)
        
        with open(self.trace_file_path, 'r') as f:
            for line in f:
                event_data = json.loads(line.strip())
                event = GraphEvent(event_data)
                self.events.append(event)
                self.frame_to_events[event.frame_number].append(event)
        
        frames = list(self.frame_to_events.keys())
        self.min_frame = min(frames) if frames else 0
        self.max_frame = max(frames) if frames else 0
    
    def get_events_for_frame(self, frame_number: int) -> List[GraphEvent]:
        """Get all events for a specific frame."""
        return self.frame_to_events.get(frame_number, [])
    
    def _process_event(self, event: GraphEvent) -> None:
        """Process a single event and update the graph accordingly."""
        if event.event_type == "node_added":
            self.graph.add_node(
                event.data["node_id"],
                label=event.data["label"],
                position=event.data["position"]
            )
        elif event.event_type == "edge_added":
            self.graph.add_edge(
                event.data["source_id"],
                event.data["target_id"],
                edge_type=event.data["edge_type"]
            )
    
    def build_graph_until_frame(self, frame_number: int) -> nx.DiGraph:
        """Build the graph state up to a specific frame."""
        if frame_number < self.last_built_frame:
            self.graph = nx.DiGraph()
            self.last_built_frame = -1
        
        if frame_number > self.last_built_frame:
            for event in self.events:
                if event.frame_number > frame_number:
                    break
                if event.frame_number > self.last_built_frame:
                    self._process_event(event)
            self.last_built_frame = frame_number
        
        return self.graph

class InteractiveGraphVisualizer:
    """Interactive dashboard for visualizing graph construction."""
    
    def __init__(self, trace_file_path: str, video_path: Optional[str] = None):
        self.playback = GraphPlayback(trace_file_path)
        self.video_path = video_path
        self.video_capture = None
        self.play_interval_ms = 1000 // 24  # 24 FPS playback speed
        self.video_lock = threading.Lock()
        self.frame_cache = {}
        self.max_cache_size = 100  # Maximum number of frames to keep in memory
        
        self._setup_video_capture()
        self.app = self._create_dashboard()
    
    def _setup_video_capture(self) -> None:
        """Set up video capture if a valid path is provided."""
        if self.video_path and Path(self.video_path).exists():
            self.video_capture = cv2.VideoCapture(self.video_path)
            # Set OpenCV to use a single thread for video decoding
            cv2.setNumThreads(1)
    
    def _get_video_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Thread-safe method to get a video frame."""
        if self.video_capture is None:
            return None
            
        # Check cache first
        if frame_number in self.frame_cache:
            return self.frame_cache[frame_number]
            
        with self.video_lock:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = self.video_capture.read()
            
            if not success:
                return None
                
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update cache
            if len(self.frame_cache) >= self.max_cache_size:
                # Remove oldest frame
                oldest_frame = min(self.frame_cache.keys())
                del self.frame_cache[oldest_frame]
            
            self.frame_cache[frame_number] = frame_rgb
            return frame_rgb
    
    def _create_dashboard(self) -> dash.Dash:
        """Create the Dash application."""
        app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            update_title=None,  # Disable the "Updating..." message
            suppress_callback_exceptions=True  # Allow for dynamic callbacks
        )
        
        # Enable Dash's callback queue to prevent concurrent callback execution
        app.enable_dev_tools(
            dev_tools_hot_reload=False,
            dev_tools_ui=False,
            dev_tools_serve_dev_bundles=False
        )
        
        # Build layout
        app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks(app)
        
        return app
    
    def _create_layout(self) -> dbc.Container:
        """Create the application layout."""
        return dbc.Container([
            # Store for play state
            dcc.Store(id='play-state', data={'is_playing': False, 'last_update': 0}),
            
            # Interval for auto-playback
            dcc.Interval(
                id='auto-advance',
                interval=self.play_interval_ms,
                disabled=True,
                max_intervals=-1  # Run indefinitely when enabled
            ),
            
            # Visualization panels
            dbc.Row([
                # Left column - Video display
                dbc.Col([
                    html.H4("Video Feed", className="text-center"),
                    dcc.Graph(id="video-display"),
                ], width=6),
                
                # Right column - Graph display
                dbc.Col([
                    html.H4("Graph Visualization", className="text-center"),
                    dcc.Graph(id="graph-display"),
                ], width=6),
            ]),
            
            # Navigation controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                # Playback controls
                                dbc.Col([
                                    html.Div([
                                        dbc.Button("← Prev", id="prev-frame", n_clicks=0, color="primary", className="me-2"),
                                        dbc.Button("Play", id="play-pause", n_clicks=0, color="success", className="me-2"),
                                        dbc.Button("Next →", id="next-frame", n_clicks=0, color="primary"),
                                    ], className="d-flex justify-content-center"),
                                ], width=4),
                                
                                # Current frame display
                                dbc.Col([
                                    html.Div([
                                        html.Span("Current Frame: ", className="me-2"),
                                        html.Strong(id="current-frame-display"),
                                    ], className="d-flex justify-content-center align-items-center h-100"),
                                ], width=4),
                                
                                # Frame slider
                                dbc.Col([
                                    dcc.Slider(
                                        id="frame-slider",
                                        min=self.playback.min_frame,
                                        max=self.playback.max_frame,
                                        value=self.playback.min_frame,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ], width=4),
                            ]),
                        ])
                    ], className="mb-3"),
                ], width=12),
            ]),
        ], fluid=True)
    
    def _register_callbacks(self, app: dash.Dash) -> None:
        """Register all necessary callbacks."""
        # Callback for play/pause toggling
        @app.callback(
            [Output("play-state", "data"),
             Output("auto-advance", "disabled"),
             Output("play-pause", "children")],
            [Input("play-pause", "n_clicks")],
            [State("play-state", "data")]
        )
        def toggle_play_state(play_clicks, current_state):
            if not play_clicks:
                # Initial state
                return current_state, True, "Play"
                
            # Toggle the play state
            is_playing = not current_state.get('is_playing', False)
            new_state = {
                'is_playing': is_playing, 
                'last_update': current_state.get('last_update', 0) + 1
            }
            
            return new_state, not is_playing, "Pause" if is_playing else "Play"
        
        # Callback for frame navigation and display updates
        @app.callback(
            [Output("video-display", "figure"),
             Output("graph-display", "figure"),
             Output("current-frame-display", "children"),
             Output("frame-slider", "value")],
            [Input("frame-slider", "value"),
             Input("prev-frame", "n_clicks"),
             Input("next-frame", "n_clicks"),
             Input("auto-advance", "n_intervals")],
            [State("play-state", "data"),
             State("frame-slider", "value")]
        )
        def update_displays(slider_frame, prev_clicks, next_clicks, 
                          n_intervals, play_state, current_frame):
            frame_number = self._determine_frame_number(
                dash.callback_context,
                slider_frame,
                current_frame,
                play_state
            )
            
            return (
                self._create_video_figure(frame_number),
                self._create_graph_figure(frame_number),
                str(frame_number),
                frame_number
            )
    
    def _determine_frame_number(self, ctx, slider_frame, current_frame, play_state) -> int:
        """Determine the current frame number based on user interaction."""
        if not ctx.triggered:
            return slider_frame
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        is_playing = play_state.get('is_playing', False)
        
        if trigger_id == "prev-frame":
            return max(self.playback.min_frame, current_frame - 1)
        elif trigger_id == "next-frame":
            return min(self.playback.max_frame, current_frame + 1)
        elif trigger_id == "auto-advance" and is_playing:
            frame_number = current_frame + 1
            if frame_number > self.playback.max_frame:
                frame_number = self.playback.min_frame
            return frame_number
        else:
            return slider_frame
    
    def _create_empty_figure(self, height: int = 400) -> go.Figure:
        """Create an empty figure with standard layout."""
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0),
            height=height
        )
        return fig

    def _add_gaze_overlay(self, fig: go.Figure, frame_number: int, frame_width: int, frame_height: int) -> None:
        """Add gaze position overlay to the figure."""
        events = self.playback.get_events_for_frame(frame_number)
        for event in events:
            if event.event_type == "frame_processed":
                pos = event.data["gaze_position"]
                if pos[0] == 0.0 and pos[1] == 0.0:
                    continue
                    
                x, y = pos[0] * frame_width, pos[1] * frame_height
                gaze_type = event.data["gaze_type"]
                color = "blue" if gaze_type == 1 else "red" if gaze_type == 2 else "gray"
                size = 15 if gaze_type in [1, 2] else 10
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers",
                    marker=dict(size=size, color=color),
                    showlegend=False
                ))

    def _create_video_figure(self, frame_number: int) -> go.Figure:
        """Create video frame figure with gaze overlay."""
        fig = self._create_empty_figure()
        
        frame = self._get_video_frame(frame_number)
        if frame is None:
            return fig
            
        frame_height, frame_width = frame.shape[:2]
        
        fig.add_trace(go.Image(z=frame))
        self._add_gaze_overlay(fig, frame_number, frame_width, frame_height)
        
        return fig
    
    def _get_current_node_id(self, frame_number: int) -> Optional[Any]:
        """Extract current node ID from frame events."""
        events = self.playback.get_events_for_frame(frame_number)
        for event in events:
            if event.event_type == "frame_processed" and 'node_id' in event.data:
                return event.data["node_id"]
        return None
    
    def _add_edges_to_figure(self, fig: go.Figure, G: nx.DiGraph, pos: Dict) -> None:
        """Add edges to the graph figure."""
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                showlegend=False
            ))
    
    def _add_nodes_to_figure(self, fig: go.Figure, G: nx.DiGraph, pos: Dict, current_node_id: Any) -> None:
        """Add nodes to the graph figure."""
        node_x, node_y, node_text, node_colors = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Node {node}: {G.nodes[node]['label']}")
            node_colors.append('#ff0000' if node == current_node_id else '#1f77b4')
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=20, color=node_colors),
            text=node_text,
            hoverinfo='text',
            showlegend=False
        ))
    
    def _create_graph_figure(self, frame_number: int) -> go.Figure:
        """Create graph visualization figure."""
        G = self.playback.build_graph_until_frame(frame_number)
        fig = go.Figure()
        
        if len(G.nodes) > 0:
            # Use Kamada-Kawai layout for better node placement
            pos = nx.kamada_kawai_layout(G)
            
            # Get current node ID
            current_node_id = self._get_current_node_id(frame_number)
            
            # Add edges and nodes to figure
            self._add_edges_to_figure(fig, G, pos)
            self._add_nodes_to_figure(fig, G, pos, current_node_id)
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=400
        )
        
        return fig
    
    def run_server(self, debug: bool = False, port: int = 8050) -> None:
        """Run the visualization server."""
        self.app.run(debug=debug, port=port)

def visualize_graph_construction(
    trace_file: str,
    video_path: Optional[str] = None,
    port: int = 8050,
    debug: bool = False
) -> None:
    """Launch the interactive visualization dashboard."""
    visualizer = InteractiveGraphVisualizer(trace_file, video_path)
    visualizer.run_server(debug=debug, port=port) 