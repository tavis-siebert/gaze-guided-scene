from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import json
import numpy as np
import cv2
import networkx as nx
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import threading
from egtea_gaze.constants import (
    GAZE_TYPE_UNTRACKED,
    GAZE_TYPE_FIXATION,
    GAZE_TYPE_SACCADE,
    GAZE_TYPE_UNKNOWN,
    GAZE_TYPE_TRUNCATED
)

class GraphVisualizer:
    @staticmethod
    def format_node_info(node: Any, prev_obj: str, theta: Any) -> Dict[str, Any]:
        return {
            'object': getattr(node, 'object_label', str(node)),
            'from': prev_obj,
            'angle': theta
        }
    
    @staticmethod
    def print_node_info(node_info: Dict[str, Any]) -> None:
        print('-----------------')
        print(f'Object: {node_info["object"]}')
        print(f'Visited from: {node_info["from"]}')
        print(f'Angle from prev: {node_info["angle"]}')
    
    @staticmethod
    def print_levels(start_node: Any, use_degrees: bool = True, edges: List = None, graph: 'Graph' = None) -> None:
        """
        Print the graph structure by levels (BFS traversal).
        
        Args:
            start_node: The node to start BFS from
            use_degrees: Whether to display angles in degrees
            edges: List of edges in the graph
            graph: Optional Graph instance (preferred over edges list)
        """
        # Create adjacency map for edge lookup
        adjacency = defaultdict(list)
        visited_nodes = set([start_node])
        nodes_by_id = {start_node.id: start_node}
        
        # Initialize with the given edges
        if edges:
            for edge in edges:
                target_node = None
                
                # If graph is available, use it to look up nodes
                if graph:
                    target_node = graph.get_node_by_id(edge.target_id)
                else:
                    # This is a fallback when graph instance isn't provided
                    # Note: this might not find all target nodes if they haven't been visited yet
                    if edge.target_id in nodes_by_id:
                        target_node = nodes_by_id[edge.target_id]
                
                if target_node:
                    adjacency[edge.source_id].append((target_node, edge.angle, edge.distance))
                    nodes_by_id[target_node.id] = target_node
                
        queue = deque([(start_node, 'none', 'none')])
        
        curr_depth = 0
        while queue:
            level_size = len(queue)
            print(f'Depth: {curr_depth}')
            
            for _ in range(level_size):
                node, prev_obj, theta = queue.popleft()
                
                # Convert angle to degrees if requested
                if theta != 'none' and use_degrees:
                    theta = f"{(theta * 180.0 / np.pi):.2f}°"
                
                node_info = GraphVisualizer.format_node_info(node, prev_obj, theta)
                GraphVisualizer.print_node_info(node_info)
                
                # Use adjacency map to get neighbors
                for neighbor, angle, distance in adjacency.get(node.id, []):
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        nodes_by_id[neighbor.id] = neighbor
                        queue.append((neighbor, getattr(node, 'object_label', str(node)), angle))
            
            print('================')
            curr_depth += 1

class GraphEvent:
    def __init__(self, event_data: Dict[str, Any]):
        self.event_type = event_data["event_type"]
        self.frame_number = event_data["frame_number"]
        self.timestamp = event_data["timestamp"]
        self.event_id = event_data["event_id"]
        self.data = event_data["data"]

class GraphPlayback:
    def __init__(self, trace_file_path: str):
        self.trace_file_path = Path(trace_file_path)
        self.graph = nx.DiGraph()
        self.last_built_frame = -1
        self.last_added_node = None
        self.last_added_edge = None
        self.object_detections = {}  # Frame number -> detection event
        self._load_events()
    
    def _load_events(self) -> None:
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
        return self.frame_to_events.get(frame_number, [])
    
    def get_object_detection(self, frame_number: int) -> Optional[GraphEvent]:
        """Get the object detection event for a specific frame, if any."""
        return self.object_detections.get(frame_number)
    
    def _process_event(self, event: GraphEvent) -> None:
        if event.event_type == "node_added":
            node_id = event.data["node_id"]
            label = event.data["label"]
            features = event.data.get("features", {})
            
            # Note: position is now calculated by the layout algorithm instead of stored
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
    def __init__(self, trace_file_path: str, video_path: Optional[str] = None):
        self.playback = GraphPlayback(trace_file_path)
        self.video_path = video_path
        self.video_capture = None
        self.play_interval_ms = 1000 // 24
        self.video_lock = threading.Lock()
        self.frame_cache = {}
        self.max_cache_size = 100
        self.edge_hover_points = 20  # Number of hover points per edge
        
        self.gaze_type_info = {
            GAZE_TYPE_UNTRACKED: {"color": "gray", "label": "Untracked"},
            GAZE_TYPE_FIXATION: {"color": "blue", "label": "Fixation"},
            GAZE_TYPE_SACCADE: {"color": "red", "label": "Saccade"},
            GAZE_TYPE_UNKNOWN: {"color": "purple", "label": "Unknown"},
            GAZE_TYPE_TRUNCATED: {"color": "orange", "label": "Truncated"}
        }
        
        self.node_background = {
            "default": "gray",
            "last_added": "blue"
        }
        self.node_border = {
            "default": "black",
            "current": self.gaze_type_info[GAZE_TYPE_FIXATION]["color"]
        }
        
        self._setup_video_capture()
        self.app = self._create_dashboard()
    
    def _setup_video_capture(self) -> None:
        if self.video_path and Path(self.video_path).exists():
            self.video_capture = cv2.VideoCapture(self.video_path)
            cv2.setNumThreads(1)
    
    def _get_video_frame(self, frame_number: int) -> Optional[np.ndarray]:
        if self.video_capture is None:
            return None
            
        if frame_number in self.frame_cache:
            return self.frame_cache[frame_number]
            
        with self.video_lock:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = self.video_capture.read()
            
            if not success:
                return None
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if len(self.frame_cache) >= self.max_cache_size:
                oldest_frame = min(self.frame_cache.keys())
                del self.frame_cache[oldest_frame]
            
            self.frame_cache[frame_number] = frame_rgb
            return frame_rgb
    
    def _create_dashboard(self) -> dash.Dash:
        app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            update_title=None,
            suppress_callback_exceptions=True
        )
        
        app.enable_dev_tools(
            dev_tools_hot_reload=False,
            dev_tools_ui=False,
            dev_tools_serve_dev_bundles=False
        )
        
        app.layout = self._create_layout()
        self._register_callbacks(app)
        
        return app
    
    def _create_layout(self) -> dbc.Container:
        return dbc.Container([
            dcc.Store(id='play-state', data={'is_playing': False, 'last_update': 0}),
            dcc.Interval(
                id='auto-advance',
                interval=self.play_interval_ms,
                disabled=True,
                max_intervals=-1
            ),
            dbc.Row([
                dbc.Col([
                    html.H4("Video Feed", className="text-center"),
                    dcc.Graph(id="video-display"),
                ], width=6),
                dbc.Col([
                    html.H4("Graph Visualization", className="text-center"),
                    dcc.Graph(id="graph-display"),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Object Detection", className="text-center"),
                    html.Div(id="detection-stats", className="border p-2")
                ], width=12),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        dbc.Button("← Prev", id="prev-frame", n_clicks=0, color="primary", className="me-2"),
                                        dbc.Button("Play", id="play-pause", n_clicks=0, color="success", className="me-2"),
                                        dbc.Button("Next →", id="next-frame", n_clicks=0, color="primary"),
                                    ], className="d-flex justify-content-center"),
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.Span("Current Frame: ", className="me-2"),
                                        html.Strong(id="current-frame-display"),
                                    ], className="d-flex justify-content-center align-items-center h-100"),
                                ], width=4),
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
        @app.callback(
            [Output("play-state", "data"),
             Output("auto-advance", "disabled"),
             Output("play-pause", "children")],
            [Input("play-pause", "n_clicks")],
            [State("play-state", "data")]
        )
        def toggle_play_state(play_clicks, current_state):
            if not play_clicks:
                return current_state, True, "Play"
                
            is_playing = not current_state.get('is_playing', False)
            new_state = {
                'is_playing': is_playing, 
                'last_update': current_state.get('last_update', 0) + 1
            }
            
            return new_state, not is_playing, "Pause" if is_playing else "Play"
        
        @app.callback(
            [Output("video-display", "figure"),
             Output("graph-display", "figure"),
             Output("detection-stats", "children"),
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
                self._create_detection_stats(frame_number),
                str(frame_number),
                frame_number
            )
    
    def _determine_frame_number(self, ctx, slider_frame, current_frame, play_state) -> int:
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
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0),
            height=height
        )
        return fig

    def _add_gaze_overlay(self, fig: go.Figure, frame_number: int, frame_width: int, frame_height: int) -> None:
        events = self.playback.get_events_for_frame(frame_number)
        
        # Add gaze point markers
        for event in events:
            if event.event_type == "frame_processed":
                pos = event.data["gaze_position"]
                if pos is None or (pos[0] == 0.0 and pos[1] == 0.0):
                    continue
                    
                x, y = pos[0] * frame_width, pos[1] * frame_height
                gaze_type = event.data["gaze_type"]
                gaze_info = self.gaze_type_info.get(gaze_type, {
                    "color": "black", 
                    "label": f"Other ({gaze_type})"
                })
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers",
                    marker=dict(size=15, color=gaze_info["color"]),
                    hovertext=gaze_info["label"],
                    hoverinfo='text',
                    showlegend=False
                ))
        
        # Add object detection bounding boxes
        detection_event = self.playback.get_object_detection(frame_number)
        if detection_event:
            bbox = detection_event.data["bounding_box"]
            most_likely_label = detection_event.data["detected_object"]
            current_label = detection_event.data.get("current_detected_label", most_likely_label)
            potential_labels = detection_event.data.get("potential_labels", {})
            
            # Create label text showing both current and most likely label if they differ
            if current_label == most_likely_label:
                label_text = f"{current_label.capitalize()}"
            else:
                label_text = f"{current_label.capitalize()} → {most_likely_label.capitalize()}"
            
            # Create hover text showing detected object and top potential label counts
            sorted_labels = sorted(potential_labels.items(), key=lambda x: x[1], reverse=True)
            potential_labels_text = "<br>".join([f"{obj.capitalize()}: {count}" for obj, count in sorted_labels[:5]])
            
            hover_text = (
                f"Current: {current_label.capitalize()}<br>"
                f"Most likely: {most_likely_label.capitalize()}<br><br>"
                f"Potential labels:<br>{potential_labels_text}"
            )
            
            # Extract bounding box coordinates [x, y, width, height]
            x, y, width, height = bbox
            x0, y0, x1, y1 = x, y, x + width, y + height
            
            # Add semi-transparent fill for the bounding box
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                fillcolor=f'rgba(0, 0, 255, 0.1)',  # Blue with 10% opacity
                mode="lines",
                line=dict(width=2, color=self.gaze_type_info[GAZE_TYPE_FIXATION]["color"]),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ))
            
            # Add label at top of bounding box
            fig.add_trace(go.Scatter(
                x=[(x0 + x1) / 2],
                y=[y0 - 10],
                mode="text",
                text=[label_text],
                textposition="top center",
                textfont=dict(
                    size=14, 
                    color=self.gaze_type_info[GAZE_TYPE_FIXATION]["color"],
                    family="Arial Black"
                ),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ))
            
            # Add a smaller box in top-right corner showing confidence level
            if sorted_labels:
                top_confidence = sorted_labels[0][1]
                total_votes = sum(count for _, count in sorted_labels)
                confidence_pct = (top_confidence / total_votes) * 100 if total_votes > 0 else 0
                
                fig.add_trace(go.Scatter(
                    x=[x1 - 50],
                    y=[y0 + 20],
                    mode="text",
                    text=[f"{confidence_pct:.0f}%"],
                    textposition="middle center",
                    textfont=dict(
                        size=12, 
                        color="white",
                        family="Arial"
                    ),
                    marker=dict(
                        size=30,
                        opacity=0.8,
                        color=self.gaze_type_info[GAZE_TYPE_FIXATION]["color"]
                    ),
                    hoverinfo='text',
                    hovertext=f"Confidence: {confidence_pct:.1f}%",
                    showlegend=False
                ))

    def _create_video_figure(self, frame_number: int) -> go.Figure:
        fig = self._create_empty_figure()
        
        frame = self._get_video_frame(frame_number)
        if frame is None:
            return fig
            
        frame_height, frame_width = frame.shape[:2]
        fig.add_trace(go.Image(z=frame))
        self._add_gaze_overlay(fig, frame_number, frame_width, frame_height)
        
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
            autosize=False,
            width=frame_width,
            height=frame_height
        )
        
        return fig
    
    def _get_current_node_id(self, frame_number: int) -> Optional[Any]:
        events = self.playback.get_events_for_frame(frame_number)
        for event in events:
            if event.event_type == "frame_processed" and 'node_id' in event.data:
                return event.data["node_id"]
        return None
    
    def _format_node_label(self, label: str) -> str:
        words = label.split('_')
        capitalized_words = [word.capitalize() for word in words]
        return '<br>'.join(capitalized_words)
        
    def _format_feature_text(self, features: Dict[str, Any]) -> str:
        if not features:
            return ""
            
        text_parts = []
        for key, value in features.items():
            if isinstance(value, (int, float)):
                value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
            else:
                value_str = str(value)
                
            # Convert snake_case to Title Case
            key_parts = key.split('_')
            key_label = ' '.join(part.capitalize() for part in key_parts)
            
            text_parts.append(f"{key_label}: {value_str}")
            
        return "<br>".join(text_parts)

    def _add_edges_to_figure(self, fig: go.Figure, G: nx.DiGraph, pos: Dict) -> None:
        regular_edge_x, regular_edge_y = [], []
        last_edge_x, last_edge_y = [], []
        edge_middle_x, edge_middle_y, edge_hover_texts = [], [], []
        
        for edge in G.edges(data=True):
            source, target = edge[0], edge[1]
            edge_data = edge[2]
            
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_type = edge_data.get('edge_type', 'unknown')
            features = edge_data.get('features', {})
            
            edge_info = f"Edge: {source} → {target}<br>Type: {edge_type}"
            if features:
                feature_text = self._format_feature_text(features)
                edge_info += f"<br><br>{feature_text}"
            
            # Add the main edge lines
            if (source, target) == self.playback.last_added_edge:
                last_edge_x.extend([x0, x1, None])
                last_edge_y.extend([y0, y1, None])
            else:
                regular_edge_x.extend([x0, x1, None])
                regular_edge_y.extend([y0, y1, None])
            
            # Add intermediate points for better hover detection
            middle_x, middle_y = self._generate_intermediate_points(x0, x1, y0, y1, self.edge_hover_points)
            edge_middle_x.extend(middle_x)
            edge_middle_y.extend(middle_y)
            edge_hover_texts.extend([edge_info] * len(middle_x))
        
        # Add the main edge lines
        if regular_edge_x:
            fig.add_trace(go.Scatter(
                x=regular_edge_x, y=regular_edge_y,
                mode='lines',
                line=dict(width=2.5, color='#888'),
                hoverinfo='none',
                showlegend=False
            ))
        
        if last_edge_x:
            fig.add_trace(go.Scatter(
                x=last_edge_x, y=last_edge_y,
                mode='lines',
                line=dict(width=4, color='blue'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add hover points along edges
        if edge_middle_x:
            fig.add_trace(go.Scatter(
                x=edge_middle_x, y=edge_middle_y,
                mode='markers',
                marker=dict(size=6, opacity=0.1),
                hoverinfo='text',
                hovertext=edge_hover_texts,
                hovertemplate='%{hovertext}<extra></extra>',
                showlegend=False
            ))
    
    def _generate_intermediate_points(self, x0, x1, y0, y1, qty):
        """Generate intermediate points along an edge for better hover detection."""
        middle_x = self._queue(x0, x1, qty + 2)
        middle_y = self._queue(y0, y1, qty + 2)
        # Remove first and last points (they are the nodes)
        middle_x.pop(0)
        middle_x.pop()
        middle_y.pop(0)
        middle_y.pop()
        return middle_x, middle_y
    
    def _queue(self, a, b, qty):
        """Generate a specified number of points between a and b."""
        q = deque()
        q.append((0, qty - 1))  # indexing starts at 0
        pts = [0] * qty
        pts[0] = a
        pts[-1] = b  # first value is a, last is b
        while q:
            left, right = q.popleft()  # remove working segment from queue
            center = (left + right + 1) // 2  # creates index values for pts
            pts[center] = (pts[left] + pts[right]) / 2
            if right - left > 2:  # stop when qty met
                q.append((left, center))
                q.append((center, right))
        return pts

    def _add_nodes_to_figure(self, fig: go.Figure, G: nx.DiGraph, pos: Dict, current_node_id: Any) -> None:
        node_x, node_y, node_text, node_hover_text = [], [], [], []
        node_colors, node_border_colors = [], []
        
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            raw_label = data['label']
            formatted_label = self._format_node_label(raw_label)
            node_text.append(formatted_label)
            
            # Create hover text with node info and features
            hover_label = ' '.join([word.capitalize() for word in raw_label.split('_')])
            hover_text = f"Node {node}: {hover_label}"
            
            # Add features to hover text if available
            features = data.get('features', {})
            if features:
                feature_text = self._format_feature_text(features)
                hover_text += f"<br><br>{feature_text}"
                
            node_hover_text.append(hover_text)
            
            is_current = node == current_node_id
            is_last_added = node == self.playback.last_added_node
            
            node_border_colors.append(self.node_border["current"] if is_current else self.node_border["default"])
            node_colors.append(self.node_background["last_added"] if is_last_added else self.node_background["default"])
        
        base_size = 60
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=base_size,
                color=node_colors,
                line=dict(width=3, color=node_border_colors)
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=11, color='white'),
            hovertext=node_hover_text,
            hoverinfo='text',
            showlegend=False
        ))
    
    def _create_graph_figure(self, frame_number: int) -> go.Figure:
        G = self.playback.build_graph_until_frame(frame_number)
        fig = go.Figure()
        
        if len(G.nodes) > 0:
            pos = nx.kamada_kawai_layout(G)
            current_node_id = self._get_current_node_id(frame_number)
            
            self._add_edges_to_figure(fig, G, pos)
            self._add_nodes_to_figure(fig, G, pos, current_node_id)
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig

    def _create_detection_stats(self, frame_number: int) -> List:
        """Create a panel showing object detection statistics for the current frame."""
        events = self.playback.get_events_for_frame(frame_number)
        detection_events = [e for e in events if e.event_type == "gaze_object_detected"]
        
        if not detection_events:
            return [html.P("No object detections in this frame", className="text-muted text-center")]
        
        # Use the most recent detection event for this frame
        event = detection_events[-1]
        most_likely_label = event.data["detected_object"]
        current_label = event.data.get("current_detected_label", most_likely_label)
        bbox = event.data["bounding_box"]
        potential_labels = event.data.get("potential_labels", {})
        
        # Sort labels by count (descending)
        sorted_labels = sorted(potential_labels.items(), key=lambda x: x[1], reverse=True)
        
        # Create color bars for the potential labels
        max_count = max(count for _, count in sorted_labels) if sorted_labels else 1
        
        # Calculate confidence percentage for most likely label
        if sorted_labels:
            top_confidence = sorted_labels[0][1]
            total_votes = sum(count for _, count in sorted_labels)
            confidence_pct = (top_confidence / total_votes) * 100 if total_votes > 0 else 0
        else:
            confidence_pct = 0
        
        children = [
            dbc.Row([
                dbc.Col([
                    html.H5("Current Detection:", className="text-primary mb-0")
                ], width=5),
                dbc.Col([
                    html.H5(f"{current_label.capitalize()}", className="mb-0")
                ], width=7)
            ], className="mb-2"),
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
            html.Div([
                html.Div([
                    html.Strong("Bounding Box [x, y, width, height]: "),
                    html.Span(f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                ], className="mb-2"),
                html.Strong("Potential Labels:", className="d-block mb-2"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Span(label.capitalize(), className="align-middle")
                        ], width=3),
                        dbc.Col([
                            dbc.Progress(
                                value=(count / max_count) * 100,
                                label=f"{count}",
                                color="info" if label == most_likely_label else 
                                      "success" if label == current_label else "secondary",
                                className="mb-2"
                            )
                        ], width=9)
                    ]) for label, count in sorted_labels[:10]  # Show top 10 labels
                ])
            ])
        ]
        
        return children

def visualize_graph_construction(
    trace_file: str,
    video_path: Optional[str] = None,
    port: int = 8050,
    debug: bool = False
) -> None:
    visualizer = InteractiveGraphVisualizer(trace_file, video_path)
    visualizer.app.run(debug=debug, port=port, use_reloader=False)