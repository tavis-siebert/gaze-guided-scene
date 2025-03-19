"""Main dashboard component that integrates all visualization components."""
from typing import Optional
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from graph.dashboard.graph_constants import DEFAULT_PLAY_INTERVAL_MS
from graph.dashboard.graph_playback import GraphPlayback
from graph.dashboard.video_display import VideoDisplay
from graph.dashboard.graph_display import GraphDisplay
from graph.dashboard.playback_controls import PlaybackControls


class Dashboard:
    """Main dashboard component that integrates all visualization elements.
    
    This component creates and manages the Dash application, registers
    callbacks, and coordinates the interaction between components.
    
    Attributes:
        playback: GraphPlayback instance for managing events and graph state
        video_display: VideoDisplay component for frame visualization
        graph_display: GraphDisplay component for graph visualization
        playback_controls: PlaybackControls component for playback interaction
        app: Dash application instance
    """
    
    def __init__(
        self, 
        trace_file_path: str, 
        video_path: Optional[str] = None,
        play_interval_ms: int = DEFAULT_PLAY_INTERVAL_MS
    ):
        """Initialize the dashboard with data sources.
        
        Args:
            trace_file_path: Path to the trace file with graph events
            video_path: Optional path to the video file
            play_interval_ms: Interval in ms between frames during playback
        """
        self.playback = GraphPlayback(trace_file_path)
        self.video_display = VideoDisplay(video_path)
        self.graph_display = GraphDisplay()
        self.playback_controls = PlaybackControls()
        self.play_interval_ms = play_interval_ms
        
        self.app = self._create_app()
    
    def _create_app(self) -> dash.Dash:
        """Create and configure the Dash application.
        
        Returns:
            Configured Dash application
        """
        app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            update_title=None,
            suppress_callback_exceptions=True
        )
        
        # Configure development options
        app.enable_dev_tools(
            dev_tools_hot_reload=False,
            dev_tools_ui=False,
            dev_tools_serve_dev_bundles=False
        )
        
        # Create application layout
        app.layout = self._create_layout()
        
        # Register component callbacks
        self._register_callbacks(app)
        
        return app
    
    def _create_layout(self) -> dbc.Container:
        """Create the dashboard layout.
        
        Returns:
            Dash Bootstrap Container with the complete dashboard layout
        """
        return dbc.Container([
            # Hidden state components
            dcc.Store(id="play-state", data={'is_playing': False, 'last_update': 0}),
            dcc.Interval(
                id="auto-advance",
                interval=self.play_interval_ms,
                disabled=True,
                max_intervals=-1
            ),
            
            # Video and graph visualization row
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
            
            # Playback controls row
            dbc.Row([
                dbc.Col([
                    self.playback_controls.create_layout(
                        self.playback.min_frame,
                        self.playback.max_frame,
                        self.playback.min_frame
                    )
                ], width=12),
            ]),
            
            # Hidden element for callback state
            html.Div(id="frame-state", style={"display": "none"}),
        ], fluid=True)
    
    def _register_callbacks(self, app: dash.Dash) -> None:
        """Register callbacks for all dashboard components.
        
        Args:
            app: Dash application instance
        """
        # Register component-specific callbacks
        self.playback_controls.register_callbacks(app)
        
        # Main update callback for visualization components
        @app.callback(
            [Output("video-display", "figure"),
             Output("graph-display", "figure"),
             Output("current-frame-display", "children"),
             Output("frame-slider", "value"),
             Output("frame-state", "children")],
            [Input("frame-slider", "value"),
             Input("prev-frame", "n_clicks"),
             Input("next-frame", "n_clicks"),
             Input("auto-advance", "n_intervals")],
            [State("play-state", "data"),
             State("frame-slider", "value")]
        )
        def update_displays(slider_frame, prev_clicks, next_clicks, 
                          n_intervals, play_state, current_frame):
            """Update all display components based on the current frame.
            
            Returns:
                Tuple of (video figure, graph figure, current frame text, 
                         frame slider value, frame state)
            """
            # Determine the frame number to display
            frame_number = self.playback_controls.determine_frame_number(
                dash.callback_context,
                slider_frame,
                current_frame,
                play_state,
                self.playback.min_frame,
                self.playback.max_frame
            )
            
            # Build the graph up to the current frame
            graph = self.playback.build_graph_until_frame(frame_number)
            
            # Get current node ID for highlighting
            events = self.playback.get_events_for_frame(frame_number)
            current_node_id = self.graph_display.get_current_node_id(events)
            
            # Create visualization components
            return (
                self.video_display.create_figure(frame_number, self.playback),
                self.graph_display.create_figure(
                    graph, 
                    current_node_id, 
                    self.playback.last_added_node, 
                    self.playback.last_added_edge
                ),
                str(frame_number),
                frame_number,
                str(frame_number)  # Hidden state
            )
    
    def run(self, port: int = 8050, debug: bool = False) -> None:
        """Run the dashboard server.
        
        Args:
            port: Port number to run the server on
            debug: Whether to run in debug mode
        """
        self.app.run(debug=debug, port=port, use_reloader=False) 