"""Main dashboard component for the graph visualization."""
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
from graph.dashboard.meta_info import MetaInfoBar


class Dashboard:
    def __init__(
        self, 
        trace_file_path: str, 
        video_path: Optional[str] = None,
        play_interval_ms: int = DEFAULT_PLAY_INTERVAL_MS
    ):
        """Initialize the dashboard.
        
        Args:
            trace_file_path: Path to the trace file
            video_path: Optional path to the video file
            play_interval_ms: Interval between frame updates in milliseconds
        """
        self.playback = GraphPlayback(trace_file_path)
        self.video_display = VideoDisplay(video_path)
        self.graph_display = GraphDisplay()
        self.playback_controls = PlaybackControls(
            min_frame=self.playback.min_frame,
            max_frame=self.playback.max_frame,
            current_frame=self.playback.min_frame,
            graph_playback=self.playback
        )
        self.meta_info = MetaInfoBar(video_path, trace_file_path)
        self.play_interval_ms = play_interval_ms
        
        self.app = self._create_app()
    
    def _create_app(self) -> dash.Dash:
        """Create and configure the Dash application.
        
        Returns:
            Configured Dash application instance
        """
        app = dash.Dash(
            __name__, 
            external_stylesheets=[
                dbc.themes.FLATLY,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
            ],
            update_title=None,
            suppress_callback_exceptions=True,
            title="Gaze-Guided Scene Graph"
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
        """Create the main dashboard layout.
        
        Returns:
            Dash Bootstrap Container with the dashboard layout
        """
        return dbc.Container([
            dcc.Store(id="play-state", data={'is_playing': False, 'last_update': 0}),
            dcc.Interval(id="auto-advance", interval=self.play_interval_ms, disabled=True, max_intervals=-1),
            
            dbc.Row([
                dbc.Col([
                    self.video_display.create_layout()
                ], width=6, className="d-flex"),
                
                dbc.Col([
                    self.graph_display.create_layout()
                ], width=6, className="d-flex"),
            ], className="mb-3 g-3"),
            
            dbc.Row(dbc.Col(
                self.playback_controls.create_layout()
            ), className="mb-2"),
            
            dbc.Row(dbc.Col(
                self.meta_info.create_layout()
            ), className="mb-2"),
            
            html.Div(id="frame-state", style={"display": "none"}),
        ], fluid=True, className="p-3")
    
    def _register_callbacks(self, app: dash.Dash) -> None:
        """Register callbacks for the dashboard.
        
        Args:
            app: Dash application instance
        """
        self.playback_controls.register_callbacks(app)
        
        @app.callback(
            [Output(f"{self.video_display.component_id}-graph", "figure"),
             Output(f"{self.graph_display.component_id}-graph", "figure"),
             Output(f"{self.playback_controls.component_id}-frame", "value"),
             Output("frame-state", "children"),
             Output(f"{self.playback_controls.component_id}-current-time", "children")],
            [Input(f"{self.playback_controls.component_id}-frame", "value"),
             Input(f"{self.playback_controls.component_id}-prev", "n_clicks"),
             Input(f"{self.playback_controls.component_id}-next", "n_clicks"),
             Input("auto-advance", "n_intervals")],
            [State("play-state", "data"),
             State(f"{self.playback_controls.component_id}-frame", "value"),
             State(f"{self.playback_controls.component_id}-speed", "value")]
        )
        def update_displays(slider_frame, prev_clicks, next_clicks, 
                            n_intervals, play_state, current_frame, playback_speed):
            frame_number = self.playback_controls.determine_frame_number(
                dash.callback_context, slider_frame, current_frame, play_state,
                self.playback.min_frame, self.playback.max_frame, playback_speed
            )
            
            graph = self.playback.build_graph_until_frame(frame_number)
            events = self.playback.get_events_for_frame(frame_number)
            current_node_id = self.graph_display.get_current_node_id(events)
            
            # Convert frame number to time string
            current_time = self.playback_controls.frame_to_time_str(frame_number)
            
            return (
                self.video_display.get_figure(frame_number, self.playback),
                self.graph_display.get_figure(graph),
                frame_number,
                str(frame_number),
                current_time
            )
    
    def run(self, port: int = 8050, debug: bool = False) -> None:
        """Run the dashboard server.
        
        Args:
            port: Port number to run the server on
            debug: Whether to run in debug mode
        """
        self.app.run(debug=debug, port=port, use_reloader=debug)
