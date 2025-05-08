"""Main dashboard component for the graph visualization."""
from typing import Optional
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from gazegraph.graph.dashboard.utils.constants import DEFAULT_PLAY_INTERVAL_MS
from gazegraph.graph.dashboard.playback import Playback
from gazegraph.graph.dashboard.components.video_display import VideoDisplay
from gazegraph.graph.dashboard.components.graph_display import GraphDisplay
from gazegraph.graph.dashboard.components.playback_controls import PlaybackControls
from gazegraph.graph.dashboard.components.meta_info import MetaInfo
from gazegraph.graph.dashboard.components.snapshot import Snapshot


class Dashboard:
    def __init__(
        self, 
        trace_file_path: str, 
        video_path: Optional[str] = None,
        play_interval_ms: int = DEFAULT_PLAY_INTERVAL_MS,
        action_mapping_path: str = None,
        verb_idx_file: Optional[str] = None,
        noun_idx_file: Optional[str] = None,
        train_split_file: Optional[str] = None,
        val_split_file: Optional[str] = None
    ):
        """Initialize the dashboard.
        
        Args:
            trace_file_path: Path to the trace file
            video_path: Optional path to the video file
            play_interval_ms: Interval between frame updates in milliseconds
            action_mapping_path: Path to the action mapping CSV file
            verb_idx_file: Path to the verb index mapping file
            noun_idx_file: Path to the noun index mapping file
            train_split_file: Path to the training data split file
            val_split_file: Path to the validation data split file
        """
        self.playback = Playback(trace_file_path)
        self.video_display = VideoDisplay(
            video_path, 
            playback=self.playback,
            verb_idx_file=verb_idx_file,
            noun_idx_file=noun_idx_file,
            train_split_file=train_split_file,
            val_split_file=val_split_file
        )
        self.graph_display = GraphDisplay(playback=self.playback)
        self.playback_controls = PlaybackControls(
            min_frame=self.playback.min_frame,
            max_frame=self.playback.max_frame,
            current_frame=self.playback.min_frame,
            playback=self.playback
        )
        self.meta_info = MetaInfo(video_path, trace_file_path)
        
        # Initialize the snapshot component if action mapping is provided
        self.snapshot = None
        if action_mapping_path:
            self.snapshot = Snapshot(playback=self.playback, action_mapping_path=action_mapping_path)
            
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
            
            dbc.Row([
                dbc.Col([
                    self.playback_controls.create_layout()
                ], width=9 if self.snapshot else 12),
                
                # Only add the snapshot column if the component is available
                *([
                    dbc.Col([
                        self.snapshot.create_layout()
                    ], width=3)
                ] if self.snapshot else []),
            ], className="mb-2"),
            
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
        
        # Register snapshot callbacks if the component is available
        if self.snapshot:
            self.snapshot.register_callbacks(app)
        
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
            
            # Convert frame number to time string
            current_time = self.playback_controls.frame_to_time_str(frame_number)
            
            return (
                self.video_display.get_figure(frame_number),
                self.graph_display.get_figure(frame_number),
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
