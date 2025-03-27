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
        self.playback = GraphPlayback(trace_file_path)
        self.video_display = VideoDisplay(video_path)
        self.graph_display = GraphDisplay()
        self.playback_controls = PlaybackControls()
        self.play_interval_ms = play_interval_ms
        
        self.app = self._create_app()
    
    def _create_app(self) -> dash.Dash:
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
        return dbc.Container([
            dcc.Store(id="play-state", data={'is_playing': False, 'last_update': 0}),
            dcc.Interval(id="auto-advance", interval=self.play_interval_ms, disabled=True, max_intervals=-1),
            
            dbc.Row([
                dbc.Col([
                    self.video_display.create_card()
                ], width=6, className="d-flex"),
                
                dbc.Col([
                    self.graph_display.create_card()
                ], width=6, className="d-flex"),
            ], className="mb-3 g-3"),
            
            dbc.Row(dbc.Col(
                self.playback_controls.create_layout(
                    self.playback.min_frame, 
                    self.playback.max_frame, 
                    self.playback.min_frame,
                    graph_playback=self.playback
                )
            ), className="mb-2"),
            
            dbc.Row(dbc.Col(
                MetaInfoBar(self.video_display.video_path, self.playback.trace_file_path)
            ), className="mb-2"),
            
            html.Div(id="frame-state", style={"display": "none"}),
        ], fluid=True, className="p-3")
    
    def _register_callbacks(self, app: dash.Dash) -> None:
        self.playback_controls.register_callbacks(app)
        
        @app.callback(
            [Output("video-display", "figure"),
             Output("graph-display", "figure"),
             Output("frame-slider", "value"),
             Output("frame-state", "children")],
            [Input("frame-slider", "value"),
             Input("prev-frame", "n_clicks"),
             Input("next-frame", "n_clicks"),
             Input("auto-advance", "n_intervals")],
            [State("play-state", "data"),
             State("frame-slider", "value"),
             State("playback-speed", "value")]
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
            
            return (
                self.video_display.create_figure(frame_number, self.playback),
                self.graph_display.create_figure(
                    graph, current_node_id, 
                    self.playback.last_added_node, self.playback.last_added_edge
                ),
                frame_number,
                str(frame_number)
            )
    
    def run(self, port: int = 8050, debug: bool = False) -> None:
        self.app.run(debug=debug, port=port, use_reloader=debug)
