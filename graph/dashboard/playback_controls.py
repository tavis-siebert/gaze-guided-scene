"""Playback controls component for the graph visualization dashboard."""
from typing import Callable, Dict, Any
from dash import html, dcc
import dash_bootstrap_components as dbc

from graph.dashboard.graph_constants import (
    PLAYBACK_SPEEDS, PLAYBACK_SPEED_MIN, PLAYBACK_SPEED_MAX,
    PLAYBACK_SPEED_DEFAULT, PLAYBACK_SPEED_MARKS
)


class PlaybackControls:
    """Component for controlling playback of the graph visualization.
    
    This component provides buttons for play/pause, navigation, and a slider
    for jumping to specific frames in the visualization.
    """
    
    def create_layout(self, min_frame: int, max_frame: int, current_frame: int = None) -> dbc.Card:
        """Create a layout with playback controls.
        
        Args:
            min_frame: Minimum frame number
            max_frame: Maximum frame number
            current_frame: Current frame number (defaults to min_frame if None)
            
        Returns:
            Dash Bootstrap Card component with playback controls
        """
        if current_frame is None:
            current_frame = min_frame
            
        return dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Playback buttons and speed control
                    dbc.Col([
                        html.Div([
                            dbc.Button(
                                "← Prev", 
                                id="prev-frame", 
                                n_clicks=0, 
                                color="primary", 
                                className="me-2"
                            ),
                            dbc.Button(
                                "Play", 
                                id="play-pause", 
                                n_clicks=0, 
                                color="success", 
                                className="me-2"
                            ),
                            dbc.Button(
                                "Next →", 
                                id="next-frame", 
                                n_clicks=0, 
                                color="primary",
                                className="me-2"
                            ),
                            html.Div([
                                html.Span("Speed: ", className="me-2"),
                                dcc.Slider(
                                    id="playback-speed",
                                    min=PLAYBACK_SPEED_MIN,
                                    max=PLAYBACK_SPEED_MAX,
                                    value=PLAYBACK_SPEED_DEFAULT,
                                    marks=PLAYBACK_SPEED_MARKS,
                                    step=1,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ], style={"width": "200px", "display": "inline-block", "vertical-align": "middle"}),
                        ], className="d-flex justify-content-center align-items-center"),
                    ], width=4),
                    
                    # Current frame display
                    dbc.Col([
                        html.Div([
                            html.Span("Current Frame: ", className="me-2"),
                            html.Strong(str(current_frame), id="current-frame-display"),
                        ], className="d-flex justify-content-center align-items-center h-100"),
                    ], width=4),
                    
                    # Frame slider
                    dbc.Col([
                        dcc.Slider(
                            id="frame-slider",
                            min=min_frame,
                            max=max_frame,
                            value=current_frame,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], width=4),
                ]),
            ])
        ], className="mb-3")
    
    def register_callbacks(self, app) -> None:
        """Register callbacks for the playback controls.
        
        Args:
            app: Dash application instance
        """
        from dash.dependencies import Input, Output, State
        
        @app.callback(
            [Output("play-state", "data"),
             Output("auto-advance", "disabled"),
             Output("play-pause", "children")],
            [Input("play-pause", "n_clicks")],
            [State("play-state", "data")]
        )
        def toggle_play_state(play_clicks, current_state):
            """Toggle the play/pause state when the button is clicked.
            
            Args:
                play_clicks: Number of times the button has been clicked
                current_state: Current play state
                
            Returns:
                Tuple of (new state, interval disabled, button text)
            """
            if not play_clicks:
                return current_state, True, "Play"
                
            is_playing = not current_state.get('is_playing', False)
            new_state = {
                'is_playing': is_playing, 
                'last_update': current_state.get('last_update', 0) + 1
            }
            
            return new_state, not is_playing, "Pause" if is_playing else "Play"
    
    def determine_frame_number(
        self, 
        ctx, 
        slider_frame: int, 
        current_frame: int, 
        play_state: Dict[str, Any],
        min_frame: int,
        max_frame: int,
        playback_speed: int = PLAYBACK_SPEED_DEFAULT
    ) -> int:
        """Determine the next frame number based on user interactions.
        
        Args:
            ctx: Dash callback context
            slider_frame: Current slider value
            current_frame: Current frame number
            play_state: Current play state
            min_frame: Minimum frame number
            max_frame: Maximum frame number
            playback_speed: Current playback speed setting
            
        Returns:
            Next frame number to display
        """
        if not ctx.triggered:
            return slider_frame
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        is_playing = play_state.get('is_playing', False)
        
        if trigger_id == "prev-frame":
            return max(min_frame, current_frame - playback_speed)
        elif trigger_id == "next-frame":
            return min(max_frame, current_frame + playback_speed)
        elif trigger_id == "auto-advance" and is_playing:
            frame_number = current_frame + playback_speed
            if frame_number > max_frame:
                frame_number = min_frame
            return frame_number
        else:
            return slider_frame 