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
                    # Navigation buttons
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button("←", id="prev-frame", color="secondary", size="m"),
                            dbc.Button("▶️", id="play-pause", color="success", size="m"),
                            dbc.Button("→", id="next-frame", color="secondary", size="m"),
                        ]),
                    ], width="auto", className="me-3"),
                    
                    # Frame slider
                    dbc.Col([
                        dcc.Slider(
                            id="frame-slider",
                            min=min_frame,
                            max=max_frame,
                            value=current_frame,
                            step=1,
                            marks={str(min_frame): str(min_frame), str(max_frame): str(max_frame)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], className="flex-grow-1"),
                    
                    # Speed control
                    dbc.Col([
                        html.Div([
                            dcc.Slider(
                                id="playback-speed",
                                min=PLAYBACK_SPEED_MIN,
                                max=PLAYBACK_SPEED_MAX,
                                value=PLAYBACK_SPEED_DEFAULT,
                                marks=PLAYBACK_SPEED_MARKS,
                                step=1,
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], style={"width": "300px"}),
                    ], width="auto"),
                ], className="align-items-center"),
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
                return current_state, True, "▶️"
                
            is_playing = not current_state.get('is_playing', False)
            new_state = {
                'is_playing': is_playing, 
                'last_update': current_state.get('last_update', 0) + 1
            }
            
            return new_state, not is_playing, "⏸️" if is_playing else "▶️"
    
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
                return max_frame
            return frame_number
        else:
            return slider_frame 