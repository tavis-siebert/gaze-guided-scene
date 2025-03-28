"""Playback controls component for the graph visualization dashboard."""
from typing import Callable, Dict, Any, List
from dash import html, dcc
import dash_bootstrap_components as dbc

from graph.dashboard.components.base import BaseComponent
from graph.dashboard.utils.constants import (
    PLAYBACK_SPEEDS, PLAYBACK_SPEED_MIN, PLAYBACK_SPEED_MAX,
    PLAYBACK_SPEED_DEFAULT, PLAYBACK_SPEED_MARKS, FPS
)


class PlaybackControls(BaseComponent):
    """Component for controlling playback of the graph visualization.
    
    This component provides buttons for play/pause, navigation, and a slider
    for jumping to specific frames in the visualization.
    """
    
    def __init__(
        self,
        min_frame: int,
        max_frame: int,
        current_frame: int = None,
        graph_playback = None,
        **kwargs
    ):
        """Initialize the playback controls component.
        
        Args:
            min_frame: Minimum frame number
            max_frame: Maximum frame number
            current_frame: Current frame number (defaults to min_frame if None)
            graph_playback: GraphPlayback instance for marking frames with node additions
            **kwargs: Additional arguments to pass to BaseComponent
        """
        self.min_frame = min_frame
        self.max_frame = max_frame
        self.current_frame = current_frame if current_frame is not None else min_frame
        self.graph_playback = graph_playback
        
        super().__init__(component_id="playback-controls", **kwargs)
    
    def create_layout(self) -> dbc.Card:
        """Create the component's layout.
        
        Returns:
            Dash Bootstrap Card component with playback controls
        """
        # Create basic marks for min and max frames
        slider_marks = {str(self.min_frame): str(self.min_frame), 
                       str(self.max_frame): str(self.max_frame)}
        
        # Add marks for frames with node additions if graph_playback is provided
        if self.graph_playback:
            node_frames = self.get_node_addition_frames(self.graph_playback)
            for frame in node_frames:
                # Only add label for node frames if they're not too close to other marks
                if frame != self.min_frame and frame != self.max_frame:
                    slider_marks[str(frame)] = {
                        "label": "",  # Empty label to hide text but keep marker
                    }
            
        # Convert frames to time strings
        current_time_str = self.frame_to_time_str(self.current_frame)
        total_time_str = self.frame_to_time_str(self.max_frame)
        
        return dbc.Card([
            dbc.CardHeader("Playback Controls"),
            dbc.CardBody([
                dbc.Row([
                    # Navigation buttons
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button("←", id=f"{self.component_id}-prev", color="secondary", size="m"),
                            dbc.Button("▶️", id=f"{self.component_id}-play-pause", color="success", size="m"),
                            dbc.Button("→", id=f"{self.component_id}-next", color="secondary", size="m"),
                        ]),
                    ], width="auto", className="me-3"),
                    
                    # Speed control
                    dbc.Col([
                        html.Div([
                            dcc.Slider(
                                id=f"{self.component_id}-speed",
                                min=PLAYBACK_SPEED_MIN,
                                max=PLAYBACK_SPEED_MAX,
                                value=PLAYBACK_SPEED_DEFAULT,
                                marks=PLAYBACK_SPEED_MARKS,
                                step=1,
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], style={"width": "300px"}),
                    ], width="auto", className="me-3"),
                    
                    # Current time display
                    dbc.Col([
                        html.Div(
                            id=f"{self.component_id}-current-time",
                            children=current_time_str,
                            className="text-end me-2"
                        ),
                    ], width="auto", style={"minWidth": "50px"}),
                    
                    # Frame slider
                    dbc.Col([
                        dcc.Slider(
                            id=f"{self.component_id}-frame",
                            min=self.min_frame,
                            max=self.max_frame,
                            value=self.current_frame,
                            step=1,
                            marks=slider_marks,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], className="flex-grow-1"),
                    
                    # Total time display
                    dbc.Col([
                        html.Div(
                            id=f"{self.component_id}-total-time",
                            children=total_time_str,
                            className="text-start ms-2"
                        ),
                    ], width="auto", style={"minWidth": "50px"}),
                ], className="align-items-center"),
            ])
        ], className="mb-3")
    
    def get_node_addition_frames(self, graph_playback) -> List[int]:
        """Get a list of frame numbers where new unique nodes were added.
        
        Args:
            graph_playback: GraphPlayback instance containing event data
            
        Returns:
            List of frame numbers where unique node addition events occurred
        """
        node_frames = {}  # Maps node_id to first frame it appears in
        
        # Sort frames to ensure we find the first occurrence of each node
        sorted_frames = sorted(graph_playback.frame_to_events.keys())
        
        for frame_num in sorted_frames:
            events = graph_playback.frame_to_events[frame_num]
            for event in events:
                if event.event_type == "node_added":
                    node_id = event.data["node_id"]
                    # Only record the first occurrence of this node_id
                    if node_id not in node_frames:
                        node_frames[node_id] = frame_num
        
        # Return sorted list of unique frames where new nodes appeared
        return sorted(set(node_frames.values()))
    
    def frame_to_time_str(self, frame_number: int) -> str:
        """Convert frame number to time string in MM:SS format.
        
        Args:
            frame_number: Frame number to convert
            
        Returns:
            Time string in MM:SS format
        """
        total_seconds = int(frame_number / FPS)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def register_callbacks(self, app) -> None:
        """Register callbacks for the playback controls.
        
        Args:
            app: Dash application instance
        """
        from dash.dependencies import Input, Output, State
        
        @app.callback(
            [Output("play-state", "data"),
             Output("auto-advance", "disabled"),
             Output(f"{self.component_id}-play-pause", "children")],
            [Input(f"{self.component_id}-play-pause", "n_clicks")],
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
        
        if trigger_id == f"{self.component_id}-prev":
            return max(min_frame, current_frame - playback_speed)
        elif trigger_id == f"{self.component_id}-next":
            return min(max_frame, current_frame + playback_speed)
        elif trigger_id == "auto-advance" and is_playing:
            frame_number = current_frame + playback_speed
            if frame_number > max_frame:
                return max_frame
            return frame_number
        else:
            return slider_frame 