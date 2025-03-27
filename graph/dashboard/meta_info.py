"""Meta information display component for the dashboard."""
from pathlib import Path
from dash import html
import dash_bootstrap_components as dbc


class MetaInfoBar(html.Div):
    """Component for displaying meta information about the visualization.
    
    This component shows the video and trace file names in a clean, minimal format
    using badges with icons.
    
    Attributes:
        video_path: Path to the video file
        trace_path: Path to the trace file
    """
    
    def __init__(self, video_path: str, trace_path: str):
        """Initialize the meta information bar.
        
        Args:
            video_path: Path to the video file
            trace_path: Path to the trace file
        """
        super().__init__(
            dbc.Row([
                dbc.Col([
                    dbc.Badge([
                        html.I(className="fas fa-film me-2"),
                        html.Span(Path(video_path).name)
                    ], color="light", text_color="dark", className="me-3"),
                    dbc.Badge([
                        html.I(className="fas fa-file-code me-2"),
                        html.Span(Path(trace_path).name)
                    ], color="light", text_color="dark"),
                ], width="auto")
            ])
        )