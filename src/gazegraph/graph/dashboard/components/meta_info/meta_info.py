"""Meta information display component for the dashboard."""

import dash_bootstrap_components as dbc
from dash import html
from pathlib import Path

from gazegraph.graph.dashboard.components.base import BaseComponent


class MetaInfo(BaseComponent):
    """Component for displaying meta information about the visualization.

    This component shows the video and trace file names in a clean, minimal format
    using badges with icons.

    Attributes:
        video_path: Path to the video file
        trace_path: Path to the trace file
    """

    def __init__(self, video_path: str, trace_path: str, **kwargs):
        """Initialize the meta information bar.

        Args:
            video_path: Path to the video file
            trace_path: Path to the trace file
            **kwargs: Additional arguments to pass to BaseComponent
        """
        self.video_path = video_path
        self.trace_path = trace_path
        super().__init__(component_id="meta-info", **kwargs)

    def create_layout(self) -> html.Div:
        """Create the component's layout.

        Returns:
            Dash component with meta information
        """
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Badge(
                                    [
                                        html.I(className="fas fa-film me-2"),
                                        html.Span(Path(self.video_path).name),
                                    ],
                                    color="light",
                                    text_color="dark",
                                    className="me-3",
                                ),
                                dbc.Badge(
                                    [
                                        html.I(className="fas fa-file-code me-2"),
                                        html.Span(Path(self.trace_path).name),
                                    ],
                                    color="light",
                                    text_color="dark",
                                ),
                            ],
                            width="auto",
                        )
                    ]
                )
            ],
            className="mb-2",
        )
