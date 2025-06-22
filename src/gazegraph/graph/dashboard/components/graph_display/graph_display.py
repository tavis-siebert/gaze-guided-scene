"""Graph display component for visualizing the scene graph."""

from typing import Dict
import networkx as nx
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc

from gazegraph.graph.dashboard.components.base import BaseComponent
from gazegraph.graph.dashboard.utils.constants import (
    FIGURE_HEIGHT,
    FIGURE_MARGIN,
    FIGURE_BG_COLOR,
    FIGURE_PAPER_BG_COLOR,
    FIGURE_HOVER_LABEL,
    MAX_EDGE_HOVER_POINTS,
)
from gazegraph.graph.dashboard.components.graph_display.layout import (
    compute_graph_layout,
)
from gazegraph.graph.dashboard.components.graph_display.plotting import (
    create_empty_graph,
    add_edges_to_figure,
    add_nodes_to_figure,
)
from gazegraph.logger import get_logger

logger = get_logger(__name__)


class GraphDisplay(BaseComponent):
    """Component for displaying the graph visualization.

    This component manages the creation of graph figures and handles
    styling and interaction for the graph visualization.

    Attributes:
        _cached_figure: Cached figure for the last processed graph
        _last_graph_hash: Hash of the last processed graph
    """

    def __init__(self, playback=None, **kwargs):
        """Initialize the graph display component.

        Args:
            playback: Playback instance for accessing the last added node
            **kwargs: Additional arguments to pass to BaseComponent
        """
        self._cached_figure = None
        self._last_graph_hash = None
        self.playback = playback

        super().__init__(component_id="graph-display", **kwargs)

    def create_layout(self) -> dbc.Card:
        """Create the component's layout.

        Returns:
            Dash Bootstrap Card component with graph display
        """
        return dbc.Card(
            [
                dbc.CardHeader("Graph Visualization"),
                dbc.CardBody(
                    [
                        dcc.Graph(
                            id=f"{self.component_id}-graph",
                            style={"height": "60vh"},
                            config={"responsive": True},
                        )
                    ]
                ),
            ],
            className="shadow-sm h-100 w-100",
        )

    def _get_graph_hash(self, G: nx.DiGraph) -> str:
        """Generate a hash of the graph structure for caching.

        Args:
            G: NetworkX directed graph

        Returns:
            String hash of the graph structure
        """
        graph_str = f"{len(G.nodes)}_{len(G.edges)}"
        for node in sorted(G.nodes(data=True)):
            graph_str += f"_{node[0]}_{node[1].get('label', '')}"
        for edge in sorted(G.edges(data=True)):
            graph_str += f"_{edge[0]}_{edge[1]}_{edge[2].get('edge_type', '')}"
        return graph_str

    def _create_figure(
        self, G: nx.DiGraph, pos: Dict, most_recent_node=None, last_added_edge=None
    ) -> go.Figure:
        """Create a complete graph visualization figure.

        Args:
            G: NetworkX directed graph
            pos: Dictionary mapping node IDs to positions
            most_recent_node: ID of the most recently updated or added node to highlight
            last_added_edge: Tuple of (source_id, target_id) for the last added edge to highlight

        Returns:
            Plotly figure with graph visualization
        """
        fig = go.Figure()

        add_edges_to_figure(fig, G, pos, MAX_EDGE_HOVER_POINTS, last_added_edge)
        add_nodes_to_figure(fig, G, pos, most_recent_node)

        fig.update_layout(
            showlegend=False,
            margin=FIGURE_MARGIN,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=FIGURE_HEIGHT,
            plot_bgcolor=FIGURE_BG_COLOR,
            paper_bgcolor=FIGURE_PAPER_BG_COLOR,
            hoverlabel=FIGURE_HOVER_LABEL,
        )

        return fig

    def get_figure(self, frame_number: int) -> go.Figure:
        """Get a complete graph visualization figure for the specified frame.

        Args:
            frame_number: Frame number to build the graph up to

        Returns:
            Plotly figure with graph visualization
        """
        if not self.playback:
            return create_empty_graph()

        # Build the graph using playback
        G = self.playback.build_graph_until_frame(frame_number)

        if len(G.nodes) == 0:
            return create_empty_graph()

        current_graph_hash = self._get_graph_hash(G)
        if (
            current_graph_hash == self._last_graph_hash
            and self._cached_figure is not None
        ):
            return go.Figure(self._cached_figure)

        pos = compute_graph_layout(G)
        most_recent_node = self.playback.most_recent_node
        last_added_edge = self.playback.last_added_edge
        fig = self._create_figure(G, pos, most_recent_node, last_added_edge)

        self._cached_figure = fig
        self._last_graph_hash = current_graph_hash

        return fig
