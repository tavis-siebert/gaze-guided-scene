"""Graph display component for visualizing the scene graph."""
from typing import Dict, Any, List, Optional
import networkx as nx
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html

from graph.dashboard.base_component import BaseComponent
from graph.dashboard.graph_constants import (
    FIGURE_HEIGHT, FIGURE_MARGIN, FIGURE_BG_COLOR, FIGURE_PAPER_BG_COLOR,
    FIGURE_HOVER_LABEL, MAX_EDGE_HOVER_POINTS
)
from graph.dashboard.graph_layout import compute_graph_layout
from graph.dashboard.graph_plotting import create_empty_graph, add_edges_to_figure, add_nodes_to_figure
from logger import get_logger

logger = get_logger(__name__)


class GraphDisplay(BaseComponent):
    """Component for displaying the graph visualization.
    
    This component manages the creation of graph figures and handles
    styling and interaction for the graph visualization.
    
    Attributes:
        _cached_figure: Cached figure for the last processed graph
        _last_graph_hash: Hash of the last processed graph
    """
    
    def __init__(self, **kwargs):
        """Initialize the graph display component.
        
        Args:
            **kwargs: Additional arguments to pass to BaseComponent
        """
        self._cached_figure = None
        self._last_graph_hash = None
        
        super().__init__(component_id="graph-display", **kwargs)
    
    def create_layout(self) -> dbc.Card:
        """Create the component's layout.
        
        Returns:
            Dash Bootstrap Card component with graph display
        """
        return dbc.Card([
            dbc.CardHeader("Graph Visualization"),
            dbc.CardBody([
                dcc.Graph(
                    id=f"{self.component_id}-graph",
                    style={"height": "60vh"},
                    config={"responsive": True}
                )
            ])
        ], className="shadow-sm h-100 w-100")
    
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
    
    def _create_figure(self, G: nx.DiGraph, pos: Dict) -> go.Figure:
        """Create a complete graph visualization figure.
        
        Args:
            G: NetworkX directed graph
            pos: Dictionary mapping node IDs to positions
            
        Returns:
            Plotly figure with graph visualization
        """
        fig = go.Figure()
        
        add_edges_to_figure(fig, G, pos, MAX_EDGE_HOVER_POINTS)
        add_nodes_to_figure(fig, G, pos)
        
        fig.update_layout(
            showlegend=False,
            margin=FIGURE_MARGIN,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=FIGURE_HEIGHT,
            plot_bgcolor=FIGURE_BG_COLOR,
            paper_bgcolor=FIGURE_PAPER_BG_COLOR,
            hoverlabel=FIGURE_HOVER_LABEL
        )
        
        return fig
    
    def get_figure(self, G: nx.DiGraph) -> go.Figure:
        """Get a complete graph visualization figure.
        
        Args:
            G: NetworkX directed graph to visualize
            
        Returns:
            Plotly figure with graph visualization
        """
        if len(G.nodes) == 0:
            return create_empty_graph()
        
        current_graph_hash = self._get_graph_hash(G)
        if current_graph_hash == self._last_graph_hash and self._cached_figure is not None:
            return go.Figure(self._cached_figure)
        
        pos = compute_graph_layout(G)
        fig = self._create_figure(G, pos)
        
        self._cached_figure = fig
        self._last_graph_hash = current_graph_hash
        
        return fig
    
    def get_current_node_id(self, events: List) -> Optional[Any]:
        """Extract the current node ID from frame processing events.
        
        Args:
            events: List of events for the current frame
            
        Returns:
            Current node ID or None if not found
        """
        for event in events:
            if event.event_type == "frame_processed" and 'node_id' in event.data:
                return event.data["node_id"]
        return None 