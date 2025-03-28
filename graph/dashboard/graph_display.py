"""Graph display component for visualizing the scene graph."""
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc

from graph.dashboard.graph_constants import (
    NODE_BACKGROUND, NODE_BORDER, NODE_BASE_SIZE, NODE_FONT_SIZE,
    NODE_FONT_COLOR, EDGE_WIDTH, EDGE_COLOR, EDGE_HOVER_OPACITY,
    EDGE_HOVER_SIZE, EDGE_LABEL_FONT_SIZE, EDGE_LABEL_COLOR,
    FIGURE_HEIGHT, FIGURE_MARGIN, FIGURE_BG_COLOR, FIGURE_PAPER_BG_COLOR,
    FIGURE_HOVER_LABEL, MAX_EDGE_HOVER_POINTS
)
from graph.dashboard.utils import (
    format_node_label, format_feature_text, generate_intermediate_points,
    get_angle_symbol
)
from graph.dashboard.layout_utils import compute_graph_layout
from graph.dashboard.svg_utils import ICON_X_POINTS, ICON_Y_POINTS
from logger import get_logger

logger = get_logger(__name__)


def create_empty_figure() -> go.Figure:
    """Create an empty figure with appropriate layout.
    
    Returns:
        Empty Plotly figure with placeholder message
    """
    fig = go.Figure()
    
    # Add the SVG shape using precomputed points
    fig.add_trace(go.Scatter(
        x=ICON_X_POINTS, y=ICON_Y_POINTS,
        mode='lines',
        line=dict(width=1, color='#555'),
        fill='toself',
        fillcolor='#666',
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add placeholder text
    fig.add_annotation(
        text="Empty Graph",
        xref="paper", yref="paper",
        x=0.5, y=0.4,
        showarrow=False,
        font=dict(size=16, color="#444"),
        align="center"
    )
    
    # Configure layout
    fig.update_layout(
        showlegend=False,
        margin=FIGURE_MARGIN,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
        height=FIGURE_HEIGHT,
        plot_bgcolor=FIGURE_BG_COLOR,
        paper_bgcolor=FIGURE_PAPER_BG_COLOR
    )
    return fig


def add_edges_to_figure(fig: go.Figure, G: nx.DiGraph, 
                       pos: Dict, last_added_edge: Optional[Tuple], max_edge_hover_points: int) -> None:
    """Add edges to the graph figure.
    
    Args:
        fig: The Plotly figure to add edges to
        G: NetworkX directed graph
        pos: Dictionary mapping node IDs to positions
        last_added_edge: Tuple of (source_id, target_id) for the most recently added edge
        max_edge_hover_points: Maximum number of edges for which to render hover points
    """
    edge_x, edge_y = [], []
    edge_middle_x, edge_middle_y, edge_hover_texts = [], [], []
    edge_labels_x, edge_labels_y, edge_labels_text = [], [], []
    
    for edge in G.edges(data=True):
        source, target = edge[0], edge[1]
        edge_data = edge[2]
        
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        edge_type = edge_data.get('edge_type', 'unknown')
        features = edge_data.get('features', {})
        
        # Create hover text
        edge_info = f"Edge: {source} → {target}<br>Type: {edge_type}"
        if features:
            feature_text = format_feature_text(features)
            edge_info += f"<br><br>{feature_text}"
        
        # Add main edge lines
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Add hover points if under threshold
        if len(G.edges()) <= max_edge_hover_points:
            middle_x, middle_y = generate_intermediate_points(x0, x1, y0, y1, qty=5)
            edge_middle_x.extend(middle_x)
            edge_middle_y.extend(middle_y)
            edge_hover_texts.extend([edge_info] * len(middle_x))
        
        # Add angle label if available
        if 'angle_degrees' in features:
            angle = features['angle_degrees']
            symbol = get_angle_symbol(angle)
            label_x = (x0 + x1) / 2
            label_y = (y0 + y1) / 2
            edge_labels_x.append(label_x)
            edge_labels_y.append(label_y)
            edge_labels_text.append(symbol)
    
    # Add edges
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=EDGE_WIDTH, color=EDGE_COLOR),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add hover points
    if edge_middle_x:
        fig.add_trace(go.Scatter(
            x=edge_middle_x, y=edge_middle_y,
            mode='markers',
            marker=dict(size=EDGE_HOVER_SIZE, opacity=EDGE_HOVER_OPACITY),
            hoverinfo='text',
            hovertext=edge_hover_texts,
            hovertemplate='%{hovertext}<extra></extra>',
            showlegend=False
        ))
    
    # Add angle labels
    if edge_labels_x:
        fig.add_trace(go.Scatter(
            x=edge_labels_x, y=edge_labels_y,
            mode='text',
            text=edge_labels_text,
            textposition="middle center",
            textfont=dict(size=EDGE_LABEL_FONT_SIZE, color=EDGE_LABEL_COLOR),
            hoverinfo='none',
            showlegend=False
        ))


def add_nodes_to_figure(fig: go.Figure, G: nx.DiGraph, pos: Dict, 
                       current_node_id: Optional[Any], last_added_node: Optional[Any]) -> None:
    """Add nodes to the graph figure.
    
    Args:
        fig: The Plotly figure to add nodes to
        G: NetworkX directed graph
        pos: Dictionary mapping node IDs to positions
        current_node_id: ID of the currently active node (if any)
        last_added_node: ID of the most recently added node (if any)
    """
    node_x, node_y, node_text, node_hover_text = [], [], [], []
    
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Format labels
        raw_label = data['label']
        formatted_label = format_node_label(raw_label)
        node_text.append(formatted_label)
        
        # Create hover text
        hover_label = ' '.join([word.capitalize() for word in raw_label.split('_')])
        hover_text = f"Node {node}: {hover_label}"
        
        # Add features to hover text
        features = data.get('features', {})
        if features:
            feature_text = format_feature_text(features)
            hover_text += f"<br><br>{feature_text}"
            
        node_hover_text.append(hover_text)
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=NODE_BASE_SIZE,
            color=NODE_BACKGROUND,
            line=dict(width=3, color=NODE_BORDER)
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=NODE_FONT_SIZE, color=NODE_FONT_COLOR),
        hovertext=node_hover_text,
        hoverinfo='text',
        showlegend=False
    ))


class GraphDisplay:
    """Component for displaying the graph visualization.
    
    This component manages the creation of graph figures and handles
    styling and interaction for the graph visualization.
    
    Attributes:
        _cached_positions: Dictionary mapping graph hash to node positions
        _base_figure: Cached base figure without highlights
        _last_graph_hash: Hash of the last processed graph
        _node_trace_indices: Dictionary mapping node IDs to their trace indices
        _edge_trace_indices: Dictionary mapping edge tuples to their trace indices
        _current_positions: Dictionary mapping node IDs to their current positions
    """
    
    def __init__(self):
        """Initialize the graph display component."""
        self._cached_positions = {}
        self._base_figure = None
        self._last_graph_hash = None
        self._node_trace_indices = {}
        self._edge_trace_indices = {}
        self._current_positions = {}
    
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
    
    def _create_base_figure(self, G: nx.DiGraph, pos: Dict) -> go.Figure:
        """Create the base figure without any highlights.
        
        Args:
            G: NetworkX directed graph
            pos: Dictionary mapping node IDs to positions
            
        Returns:
            Plotly figure with base graph visualization
        """
        fig = go.Figure()
        
        # Add edges first (so they appear behind nodes)
        add_edges_to_figure(fig, G, pos, None, MAX_EDGE_HOVER_POINTS)
        # Add nodes
        add_nodes_to_figure(fig, G, pos, None, None)
        
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
    
    def create_figure(self, G: nx.DiGraph, current_node_id: Optional[Any], 
                     last_added_node: Optional[Any], last_added_edge: Optional[Tuple]) -> go.Figure:
        """Create a complete graph visualization figure.
        
        Args:
            G: NetworkX directed graph to visualize
            current_node_id: ID of the currently active node (if any)
            last_added_node: ID of the most recently added node (if any)
            last_added_edge: Tuple of (source_id, target_id) for the most recently added edge
            
        Returns:
            Plotly figure with graph visualization
        """
        if len(G.nodes) == 0:
            return create_empty_figure()
        
        # Check if we need to update the base figure
        current_graph_hash = self._get_graph_hash(G)
        if (self._base_figure is None or 
            current_graph_hash != self._last_graph_hash):
            
            # Compute new layout
            self._current_positions = compute_graph_layout(G)
            
            # Create new base figure
            self._base_figure = self._create_base_figure(G, self._current_positions)
            self._last_graph_hash = current_graph_hash
            
            # Update trace indices
            self._node_trace_indices = {
                node: i for i, node in enumerate(G.nodes())
            }
            self._edge_trace_indices = {
                (edge[0], edge[1]): i for i, edge in enumerate(G.edges())
            }
        
        # Create a copy of the base figure for this update
        return go.Figure(self._base_figure)
    
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
    
    def create_card(self) -> dbc.Card:
        """Create a card containing the graph display.
        
        Returns:
            Dash Bootstrap Card component with graph display
        """
        return dbc.Card([
            dbc.CardHeader("Graph Visualization"),
            dbc.CardBody(dcc.Graph(id="graph-display"))
        ], className="shadow-sm h-100 w-100") 