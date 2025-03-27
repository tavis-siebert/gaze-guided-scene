"""Graph display component for visualizing the scene graph."""
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc
from svg.path import parse_path
import numpy as np

from graph.dashboard.graph_constants import NODE_BACKGROUND, NODE_BORDER
from graph.dashboard.utils import format_node_label, format_feature_text, generate_intermediate_points
from logger import get_logger

logger = get_logger(__name__)


# Precompute SVG path points for the diagram-project icon
def _precompute_svg_points():
    """Precompute the normalized points for the SVG diagram-project icon.
    
    Returns:
        Tuple of x and y normalized coordinates for the SVG path
    """
    # SVG path data for diagram-project icon (FontAwesome diagram-project)
    path_data = "M0 80C0 53.5 21.5 32 48 32l96 0c26.5 0 48 21.5 48 48l0 16 192 0 0-16c0-26.5 21.5-48 48-48l96 0c26.5 0 48 21.5 48 48l0 96c0 26.5-21.5 48-48 48l-96 0c-26.5 0-48-21.5-48-48l0-16-192 0 0 16c0 1.7-.1 3.4-.3 5L272 288l96 0c26.5 0 48 21.5 48 48l0 96c0 26.5-21.5 48-48 48l-96 0c-26.5 0-48-21.5-48-48l0-96c0-1.7 .1-3.4 .3-5L144 224l-96 0c-26.5 0-48-21.5-48-48L0 80z"
    
    # Parse and sample points from SVG path
    path = parse_path(path_data)
    n_samples = 250
    points = np.array([(path.point(i/n_samples).real, path.point(i/n_samples).imag) for i in range(n_samples)])
    
    # Normalize to fit in center 10% of figure
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    x_norm = 0.45 + 0.1 * (points[:, 0] - min_coords[0]) / (max_coords[0] - min_coords[0])
    y_norm = 0.45 + 0.1 * (points[:, 1] - min_coords[1]) / (max_coords[1] - min_coords[1])
    
    return x_norm, y_norm

# Precompute the SVG path points at module initialization
ICON_X_POINTS, ICON_Y_POINTS = _precompute_svg_points()


def get_angle_symbol(angle_degrees: float) -> str:
    """Map angle degrees to corresponding arrow symbol.
    
    Args:
        angle_degrees: Angle in degrees
        
    Returns:
        Arrow symbol representing the angle bin
    """
    symbols = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
    bin_size = 360 / len(symbols)
    bin_index = int((angle_degrees % 360) / bin_size)
    return symbols[bin_index]


class GraphDisplay:
    """Component for displaying the graph visualization.
    
    This component manages the creation of graph figures and handles
    styling and interaction for the graph visualization.
    
    Attributes:
        edge_hover_points: Number of hover points to generate per edge
    """
    
    def __init__(self, edge_hover_points: int = 20):
        """Initialize the graph display component.
        
        Args:
            edge_hover_points: Number of hover points to generate per edge for better interaction
        """
        self.edge_hover_points = edge_hover_points
    
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
        fig = go.Figure()
        
        if len(G.nodes) == 0:
            return self._create_empty_figure()
            
        # Use Kamada-Kawai layout for nice node positioning
        pos = nx.kamada_kawai_layout(G)
        
        self._add_edges_to_figure(fig, G, pos, last_added_edge)
        self._add_nodes_to_figure(fig, G, pos, current_node_id, last_added_node)
        
        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
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
    
    def _create_empty_figure(self) -> go.Figure:
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
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    def _add_edges_to_figure(self, fig: go.Figure, G: nx.DiGraph, 
                            pos: Dict, last_added_edge: Optional[Tuple]) -> None:
        """Add edges to the graph figure.
        
        Args:
            fig: The Plotly figure to add edges to
            G: NetworkX directed graph
            pos: Dictionary mapping node IDs to positions
            last_added_edge: Tuple of (source_id, target_id) for the most recently added edge
        """
        regular_edge_x, regular_edge_y = [], []
        last_edge_x, last_edge_y = [], []
        edge_middle_x, edge_middle_y, edge_hover_texts = [], [], []
        edge_labels_x, edge_labels_y, edge_labels_text = [], [], []
        
        for edge in G.edges(data=True):
            source, target = edge[0], edge[1]
            edge_data = edge[2]
            
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_type = edge_data.get('edge_type', 'unknown')
            features = edge_data.get('features', {})
            
            # Create hover text with edge information
            edge_info = f"Edge: {source} → {target}<br>Type: {edge_type}"
            if features:
                feature_text = format_feature_text(features)
                edge_info += f"<br><br>{feature_text}"
            
            # Add the main edge lines - highlight the last added edge
            if (source, target) == last_added_edge:
                last_edge_x.extend([x0, x1, None])
                last_edge_y.extend([y0, y1, None])
            else:
                regular_edge_x.extend([x0, x1, None])
                regular_edge_y.extend([y0, y1, None])
            
            # Add intermediate points for better hover detection
            middle_x, middle_y = generate_intermediate_points(
                x0, x1, y0, y1, self.edge_hover_points
            )
            edge_middle_x.extend(middle_x)
            edge_middle_y.extend(middle_y)
            edge_hover_texts.extend([edge_info] * len(middle_x))
            
            # Add angle label if angle_degrees feature exists
            if 'angle_degrees' in features:
                angle = features['angle_degrees']
                symbol = get_angle_symbol(angle)
                # Position label at the middle of the edge
                label_x = (x0 + x1) / 2
                label_y = (y0 + y1) / 2
                edge_labels_x.append(label_x)
                edge_labels_y.append(label_y)
                edge_labels_text.append(symbol)
        
        # Add regular edges
        if regular_edge_x:
            fig.add_trace(go.Scatter(
                x=regular_edge_x, y=regular_edge_y,
                mode='lines',
                line=dict(width=2.5, color='#888'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add last added edge (highlighted)
        if last_edge_x:
            fig.add_trace(go.Scatter(
                x=last_edge_x, y=last_edge_y,
                mode='lines',
                line=dict(width=4, color='blue'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add hover points along edges
        if edge_middle_x:
            fig.add_trace(go.Scatter(
                x=edge_middle_x, y=edge_middle_y,
                mode='markers',
                marker=dict(size=6, opacity=0.1),
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
                textfont=dict(size=18, color='#000'),
                hoverinfo='none',
                showlegend=False
            ))
    
    def _add_nodes_to_figure(self, fig: go.Figure, G: nx.DiGraph, pos: Dict, 
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
        node_colors, node_border_colors = [], []
        
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Format the node label
            raw_label = data['label']
            formatted_label = format_node_label(raw_label)
            node_text.append(formatted_label)
            
            # Create hover text with node info and features
            hover_label = ' '.join([word.capitalize() for word in raw_label.split('_')])
            hover_text = f"Node {node}: {hover_label}"
            
            # Add features to hover text if available
            features = data.get('features', {})
            if features:
                feature_text = format_feature_text(features)
                hover_text += f"<br><br>{feature_text}"
                
            node_hover_text.append(hover_text)
            
            # Determine node styling based on state
            is_current = node == current_node_id
            is_last_added = node == last_added_node
            
            node_border_colors.append(
                NODE_BORDER["current"] if is_current else NODE_BORDER["default"]
            )
            node_colors.append(
                NODE_BACKGROUND["last_added"] if is_last_added else NODE_BACKGROUND["default"]
            )
        
        # Add the nodes
        base_size = 60
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=base_size,
                color=node_colors,
                line=dict(width=3, color=node_border_colors)
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=11, color='white'),
            hovertext=node_hover_text,
            hoverinfo='text',
            showlegend=False
        ))
    
    def create_card(self) -> dbc.Card:
        """Create a card containing the graph display.
        
        Returns:
            Dash Bootstrap Card component with graph display
        """
        return dbc.Card([
            dbc.CardHeader("Graph Visualization"),
            dbc.CardBody(dcc.Graph(id="graph-display"))
        ], className="shadow-sm h-100 w-100") 