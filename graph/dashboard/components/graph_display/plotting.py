"""Helper functions for plotting graphs using Plotly."""
from typing import Dict, List
import networkx as nx
import plotly.graph_objects as go

from graph.dashboard.utils.constants import (
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
from graph.dashboard.utils.svg import ICON_X_POINTS, ICON_Y_POINTS
from logger import get_logger

logger = get_logger(__name__)


def create_empty_graph() -> go.Figure:
    """Create an empty graph visualization with placeholder message.
    
    Returns:
        Plotly figure with empty graph visualization
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ICON_X_POINTS, y=ICON_Y_POINTS,
        mode='lines',
        line=dict(width=1, color='#555'),
        fill='toself',
        fillcolor='#666',
        hoverinfo='none',
        showlegend=False
    ))
    
    fig.add_annotation(
        text="Empty Graph",
        xref="paper", yref="paper",
        x=0.5, y=0.4,
        showarrow=False,
        font=dict(size=16, color="#444"),
        align="center"
    )
    
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
                       pos: Dict, max_edge_hover_points: int) -> None:
    """Add edges to the graph figure.
    
    Args:
        fig: The Plotly figure to add edges to
        G: NetworkX directed graph
        pos: Dictionary mapping node IDs to positions
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
        
        edge_info = f"Edge: {source} → {target}<br>Type: {edge_type}"
        if features:
            feature_text = format_feature_text(features)
            edge_info += f"<br><br>{feature_text}"
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        if len(G.edges()) <= max_edge_hover_points:
            middle_x, middle_y = generate_intermediate_points(x0, x1, y0, y1, qty=5)
            edge_middle_x.extend(middle_x)
            edge_middle_y.extend(middle_y)
            edge_hover_texts.extend([edge_info] * len(middle_x))
        
        if 'angle_degrees' in features:
            angle = features['angle_degrees']
            symbol = get_angle_symbol(angle)
            label_x = (x0 + x1) / 2
            label_y = (y0 + y1) / 2
            edge_labels_x.append(label_x)
            edge_labels_y.append(label_y)
            edge_labels_text.append(symbol)
    
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=EDGE_WIDTH, color=EDGE_COLOR),
            hoverinfo='none',
            showlegend=False
        ))
    
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


def add_nodes_to_figure(fig: go.Figure, G: nx.DiGraph, pos: Dict) -> None:
    """Add nodes to the graph figure.
    
    Args:
        fig: The Plotly figure to add nodes to
        G: NetworkX directed graph
        pos: Dictionary mapping node IDs to positions
    """
    node_x, node_y, node_text, node_hover_text = [], [], [], []
    
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        raw_label = data['label']
        formatted_label = format_node_label(raw_label)
        node_text.append(formatted_label)
        
        hover_label = ' '.join([word.capitalize() for word in raw_label.split('_')])
        hover_text = f"Node {node}: {hover_label}"
        
        features = data.get('features', {})
        if features:
            feature_text = format_feature_text(features)
            hover_text += f"<br><br>{feature_text}"
            
        node_hover_text.append(hover_text)
    
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