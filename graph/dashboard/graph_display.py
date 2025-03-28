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
        max_angle_nodes: Maximum number of nodes for which to use angle-based initialization
        _cached_positions: Dictionary mapping graph hash to node positions
        _base_figure: Cached base figure without highlights
        _last_graph_hash: Hash of the last processed graph
        _last_current_node: Last processed current node ID
        _last_added_node: Last processed added node ID
        _last_added_edge: Last processed added edge
        _node_trace_indices: Dictionary mapping node IDs to their trace indices
        _edge_trace_indices: Dictionary mapping edge tuples to their trace indices
        _current_positions: Dictionary mapping node IDs to their current positions
    """
    
    def __init__(self, edge_hover_points: int = 20, max_angle_nodes: int = 25):
        """Initialize the graph display component.
        
        Args:
            edge_hover_points: Number of hover points to generate per edge for better interaction
            max_angle_nodes: Maximum number of nodes for which to use angle-based initialization
        """
        self.edge_hover_points = edge_hover_points
        self.max_angle_nodes = max_angle_nodes
        self._cached_positions = {}
        self._base_figure = None
        self._last_graph_hash = None
        self._last_current_node = None
        self._last_added_node = None
        self._last_added_edge = None
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
        # Create a string representation of the graph structure
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
        self._add_edges_to_figure(fig, G, pos, None)
        # Add nodes
        self._add_nodes_to_figure(fig, G, pos, None, None)
        
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
    
    def _update_highlights(self, fig: go.Figure, G: nx.DiGraph,
                          current_node_id: Optional[Any], last_added_node: Optional[Any],
                          last_added_edge: Optional[Tuple]) -> None:
        """Update the highlights in the figure without recreating it.
        
        Args:
            fig: The Plotly figure to update
            G: NetworkX directed graph
            current_node_id: ID of the currently active node (if any)
            last_added_node: ID of the most recently added node (if any)
            last_added_edge: Tuple of (source_id, target_id) for the most recently added edge
        """
        # Update node highlights
        for node, data in G.nodes(data=True):
            is_current = node == current_node_id
            is_last_added = node == last_added_node
            
            # Update node color and border
            node_idx = self._node_trace_indices.get(node)
            if node_idx is not None and node_idx < len(fig.data):
                fig.data[node_idx].marker.color = (
                    NODE_BACKGROUND["last_added"] if is_last_added else NODE_BACKGROUND["default"]
                )
                fig.data[node_idx].marker.line.color = (
                    NODE_BORDER["current"] if is_current else NODE_BORDER["default"]
                )
        
        # Update edge highlights
        for edge in G.edges():
            edge_key = (edge[0], edge[1])
            is_last_added = edge_key == last_added_edge
            
            # Update edge color and width
            edge_idx = self._edge_trace_indices.get(edge_key)
            if edge_idx is not None and edge_idx < len(fig.data):
                fig.data[edge_idx].line.color = 'blue' if is_last_added else '#888'
                fig.data[edge_idx].line.width = 4 if is_last_added else 2.5
    
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
            return self._create_empty_figure()
        
        # Check if we need to update the base figure
        current_graph_hash = self._get_graph_hash(G)
        if (self._base_figure is None or 
            current_graph_hash != self._last_graph_hash):
            
            # Initialize positions based on edge angles only for small graphs
            if len(G.nodes) < self.max_angle_nodes:
                self._current_positions = self._initialize_positions_from_angles(G)
            else:
                self._current_positions = None
            
            try:
                # Try Kamada-Kawai layout with initial positions for optimization
                self._current_positions = nx.kamada_kawai_layout(G, pos=self._current_positions)
            except Exception as e:
                # Fall back to spring layout if Kamada-Kawai fails
                logger.warning(f"Kamada-Kawai layout failed: {e}. Falling back to spring layout.")
                self._current_positions = nx.spring_layout(G, pos=self._current_positions, iterations=50, seed=42)
            
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
        fig = go.Figure(self._base_figure)
        
        # Update highlights if state has changed
        state_changed = (
            current_node_id != self._last_current_node or
            last_added_node != self._last_added_node or
            last_added_edge != self._last_added_edge
        )
        
        if state_changed:
            self._update_highlights(fig, G, current_node_id, last_added_node, last_added_edge)
            self._last_current_node = current_node_id
            self._last_added_node = last_added_node
            self._last_added_edge = last_added_edge
        
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
        
        # Track edge indices for each edge
        edge_indices = {}
        current_index = 0
        
        # First pass: collect all edge data
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
            # Store indices for regular edges
            for edge in G.edges():
                if edge != last_added_edge:
                    edge_indices[edge] = current_index
                    current_index += 1
        
        # Add last added edge (highlighted)
        if last_edge_x:
            fig.add_trace(go.Scatter(
                x=last_edge_x, y=last_edge_y,
                mode='lines',
                line=dict(width=4, color='blue'),
                hoverinfo='none',
                showlegend=False
            ))
            # Store index for last added edge
            if last_added_edge:
                edge_indices[last_added_edge] = current_index
                current_index += 1
        
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
        
        # Store edge indices for later use
        self._edge_trace_indices = edge_indices
    
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
    
    def _initialize_positions_from_angles(self, G: nx.DiGraph) -> Dict:
        """Initialize node positions based on edge angle features.
        
        This creates an initial layout where connected nodes are positioned
        according to their angular relationships, which serves as a starting
        point for the Kamada-Kawai algorithm.
        
        Args:
            G: NetworkX directed graph to lay out
            
        Returns:
            Dictionary mapping node IDs to (x,y) positions
        """
        pos = {}
        processed_nodes = set()
        
        # Start with a node that has outgoing edges with angle information
        start_node = None
        for node in G.nodes():
            if any('angle_degrees' in G[node][target].get('features', {}) 
                   for target in G.successors(node)):
                start_node = node
                break
        
        # If no suitable start node found, fall back to first node
        if start_node is None and G.nodes:
            start_node = list(G.nodes())[0]
        
        if start_node is None:
            return {}
        
        # Create deterministic jitter based on node hash
        def get_jitter(node_id, scale=0.05):
            """Generate deterministic jitter based on node ID."""
            # Use hash of node ID to create consistent jitter values
            node_hash = hash(str(node_id))
            # Use modulo to constrain values within desired range
            jitter_x = ((node_hash % 1000) / 1000.0 - 0.5) * scale
            jitter_y = ((node_hash // 1000 % 1000) / 1000.0 - 0.5) * scale
            return jitter_x, jitter_y
            
        # Place the start node at the center with small deterministic jitter
        jitter_x, jitter_y = get_jitter(start_node, scale=0.02)
        pos[start_node] = (jitter_x, jitter_y)
        processed_nodes.add(start_node)
        
        # Process nodes in breadth-first order to propagate positions
        nodes_to_process = [start_node]
        radius = 1.0  # Distance from center for first layer
        
        while nodes_to_process:
            current_nodes = nodes_to_process.copy()
            nodes_to_process = []
            
            for node in current_nodes:
                # Process outgoing edges with angle information
                for target in G.successors(node):
                    if target in processed_nodes:
                        continue
                    
                    edge_data = G[node][target]
                    angle_degrees = edge_data.get('features', {}).get('angle_degrees')
                    
                    if angle_degrees is not None:
                        # Convert angle to radians (adjust as needed for your angle convention)
                        angle_rad = np.radians(angle_degrees)
                        # Get deterministic jitter for this target node
                        jitter_x, jitter_y = get_jitter(target)
                        # Position target based on angle and radius with deterministic jitter
                        x = pos[node][0] + radius * np.cos(angle_rad) + jitter_x
                        y = pos[node][1] + radius * np.sin(angle_rad) + jitter_y
                        pos[target] = (x, y)
                        processed_nodes.add(target)
                        nodes_to_process.append(target)
                
                # Process incoming edges with angle information
                for source in G.predecessors(node):
                    if source in processed_nodes:
                        continue
                    
                    edge_data = G[source][node]
                    angle_degrees = edge_data.get('features', {}).get('angle_degrees')
                    
                    if angle_degrees is not None:
                        # For incoming edges, use opposite angle
                        angle_rad = np.radians((angle_degrees + 180) % 360)
                        # Get deterministic jitter for this source node
                        jitter_x, jitter_y = get_jitter(source)
                        # Position source based on angle and radius with deterministic jitter
                        x = pos[node][0] + radius * np.cos(angle_rad) + jitter_x
                        y = pos[node][1] + radius * np.sin(angle_rad) + jitter_y
                        pos[source] = (x, y)
                        processed_nodes.add(source)
                        nodes_to_process.append(source)
            
            # Increase radius for next layer to avoid overlaps
            radius += 0.5
        
        # For any remaining nodes without angle information, place them using deterministic positions
        for node in G.nodes():
            if node not in pos:
                # Use deterministic values based on node hash
                node_hash = hash(str(node))
                x = -2 + 4 * ((node_hash % 1000) / 1000.0)
                y = -2 + 4 * ((node_hash // 1000 % 1000) / 1000.0)
                pos[node] = (x, y)
        
        return pos
    
    def create_card(self) -> dbc.Card:
        """Create a card containing the graph display.
        
        Returns:
            Dash Bootstrap Card component with graph display
        """
        return dbc.Card([
            dbc.CardHeader("Graph Visualization"),
            dbc.CardBody(dcc.Graph(id="graph-display"))
        ], className="shadow-sm h-100 w-100") 