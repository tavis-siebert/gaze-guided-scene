"""Utility functions for SVG path processing."""
from typing import Tuple
import numpy as np
from svg.path import parse_path

from gazegraph.graph.dashboard.utils.constants import DIAGRAM_PROJECT_ICON_PATH

def precompute_svg_points(n_samples: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute normalized points for the SVG diagram-project icon.
    
    Args:
        n_samples: Number of points to sample from the SVG path
        
    Returns:
        Tuple of x and y normalized coordinates for the SVG path
    """
    path = parse_path(DIAGRAM_PROJECT_ICON_PATH)
    points = np.array([(path.point(i/n_samples).real, path.point(i/n_samples).imag) 
                      for i in range(n_samples)])
    
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    x_norm = 0.45 + 0.1 * (points[:, 0] - min_coords[0]) / (max_coords[0] - min_coords[0])
    y_norm = 0.45 + 0.1 * (points[:, 1] - min_coords[1]) / (max_coords[1] - min_coords[1])
    
    return x_norm, y_norm

# Precompute the SVG path points at module initialization
ICON_X_POINTS, ICON_Y_POINTS = precompute_svg_points() 