"""Utility functions for graph visualization components."""

from typing import Dict, Any, List, Tuple
from collections import deque


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


def format_node_info(node: Any, prev_obj: str, theta: Any) -> Dict[str, Any]:
    """Format node information for display.

    Args:
        node: The node object
        prev_obj: The previous object label
        theta: The angle from the previous node

    Returns:
        Dictionary with formatted node information
    """
    return {
        "object": getattr(node, "object_label", str(node)),
        "from": prev_obj,
        "angle": theta,
    }


def format_label(label: str, line_break: bool = False) -> str:
    """Format a label for display with consistent capitalization.

    Args:
        label: The raw label (e.g., 'coffee_cup')
        line_break: If True, insert line breaks between words for multi-line display
                    If False, join words with spaces for single-line display

    Returns:
        Formatted label with capitalized words
    """
    words = label.split("_")
    capitalized_words = [word.capitalize() for word in words]

    if line_break:
        return "<br>".join(capitalized_words)
    else:
        return " ".join(capitalized_words)


def format_node_label(label: str) -> str:
    """Format a node label for display.

    Args:
        label: The raw node label

    Returns:
        Formatted label with capitalized words separated by line breaks
    """
    return format_label(label, line_break=True)


def format_feature_text(features: Dict[str, Any]) -> str:
    """Format feature dictionary as HTML text.

    Args:
        features: Dictionary of feature key-value pairs

    Returns:
        HTML-formatted string with feature information
    """
    if not features:
        return ""

    text_parts = []
    for key, value in features.items():
        if isinstance(value, (int, float)):
            value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
        else:
            value_str = str(value)

        # Convert snake_case to Title Case
        key_parts = key.split("_")
        key_label = " ".join(part.capitalize() for part in key_parts)

        text_parts.append(f"{key_label}: {value_str}")

    return "<br>".join(text_parts)


def generate_intermediate_points(
    x0: float, x1: float, y0: float, y1: float, qty: int
) -> Tuple[List[float], List[float]]:
    """Generate intermediate points along an edge for better hover detection.

    Args:
        x0: x-coordinate of starting point
        x1: x-coordinate of ending point
        y0: y-coordinate of starting point
        y1: y-coordinate of ending point
        qty: Number of intermediate points to generate

    Returns:
        Tuple of (x_coordinates, y_coordinates) lists
    """
    middle_x = _generate_equidistant_points(x0, x1, qty + 2)
    middle_y = _generate_equidistant_points(y0, y1, qty + 2)
    # Remove first and last points (they are the nodes)
    middle_x.pop(0)
    middle_x.pop()
    middle_y.pop(0)
    middle_y.pop()
    return middle_x, middle_y


def _generate_equidistant_points(a: float, b: float, qty: int) -> List[float]:
    """Generate a specified number of points between a and b.

    Args:
        a: Starting value
        b: Ending value
        qty: Number of points to generate (including start and end)

    Returns:
        List of equidistant points from a to b
    """
    q = deque()
    q.append((0, qty - 1))  # indexing starts at 0
    pts = [0] * qty
    pts[0] = a
    pts[-1] = b  # first value is a, last is b
    while q:
        left, right = q.popleft()  # remove working segment from queue
        center = (left + right + 1) // 2  # creates index values for pts
        pts[center] = (pts[left] + pts[right]) / 2
        if right - left > 2:  # stop when qty met
            q.append((left, center))
            q.append((center, right))
    return pts
