# Graph Dashboard Components

This directory contains modular dashboard components for the graph visualization, which follows a component-based architecture using Plotly Dash.

## Component Architecture

The visualization follows a component-based approach that separates concerns:

- **Dashboard** - Main component that integrates all others and manages the Dash application
- **GraphPlayback** - Handles trace file loading and graph state management
- **VideoDisplay** - Manages video frame loading, caching, and overlay visualization
- **GraphDisplay** - Handles graph visualization and interaction
- **DetectionDisplay** - Creates the object detection panel with statistics
- **PlaybackControls** - Provides controls for playback navigation

## Design Patterns

The implementation uses several design patterns:

1. **Component Pattern** - Each component encapsulates its own functionality and UI
2. **Composition Pattern** - The Dashboard composes other components rather than inheriting
3. **Callback Pattern** - Components register their own callbacks with the Dash app
4. **Factory Pattern** - Components provide factory methods for creating layouts
5. **State Management** - Application state is stored using dcc.Store and propagated via callbacks

## File Structure

- `__init__.py` - Exposes component classes
- `graph_constants.py` - Shared constants and styling information
- `utils.py` - Helper functions used across components
- `graph_event.py` - Data class for events in the trace file
- `graph_playback.py` - Graph construction and event management
- `video_display.py` - Video frame display and gaze overlay
- `graph_display.py` - Graph visualization with nodes and edges
- `detection_display.py` - Object detection information panel
- `playback_controls.py` - Frame navigation controls
- `dashboard.py` - Main component integrating all others

## Usage

The Dashboard component is the main entry point and can be used as follows:

```python
from graph.dashboard.dashboard import Dashboard

# Create and run the dashboard
dashboard = Dashboard("path/to/trace_file.jsonl", "path/to/video.mp4")
dashboard.run(port=8050, debug=False)
```

Alternatively, use the simpler API in visualizer.py:

```python
from graph.visualizer import visualize_graph_construction

visualize_graph_construction("path/to/trace_file.jsonl", "path/to/video.mp4")
``` 