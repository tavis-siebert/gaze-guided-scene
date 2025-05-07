"""Snapshot component for displaying action predictions."""
from typing import Dict, Any, List, Optional
import csv
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import dash

from gaze_guided_scene.graph.dashboard.playback import Playback


class Snapshot:
    """Component for displaying action prediction snapshots."""
    
    def __init__(self, playback: Playback, action_mapping_path: str):
        """Initialize the snapshot component.
        
        Args:
            playback: Playback instance for accessing trace events
            action_mapping_path: Path to the action mapping CSV file
        """
        self.component_id = "snapshot"
        self.playback = playback
        self._load_action_mapping(action_mapping_path)
        
    def _load_action_mapping(self, action_mapping_path: str) -> None:
        """Load action mapping from a CSV file.
        
        Args:
            action_mapping_path: Path to the action mapping CSV file
        """
        self.action_mapping = {}
        
        try:
            # Try to load the CSV file
            with open(action_mapping_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # The ego_topo_action_id is the identifier
                    ego_topo_id = int(row.get("ego_topo_action_id", -1))
                    verb_name = row.get("verb_name", "")
                    noun_name = row.get("noun_name", "")
                    
                    # Use the action_description column if available, otherwise construct it
                    description = row.get("action_description")
                    if not description and verb_name and noun_name:
                        description = f"{verb_name} {noun_name}"
                        
                    if ego_topo_id >= 0 and description:
                        self.action_mapping[ego_topo_id] = description
        except Exception as e:
            print(f"Error loading action mapping from {action_mapping_path}: {e}")
            # Create an empty mapping if file can't be loaded
            self.action_mapping = {}
    
    def create_layout(self) -> dbc.Card:
        """Create the layout for the snapshot component.
        
        Returns:
            Dash Bootstrap Card component with the snapshot layout
        """
        return dbc.Card([
            dbc.CardHeader("Checkpoint", className="fw-bold"),
            dbc.CardBody([
                html.Div(id=f"{self.component_id}-content", className="snapshot-content")
            ], className="p-2")
        ], className="h-100")
    
    def get_snapshot_content(self, frame_number: int) -> List[html.Div]:
        """Get the snapshot content for a specific frame.
        
        Args:
            frame_number: The frame number to get the snapshot for
            
        Returns:
            List of Dash HTML components for displaying snapshot data
        """
        checkpoint_event = self.playback.get_checkpoint(frame_number)
        
        if not checkpoint_event:
            return [html.Div("No checkpoint data available for this frame.", className="text-muted")]
        
        action_labels = checkpoint_event.data.get("action_labels", {})
        next_action_id = action_labels.get("next_action")
        future_actions_ordered = action_labels.get("future_actions_ordered", [])
        
        # Convert to human-readable descriptions
        next_action_text = self.action_mapping.get(next_action_id, f"Unknown Action ({next_action_id})")
        LIMIT = 5
        future_actions_text = [
            self.action_mapping.get(action_id, f"Unknown Action ({action_id})")
            for action_id in future_actions_ordered[:LIMIT]
        ]
        
        content = [
            html.Div([
                html.Strong(f"Next {LIMIT} Actions:"),
                html.Ol([
                    html.Li(action_text) for action_text in future_actions_text
                ], className="mb-0 ps-4 mt-1")
            ]),
            
            html.Div([
                html.Strong("Graph Stats: "),
                html.Span(f"{checkpoint_event.data.get('node_count', 0)} nodes, {checkpoint_event.data.get('edge_count', 0)} edges", className="ms-1")
            ], className="mt-2 text-muted small")
        ]
        
        return content
    
    def register_callbacks(self, app: dash.Dash) -> None:
        """Register callbacks for the snapshot component.
        
        Args:
            app: Dash application instance
        """
        @app.callback(
            dash.dependencies.Output(f"{self.component_id}-content", "children"),
            [dash.dependencies.Input("frame-state", "children")]
        )
        def update_snapshot(frame_state):
            if frame_state is None:
                return [html.Div("No frame data available.", className="text-muted")]
            
            try:
                frame_number = int(frame_state)
                return self.get_snapshot_content(frame_number)
            except (ValueError, TypeError):
                return [html.Div("Invalid frame data.", className="text-muted")] 