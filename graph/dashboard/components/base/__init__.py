"""Base component class for dashboard components."""
from dash import html
import dash_bootstrap_components as dbc


class BaseComponent(html.Div):
    """Base class for all dashboard components.
    
    This class provides common functionality and structure for dashboard components.
    Components should inherit from this class and implement their specific layout
    in the create_layout method.
    
    Attributes:
        component_id: Unique identifier for the component
    """
    
    def __init__(self, component_id: str, **kwargs):
        """Initialize the base component.
        
        Args:
            component_id: Unique identifier for the component
            **kwargs: Additional arguments to pass to html.Div
        """
        self.component_id = component_id
        super().__init__(**kwargs)
    
    def create_layout(self) -> html.Div:
        """Create the component's layout.
        
        This method should be implemented by all child components to define their
        specific layout structure.
        
        Returns:
            Dash component defining the component's layout
        """
        raise NotImplementedError("Child components must implement create_layout") 