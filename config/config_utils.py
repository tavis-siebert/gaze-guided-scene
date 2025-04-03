import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
import getpass
from dataclasses import dataclass

# Get absolute path to repository root directory (parent of config directory)
REPO_ROOT = str(Path(__file__).parent.parent.absolute())

class DotDict:
    """Dictionary subclass that enables dot notation access to nested dictionaries."""
    def __init__(self, dictionary: Dict[str, Any]):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to regular dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def resolve_env_vars(value: str) -> str:
    """Resolve environment variables and special variables in string values."""
    if isinstance(value, str):
        # First replace ${USER} with actual username since some environments might not have USER set
        value = value.replace("${USER}", getpass.getuser())
        # Replace ${REPO_ROOT} with the absolute path to repository root
        value = value.replace("${REPO_ROOT}", REPO_ROOT)
        # Then handle any other environment variables
        return os.path.expandvars(value)
    return value

def extract_reference(value: str) -> Tuple[str, str, str]:
    """Extract reference path and surrounding text from a string containing ${...}."""
    start = value.find("${") + 2
    end = value.find("}")
    if end == -1:
        raise ValueError(f"Malformed reference in config: {value} - missing closing brace")
    
    prefix = value[:start-2]
    ref_path = value[start:end]
    suffix = value[end+1:]
    return prefix, ref_path, suffix

def get_reference_value(ref_path: str, full_config: Dict[str, Any]) -> Any:
    """Get the value from the config using a dot-notation reference path."""
    if "." not in ref_path:
        return None  # Skip if it looks like an unresolved env var
        
    ref_value = full_config
    for part in ref_path.split("."):
        if not isinstance(ref_value, dict):
            raise ValueError(f"Invalid reference path '{ref_path}' - '{part}' is not a dictionary")
        if part not in ref_value:
            raise KeyError(f"Reference '{ref_path}' not found - '{part}' does not exist")
        ref_value = ref_value[part]
    return ref_value

def resolve_string_value(value: str, full_config: Dict[str, Any]) -> str:
    """Resolve all references in a string value until no more ${...} patterns exist."""
    while "${" in value:
        # First check if it's an environment variable
        resolved = resolve_env_vars(value)
        if resolved != value:
            value = resolved
            continue
            
        try:
            prefix, ref_path, suffix = extract_reference(value)
            ref_value = get_reference_value(ref_path, full_config)
            
            if ref_value is None:  # Unresolved env var
                break
                
            value = f"{prefix}{ref_value}{suffix}"
            # Resolve any environment variables in the result
            value = resolve_env_vars(value)
        except Exception as e:
            raise ValueError(f"Error resolving reference in '{value}': {str(e)}")
    
    return value

def resolve_references(config: Dict[str, Any], full_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Recursively resolve references in the config using ${path.to.value} syntax."""
    if full_config is None:
        full_config = config

    resolved_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved_config[key] = resolve_references(value, full_config)
        elif isinstance(value, list):
            resolved_config[key] = [
                resolve_references(item, full_config) if isinstance(item, dict)
                else resolve_string_value(item, full_config) if isinstance(item, str)
                else item
                for item in value
            ]
        elif isinstance(value, str):
            resolved_config[key] = resolve_string_value(value, full_config)
        else:
            resolved_config[key] = value
    
    return resolved_config

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override values taking precedence."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result

def load_config(config_path: Optional[str] = None) -> DotDict:
    """Load configuration from YAML file and resolve all references.
    
    Supports extending from a base config using the 'extends' key.
    If no config_path is provided, the default config will be loaded.
    
    Args:
        config_path: Path to the config file to load. If None, loads default config.
        
    Returns:
        DotDict: Configuration object that supports both dictionary and dot notation access
    """
    if config_path is None:
        # Use student_cluster_config.yaml as fallback if no path is specified
        config_path = Path(__file__).parent / "student_cluster_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle extends key
    if 'extends' in config:
        base_config_path = config.pop('extends')
        # Resolve relative paths
        if not os.path.isabs(base_config_path):
            base_config_path = os.path.join(os.path.dirname(config_path), base_config_path)
        
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configs with current config overriding base
        config = deep_merge(base_config, config)

    resolved_config = resolve_references(config)
    return DotDict(resolved_config)