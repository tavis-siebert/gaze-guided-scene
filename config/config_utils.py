import os
import yaml
from pathlib import Path
from typing import Dict, Any
import getpass
from dataclasses import dataclass

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
    """Resolve environment variables in string values."""
    if isinstance(value, str):
        # First replace ${USER} with actual username since some environments might not have USER set
        value = value.replace("${USER}", getpass.getuser())
        # Then handle any other environment variables
        return os.path.expandvars(value)
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
            resolved_config[key] = [resolve_references(item, full_config) 
                                  if isinstance(item, dict) 
                                  else resolve_env_vars(item) 
                                  for item in value]
        elif isinstance(value, str) and "${" in value:
            try:
                # First try to resolve as environment variable
                resolved_value = resolve_env_vars(value)
                if resolved_value != value:  # If env var was resolved
                    resolved_config[key] = resolved_value
                    continue
                
                # If not an env var, try to resolve as config reference
                start = value.find("${") + 2
                end = value.find("}")
                if end == -1:
                    raise ValueError(f"Malformed reference in config: {value} - missing closing brace")
                
                ref_path = value[start:end]
                # Skip if it looks like an unresolved env var
                if "." not in ref_path:
                    resolved_config[key] = value
                    continue
                    
                parts = ref_path.split(".")
                
                # Get the referenced value
                ref_value = full_config
                for part in parts:
                    if not isinstance(ref_value, dict):
                        raise ValueError(f"Invalid reference path '{ref_path}' - '{part}' is not a dictionary")
                    if part not in ref_value:
                        raise KeyError(f"Reference '{ref_path}' not found - '{part}' does not exist")
                    ref_value = ref_value[part]
                
                # Replace the reference in the original string
                prefix = value[:start-2]
                suffix = value[end+1:]
                resolved_value = f"{prefix}{ref_value}{suffix}"
                resolved_config[key] = resolve_env_vars(resolved_value)
            except Exception as e:
                raise ValueError(f"Error resolving reference in '{key}': {str(e)}")
        else:
            resolved_config[key] = resolve_env_vars(value)
    
    return resolved_config

def load_config(config_path: str = None) -> DotDict:
    """Load configuration from YAML file and resolve all references.
    
    Returns:
        DotDict: Configuration object that supports both dictionary and dot notation access
    """
    if config_path is None:
        config_path = Path(__file__).parent / "default_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    resolved_config = resolve_references(config)
    return DotDict(resolved_config) 