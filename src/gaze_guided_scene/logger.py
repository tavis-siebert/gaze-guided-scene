import logging
import os
import sys
from pathlib import Path

# Global variables to track configuration state
_log_file = None
_formatter = None
_configured = False

def configure_root_logger(log_level=None, log_file=None):
    """
    Configure the root logger with console and optional file handlers.
    
    Args:
        log_level (str, optional): Logging level. Defaults to INFO if not specified.
        log_file (str, optional): Path to log file. If None, logs to console only.
    """
    global _log_file, _formatter, _configured
    
    if _configured:
        return
    
    # Set log level from environment variable or parameter
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    _formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        _log_file = log_file
        log_path = Path(log_file)
        # Create directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(_formatter)
        root_logger.addHandler(file_handler)
    
    _configured = True

def get_logger(name=None):
    """
    Get a logger with the specified name.
    
    Args:
        name (str, optional): Logger name. If None, returns the root logger.
        
    Returns:
        logging.Logger: Logger instance
    """
    if not _configured:
        configure_root_logger()
    
    if name is None:
        return logging.getLogger()
    
    return logging.getLogger(name)