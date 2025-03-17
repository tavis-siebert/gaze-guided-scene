import logging
import os
import sys
from pathlib import Path

def setup_logger(name=None, log_level=None, log_file=None):
    """
    Set up and configure the logger.
    
    Args:
        name (str, optional): Logger name. If None, returns the root logger.
        log_level (str, optional): Logging level. Defaults to INFO if not specified.
        log_file (str, optional): Path to log file. If None, logs to console only.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get the logger
    logger = logging.getLogger(name)
    
    # If handlers are already configured, return the logger
    if logger.handlers:
        return logger
    
    # Set log level from environment variable or parameter
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logger.setLevel(numeric_level)
    
    # Prevent propagation to the root logger to avoid duplicate logs
    if name is not None:
        logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        # Create directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a default root logger
root_logger = setup_logger()

# Convenience functions to get loggers for different modules
def get_logger(name=None):
    """Get a logger with the specified name."""
    if name is None:
        return root_logger
    return setup_logger(name) 