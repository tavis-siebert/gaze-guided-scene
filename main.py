#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from config.config_utils import load_config, DotDict
from logger import get_logger, configure_root_logger

logger = None

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gaze-guided scene understanding toolkit")
    parser.add_argument("--config", type=str, default="config/student_cluster_config.yaml",
                       help="Path to custom config file. Defaults to config/student_cluster_config.yaml")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level", default="INFO")
    parser.add_argument("--log-file", type=str,
                       help="Path to log file. If not specified, logs to console only", default="logs/main.log")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Setup command
    setup_parser = subparsers.add_parser("setup-scratch", help="Setup scratch directories and download required files")
    setup_parser.add_argument("--dropbox-token", type=str,
                          help="Dropbox access token (defaults to DROPBOX_TOKEN env var)")
    
    # Dataset building command
    build_parser = subparsers.add_parser("build", help="Build the dataset")
    build_parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="gpu",
                            help="Device to use for processing (default: gpu)")
    build_parser.add_argument("--videos", type=str, nargs="+", 
                            help="Specific video names to process (e.g., OP01-R04-ContinentalBreakfast). If not specified, all videos will be processed.")
    
    return parser

def get_dropbox_token(args: argparse.Namespace) -> str:
    """Get Dropbox token from command line args or environment variable."""
    # Load environment variables from .env file
    load_dotenv()
    
    if hasattr(args, "dropbox_token") and args.dropbox_token:
        return args.dropbox_token
    
    token = os.environ.get("DROPBOX_TOKEN")
    if not token:
        logger.error(
            "Dropbox token not found, which is required for downloading the EGTEA Gaze+ dataset (see README). "
            "Either set DROPBOX_TOKEN in .env file "
            "or provide --dropbox-token argument"
        )
        raise ValueError("Dropbox token not found")
    return token

def main():
    global logger
    parser = setup_parser()
    args = parser.parse_args()
    
    # Configure root logger based on command line arguments
    configure_root_logger(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    # Get the main logger
    logger = get_logger("main")
    logger.info(f"Logging configured with level: {args.log_level}, file: {args.log_file or 'console only'}")
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    if args.command == "setup-scratch":
        from scripts.setup_scratch import setup_scratch
        dropbox_token = get_dropbox_token(args)
        logger.info("Starting scratch setup process")
        setup_scratch(config, access_token=dropbox_token)
    elif args.command == "build":
        from datasets.build_dataset import build_dataset
        logger.info("Starting dataset building process")
        use_gpu = args.device.lower() == "gpu"
        build_dataset(config, use_gpu=use_gpu, videos=args.videos)

if __name__ == "__main__":
    main() 