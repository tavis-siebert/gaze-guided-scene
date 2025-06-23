#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from gazegraph.config.config_utils import load_config
from gazegraph.logger import get_logger, configure_root_logger
import multiprocessing as mp

logger = None

mp.set_start_method("spawn", force=True)

def setup_parser() -> argparse.ArgumentParser:
    load_dotenv()

    package_dir = Path(__file__).parent
    default_config_path = os.environ.get(
        "CONFIG_PATH", str(package_dir / "config" / "student_cluster_config.yaml")
    )

    parser = argparse.ArgumentParser(
        description="Gaze-guided scene understanding toolkit"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="Path to custom config file. Defaults to path in CONFIG_PATH env var or config/student_cluster_config.yaml",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
        default="INFO",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file. If not specified, logs to console only",
        default="logs/main.log",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup-scratch", help="Setup scratch directories and download required files"
    )
    setup_parser.add_argument(
        "--dropbox-token",
        type=str,
        help="Dropbox access token (defaults to DROPBOX_TOKEN env var)",
    )

    # Graph building command
    build_parser = subparsers.add_parser(
        "build-graphs", help="Build scene graphs from videos"
    )
    build_parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Device to use for processing (default: gpu)",
    )
    build_parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        help="Specific video names to process (e.g., OP01-R04-ContinentalBreakfast). If not specified, all videos will be processed.",
    )
    build_parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Enable graph construction tracing for visualization",
    )
    build_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing graph checkpoints"
    )

    # Training command
    train_parser = subparsers.add_parser(
        "train", help="Train a GNN on a specified task"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Device to use for processing (default: gpu)",
    )
    train_parser.add_argument(
        "--task",
        type=str,
        choices=[
            "future_actions",
            "next_action",
            "action_recognition",
            "object_recognition",
        ],
        required=True,
        help="Task to train the model on",
    )
    train_parser.add_argument(
        "--graph-type", 
        type=str, 
        choices=["object-graph", "action-graph", 'action-object-graph'],
        default="object-graph",
        help="Type of graph dataset to use (default: object-graph)"
    )
    train_parser.add_argument(
        "--object-node-feature",
        type=str,
        choices=["one-hot", "roi-embeddings", "object-label-embeddings"],
        default="object-label-embeddings",
        help="Type of object node features to use (default: one-hot)",
    )
    train_parser.add_argument(
        "--action-node-feature",
        type=str,
        choices=["action-one-hot", "action-label-embedding"],
        default="action-label-embedding",
        help="Type of action node features to use (default: action-label-embedding)",
    )
    train_parser.add_argument(
        "--load-cached",
        action="store_true",
        help="Load cached GraphDataset from files in data/datasets/",
    )


    # Visualization command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize graph construction process"
    )
    visualize_parser.add_argument(
        "--video-name",
        type=str,
        help="Name of the video to process (used to locate trace file if trace-path not provided)",
    )
    visualize_parser.add_argument(
        "--video-path", type=str, help="Path to the video file"
    )
    visualize_parser.add_argument(
        "--trace-path", type=str, help="Path to the trace file"
    )
    visualize_parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the server on"
    )
    visualize_parser.add_argument(
        "--debug", action="store_true", help="Whether to run in debug mode"
    )

    # label-fixations command
    label_parser = subparsers.add_parser(
        "label-fixations", help="Run YOLO-World to label fixations in videos."
    )
    label_parser.add_argument(
        "--in-pkl", type=str, help="Path to base features pickle file (e.g., from LFB)."
    )
    label_parser.add_argument(
        "--out-dir", type=str, help="Directory to save fixation label and ROI maps."
    )
    label_parser.add_argument(
        "--skip-roi", action="store_true", help="Disable CLIP/ROI vector computation to run faster."
    )

    # compose-features command
    compose_parser = subparsers.add_parser(
        "compose-features", help="Compose final feature vectors from multiple sources."
    )
    compose_parser.add_argument(
        "--base-pkl", type=str, help="Path to base 2048-D backbone features pickle."
    )
    compose_parser.add_argument(
        "--fix-pkl", type=str, help="Path to fixation_label_map.pkl from label-fixations."
    )
    compose_parser.add_argument(
        "--out-pkl", type=str, help="Path to save the final composed features pickle."
    )
    compose_parser.add_argument(
        "--mode", required=True, 
        choices=["onehot", "clip", "roi", "text+roi", "combo", "combo+roi"],
        help="Feature composition mode."
    )
    compose_parser.add_argument(
        "--clip-cache", type=str, help="Path to pre-computed CLIP text embedding cache."
    )

    return parser


def get_dropbox_token(args: argparse.Namespace) -> str:
    """Get Dropbox token from command line args or environment variable."""
    # Environment variables are already loaded in setup_parser

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


def check_gpu_availability(device_choice):
    """
    Check if GPU is available when requested and log a warning if not.

    Args:
        device_choice (str): The device choice ('gpu' or 'cpu')

    Returns:
        bool: True if GPU is requested and available, False otherwise
    """
    if device_choice.lower() != "gpu":
        return False

    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning(
                "GPU was requested but CUDA is not available. Falling back to CPU."
            )
            return False
        return True
    except ImportError:
        logger.warning(
            "GPU was requested but PyTorch is not installed. Falling back to CPU."
        )
        return False


def main():
    global logger
    parser = setup_parser()
    args = parser.parse_args()

    # Configure root logger based on command line arguments
    configure_root_logger(log_level=args.log_level, log_file=args.log_file)

    # Get the main logger
    logger = get_logger("main")
    logger.info(
        f"Logging configured with level: {args.log_level}, file: {args.log_file or 'console only'}"
    )

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    if args.command == "setup-scratch":
        from gazegraph.setup_scratch import setup_scratch

        dropbox_token = get_dropbox_token(args)
        logger.info("Starting scratch setup process")
        setup_scratch(config, access_token=dropbox_token)
    elif args.command == "train":
        from gazegraph.training.tasks import get_task

        logger.info(f"Starting training for task: {args.task}")

        # Determine the device to use
        device = "cpu"
        if args.device == "gpu" and check_gpu_availability(args.device):
            device = "cuda"
        logger.info(f"Using device: {device}")

        # Initialize task and start training
        task = None
        try:
            TaskClass = get_task(args.task)
            # Pass object node feature type and graph type to the task
            task = TaskClass(
                config=config,
                device=device,
                task_name=args.task,
                object_node_feature=args.object_node_feature,
                action_node_feature=args.action_node_feature,
                load_cached=args.load_cached,
                graph_type=args.graph_type,
            )
            logger.info(
                f"Starting training process with object node feature type: {args.object_node_feature} and graph type: {args.graph_type}"
            )
            if args.load_cached:
                logger.info("Using cached GraphDataset from files")
            task.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            # Close tensorboard writer if task was initialized
            if task is not None:
                task.close()
                logger.info("TensorBoard writer closed")
    elif args.command == "build-graphs":
        from gazegraph.graph.graph_processor import build_graphs

        logger.info("Starting graph building process")

        # Check GPU availability if requested
        use_gpu = check_gpu_availability(args.device)

        # Build graphs with overwrite flag
        build_graphs(
            config,
            use_gpu=use_gpu,
            videos=args.videos,
            enable_tracing=args.enable_tracing,
            overwrite=args.overwrite,
        )
    elif args.command == "visualize":
        from gazegraph.graph.visualizer import visualize_graph_construction

        logger.info("Starting graph visualization process")

        # Validate and resolve trace file path
        trace_file = None
        if args.trace_path:
            trace_file = Path(args.trace_path)
        elif args.video_name:
            trace_file = (
                Path(config.directories.traces) / f"{args.video_name}_trace.jsonl"
            )
        else:
            logger.error("Either --video-name or --trace-path must be provided")
            sys.exit(1)

        if not trace_file.exists():
            logger.error(f"No trace file found at: {trace_file}")
            if args.video_name:
                logger.error("To generate a trace file, run:")
                logger.error(
                    f"    ./run.sh build-graphs --videos {args.video_name} --enable-tracing"
                )
            sys.exit(1)

        # Validate and resolve video path
        video_path = args.video_path
        if video_path is None:
            if (
                args.video_name
                and hasattr(config, "dataset")
                and hasattr(config.dataset, "egtea")
            ):
                possible_video_path = (
                    Path(config.dataset.egtea.raw_videos) / f"{args.video_name}.mp4"
                )
                if possible_video_path.exists():
                    video_path = str(possible_video_path)
                    logger.info(f"Found video file at {video_path}")
                else:
                    logger.warning(
                        f"Could not find video file at expected location: {possible_video_path}"
                    )
                    logger.warning("Visualization will proceed without video display")
            else:
                # If no video_name is provided, video_path is required
                if not args.video_name:
                    logger.error(
                        "When using --trace-path directly, --video-path must also be provided"
                    )
                    logger.error(
                        "Either specify --video-name (to auto-locate video) or both --trace-path and --video-path"
                    )
                    sys.exit(1)
                else:
                    logger.warning(
                        "No video path provided. Visualization will proceed without video display."
                    )
        elif not Path(video_path).exists():
            logger.error(f"Video file does not exist at: {video_path}")
            sys.exit(1)
        else:
            logger.info(f"Using provided video file at {video_path}")

        # Launch visualization dashboard
        visualize_graph_construction(
            trace_file=str(trace_file),
            video_path=video_path,
            port=args.port,
            debug=args.debug,
            verb_idx_file=config.dataset.egtea.verb_idx_file,
            noun_idx_file=config.dataset.egtea.noun_idx_file,
            train_split_file=config.dataset.ego_topo.splits.train,
            val_split_file=config.dataset.ego_topo.splits.val,
        )
    elif args.command == "label-fixations":
        from gazegraph.processing.label_fixations import run_fixation_labeling
        
        # Use paths from config if not provided
        in_pkl = Path(args.in_pkl or config.directories.features + "/base_features.pkl")
        out_dir = Path(args.out_dir or config.directories.features)

        logger.info(f"Starting fixation labeling from {in_pkl}")
        run_fixation_labeling(config, in_pkl, out_dir, args.skip_roi)

    elif args.command == "compose-features":
        from gazegraph.processing.compose_features import run_feature_composition
        
        # Use paths from config if not provided
        base_pkl = Path(args.base_pkl or config.directories.features + "/base_features.pkl")
        fix_pkl_name = f"{base_pkl.stem}_label_map.pkl"
        fix_pkl = Path(args.fix_pkl or config.directories.features + f"/{fix_pkl_name}")
        out_pkl = Path(args.out_pkl or config.directories.features + f"/composed_features_{args.mode}.pkl")
        clip_cache = Path(args.clip_cache or config.directories.features + "/clip_text_cache.pt")

        logger.info(f"Starting feature composition in '{args.mode}' mode.")
        run_feature_composition(config, base_pkl, fix_pkl, out_pkl, args.mode, clip_cache)


if __name__ == "__main__":
    main()
