"""
Utilities for ONNX Runtime configuration and management.
"""

import os
import multiprocessing
from onnxruntime import SessionOptions
from typing import Optional

from gazegraph.logger import get_logger

logger = get_logger(__name__)


def make_session_options(num_workers: Optional[int] = None) -> SessionOptions:
    """
    Create optimized SessionOptions for ONNX Runtime with proper thread allocation.

    Args:
        num_workers: Number of parallel workers that will use ONNX Runtime.
                    If None, uses environment variable NUM_WORKERS or configuration.

    Returns:
        SessionOptions configured with appropriate thread counts
    """
    # Determine number of workers from environment variable or default
    if num_workers is None:
        # First try to get from specific environment variable
        env_workers = os.getenv("NUM_WORKERS")
        if env_workers:
            num_workers = int(env_workers)
        else:
            # Default to 4 workers if not specified
            num_workers = 4

    # Get total CPU count
    total_cpus = multiprocessing.cpu_count()

    # Calculate optimal thread count
    # Ensure at least 1 thread per session, but try to divide available cores evenly
    intra_op_threads = max(1, total_cpus // num_workers)

    # Create and configure session options
    opts = SessionOptions()
    opts.intra_op_num_threads = intra_op_threads
    # inter-op only matters if parallel execution is used (ORT_PARALLEL)
    opts.inter_op_num_threads = 1

    logger.debug(
        f"ONNX Session: configured for {num_workers} workers with {intra_op_threads} threads per worker"
    )
    return opts
