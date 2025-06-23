"""
This module processes videos to identify the most likely fixated object label for each frame,
generating a mapping from video and frame to an object label and an optional visual ROI embedding.

It features:
- Automatic checkpointing to save progress after each video.
- Smart resumption to re-use previously computed label maps when only ROI vectors are needed.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from gazegraph.config.config_utils import DotDict
from gazegraph.datasets.egtea_gaze.video_processor import Video
from gazegraph.graph.gaze import GazeType
from gazegraph.graph.object_detection import Detection, ObjectDetector
from gazegraph.logger import get_logger
from gazegraph.models.clip import ClipModel

log = get_logger(__name__)
# Suppress noisy logging from the object detector during this specific task
logging.getLogger("gazegraph.graph.object_detection").setLevel(logging.WARNING)


def _tensor_roi_to_pil(
    frame: torch.Tensor, bbox: Tuple[float, float, float, float], padding: int = 0
) -> Image.Image | None:
    """Crop C×H×W frame tensor (uint8) to a PIL Image with optional padding."""
    left, top, w, h = bbox
    left -= padding
    top -= padding
    right = left + w + 2 * padding
    bot = top + h + 2 * padding

    C, H, W = frame.shape
    left = int(max(0, min(left, W - 1)))
    top = int(max(0, min(top, H - 1)))
    right = int(max(left + 1, min(right, W)))
    bot = int(max(top + 1, min(bot, H)))
    if right <= left or bot <= top:
        return None

    roi = frame[:, top:bot, left:right].to(torch.uint8).cpu().numpy()
    if roi.size == 0:
        return None

    return (
        Image.fromarray(roi.transpose(1, 2, 0))
        if C == 3
        else Image.fromarray(roi.squeeze(0), mode="L")
    )


def extract_fixation_maps(
    video_id: str,
    wanted_frames: Set[int],
    cfg: DotDict,
    detector: ObjectDetector,
    clip: ClipModel | None,
) -> tuple[Dict[int, str | None], Dict[int, np.ndarray]]:
    """Return frame→label and frame→ROI‑vector maps for one video."""
    detector.reset()
    vid = Video(video_id, cfg)

    # Dimensionality for zero-vectors when ROI disabled
    dim = 512 if clip is None else clip.encode_texts(["dummy"])[0].shape[-1]

    label_map: Dict[int, str | None] = {}
    roi_map: Dict[int, np.ndarray] = {}

    cur_label: str | None = None
    cur_roi_vec = np.zeros(dim, dtype=np.float32)
    cluster_frames: list[int] = []
    best_det_score: float = -1
    best_det_roi_pil: Image.Image | None = None

    for frame, _, is_black_frame, gaze in vid:
        idx = gaze.frame_idx if gaze is not None else -1
        if is_black_frame or idx < vid.first_frame or idx > vid.last_frame:
            continue

        detections: list[Detection] = []
        if gaze.type == GazeType.FIXATION:
            detections = detector.detect_objects(frame, gaze, idx)
            cluster_frames.append(idx)
            if clip is not None:
                for det in detections:
                    if det.is_fixated and det.score > best_det_score:
                        best_det_score = det.score
                        best_det_roi_pil = _tensor_roi_to_pil(frame, det.bbox)

        elif gaze.type == GazeType.SACCADE:
            cur_label = (
                detector.get_fixated_object()[0]
                if detector.has_fixated_objects()
                else None
            )
            if clip is not None and best_det_roi_pil is not None:
                with torch.no_grad():
                    cur_roi_vec = (
                        clip.encode_image(best_det_roi_pil)
                        .cpu()
                        .float()
                        .numpy()
                        .squeeze()
                    )
            else:
                cur_roi_vec = np.zeros(dim, dtype=np.float32)

            for f in cluster_frames:
                if f in wanted_frames:
                    label_map[f] = cur_label
                    roi_map[f] = cur_roi_vec
            cluster_frames.clear()
            best_det_score = -1
            best_det_roi_pil = None
            detector.reset()

        if idx in wanted_frames and idx not in label_map:
            label_map[idx] = cur_label
            roi_map[idx] = cur_roi_vec

        if len(label_map) == len(wanted_frames):
            break

    # Assign label to any remaining wanted frames
    for f in wanted_frames - label_map.keys():
        label_map[f] = cur_label
        roi_map[f] = cur_roi_vec
    return label_map, roi_map


def run_fixation_labeling(
    config: DotDict, in_pkl: Path, out_dir: Path, skip_roi: bool
):
    """
    Run YOLO-World to decide the winner of every fixation and store results.

    Args:
        config: The project configuration object.
        in_pkl: Path to the input pickle file containing base features.
        out_dir: Directory to save the output label and ROI maps.
        skip_roi: If True, disables CLIP/ROI vector computation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    label_path = out_dir / f"{in_pkl.stem}_label_map.pkl"
    roi_path = out_dir / f"{in_pkl.stem}_roi_map.pkl"

    label_maps = pickle.load(label_path.open("rb")) if label_path.exists() else {}
    roi_maps = pickle.load(roi_path.open("rb")) if roi_path.exists() else {}

    if skip_roi and not roi_maps:
        log.info("Running with --skip-roi; ROI map will not be written.")
    if not skip_roi and label_maps and not roi_maps:
        log.info("Label map found – will compute missing ROI vectors only.")

    base = pickle.load(in_pkl.open("rb"))  # {video→{frame→vec}}

    if not base:
        log.warning(f"Input pickle file is empty: {in_pkl}")
        return

    first_vid = next(iter(base))
    class_list = list(Video(first_vid, config).metadata.object_label_to_id.keys())
    detector = ObjectDetector(
        model_path=Path(config.models.yolo_world.model_path),
        classes=class_list,
        config=config,
        tracer=None,
    )

    clip = None
    if not skip_roi:
        clip = ClipModel(device="cuda")

    todo = []
    for vid, frames in base.items():
        need = False
        if vid not in label_maps:
            need = True  # Never processed at all
        elif not skip_roi and vid not in roi_maps:
            need = True  # Labels present but ROI missing
        if need:
            todo.append((vid, frames))

    if not todo:
        log.info("All videos are already processed. Nothing to do.")
        return

    for vid, frames in tqdm(todo, total=len(todo), desc="Labeling Fixations"):
        lbl_map, r_map = extract_fixation_maps(vid, set(frames), config, detector, clip)
        label_maps[vid] = lbl_map
        if not skip_roi:
            roi_maps[vid] = r_map

        # Write incremental checkpoint
        with label_path.open("wb") as f:
            pickle.dump(label_maps, f, protocol=4)
        if not skip_roi:
            with roi_path.open("wb") as f:
                pickle.dump(roi_maps, f, protocol=4)

    log.info(f"All done; results in {out_dir}")
