"""
compose_features.py

Module to compose final feature vectors for gaze-graph experiments. This combines
base features with one-hot labels, CLIP text embeddings, and/or ROI embeddings.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gazegraph.config.config_utils import DotDict
from gazegraph.datasets.egtea_gaze.video_processor import Video
from gazegraph.logger import get_logger
from gazegraph.models.clip import ClipModel

log = get_logger(__name__)


def load_clip_cache(path: Path, labels: list[str], device: str = "cuda") -> dict[str, torch.Tensor]:
    """
    Build or load a {label -> D-dim tensor (cpu)} cache, where D is determined
    from the first embedding returned by ClipModel.encode_texts().
    """
    if path.exists():
        log.info(f"Loading pre-computed CLIP text cache from: {path}")
        return torch.load(path, map_location="cpu")

    log.info(f"CLIP text cache not found at {path}. Generating new cache...")
    cache: dict[str, torch.Tensor] = {}
    clip = ClipModel(device=device)

    expected_dim = None
    with torch.no_grad():
        for lbl in tqdm(labels, desc="Encoding CLIP text embeddings"):
            prompt = f"a photo of a {lbl.replace('_', ' ')}"
            emb = clip.encode_texts([prompt])

            if isinstance(emb, list):
                emb = emb[0]
            if isinstance(emb, torch.Tensor) and emb.ndim > 1:
                emb = emb.squeeze(0)

            if not isinstance(emb, torch.Tensor):
                raise RuntimeError(f"encode_texts() returned {type(emb)}")

            if expected_dim is None:
                expected_dim = emb.numel()
            elif emb.numel() != expected_dim:
                raise RuntimeError(f"Inconsistent embedding size: expected {expected_dim}, got {emb.numel()}")

            cache[lbl] = emb.cpu()

    log.info(f"Saving new CLIP text cache to {path} (dim = {expected_dim})")
    torch.save(cache, path)
    return cache


def run_feature_composition(
    config: DotDict,
    base_pkl: Path,
    fix_pkl: Path,
    out_pkl: Path,
    mode: str,
    clip_cache_path: Path,
):
    """
    Compose final features based on the selected mode.

    Args:
        config: The global configuration object.
        base_pkl: Path to the 2048-D backbone features pickle.
        fix_pkl: Path to fixation_label_map.pkl from the label-fixations step.
        out_pkl: Path to save the final composed features pickle.
        mode: The composition mode (e.g., 'onehot', 'clip', 'combo').
        clip_cache_path: Path to the pre-computed CLIP text embedding cache.
    """
    valid_modes = {"onehot", "clip", "roi", "text+roi", "combo", "combo+roi"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    log.info(f"Loading base features from: {base_pkl}")
    with open(base_pkl, "rb") as f:
        base = pickle.load(f)

    log.info(f"Loading fixation labels from: {fix_pkl}")
    with open(fix_pkl, "rb") as f:
        label_map = pickle.load(f)

    # Use the first video to determine the class list for one-hot encoding
    first_vid = next(iter(base))
    class_list = list(Video(first_vid, config).metadata.object_label_to_id.keys())
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    K = len(class_list)
    log.info(f"Found {K} classes for one-hot encoding.")

    # Prepare CLIP text embedding cache if needed for the selected mode
    if mode in {"clip", "text+roi", "combo", "combo+roi"}:
        clip_cache = load_clip_cache(clip_cache_path, class_list)
        clip_dim = next(iter(clip_cache.values())).shape[0]
        log.info(f"Loaded CLIP text embeddings with dimension {clip_dim}.")

    # Load ROI feature map if needed for the selected mode
    if mode in {"roi", "text+roi", "combo+roi"}:
        roi_pkl_path = fix_pkl.parent / f"{base_pkl.stem}_roi_map.pkl"
        if not roi_pkl_path.exists():
            raise FileNotFoundError(
                f"ROI map not found at {roi_pkl_path}. Please run 'label-fixations' without --skip-roi."
            )
        log.info(f"Loading ROI features from: {roi_pkl_path}")
        with open(roi_pkl_path, "rb") as f:
            roi_map = pickle.load(f)
        # Infer dimension from the first available ROI vector
        first_roi_vec = next(iter(next(iter(roi_map.values())).values()))
        roi_dim = first_roi_vec.shape[0]
        log.info(f"Loaded ROI features with dimension {roi_dim}.")


    # Main composition loop
    merged_features: dict[str, dict[int, np.ndarray]] = {}
    for vid_name in tqdm(base.keys(), desc=f"Composing features (mode: {mode})"):
        if vid_name not in label_map:
            log.warning(f"Skipping video '{vid_name}': not found in fixation label map.")
            continue
        
        merged_vid_features = {}
        for frame_idx, base_vec in base[vid_name].items():
            label = label_map[vid_name].get(frame_idx)
            feature_parts: list[np.ndarray] = [base_vec]

            # Append one-hot encoding
            if mode in {"onehot", "combo", "combo+roi"}:
                one_hot_vec = np.zeros(K, dtype=np.float32)
                if label and label in class_to_idx:
                    one_hot_vec[class_to_idx[label]] = 1.0
                feature_parts.append(one_hot_vec)

            # Append CLIP text embedding
            if mode in {"clip", "text+roi", "combo", "combo+roi"}:
                if label and label in clip_cache:
                    clip_tensor = clip_cache[label]
                    # Ensure tensor is 1D before converting to numpy
                    clip_vec = clip_tensor.squeeze().numpy()
                else:
                    clip_vec = np.zeros(clip_dim, dtype=np.float32)
                feature_parts.append(clip_vec)

            # Append ROI embedding
            if mode in {"roi", "text+roi", "combo+roi"}:
                roi_vec = roi_map.get(vid_name, {}).get(frame_idx, np.zeros(roi_dim, dtype=np.float32))
                feature_parts.append(roi_vec)

            merged_vid_features[frame_idx] = np.concatenate(feature_parts, axis=-1)
        merged_features[vid_name] = merged_vid_features

    log.info(f"Saving composed features to: {out_pkl}")
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(merged_features, f, protocol=4)
    log.info("Feature composition complete.")
