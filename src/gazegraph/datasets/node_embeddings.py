"""
Node embedding module for creating semantic embeddings of graph nodes.
"""

import torch
from typing import Optional, Dict, List, Tuple
from PIL import Image

from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.models.clip import ClipModel
from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.graph.graph_tracer import GraphTracer
from gazegraph.datasets.egtea_gaze.video_processor import Video
from gazegraph.logger import get_logger

logger = get_logger(__name__)


class NodeEmbeddings:
    """
    Handles creation of embeddings for various node types in scene graphs.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the node embedder.
        
        Args:
            device: Device to run models on ("cuda" or "cpu")
        """
        self.device = device
        self.clip_model = ClipModel(device=self.device)
        # Cache for ROI embeddings per visit: (video_name, object_label, visit_start, visit_end) -> list of tensors
        self._roi_visit_embedding_cache: Dict[Tuple[str, str, int, int], List[torch.Tensor]] = {}
        # Cache for object label embeddings: object_label -> tensor
        self._object_label_embedding_cache: Dict[str, torch.Tensor] = {}
        # Cache for action label embeddings: action_label -> tensor
        self._action_label_embedding_cache: Dict[str, torch.Tensor] = {}
        
    def get_action_embedding(self, action_idx: int) -> Optional[torch.Tensor]:
        """
        Get embedding for an action using CLIP text embedding.
        
        Args:
            action_idx: The index of the action
            
        Returns:
            Tensor containing the action embedding, or None if the action is not found
        """
        action_name = ActionRecord.get_action_name_by_idx(action_idx)
        if action_name is None:
            logger.warning(f"Action index {action_idx} not found in action mapping")
            return None
        
        # Check cache first
        if action_name in self._action_label_embedding_cache:
            return self._action_label_embedding_cache[action_name]
            
        # Generate new embedding
        embedding = self.clip_model.encode_texts([action_name])[0]
        
        # Cache the embedding
        self._action_label_embedding_cache[action_name] = embedding
        
        return embedding
        
    def get_object_node_embedding_roi(
        self,
        checkpoint: GraphCheckpoint,
        tracer: GraphTracer,
        video: Video,
        node_id: int
    ) -> Optional[torch.Tensor]:
        """
        Generate object embedding for a node by averaging embeddings of ROIs
        from all visits to the node. (ROI-based visual embedding)

        Args:
            checkpoint: Graph checkpoint containing node information
            tracer: Graph tracer to retrieve detections
            video: Video processor for frame access
            node_id: ID of the node to generate embeddings for

        Returns:
            Visual embedding tensor for the node or None if no suitable ROIs found
        """
        node_data = checkpoint.nodes.get(node_id)
        if not node_data:
            logger.warning(f"Node ID {node_id} not found in checkpoint")
            return None

        object_label = node_data["object_label"]
        visits = node_data["visits"]

        if not visits:
            logger.warning(f"Node {node_id} ('{object_label}') has no visits")
            return None

        all_visit_embeddings = []
        for visit_start, visit_end in visits:
            # Process each visit and collect ROI embeddings from that visit
            visit_roi_embeddings = self._get_roi_embeddings_for_visit(
                video, tracer, object_label, visit_start, visit_end
            )

            if visit_roi_embeddings:  # List of (1, D) tensors
                # Calculate mean embedding for the current visit
                mean_visit_embedding = torch.mean(torch.cat(visit_roi_embeddings, dim=0), dim=0)
                all_visit_embeddings.append(mean_visit_embedding) # List of (D) tensors
                logger.debug(
                    f"Generated embedding for visit {visit_start}-{visit_end} "
                    f"with {len(visit_roi_embeddings)} ROIs for node {node_id} ('{object_label}')"
                )

        if not all_visit_embeddings:
            logger.warning(f"No valid ROIs found across all visits for node {node_id} ('{object_label}')")
            return None

        # Calculate mean embedding across all visits
        final_node_embedding = torch.mean(torch.stack(all_visit_embeddings), dim=0)
        logger.info(
            f"Generated visual embedding for node {node_id} ('{object_label}') "
            f"from {len(all_visit_embeddings)} visits"
        )
        return final_node_embedding


    def get_object_node_embedding_label(
        self,
        checkpoint: GraphCheckpoint,
        node_id: int
    ) -> Optional[torch.Tensor]:
        """
        Generate object embedding for a node using only its label and CLIP text encoding.
        This produces a semantic embedding based solely on the node's label.

        Args:
            checkpoint: Graph checkpoint containing node information
            node_id: ID of the node to generate embeddings for

        Returns:
            Text-based embedding tensor for the node or None if node/label not found
        """
        node_data = checkpoint.nodes.get(node_id)
        if not node_data:
            logger.warning(f"Node ID {node_id} not found in checkpoint")
            return None
        object_label = node_data["object_label"]
        if not object_label:
            logger.warning(f"Node {node_id} has no object_label")
            return None
        
        # Check cache first
        if object_label in self._object_label_embedding_cache:
            return self._object_label_embedding_cache[object_label]
        
        # Generate new embedding
        embedding = self.clip_model.encode_texts([object_label])[0]  # List of 1 tensor -> single tensor
        
        # Cache the embedding
        self._object_label_embedding_cache[object_label] = embedding
        
        logger.info(f"Generated label-based embedding for node {node_id} ('{object_label}')")
        return embedding


    def _get_roi_embeddings_for_visit(
        self,
        video: Video,
        tracer: GraphTracer,
        object_label: str,
        visit_start: int,
        visit_end: int
    ) -> List[torch.Tensor]:
        """Processes all frames within a single visit and collects ROI embedding only from the frame with highest detection confidence. Caches by (video_name, object_label, visit_start, visit_end)."""
        cache_key = (video.video_name, object_label, visit_start, visit_end)
        if cache_key in self._roi_visit_embedding_cache:
            return self._roi_visit_embedding_cache[cache_key]

        best_detection = None
        best_detection_score = float('-inf')
        best_detection_frame_tensor = None
        
        try:
            video.seek_to_frame(visit_start)
        except Exception as e:
            logger.error(
                f"Error seeking to frame {visit_start} for visit {visit_start}-{visit_end} "
                f"for '{object_label}': {e}"
            )
            return []
            
        current_frame_num = visit_start

        # Process frames in the visit range
        while current_frame_num <= visit_end:
            try:
                frame_dict = next(video.stream)
            except StopIteration:
                logger.error(
                    f"Reached end of video stream while processing visit {visit_start}-{visit_end} "
                    f"for '{object_label}' at frame {current_frame_num}."
                )
                break
                
            frame_tensor = frame_dict['data']
            detections = tracer.get_detections_for_frame(current_frame_num)
            
            # Find best detection in this frame
            for det in detections:
                if det.class_name == object_label and det.is_fixated:
                    if det.score > best_detection_score:
                        best_detection_score = det.score
                        best_detection = det
                        best_detection_frame_tensor = frame_tensor
            current_frame_num += 1
        
        collected_roi_embeddings = []
        if best_detection is not None and best_detection_frame_tensor is not None:
            roi_tensor = self._extract_roi(best_detection_frame_tensor, best_detection.bbox)
            if self._is_valid_roi(roi_tensor):
                pil_image = self._convert_roi_tensor_to_pil(roi_tensor)
                roi_embedding = self.clip_model.encode_image(pil_image)  # (1, D)
                collected_roi_embeddings.append(roi_embedding)
        
        # Cache and return results
        self._roi_visit_embedding_cache[cache_key] = collected_roi_embeddings
        return collected_roi_embeddings

    def _is_valid_roi(self, roi_tensor: torch.Tensor) -> bool:
        """Check if ROI tensor is non-empty and has valid dimensions."""
        return roi_tensor is not None and roi_tensor.numel() > 0 and roi_tensor.shape[1] > 0 and roi_tensor.shape[2] > 0

    def _get_roi_embeddings_for_frame(
        self,
        frame_tensor: torch.Tensor,
        frame_num: int,
        tracer: GraphTracer,
        object_label: str
    ) -> List[torch.Tensor]:
        """Processes detections in a single frame and returns their CLIP embeddings."""
        frame_s_roi_embeddings = []
        detections = tracer.get_detections_for_frame(frame_num)
        
        matching_detections = [
            det for det in detections
            if det.class_name == object_label and det.is_fixated
        ]

        if not matching_detections:
            return frame_s_roi_embeddings

        for detection in matching_detections:
            roi_tensor = self._extract_roi(frame_tensor, detection.bbox)
            
            if not self._is_valid_roi(roi_tensor):
                logger.debug(f"Skipping invalid ROI {detection.bbox} in frame {frame_num} for '{object_label}'.")
                continue
            
            pil_image = self._convert_roi_tensor_to_pil(roi_tensor)
            roi_embedding = self.clip_model.encode_image(pil_image)
            frame_s_roi_embeddings.append(roi_embedding)
        return frame_s_roi_embeddings

    @staticmethod
    def _convert_roi_tensor_to_pil(roi_tensor: torch.Tensor) -> Image.Image:
        """Converts a C,H,W roi_tensor to a PIL Image. Expects roi_tensor on any device."""
        roi_numpy_uint8 = roi_tensor.cpu().to(torch.uint8).numpy()
        channels = roi_numpy_uint8.shape[0]

        if channels == 3:  # RGB
            pil_image = Image.fromarray(roi_numpy_uint8.transpose(1, 2, 0))
        elif channels == 1:  # Grayscale
            pil_image = Image.fromarray(roi_numpy_uint8.squeeze(0), mode='L')
        else:
            raise ValueError(f"Unsupported number of channels ({channels}) in ROI for PIL conversion. Shape: {roi_tensor.shape}")
        return pil_image

    def _extract_roi(self, frame: torch.Tensor, bbox: Tuple[float, float, float, float], padding: int = 0) -> torch.Tensor:
        """
        Extract region of interest from frame using bounding box, with optional padding.
        
        Args:
            frame: Video frame tensor
            bbox: Bounding box coordinates (left, top, width, height)
            padding: Optional padding in pixels to expand the ROI in all directions
            
        Returns:
            Tensor containing the ROI
        """
        left, top, width, height = bbox
        
        # Apply padding to expand the bounding box
        left -= padding
        top -= padding
        width += 2 * padding
        height += 2 * padding
        
        # Ensure bbox is within frame boundaries
        height_limit, width_limit = frame.shape[1:3]
        
        # Convert to integers and clip values to ensure they're within frame boundaries
        left = max(0, min(int(left), width_limit - 1))
        top = max(0, min(int(top), height_limit - 1))
        width = min(int(width), width_limit - left)
        height = min(int(height), height_limit - top)
        
        right = left + width
        bottom = top + height
        
        # Extract ROI
        roi = frame[:, top:bottom, left:right]
        
        return roi