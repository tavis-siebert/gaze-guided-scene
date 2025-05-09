"""
Node embedding module for creating semantic embeddings of graph nodes.
"""

import torch
from typing import Optional, Dict, List, Union, Tuple
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
        self.clip_model = None
        
    def _get_clip_model(self) -> ClipModel:
        """Get or initialize the text embedding model."""
        if self.clip_model is None:
            logger.info(f"Initializing CLIP model on {self.device}")
            self.clip_model = ClipModel(device=self.device)
            self.clip_model.load()
        return self.clip_model
        
    def get_action_embedding(self, action_idx: int) -> Optional[torch.Tensor]:
        """
        Get embedding for an action using CLIP text embedding.
        
        Args:
            action_idx: The index of the action
            
        Returns:
            Tensor containing the action embedding, or None if the action is not found
        """
        # Get action name
        action_name = ActionRecord.get_action_name_by_idx(action_idx)
        if action_name is None:
            logger.warning(f"Action index {action_idx} not found in action mapping")
            return None
            
        # Get text embedding
        clip_model = self._get_clip_model()
        embedding = clip_model.encode_text([action_name])[0]  # List of 1 tensor -> single tensor
        
        return embedding
        
    def get_object_node_embedding(
        self,
        checkpoint: GraphCheckpoint,
        tracer: GraphTracer,
        video: Video,
        node_id: int
    ) -> Optional[torch.Tensor]:
        """
        Generate object embedding for a node by averaging embeddings of ROIs
        from all visits to the node.

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

        self._get_clip_model() # Ensures self.clip_model is initialized

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

    def _get_roi_embeddings_for_visit(
        self,
        video: Video,
        tracer: GraphTracer,
        object_label: str,
        visit_start: int,
        visit_end: int
    ) -> List[torch.Tensor]:
        """Processes all frames within a single visit and collects ROI embeddings."""
        collected_roi_embeddings = []
        video.seek_to_frame(visit_start)
        current_frame_num = visit_start

        try:
            while current_frame_num <= visit_end:
                try:
                    frame_dict = next(video.stream)
                except StopIteration:
                    logger.debug(
                        f"Reached end of video stream while processing visit {visit_start}-{visit_end} "
                        f"for '{object_label}' at frame {current_frame_num}."
                    )
                    break  # Exit while loop for this visit
                
                frame_tensor = frame_dict['data']
                
                frame_roi_embeddings = self._get_roi_embeddings_for_frame(
                    frame_tensor, current_frame_num, tracer, object_label
                )
                collected_roi_embeddings.extend(frame_roi_embeddings)
                
                current_frame_num += 1
        except Exception as e:
            logger.warning(
                f"Error during frame iteration for visit {visit_start}-{visit_end} "
                f"for '{object_label}': {e}"
            )
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
            if pil_image is None:
                continue
            
            try:
                # self.clip_model is guaranteed to be initialized
                roi_embedding = self.clip_model.encode_image(pil_image) # Returns (1, D)
                frame_s_roi_embeddings.append(roi_embedding)
            except Exception as e:
                logger.warning(
                    f"CLIP encoding failed for ROI in frame {frame_num} for '{object_label}' "
                    f"(bbox: {detection.bbox}): {e}"
                )
                continue
                
        return frame_s_roi_embeddings

    @staticmethod
    def _convert_roi_tensor_to_pil(roi_tensor: torch.Tensor) -> Optional[Image.Image]:
        """Converts a C,H,W roi_tensor to a PIL Image. Expects roi_tensor on any device."""
        try:
            roi_numpy_uint8 = roi_tensor.cpu().to(torch.uint8).numpy()
            channels = roi_numpy_uint8.shape[0]

            if channels == 3:  # RGB
                pil_image = Image.fromarray(roi_numpy_uint8.transpose(1, 2, 0))
            elif channels == 1:  # Grayscale
                pil_image = Image.fromarray(roi_numpy_uint8.squeeze(0), mode='L')
            else:
                logger.warning(
                    f"Unsupported number of channels ({channels}) in ROI for PIL conversion. Shape: {roi_tensor.shape}"
                )
                return None
            return pil_image
        except Exception as e:
            logger.warning(f"Error creating PIL image from ROI tensor (shape: {roi_tensor.shape}): {e}")
            return None
            
    def _extract_roi(self, frame: torch.Tensor, bbox: Tuple[float, float, float, float]) -> Optional[torch.Tensor]:
        """
        Extract region of interest from frame using bounding box.
        
        Args:
            frame: Video frame tensor
            bbox: Bounding box coordinates (left, top, width, height)
            
        Returns:
            Tensor containing the ROI or None if extraction fails
        """
        left, top, width, height = bbox
        
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