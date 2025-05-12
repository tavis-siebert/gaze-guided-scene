from abc import ABC, abstractmethod
from torch_geometric.data import Data
from typing import Any
import torch

class GraphAssembler(ABC):
    """Abstract base for graph assembly strategies."""
    @abstractmethod
    def assemble(self, checkpoint: Any, y: torch.Tensor) -> Data:
        pass

class ObjectGraph(GraphAssembler):
    """Graph assembler for object graphs (EGTEA)."""
    def __init__(self, node_feature_extractor, object_node_feature, config, device):
        self.node_feature_extractor = node_feature_extractor
        self.object_node_feature = object_node_feature
        self.config = config
        self.device = device

    def assemble(self, checkpoint, y):
        node_features = self._extract_node_features(checkpoint)
        edge_index, edge_attr = self._extract_edge_features(checkpoint)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def _extract_node_features(self, checkpoint):
        # Logic adapted from GraphDataset._extract_node_features
        from gazegraph.logger import get_logger
        logger = get_logger(__name__)
        if self.object_node_feature == "roi-embeddings":
            tracer = self._get_tracer_for_checkpoint(checkpoint)
            video = self._get_video_for_checkpoint(checkpoint)
            if tracer and video:
                self.node_feature_extractor.set_context(tracer=tracer, video=video)
            else:
                logger.warning(f"Could not set ROI embedding context for checkpoint {checkpoint.video_name}")
        return self.node_feature_extractor.extract_features(checkpoint)

    def _get_tracer_for_checkpoint(self, checkpoint):
        from gazegraph.graph.graph_tracer import GraphTracer
        from pathlib import Path
        logger = __import__('gazegraph.logger', fromlist=['get_logger']).get_logger(__name__)
        video_name = checkpoint.video_name
        config = self.config
        if not config or not hasattr(config, 'directories') or not hasattr(config.directories, 'traces'):
            logger.warning("Config or directories.traces missing; cannot load trace file.")
            return None
        trace_path = Path(config.directories.traces) / f"{video_name}_trace.jsonl"
        if not trace_path.exists():
            logger.warning(f"Trace file not found at {trace_path}. ROI embeddings may not work correctly.")
            return None
        tracer = GraphTracer(trace_path.parent, video_name, enabled=False)
        return tracer

    def _get_video_for_checkpoint(self, checkpoint):
        from gazegraph.datasets.egtea_gaze.video_processor import Video
        from pathlib import Path
        logger = __import__('gazegraph.logger', fromlist=['get_logger']).get_logger(__name__)
        video_name = checkpoint.video_name
        config = self.config
        if not config or not hasattr(config, 'dataset') or not hasattr(config.dataset, 'egtea') or not hasattr(config.dataset.egtea, 'raw_videos'):
            logger.warning("Config or dataset.egtea.raw_videos missing; cannot load video file.")
            return None
        video_path = Path(config.dataset.egtea.raw_videos) / f"{video_name}.mp4"
        if not video_path.exists():
            logger.warning(f"Video file not found at {video_path}. ROI embeddings may not work correctly.")
            return None
        video = Video(video_name)
        return video

    def _extract_edge_features(self, checkpoint):
        # Logic adapted from GraphDataset._extract_edge_features
        import torch
        if not checkpoint.edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
        edge_list = [(e["source_id"], e["target_id"]) for e in checkpoint.edges]
        edge_attrs = [[e.get("angle", 0.0)] for e in checkpoint.edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        if edge_attr.shape[0] > 0 and edge_attr.max() > 0:
            edge_attr = edge_attr / (edge_attr.max() + 1e-8)
        return edge_index, edge_attr

class ActionGraph(GraphAssembler):
    """Stub for action graph assembler."""
    def assemble(self, checkpoint, y):
        raise NotImplementedError("ActionGraph assembler not implemented yet.")
