import torch

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Literal, List
from torch_geometric.data import Data, HeteroData

from gazegraph.datasets.egtea_gaze.video_processor import Video
from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.training.dataset.node_features import NodeFeatureExtractor, get_node_feature_extractor
from gazegraph.config.config_utils import DotDict
from gazegraph.datasets.node_embeddings import NodeEmbeddings
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.logger import get_logger
from gazegraph.graph.graph_tracer import GraphTracer

logger = get_logger(__name__)

class GraphAssembler(ABC):
    """Abstract base for graph assembly strategies."""
    @abstractmethod
    def assemble(self, checkpoint: GraphCheckpoint, y: torch.Tensor) -> Data | HeteroData:
        pass

class ObjectGraph(GraphAssembler):
    """Graph assembler for object graphs (EGTEA)."""
    def __init__(self, node_feature_extractor: NodeFeatureExtractor, object_node_feature: str, config: DotDict, device: str):
        self.node_feature_extractor = node_feature_extractor
        self.object_node_feature = object_node_feature
        self.config = config
        self.device = device

    def assemble(self, checkpoint: GraphCheckpoint, y: torch.Tensor) -> Data:
        node_features = self._extract_node_features(checkpoint)
        edge_index, edge_attr = self._extract_edge_features(checkpoint)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def _extract_node_features(self, checkpoint):
        # Logic adapted from GraphDataset._extract_node_features
        if self.object_node_feature == "roi-embeddings":
            tracer = self._get_tracer_for_checkpoint(checkpoint)
            video = self._get_video_for_checkpoint(checkpoint)
            if tracer and video:
                self.node_feature_extractor.set_context(tracer=tracer, video=video)
            else:
                logger.warning(f"Could not set ROI embedding context for checkpoint {checkpoint.video_name}")
        return self.node_feature_extractor.extract_features(checkpoint)

    def _get_tracer_for_checkpoint(self, checkpoint):
        video_name = checkpoint.video_name
        config = self.config
        if not config or not hasattr(config, 'directories') or not hasattr(config.directories, 'traces'):
            raise ValueError("Config or directories.traces missing; cannot load trace file.")
        trace_path = Path(config.directories.traces) / f"{video_name}_trace.jsonl"
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found at {trace_path}. ROI embeddings may not work correctly.")
        tracer = GraphTracer(trace_path.parent, video_name, enabled=False)
        return tracer

    def _get_video_for_checkpoint(self, checkpoint: GraphCheckpoint) -> Video:
        video_name = checkpoint.video_name
        config = self.config
        if not config or not hasattr(config, 'dataset') or not hasattr(config.dataset, 'egtea') or not hasattr(config.dataset.egtea, 'raw_videos'):
            raise ValueError("Config or dataset.egtea.raw_videos missing; cannot load video file.")
        video_path = Path(config.dataset.egtea.raw_videos) / f"{video_name}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path}. ROI embeddings may not work correctly.")
        video = Video(video_name)
        return video

    def _extract_edge_features(self, checkpoint: GraphCheckpoint) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_feature_dim = 1

        edge_list = [(e["source_id"], e["target_id"]) for e in checkpoint.edges] if checkpoint.edges else []
        edge_attrs = [[e.get("angle", 0.0)] for e in checkpoint.edges] if checkpoint.edges else []

        edge_index = (
            torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            if edge_list else torch.empty((2, 0), dtype=torch.long)
        )

        edge_attr = (
            torch.tensor(edge_attrs, dtype=torch.float)
            if edge_attrs else torch.empty((0, edge_feature_dim), dtype=torch.float)
        )

        if edge_attr.numel() > 0 and edge_attr.max() > 0:
            edge_attr = edge_attr / (edge_attr.max() + 1e-8)

        assert edge_index.shape == (2, len(edge_list)), f"edge_index.shape: {edge_index.shape}, expected (2, {len(edge_list)})"
        assert edge_attr.shape == (len(edge_list), edge_feature_dim), f"edge_attr.shape: {edge_attr.shape}, expected ({len(edge_list)}, {edge_feature_dim})"

        return edge_index, edge_attr

class ActionGraph(GraphAssembler):
    """Graph assembler for action graphs (EGTEA). Each node is an observed action; edges connect temporally adjacent actions."""
    def __init__(self, node_feature_extractor: NodeFeatureExtractor, action_node_feature: str, config: DotDict, device: str = "cuda", node_embeddings: NodeEmbeddings | None = None):
        self.node_feature_extractor = node_feature_extractor
        self.action_node_feature = action_node_feature
        self.node_embeddings = node_embeddings
        if self.node_embeddings is None:
            self.node_embeddings = NodeEmbeddings(config, device=device)
        self.config = config
        self.device = device

    def assemble(self, checkpoint: GraphCheckpoint, y: torch.Tensor) -> Data:
        video_name = checkpoint.video_name
        current_frame = checkpoint.frame_number
        # 1. Get all observed actions up to current frame
        past_records = ActionRecord.get_past_action_records(video_name, current_frame)
        if not past_records:
            # Return empty graph with default edge_attr
            feature_dim = self.node_feature_extractor.feature_dim
            return Data(
                x=torch.empty((0, feature_dim), device=self.device),
                edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device),
                edge_attr=torch.zeros((0, 1), dtype=torch.float, device=self.device),
                y=y
            )
        
        # 2. Build node features using the node feature extractor
        x = self.node_feature_extractor.extract_features(checkpoint, action_records=past_records)
        # Ensure x is always 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 0:
            x = x.view(1, -1)

        assert x.shape == (len(past_records), self.node_feature_extractor.feature_dim), f"x.shape: {x.shape}, expected ({len(past_records)}, {self.node_feature_extractor.feature_dim})"

        # 3. Build edges (from older to younger)
        if len(past_records) > 1:
            edge_index = torch.tensor([
                [i, i+1] for i in range(len(past_records)-1)
            ], dtype=torch.long, device=self.device).t()
            # Add edge attributes (temporal distance between actions)
            edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float, device=self.device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.zeros((0, 1), dtype=torch.float, device=self.device)

        assert edge_index.shape == (2, len(past_records)), f"edge_index.shape: {edge_index.shape}, expected (2, {len(past_records)})"
        assert edge_attr.shape == (len(past_records)-1, 1), f"edge_attr.shape: {edge_attr.shape}, expected ({len(past_records)-1}, 1)"
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class ActionObjectGraph(GraphAssembler):
    def __init__(
        self, 
        config: DotDict,
        action_feature_extractor: NodeFeatureExtractor,
        object_feature_extractor: NodeFeatureExtractor,
        action_node_feature: str,
        object_node_feature: str,
        device: str
    ):
        self.config = config
        self.device = device

        self.ActionGraph = ActionGraph(action_feature_extractor, action_node_feature, config, device)
        self.ObjectGraph = ObjectGraph(object_feature_extractor, object_node_feature, config, device)
    
    def _link_objects_to_actions(self, checkpoint: GraphCheckpoint):
        """
        Links objects to all actions they appear by
            1. Iterating through each object
            2. For each object, iterating through each action (past record) 
               and linking if any of the object's visits lie within
               the timeframe of a record
        Returns:
            edge_index (torch.LongTensor) containing the edges between objects and actions
        """
        def object_affords_action(object_visits: List[List[int]], action_record: ActionRecord):
            """Whether an object appears in a given action"""
            action_start, action_end = action_record.start_frame, action_record.end_frame
            for object_start, object_end in object_visits:
                if (object_start <= action_end) and (object_end >= action_start):
                    return True
            return False

        video_name = checkpoint.video_name
        current_frame = checkpoint.frame_number
        past_records = ActionRecord.get_past_action_records(video_name, current_frame)
        
        object_src, action_dest = [], []
        for object_id, object_node in enumerate(checkpoint.nodes.values()):
            object_visits = object_node["visits"]
            for action_id, action_record in enumerate(past_records):
                if object_affords_action(object_visits, action_record):
                    object_src.append(object_id)
                    action_dest.append(action_id)
        
        edge_index = torch.tensor([object_src, action_dest], dtype=torch.long, device=self.device)
        return edge_index
        
    def assemble(self, checkpoint: GraphCheckpoint, y: torch.Tensor) -> HeteroData:
        ActionData = self.ActionGraph.assemble(checkpoint, y)
        ObjectData = self.ObjectGraph.assemble(checkpoint, y)

        ActionObjectData = HeteroData()
        ActionObjectData["action"].x = ActionData.x
        ActionObjectData["object"].x = ObjectData.x
        
        ActionObjectData["action", "next_action", "action"].edge_index = ActionData.edge_index
        ActionObjectData["object", "next_object", "object"].edge_index = ObjectData.edge_index
        object_action_edges = self._link_objects_to_actions(checkpoint)
        ActionObjectData["object", "affords", "action"].edge_index = object_action_edges
        #TODO for now, I'm skipping edge_attrs 

        ActionObjectData.y = y

        return ActionObjectData


def create_graph_assembler(
    graph_type: Literal["object-graph", "action-graph", "action-object-graph"],
    config: DotDict,
    device: str,
    object_node_feature: str = "roi-embeddings",
    action_node_feature: str = "action-label-embedding"
) -> GraphAssembler:
    """Factory method to create the appropriate graph assembler based on the dataset type.
    
    Args:
        graph_type: Type of graph to create ("object-graph" or "action-graph")
        config: Configuration object
        device: Device to use ("cuda" or "cpu")
        object_node_feature: Type of object node features
        action_node_feature: Type of action node features
        
    Returns:
        GraphAssembler: The appropriate graph assembler instance
    """
    if graph_type == "object-graph":
        node_feature_extractor = get_node_feature_extractor(
            object_node_feature, device=device, config=config
        )
        return ObjectGraph(
            node_feature_extractor=node_feature_extractor,
            object_node_feature=object_node_feature,
            config=config,
            device=device
        )
    elif graph_type == "action-graph":
        node_feature_extractor = get_node_feature_extractor(
            action_node_feature, device=device, config=config
        )
        return ActionGraph(
            node_feature_extractor=node_feature_extractor,
            action_node_feature=action_node_feature,
            config=config,
            device=device
        )
    elif graph_type == "action-object-graph":
        action_feature_extractor = get_node_feature_extractor(
            action_node_feature, device, config
        )
        object_feature_extractor = get_node_feature_extractor(
            object_node_feature, device, config
        )
        return ActionObjectGraph(
            action_feature_extractor=action_feature_extractor,
            object_feature_extractor=object_feature_extractor,
            action_node_feature=action_node_feature,
            object_node_feature=object_node_feature,
            config=config,
            device=device
        )
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

