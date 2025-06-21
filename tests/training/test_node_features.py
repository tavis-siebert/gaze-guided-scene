import pytest
import torch
from unittest.mock import MagicMock

from gazegraph.graph.checkpoint_manager import GraphCheckpoint
from gazegraph.training.dataset.node_features import (
    OneHotNodeFeatureExtractor,
    ROIEmbeddingNodeFeatureExtractor,
    ObjectLabelEmbeddingNodeFeatureExtractor,
    OneHotActionNodeFeatureExtractor,
    get_node_feature_extractor,
)
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord


class TestNodeFeatureExtractors:
    @pytest.fixture
    def mock_checkpoint(self):
        # Create a mock checkpoint
        checkpoint = MagicMock(spec=GraphCheckpoint)
        checkpoint.nodes = {
            0: {"object_label": "cup", "visits": [(10, 20), (30, 40)]},
            1: {"object_label": "bowl", "visits": [(15, 25)]},
        }
        checkpoint.object_label_to_id = {"cup": 0, "bowl": 1}
        checkpoint.num_object_classes = 2
        checkpoint.non_black_frame_count = 100
        checkpoint.frame_number = 50
        checkpoint.video_length = 200
        checkpoint.video_name = "OP01-R01-PastaSalad"

        return checkpoint

    def test_one_hot_extractor(self, mock_config, mock_checkpoint):
        """Test the OneHotNodeFeatureExtractor for object and action graphs"""
        for device in ["cpu", "cuda"]:
            extractor = OneHotNodeFeatureExtractor(config=mock_config, device=device)
            # Object graph mode
            features = extractor.extract_features(mock_checkpoint)
            assert features.shape == (2, 7)
            assert str(features.device).startswith(device)
            assert features[0, 5] == 1  # One-hot for class 0
            assert features[0, 6] == 0  # One-hot for class 1
            assert features[1, 5] == 0  # One-hot for class 0
            assert features[1, 6] == 1  # One-hot for class 1

    def one_hot_action_extractor(self, mock_config):
        # Action graph mode
        action_one_hot_extractor = OneHotActionNodeFeatureExtractor(config=mock_config)
        MockActionRecord = type("MockActionRecord", (), {})
        recs = []
        num_action_classes = len(ActionRecord.get_action_mapping())
        for idx in range(3):
            rec = MockActionRecord()
            rec.action_idx = idx % num_action_classes
            recs.append(rec)
        action_features = action_one_hot_extractor.extract_features(
            None, action_records=recs
        )
        assert action_features.shape == (3, num_action_classes)
        for i, rec in enumerate(recs):
            assert action_features[i, rec.action_idx] == 1
            assert action_features[i].sum() == 1

    def test_action_label_embedding_extractor(self, mock_config):
        """Test ActionLabelEmbeddingNodeFeatureExtractor for action graph usage"""
        embedding_dim = 8
        mock_node_embeddings = type(
            "MockNodeEmbeddings",
            (),
            {
                "get_action_embedding": lambda self, idx: torch.full(
                    (embedding_dim,), idx, dtype=torch.float
                )
            },
        )()
        extractor = __import__(
            "gazegraph.training.dataset.node_features",
            fromlist=["ActionLabelEmbeddingNodeFeatureExtractor"],
        ).ActionLabelEmbeddingNodeFeatureExtractor(
            config=mock_config,
            device="cpu",
            embedding_dim=embedding_dim,
            node_embeddings=mock_node_embeddings,
        )
        MockActionRecord = type("MockActionRecord", (), {})
        recs = []
        for idx in range(5):
            rec = MockActionRecord()
            rec.action_idx = idx
            recs.append(rec)
        features = extractor.extract_features(None, action_records=recs)
        assert features.shape == (5, embedding_dim)
        for i, rec in enumerate(recs):
            assert torch.all(features[i] == rec.action_idx)

    def test_object_label_embedding_extractor_device(
        self, mock_config, mock_checkpoint
    ):
        """Test device consistency for ObjectLabelEmbeddingNodeFeatureExtractor"""
        mock_node_embeddings = MagicMock()
        embedding_dim = 16
        mock_embedding = torch.ones(embedding_dim)
        mock_node_embeddings.get_object_node_embedding_label.return_value = (
            mock_embedding
        )
        for device in ["cpu", "cuda"]:
            extractor = ObjectLabelEmbeddingNodeFeatureExtractor(
                config=mock_config,
                device=device,
                embedding_dim=embedding_dim,
                node_embeddings=mock_node_embeddings,
            )
            features = extractor.extract_features(mock_checkpoint)
            assert str(features.device).startswith(device)

    def test_roi_embedding_extractor_device(self, mock_config, mock_checkpoint):
        """Test device consistency for ROIEmbeddingNodeFeatureExtractor"""
        mock_node_embeddings = MagicMock()
        embedding_dim = 16
        mock_embedding = torch.ones(embedding_dim)
        mock_node_embeddings.get_object_node_embedding_roi.return_value = mock_embedding
        for device in ["cpu", "cuda"]:
            extractor = ROIEmbeddingNodeFeatureExtractor(
                config=mock_config,
                device=device,
                embedding_dim=embedding_dim,
                node_embeddings=mock_node_embeddings,
            )
            mock_tracer = MagicMock()
            mock_video = MagicMock()
            extractor.set_context(mock_tracer, mock_video)
            features = extractor.extract_features(mock_checkpoint)
            assert str(features.device).startswith(device)

    def test_roi_embedding_extractor(self, mock_config, mock_checkpoint):
        """Test the ROIEmbeddingNodeFeatureExtractor"""
        # Create a proper mock for NodeEmbeddings
        mock_node_embeddings = MagicMock()

        # Set embedding dimension for the test
        embedding_dim = 16
        mock_embedding = torch.ones(embedding_dim)
        mock_node_embeddings.get_object_node_embedding_roi.return_value = mock_embedding

        # Create the extractor with the mock node_embeddings
        extractor = ROIEmbeddingNodeFeatureExtractor(
            config=mock_config,
            device="cpu",
            embedding_dim=embedding_dim,
            node_embeddings=mock_node_embeddings,
        )

        # Mock the tracer and video
        mock_tracer = MagicMock()
        mock_video = MagicMock()
        extractor.set_context(mock_tracer, mock_video)

        # Extract features
        features = extractor.extract_features(mock_checkpoint)

        # Check shape: 2 nodes, 5 temporal features + embedding_dim
        assert features.shape == (2, 5 + embedding_dim)

        # Check that the embedding part is filled with ones
        assert torch.all(features[0, 5:] == 1)
        assert torch.all(features[1, 5:] == 1)

        # Check that the method was called with correct parameters
        mock_node_embeddings.get_object_node_embedding_roi.assert_any_call(
            mock_checkpoint, mock_tracer, mock_video, 0
        )
        mock_node_embeddings.get_object_node_embedding_roi.assert_any_call(
            mock_checkpoint, mock_tracer, mock_video, 1
        )

    def test_object_label_embedding_extractor(self, mock_config, mock_checkpoint):
        """Test the ObjectLabelEmbeddingNodeFeatureExtractor"""
        mock_node_embeddings = MagicMock()
        embedding_dim = 16
        mock_embedding = torch.ones(embedding_dim)
        mock_node_embeddings.get_object_node_embedding_label.return_value = (
            mock_embedding
        )
        extractor = ObjectLabelEmbeddingNodeFeatureExtractor(
            config=mock_config,
            device="cpu",
            embedding_dim=embedding_dim,
            node_embeddings=mock_node_embeddings,
        )
        features = extractor.extract_features(mock_checkpoint)
        # Check shape: 2 nodes, 5 temporal features + embedding_dim
        assert features.shape == (2, 5 + embedding_dim)
        # Check that the embedding part is filled with ones
        assert torch.all(features[:, 5:] == 1)
        # Check that the method was called with correct parameters
        mock_node_embeddings.get_object_node_embedding_label.assert_any_call(
            mock_checkpoint, 0
        )
        mock_node_embeddings.get_object_node_embedding_label.assert_any_call(
            mock_checkpoint, 1
        )

    def test_normalization_and_shape_consistency(self):
        """Test normalization and shape for various feature tensor edge cases"""
        from gazegraph.training.dataset.node_features import NodeFeatureExtractor

        class DummyExtractor(NodeFeatureExtractor):
            def extract_features(self, checkpoint):
                return torch.tensor([])

            @property
            def feature_dim(self):
                return 1

        extractor = DummyExtractor(config=type("DotDict", (), {})())
        # Multi-node, multi-feature
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        norm = extractor._normalize_features(t.clone())
        assert norm.shape == (2, 3)
        assert norm[0, 1] == t[0, 1] / t[:, 1].max()
        # Single-node, multi-feature
        t = torch.tensor([1.0, 2.0, 3.0])
        norm = extractor._normalize_features(t.clone())
        assert norm.shape == (1, 3)
        # Multi-node, single-feature
        t = torch.tensor([[1.0], [2.0]])
        norm = extractor._normalize_features(t.clone())
        assert norm.shape == (2, 1)
        # Single-node, single-feature
        t = torch.tensor([1.0])
        norm = extractor._normalize_features(t.clone())
        assert norm.shape == (1, 1)
        # Empty tensor
        t = torch.empty((0, 3))
        norm = extractor._normalize_features(t)
        assert norm.shape == (0, 3)

    def test_no_index_error_on_various_inputs(self):
        """No IndexError should be raised for any input shape"""
        from gazegraph.training.dataset.node_features import NodeFeatureExtractor

        class DummyExtractor(NodeFeatureExtractor):
            def extract_features(self, checkpoint):
                return torch.tensor([])

            @property
            def feature_dim(self):
                return 1

        extractor = DummyExtractor(config=type("DotDict", (), {})())
        extractor._normalize_features(torch.tensor([[1.0], [2.0]]))
        extractor._normalize_features(torch.tensor([]))
        extractor._normalize_features(torch.empty((0, 3)))

    def test_feature_extractors_output_shape(self, mock_config, mock_checkpoint):
        """All extractors return 2D output with correct feature dim"""
        mock_node_embeddings = MagicMock()
        embedding_dim = 8
        # OneHot
        extractor = OneHotNodeFeatureExtractor(config=mock_config)
        features = extractor.extract_features(mock_checkpoint)
        assert features.ndim == 2
        # ROI
        mock_node_embeddings.get_object_node_embedding_roi.return_value = torch.ones(
            embedding_dim
        )
        roi_extractor = ROIEmbeddingNodeFeatureExtractor(
            config=mock_config,
            device="cpu",
            embedding_dim=embedding_dim,
            node_embeddings=mock_node_embeddings,
        )
        roi_extractor.set_context(MagicMock(), MagicMock())
        features = roi_extractor.extract_features(mock_checkpoint)
        assert features.ndim == 2
        # ObjectLabelEmbedding
        mock_node_embeddings.get_object_node_embedding_label.return_value = torch.ones(
            embedding_dim
        )
        obj_extractor = ObjectLabelEmbeddingNodeFeatureExtractor(
            config=mock_config,
            device="cpu",
            embedding_dim=embedding_dim,
            node_embeddings=mock_node_embeddings,
        )
        features = obj_extractor.extract_features(mock_checkpoint)
        assert features.ndim == 2

    def test_visit_count_normalization(self):
        """Visit count column is normalized if max > 0"""
        from gazegraph.training.dataset.node_features import NodeFeatureExtractor

        class DummyExtractor(NodeFeatureExtractor):
            def extract_features(self, checkpoint):
                return torch.tensor([])

            @property
            def feature_dim(self):
                return 1

        extractor = DummyExtractor(config=type("DotDict", (), {})())
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 6.0]])
        norm = extractor._normalize_features(t.clone())
        assert norm[0, 1] == 2.0 / 6.0
        assert norm[1, 1] == 1.0

    def test_get_node_feature_extractor(self, mock_config):
        """Test the factory function for node feature extractors"""
        # Create a proper mock for NodeEmbeddings
        mock_node_embeddings = MagicMock()

        # Test one-hot extractor
        extractor = get_node_feature_extractor(
            "one-hot", config=mock_config, node_embeddings=mock_node_embeddings
        )
        assert isinstance(extractor, OneHotNodeFeatureExtractor)

        # Test ROI embedding extractor
        extractor = get_node_feature_extractor(
            "roi-embeddings",
            config=mock_config,
            device="cpu",
            embedding_dim=32,
            node_embeddings=mock_node_embeddings,
        )
        assert isinstance(extractor, ROIEmbeddingNodeFeatureExtractor)
        assert extractor.embedding_dim == 32

        # Test object label embedding extractor
        extractor = get_node_feature_extractor(
            "object-label-embeddings",
            config=mock_config,
            device="cpu",
            embedding_dim=64,
            node_embeddings=mock_node_embeddings,
        )
        assert isinstance(extractor, ObjectLabelEmbeddingNodeFeatureExtractor)
        assert extractor.embedding_dim == 64

        # Test invalid type
        with pytest.raises(ValueError):
            get_node_feature_extractor("invalid-type")

    def test_dimension_handling_object_label_embeddings(
        self, mock_config, mock_checkpoint
    ):
        """Test dimension handling in ObjectLabelEmbeddingNodeFeatureExtractor"""
        # Create a proper mock for NodeEmbeddings
        mock_node_embeddings = MagicMock()

        embedding_dim = 16
        extractor = ObjectLabelEmbeddingNodeFeatureExtractor(
            config=mock_config,
            device="cpu",
            embedding_dim=embedding_dim,
            node_embeddings=mock_node_embeddings,
        )

        # Mock the _extract_temporal_features and _normalize_features methods to avoid issues
        temporal_features = torch.ones(5)  # 5 temporal features
        extractor._extract_temporal_features = MagicMock(return_value=temporal_features)
        extractor._normalize_features = MagicMock(
            return_value=torch.ones(2, 5 + embedding_dim)
        )  # Final expected output shape

        # Test case 1: 1D temporal features and 1D label embedding (normal case)
        mock_embedding_1d = torch.ones(embedding_dim)
        mock_node_embeddings.get_object_node_embedding_label.return_value = (
            mock_embedding_1d
        )

        features = extractor.extract_features(mock_checkpoint)
        assert features.shape == (2, 5 + embedding_dim)

        # Test case 2: 1D temporal features and 2D label embedding
        mock_embedding_2d = torch.ones(1, embedding_dim)
        mock_node_embeddings.get_object_node_embedding_label.return_value = (
            mock_embedding_2d
        )

        features = extractor.extract_features(mock_checkpoint)
        assert features.shape == (2, 5 + embedding_dim)

    def test_dimension_handling_roi_embeddings(self, mock_config, mock_checkpoint):
        """Test dimension handling in ROIEmbeddingNodeFeatureExtractor"""
        # Create a proper mock for NodeEmbeddings
        mock_node_embeddings = MagicMock()

        embedding_dim = 16
        extractor = ROIEmbeddingNodeFeatureExtractor(
            config=mock_config,
            device="cpu",
            embedding_dim=embedding_dim,
            node_embeddings=mock_node_embeddings,
        )

        # Mock the tracer and video
        mock_tracer = MagicMock()
        mock_video = MagicMock()
        extractor.set_context(mock_tracer, mock_video)

        # Mock the _extract_temporal_features and _normalize_features methods to avoid issues
        temporal_features = torch.ones(5)  # 5 temporal features
        extractor._extract_temporal_features = MagicMock(return_value=temporal_features)
        extractor._normalize_features = MagicMock(
            return_value=torch.ones(2, 5 + embedding_dim)
        )  # Final expected output shape

        # Test case 1: 1D temporal features and 1D ROI embedding (normal case)
        mock_embedding_1d = torch.ones(embedding_dim)
        mock_node_embeddings.get_object_node_embedding_roi.return_value = (
            mock_embedding_1d
        )

        features = extractor.extract_features(mock_checkpoint)
        assert features.shape == (2, 5 + embedding_dim)

        # Test case 2: 1D temporal features and 2D ROI embedding
        mock_embedding_2d = torch.ones(1, embedding_dim)
        mock_node_embeddings.get_object_node_embedding_roi.return_value = (
            mock_embedding_2d
        )

        features = extractor.extract_features(mock_checkpoint)
        assert features.shape == (2, 5 + embedding_dim)
