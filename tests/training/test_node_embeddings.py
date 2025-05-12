"""
Unit tests for the NodeEmbeddings class.
"""

import pytest
import torch
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from gazegraph.datasets.node_embeddings import NodeEmbeddings
from gazegraph.datasets.egtea_gaze.action_record import ActionRecord
from gazegraph.graph.checkpoint_manager import CheckpointManager
from gazegraph.graph.graph_tracer import GraphTracer
from gazegraph.datasets.egtea_gaze.video_processor import Video
from gazegraph.config.config_utils import get_config


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def node_embeddings(device):
    """Fixture for a NodeEmbeddings instance configured for testing."""
    # Disable prepopulation for most tests to avoid unnecessary computation
    return NodeEmbeddings(device=device, prepopulate_caches=False)


@pytest.fixture
def test_checkpoint():
    """Fixture that loads a real checkpoint from the data directory."""
    graph_path = Path("data/graphs/train/OP01-R04-ContinentalBreakfast_graph.pth")
    checkpoints = CheckpointManager.load_checkpoints(str(graph_path))
    if not checkpoints:
        pytest.skip("Test graph checkpoint not available")
    return checkpoints[-1]


@pytest.fixture
def test_tracer():
    """Fixture that loads the trace file for the test video."""
    trace_path = Path("data/traces/OP01-R04-ContinentalBreakfast_trace.jsonl")
    if not trace_path.exists():
        pytest.skip("Test trace file not available")
    return GraphTracer("data/traces", "OP01-R04-ContinentalBreakfast", enabled=False)


@pytest.fixture
def test_video():
    """Fixture that initializes a Video object for the test video."""
    try:
        return Video("OP01-R04-ContinentalBreakfast")
    except Exception as e:
        pytest.skip(f"Test video not available: {e}")


@pytest.mark.unit
def test_initialization(node_embeddings, device):
    """Test that NodeEmbeddings initializes correctly."""
    assert node_embeddings.device == device
    assert node_embeddings.clip_model is not None


@pytest.mark.unit
def test_get_clip_model(node_embeddings):
    """Test that the CLIP model is initialized properly."""
    clip_model = node_embeddings.clip_model
    assert clip_model is not None
    assert clip_model.device == node_embeddings.device


@pytest.mark.gpu
@pytest.mark.unit
def test_get_action_embedding(node_embeddings):
    """Test action embedding generation and caching."""
    with patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_action_name_by_idx", return_value="take bowl"):
        with patch.object(node_embeddings.clip_model, "encode_texts", return_value=[torch.ones((1, 768))]) as mock_encode_texts:
            embedding = node_embeddings.get_action_embedding(0)
            assert embedding is not None
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape[1] == 768
            # Reset mock to clear call history
            mock_encode_texts.reset_mock()
            # Second call should use cache
            embedding2 = node_embeddings.get_action_embedding(0)
            # Verify encode_texts was not called again
            mock_encode_texts.assert_not_called()
            assert torch.allclose(embedding, embedding2), "Second call should return same embedding from cache"


@pytest.mark.unit
def test_get_action_embedding_invalid_action(node_embeddings):
    """Test behavior with invalid action index."""
    with patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_action_name_by_idx", 
               return_value=None):
        embedding = node_embeddings.get_action_embedding(9999)
        assert embedding is None


@pytest.mark.unit
def test_extract_roi(node_embeddings):
    """Test ROI extraction from frame tensor."""
    frame = torch.zeros((3, 100, 100), dtype=torch.uint8)
    frame[:, 30:60, 30:60] = 255  # White square in the middle
    bbox = (30, 30, 30, 30)  # left, top, width, height
    roi = node_embeddings._extract_roi(frame, bbox)
    assert roi is not None
    assert roi.shape == (3, 30, 30)
    assert roi.sum() == 3 * 30 * 30 * 255  # All pixels should be white


@pytest.mark.unit
def test_is_valid_roi(node_embeddings):
    """Test ROI validation checks."""
    valid_roi = torch.ones((3, 32, 32))
    assert node_embeddings._is_valid_roi(valid_roi) is True
    empty_roi = torch.tensor([])
    assert node_embeddings._is_valid_roi(empty_roi) is False
    none_roi = None
    assert node_embeddings._is_valid_roi(none_roi) is False


@pytest.mark.gpu
@pytest.mark.integration
def test_get_roi_embeddings_for_frame(node_embeddings):
    """Test ROI embedding retrieval for a frame with mocked clip_model."""
    frame = torch.ones((3, 224, 224), dtype=torch.uint8) * 128
    frame_num = 10
    mock_tracer = MagicMock()
    mock_detection = MagicMock(class_name="bowl", is_fixated=True, bbox=(50, 50, 100, 100))
    mock_tracer.get_detections_for_frame.return_value = [mock_detection]
    with patch.object(node_embeddings, '_convert_roi_tensor_to_pil', return_value=MagicMock()):
        node_embeddings.clip_model = MagicMock()
        node_embeddings.clip_model.encode_image.return_value = torch.ones((1, 512))
        embeddings = node_embeddings._get_roi_embeddings_for_frame(
            frame_tensor=frame,
            frame_num=frame_num,
            tracer=mock_tracer,
            object_label="bowl"
        )
    assert len(embeddings) == 1
    assert embeddings[0].shape == (1, 512)
    mock_tracer.get_detections_for_frame.assert_called_once_with(frame_num)


@pytest.mark.unit
def test_roi_embeddings_cache_behavior(node_embeddings):
    """Test ROI embedding caching per (video_name, object_label, visit_start, visit_end)."""
    mock_video = MagicMock(video_name="vid1")
    mock_tracer = MagicMock()
    object_label = "bowl"
    visit_start, visit_end = 10, 20
    dummy_detection = MagicMock(class_name=object_label, is_fixated=True, confidence=0.9, score=0.9, bbox=(0,0,32,32))
    mock_tracer.get_detections_for_frame.return_value = [dummy_detection]
    # Patch methods to avoid actual image processing
    with patch.object(node_embeddings, '_extract_roi', return_value=torch.ones((3, 32, 32))):
        with patch.object(node_embeddings, '_convert_roi_tensor_to_pil', return_value=MagicMock()):
            node_embeddings.clip_model = MagicMock()
            node_embeddings.clip_model.encode_image.return_value = torch.ones((1, 512))
            # First call: should compute and cache
            out1 = node_embeddings._get_roi_embeddings_for_visit(
                mock_video, mock_tracer, object_label, visit_start, visit_end)
            assert len(out1) == 1, "First call should return one embedding"
            # Second call: should hit cache, so encode_image not called again
            node_embeddings.clip_model.encode_image.reset_mock()
            out2 = node_embeddings._get_roi_embeddings_for_visit(
                mock_video, mock_tracer, object_label, visit_start, visit_end)
            assert out2 == out1, "Second call should return same embeddings from cache"
            node_embeddings.clip_model.encode_image.assert_not_called()
            # Different key: should compute again with a different result
            # Create a new mock for the second call to ensure different results
            node_embeddings.clip_model.encode_image.return_value = torch.ones((1, 512)) * 2
            out3 = node_embeddings._get_roi_embeddings_for_visit(
                mock_video, mock_tracer, object_label, visit_start+1, visit_end)
            assert len(out3) == len(out1), "Third call should return same number of embeddings"
            assert any(not torch.allclose(t1, t3) for t1, t3 in zip(out1, out3)), "Third call should return different embeddings"

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("has_visits", [True, False])
def test_get_object_node_embedding_roi(node_embeddings, has_visits):
    """Test object node embedding generation (ROI-based)."""
    mock_checkpoint = MagicMock()
    mock_tracer = MagicMock()
    mock_video = MagicMock()
    node_data = {
        "object_label": "bowl",
        "visits": [(10, 20)] if has_visits else []
    }
    mock_checkpoint.nodes = {1: node_data}
    
    if has_visits:
        with patch.object(node_embeddings, '_get_roi_embeddings_for_visit') as mock_get_roi:
            mock_get_roi.return_value = [torch.ones((1, 768))]
            # Use the clip_model attribute directly
            embedding = node_embeddings.get_object_node_embedding_roi(
                checkpoint=mock_checkpoint,
                tracer=mock_tracer,
                video=mock_video,
                node_id=1
            )
            assert embedding is not None
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape[0] == 768  # We assume ViT-L/14 CLIP model
            mock_get_roi.assert_called_once()
    else:
        embedding = node_embeddings.get_object_node_embedding_roi(
            checkpoint=mock_checkpoint,
            tracer=mock_tracer,
            video=mock_video,
            node_id=1
        )
        assert embedding is None


@pytest.mark.unit
@pytest.mark.gpu
def test_get_object_node_embedding_label(node_embeddings):
    """Test object node embedding generation (label-based)."""
    mock_checkpoint = MagicMock()
    node_data = {
        "object_label": "bowl",
        "visits": [(10, 20)]
    }
    mock_checkpoint.nodes = {1: node_data}
    with patch.object(node_embeddings.clip_model, 'encode_texts', return_value=[torch.ones((1, 768))]) as mock_encode_texts:
        embedding = node_embeddings.get_object_node_embedding_label(
            checkpoint=mock_checkpoint,
            node_id=1
        )
    assert embedding is not None
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape[1] == 768  # Should match CLIP output shape

    # Test missing label
    node_data_no_label = {"object_label": "", "visits": [(10, 20)]}
    mock_checkpoint.nodes = {2: node_data_no_label}
    embedding = node_embeddings.get_object_node_embedding_label(
        checkpoint=mock_checkpoint,
        node_id=2
    )
    assert embedding is None

    # Test missing node
    mock_checkpoint.nodes = {}
    embedding = node_embeddings.get_object_node_embedding_label(
        checkpoint=mock_checkpoint,
        node_id=3
    )
    assert embedding is None


@pytest.mark.integration
@pytest.mark.gpu
def test_object_node_embedding_with_real_data(node_embeddings, test_checkpoint, test_tracer, test_video):
    """Test embedding generation with real checkpoint, tracer, and video data."""
    # Find a node with visits
    node_id = None
    for nid, node in test_checkpoint.nodes.items():
        if 'visits' in node and node['visits']:
            node_id = nid
            break
    
    if node_id is None:
        pytest.skip("No nodes with visits found in test checkpoint")
    
    # Test ROI-based embedding
    embedding = node_embeddings.get_object_node_embedding_roi(
        checkpoint=test_checkpoint,
        tracer=test_tracer,
        video=test_video,
        node_id=node_id
    )
    if embedding is not None:
        assert isinstance(embedding, torch.Tensor)
        assert embedding.dim() == 1
        assert embedding.shape[0] == 768  # We assume ViT-L/14 CLIP model


@pytest.mark.unit
def test_object_label_embedding_cache(node_embeddings):
    """Test caching of object label embeddings with mocked clip_model."""
    mock_checkpoint = MagicMock()
    mock_node_data = {"object_label": "bowl"}
    mock_checkpoint.nodes = {1: mock_node_data}
    node_embeddings.clip_model.encode_texts = MagicMock()
    node_embeddings.clip_model.encode_texts.side_effect = [
        torch.ones((1, 768)), # First call
        torch.ones((1, 768)) * 2 # Second call (should not be used due to caching)
    ]
    # First call: should compute and cache
    embedding1 = node_embeddings.get_object_node_embedding_label(mock_checkpoint, 1)
    assert embedding1 is not None
    # Second call with same object label: should hit cache
    embedding2 = node_embeddings.get_object_node_embedding_label(mock_checkpoint, 1)
    assert torch.allclose(embedding1, embedding2)
    # Verify encode_texts was called only once
    assert node_embeddings.clip_model.encode_texts.call_count == 1
    # Different object label: should compute again
    mock_node_data2 = {"object_label": "spoon"}
    mock_checkpoint.nodes = {2: mock_node_data2}
    embedding3 = node_embeddings.get_object_node_embedding_label(mock_checkpoint, 2)
    assert node_embeddings.clip_model.encode_texts.call_count == 2


@pytest.mark.unit
def test_action_label_embedding_cache(node_embeddings):
    """Test caching of action label embeddings with mocked clip_model."""
    with patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_action_name_by_idx") as mock_get_action_name:
        mock_get_action_name.side_effect = ["take bowl", "take bowl", "put spoon"]
        node_embeddings.clip_model.encode_texts = MagicMock()
        node_embeddings.clip_model.encode_texts.side_effect = [
            torch.ones((1, 768)), # First call
            torch.ones((1, 768)) * 2 # Second call (should not be used due to caching) 
        ]
        # First call: should compute and cache
        embedding1 = node_embeddings.get_action_embedding(0)
        assert embedding1 is not None
        # Second call with same action label: should hit cache
        embedding2 = node_embeddings.get_action_embedding(0)
        assert torch.allclose(embedding1, embedding2)
        # Verify encode_texts was called only once
        assert node_embeddings.clip_model.encode_texts.call_count == 1
        # Different action label: should compute again
        embedding3 = node_embeddings.get_action_embedding(1)
        # Verify encode_texts was called again
        assert node_embeddings.clip_model.encode_texts.call_count == 2


@pytest.mark.integration
@pytest.mark.gpu
def test_roi_image_classification(node_embeddings, test_checkpoint, test_tracer, test_video):
    """Test that the best detection in the visit range is (mostly) correctly classified (top 3)."""
    object_labels = ActionRecord.get_noun_names()
    object_labels = [f"a photo of a {label.replace('_', ' ')}" for label in object_labels]
    clip_model = node_embeddings.clip_model
    node_ids = [nid for nid, node in test_checkpoint.nodes.items() 
               if 'visits' in node and node['visits']]
    
    assert len(node_ids) > 0, "No valid nodes with visits found in checkpoint"

    correct_count = 0
    total_count = 0

    for node_id in node_ids:
        node_data = test_checkpoint.nodes[node_id]
        object_label = node_data["object_label"]
        visit_start, visit_end = node_data["visits"][0]

        # Get all detections for the visit range
        detections = []
        for frame_idx in range(visit_start, visit_end + 1):
            detections.extend(test_tracer.get_detections_for_frame(frame_idx))

        # Filter matching detections
        matching_dets = [det for det in detections 
                        if det.class_name == object_label and det.is_fixated]
        print(f"Matching detections for object {object_label} (node {node_id}):")
        for det in matching_dets:
            print(f"  - Frame {det.frame_idx}, BBox: {det.bbox}, Score: {det.score}")
        assert len(matching_dets) > 0, f"No matching detections found for object {object_label} (node {node_id})"

        # Get the best detection
        best_det = max(matching_dets, key=lambda x: x.score)
        print(f"Best detection for object {object_label} (node {node_id}):")
        print(f"  - Frame {best_det.frame_idx}, BBox: {best_det.bbox}, Score: {best_det.score}")
        test_video.seek_to_frame(best_det.frame_idx)

        # Extract ROI from best detection
        frame = next(test_video.stream)
        frame_tensor = frame['data']
        # save frame to file
        frame_pil = Image.fromarray(frame_tensor.numpy().transpose(1, 2, 0))
        frame_pil.save(f"data/tests/out/test_node_embeddings_frame_{object_label}_node_{node_id}.png")
        roi_tensor = node_embeddings._extract_roi(frame_tensor, best_det.bbox, padding=10)
        pil_image = NodeEmbeddings._convert_roi_tensor_to_pil(roi_tensor)
        # save pil_image to file
        pil_image.save(f"data/tests/out/test_node_embeddings_roi_{object_label}_node_{node_id}.png")

        # Classify ROI
        scores, best_label = clip_model.classify(object_labels, pil_image)
        
        # Convert scores to tensor and apply softmax
        scores_tensor = torch.tensor(scores)
        scores = torch.nn.functional.softmax(scores_tensor, dim=0).tolist()
        
        # Sort scores and labels
        sorted_results = sorted(zip(scores, object_labels), key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_labels = zip(*sorted_results)

        # Print results
        print(f"Top 3 labels for object {object_label} (node {node_id}):")
        for label, score in list(zip(sorted_labels, sorted_scores))[:3]:
            print(f"  - {label}: {score}")

        expected_label = f"a photo of a {object_label.replace('_', ' ')}"
        if expected_label in sorted_labels[:3]:
            print(f"Correctly classified object {object_label} (node {node_id})")
            correct_count += 1
        else:
            print(f"Incorrectly classified object {object_label} (node {node_id})")
        total_count += 1
    
    accuracy = correct_count / total_count
    print(f"Accuracy: {accuracy}")
    assert accuracy > 0.5, f"Accuracy is too low: {accuracy}"


@pytest.fixture
def cleanup_cache_files():
    """Fixture to clean up cache files before and after tests."""
    # Get cache paths from config
    config = get_config()
    object_path = Path(config.dataset.embeddings.object_label_embedding_path)
    action_path = Path(config.dataset.embeddings.action_label_embedding_path)
    
    # Remove cache files before test if they exist
    if object_path.exists():
        object_path.unlink()
    if action_path.exists():
        action_path.unlink()
    
    yield
    
    # Clean up after test
    if object_path.exists():
        object_path.unlink()
    if action_path.exists():
        action_path.unlink()


@pytest.mark.unit
def test_prepopulate_caches(device, cleanup_cache_files):
    """Test that caches are prepopulated with all noun and action labels."""
    # Mock the ActionRecord methods
    mock_noun_names = ["bowl", "spoon", "cup"]
    mock_action_names = {0: "take bowl", 1: "put spoon", 2: "wash cup"}
    
    with patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_noun_names", return_value=mock_noun_names), \
         patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_action_names", return_value=mock_action_names):
        
        # Create mock clip model that returns predictable embeddings
        mock_clip_model = MagicMock()
        mock_clip_model.encode_texts.side_effect = lambda texts: [torch.ones((1, 768)) * i for i, _ in enumerate(texts)]
        
        with patch("gazegraph.models.clip.ClipModel", return_value=mock_clip_model):
            # Initialize with prepopulate_caches=True
            node_embeddings = NodeEmbeddings(device=device, prepopulate_caches=True)
            
            # Check that caches were populated
            assert len(node_embeddings._object_label_embedding_cache) == len(mock_noun_names)
            assert len(node_embeddings._action_label_embedding_cache) == len(mock_action_names)
            
            # Check that cache files were created
            assert node_embeddings.object_label_embedding_path.exists()
            assert node_embeddings.action_label_embedding_path.exists()


@pytest.mark.unit
def test_load_caches(device, cleanup_cache_files):
    """Test that caches are loaded from files if they exist."""
    # Get config for paths
    config = get_config()
    object_path = Path(config.dataset.embeddings.object_label_embedding_path)
    action_path = Path(config.dataset.embeddings.action_label_embedding_path)
    
    # Create mock cache data
    mock_object_cache = {"bowl": torch.ones((1, 768)), "spoon": torch.ones((1, 768)) * 2}
    mock_action_cache = {"take bowl": torch.ones((1, 768)) * 3, "put spoon": torch.ones((1, 768)) * 4}
    
    # Create cache directory if it doesn't exist
    os.makedirs(object_path.parent, exist_ok=True)
    
    # Save mock caches to files
    torch.save(mock_object_cache, object_path)
    torch.save(mock_action_cache, action_path)
    
    # Initialize with prepopulate_caches=False to test only loading
    node_embeddings = NodeEmbeddings(device=device, prepopulate_caches=False)
    
    # Check that caches were loaded correctly
    assert len(node_embeddings._object_label_embedding_cache) == len(mock_object_cache)
    assert len(node_embeddings._action_label_embedding_cache) == len(mock_action_cache)
    
    # Check that the loaded embeddings match the mock data
    for label, embedding in mock_object_cache.items():
        assert label in node_embeddings._object_label_embedding_cache
        assert torch.allclose(node_embeddings._object_label_embedding_cache[label], embedding)
    
    for label, embedding in mock_action_cache.items():
        assert label in node_embeddings._action_label_embedding_cache
        assert torch.allclose(node_embeddings._action_label_embedding_cache[label], embedding)


@pytest.mark.unit
def test_prepopulate_and_load_caches_integration(device, cleanup_cache_files):
    """Test the full cycle of prepopulating, saving, and loading caches."""
    # Mock the ActionRecord methods
    mock_noun_names = ["bowl", "spoon", "cup"]
    mock_action_names = {0: "take bowl", 1: "put spoon", 2: "wash cup"}
    
    with patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_noun_names", return_value=mock_noun_names), \
         patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_action_names", return_value=mock_action_names):
        
        # First instance: prepopulate and save caches
        first_instance = NodeEmbeddings(device=device, prepopulate_caches=True)
        
        # Check that cache files were created
        assert first_instance.object_label_embedding_path.exists()
        assert first_instance.action_label_embedding_path.exists()
        
        # Store the original cache contents for comparison
        original_object_cache = first_instance._object_label_embedding_cache.copy()
        original_action_cache = first_instance._action_label_embedding_cache.copy()
        
        # Second instance: should load from cache files
        # Mock the methods to return empty values to ensure we're loading from cache
        with patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_noun_names", return_value=[]), \
             patch("gazegraph.datasets.egtea_gaze.action_record.ActionRecord.get_action_names", return_value={}):
            
            second_instance = NodeEmbeddings(device=device, prepopulate_caches=True)
            
            # Check that caches were loaded correctly
            assert len(second_instance._object_label_embedding_cache) == len(original_object_cache)
            assert len(second_instance._action_label_embedding_cache) == len(original_action_cache)
            
            # Check that the loaded embeddings match the original data
            for label, embedding in original_object_cache.items():
                assert label in second_instance._object_label_embedding_cache
                assert torch.allclose(second_instance._object_label_embedding_cache[label], embedding)
            
            for label, embedding in original_action_cache.items():
                assert label in second_instance._action_label_embedding_cache
                assert torch.allclose(second_instance._action_label_embedding_cache[label], embedding)