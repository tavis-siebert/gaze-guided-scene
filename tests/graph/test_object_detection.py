import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from gazegraph.graph.object_detection import ObjectDetector, Detection, ScoreComponents
from gazegraph.graph.gaze import GazePoint, GazeType
from gazegraph.config.config_utils import DotDict


class TestScoreComponents:
    def test_to_dict(self):
        components = ScoreComponents(
            confidence=0.8,
            stability=0.7,
            gaze_proximity=0.9,
            fixation_ratio=0.5,
            duration_weighted=0.6,
            gaze_distance=10.0
        )
        
        result = components.to_dict()
        
        assert result["confidence"] == 0.8
        assert result["stability"] == 0.7
        assert result["gaze_proximity"] == 0.9
        assert result["fixation_ratio"] == 0.5
        assert result["duration_weighted"] == 0.6
        assert result["gaze_distance"] == 10.0
    
    def test_from_dict(self):
        data = {
            "confidence": 0.8,
            "stability": 0.7,
            "gaze_proximity": 0.9,
            "fixation_ratio": 0.5,
            "duration_weighted": 0.6,
            "gaze_distance": 10.0
        }
        
        components = ScoreComponents.from_dict(data)
        
        assert components.confidence == 0.8
        assert components.stability == 0.7
        assert components.gaze_proximity == 0.9
        assert components.fixation_ratio == 0.5
        assert components.duration_weighted == 0.6
        assert components.gaze_distance == 10.0
    
    def test_from_dict_partial(self):
        data = {
            "confidence": 0.8,
            "stability": 0.7
        }
        
        components = ScoreComponents.from_dict(data)
        
        assert components.confidence == 0.8
        assert components.stability == 0.7
        assert components.gaze_proximity == 0.0
        assert components.fixation_ratio == 0.0
        assert components.duration_weighted == 0.0
        assert components.gaze_distance == 0.0


class TestDetection:
    def test_to_dict(self):
        components = ScoreComponents(
            confidence=0.8,
            stability=0.7,
            gaze_proximity=0.9
        )
        
        detection = Detection(
            bbox=(10, 20, 30, 40),
            class_name="cup",
            score=0.95,
            class_id=1,
            frame_idx=5,
            is_fixated=True,
            is_top_scoring=True,
            fixation_score=0.85,
            components=components
        )
        
        result = detection.to_dict()
        
        assert result["detection"]["bbox"] == (10, 20, 30, 40)
        assert result["detection"]["class_name"] == "cup"
        assert result["detection"]["score"] == 0.95
        assert result["detection"]["class_id"] == 1
        assert result["detection"]["frame_idx"] == 5
        
        assert result["fixation"]["is_fixated"] == True
        assert result["fixation"]["is_top_scoring"] == True
        assert result["fixation"]["score"] == 0.85
        
        assert result["fixation"]["components"]["confidence"] == 0.8
        assert result["fixation"]["components"]["stability"] == 0.7
        assert result["fixation"]["components"]["gaze_proximity"] == 0.9
    
    def test_from_dict(self):
        data = {
            "detection": {
                "bbox": [10, 20, 30, 40],
                "class_name": "cup",
                "score": 0.95,
                "class_id": 1,
                "frame_idx": 5
            },
            "fixation": {
                "is_fixated": True,
                "is_top_scoring": True,
                "score": 0.85,
                "components": {
                    "confidence": 0.8,
                    "stability": 0.7,
                    "gaze_proximity": 0.9
                }
            }
        }
        
        detection = Detection.from_dict(data)
        
        assert detection.bbox == (10, 20, 30, 40)
        assert detection.class_name == "cup"
        assert detection.score == 0.95
        assert detection.class_id == 1
        assert detection.frame_idx == 5
        assert detection.is_fixated == True
        assert detection.is_top_scoring == True
        assert detection.fixation_score == 0.85
        assert detection.components.confidence == 0.8
        assert detection.components.stability == 0.7
        assert detection.components.gaze_proximity == 0.9
    
    def test_from_dict_override_frame_idx(self):
        data = {
            "detection": {
                "bbox": [10, 20, 30, 40],
                "class_name": "cup",
                "score": 0.95,
                "class_id": 1,
                "frame_idx": 5
            },
            "fixation": {
                "is_fixated": True,
                "is_top_scoring": False,
                "score": 0.85,
                "components": {}
            }
        }
        
        detection = Detection.from_dict(data, frame_idx=10)
        
        assert detection.frame_idx == 10


@pytest.fixture
def mock_config():
    config = DotDict({
        "models": {
            "yolo_world": {
                "backend": "torch"
            }
        },
        "graph": {
            "min_fixation_frame_ratio": 0.3,
            "fixated_object_detection": {
                "weights": {
                    "bbox_stability": 0.3,
                    "gaze_proximity": 0.3,
                    "confidence": 0.3,
                    "duration": 0.3
                },
                "thresholds": {
                    "bbox_stability": 0.5,
                    "gaze_proximity": 0.3,
                    "confidence": 0.6
                }
            }
        }
    })
    return config


@pytest.fixture
def mock_detector(mock_config):
    with patch("gazegraph.graph.object_detection.YOLOWorldModel") as mock_model_class:
        mock_model = MagicMock()
        mock_model.conf_threshold = 0.6
        mock_model.iou_threshold = 0.5
        mock_model_class.create.return_value = mock_model
        
        obj_labels = {0: "cup", 1: "bowl", 2: "spoon"}
        labels_to_int = {"cup": 0, "bowl": 1, "spoon": 2}
        
        detector = ObjectDetector(
            model_path=Path("dummy/path"),
            obj_labels=obj_labels,
            labels_to_int=labels_to_int,
            config=mock_config,
            tracer=None
        )
        
        yield detector


class TestObjectDetector:
    def test_init(self, mock_detector, mock_config):
        assert mock_detector.min_fixation_frame_ratio == mock_config.graph.min_fixation_frame_ratio
        assert mock_detector.bbox_stability_weight == mock_config.graph.fixated_object_detection.weights.bbox_stability
        assert mock_detector.gaze_proximity_weight == mock_config.graph.fixated_object_detection.weights.gaze_proximity
        assert mock_detector.confidence_weight == mock_config.graph.fixated_object_detection.weights.confidence
        assert mock_detector.duration_weight == mock_config.graph.fixated_object_detection.weights.duration
        assert mock_detector.bbox_stability_threshold == mock_config.graph.fixated_object_detection.thresholds.bbox_stability
        assert mock_detector.gaze_proximity_threshold == mock_config.graph.fixated_object_detection.thresholds.gaze_proximity
        assert mock_detector.confidence_threshold == mock_config.graph.fixated_object_detection.thresholds.confidence
    
    def test_reset(self, mock_detector):
        mock_detector.fixated_objects_found = True
        mock_detector.all_detections = [1, 2, 3]
        mock_detector.gaze_points = [(0, 0, 0)]
        mock_detector.total_frames = 10
        mock_detector.fixation_scores = {"cup": 0.8}
        
        mock_detector.reset()
        
        assert mock_detector.fixated_objects_found == False
        assert mock_detector.all_detections == []
        assert mock_detector.gaze_points == []
        assert mock_detector.total_frames == 0
        assert mock_detector.fixation_scores == {}
    
    def test_compute_bbox_iou(self, mock_detector):
        # Completely overlapping boxes
        box1 = (10, 10, 20, 20)
        box2 = (10, 10, 20, 20)
        assert mock_detector._compute_bbox_iou(box1, box2) == 1.0
        
        # No overlap
        box1 = (10, 10, 20, 20)
        box2 = (40, 40, 20, 20)
        assert mock_detector._compute_bbox_iou(box1, box2) == 0.0
        
        # Partial overlap
        box1 = (10, 10, 20, 20)
        box2 = (20, 20, 20, 20)
        iou = mock_detector._compute_bbox_iou(box1, box2)
        assert 0.0 < iou < 1.0
    
    def test_compute_mean_iou(self, mock_detector):
        # Single box should have perfect stability
        bboxes = [(10, 10, 20, 20)]
        assert mock_detector._compute_mean_iou(bboxes) == 1.0
        
        # Same boxes should have perfect stability
        bboxes = [(10, 10, 20, 20), (10, 10, 20, 20), (10, 10, 20, 20)]
        assert mock_detector._compute_mean_iou(bboxes) == 1.0
        
        # Different boxes should have lower stability
        bboxes = [(10, 10, 20, 20), (15, 15, 20, 20), (20, 20, 20, 20)]
        assert mock_detector._compute_mean_iou(bboxes) < 1.0
    
    def test_compute_bbox_center(self, mock_detector):
        bbox = (10, 20, 30, 40)
        center = mock_detector._compute_bbox_center(bbox)
        assert center == (25, 40)
    
    def test_compute_distance(self, mock_detector):
        point1 = (0, 0)
        point2 = (3, 4)
        assert mock_detector._compute_distance(point1, point2) == 5.0
    
    def test_compute_gaze_distance(self, mock_detector):
        bbox = (10, 10, 20, 20)  # center is at (20, 20)
        gaze = (10, 10)  # corner
        
        # Distance from corner to center
        expected_distance = np.sqrt((10 - 20)**2 + (10 - 20)**2)
        assert mock_detector._compute_gaze_distance(bbox, gaze) == expected_distance
        
        # Gaze at center should have zero distance
        gaze = (20, 20)
        assert mock_detector._compute_gaze_distance(bbox, gaze) == 0.0
    
    def test_compute_mean_gaze_distance(self, mock_detector):
        bboxes = [(10, 10, 20, 20), (12, 12, 20, 20)]  # centers at (20, 20) and (22, 22)
        gaze_points = [(10, 10, 0), (12, 12, 1)]  # frame_idx 0 and 1
        
        # Distance from (10, 10) to (20, 20) = 14.14
        # Distance from (12, 12) to (22, 22) = 14.14
        # Mean = 14.14
        expected_mean = np.sqrt((10 - 20)**2 + (10 - 20)**2)
        assert abs(mock_detector._compute_mean_gaze_distance(bboxes, gaze_points) - expected_mean) < 0.1
        
        # Empty inputs should return infinity
        assert mock_detector._compute_mean_gaze_distance([], gaze_points) == float('inf')
        assert mock_detector._compute_mean_gaze_distance(bboxes, []) == float('inf')
    
    def test_compute_geometric_mean(self, mock_detector):
        # Test with uniform values
        values = [2, 2, 2, 2]
        assert mock_detector._compute_geometric_mean(values) == 2.0
        
        # Test with varying values
        values = [1, 10, 100]
        # (1 * 10 * 100)^(1/3) = 10^2^(1/3) = 10^(2/3) = 10^0.667 = 4.64
        expected = (1 * 10 * 100)**(1/3)
        assert abs(mock_detector._compute_geometric_mean(values) - expected) < 0.01
        
        # Test with empty list
        assert mock_detector._compute_geometric_mean([]) == 0.0
    
    def test_compute_fixation_scores(self, mock_detector):
        # Set up real method mocking
        with patch.object(ObjectDetector, "_compute_mean_iou", return_value=0.7), \
             patch.object(ObjectDetector, "_compute_geometric_mean", return_value=0.8), \
             patch.object(ObjectDetector, "_compute_mean_gaze_distance", return_value=10.0):
            
            # Temporarily lower the thresholds to ensure detections pass
            orig_conf_threshold = mock_detector.confidence_threshold
            orig_stability_threshold = mock_detector.bbox_stability_threshold
            orig_gaze_threshold = mock_detector.gaze_proximity_threshold
            
            mock_detector.confidence_threshold = 0.5  # Lower than 0.8
            mock_detector.bbox_stability_threshold = 0.5  # Lower than 0.7
            mock_detector.gaze_proximity_threshold = 0.05  # Lower than 1.0/(1.0+10.0) = 0.09
            
            try:
                # Create detections for testing
                mock_detector.all_detections = [
                    Detection(
                        bbox=(10, 10, 20, 20),
                        class_name="cup",
                        score=0.8,
                        class_id=0,
                        frame_idx=0,
                        is_fixated=True
                    ),
                    Detection(
                        bbox=(12, 12, 20, 20),
                        class_name="cup",
                        score=0.9,
                        class_id=0,
                        frame_idx=1,
                        is_fixated=True
                    ),
                    Detection(
                        bbox=(40, 40, 20, 20),
                        class_name="bowl",
                        score=0.7,
                        class_id=1,
                        frame_idx=0,
                        is_fixated=True
                    ),
                    Detection(
                        bbox=(42, 42, 20, 20),
                        class_name="bowl",
                        score=0.6,
                        class_id=1,
                        frame_idx=1,
                        is_fixated=False  # This one should be ignored for bowl
                    )
                ]
                
                # Set up gaze points
                mock_detector.gaze_points = [(15, 15, 0), (20, 20, 1)]
                
                # Set total frames
                mock_detector.total_frames = 3  # For fixation ratio calculation
                
                # Run the method
                scores = mock_detector._compute_fixation_scores()
                
                # Check that two objects were scored (cup and bowl)
                assert len(scores) == 2
                assert "cup" in scores
                assert "bowl" in scores
                
                # Check cup components
                cup_scores = scores["cup"]
                assert "fixation_ratio" in cup_scores
                assert cup_scores["fixation_ratio"] == 2/3  # 2 frames out of 3
                assert "confidence" in cup_scores
                assert cup_scores["confidence"] == 0.8  # From mock
                assert "stability" in cup_scores
                assert cup_scores["stability"] == 0.7  # From mock
                assert "gaze_proximity" in cup_scores
                assert cup_scores["gaze_proximity"] == 1.0 / (1.0 + 10.0)
                assert "duration_weighted" in cup_scores
                assert cup_scores["duration_weighted"] == 0.8 * (2/3)
                assert "final_score" in cup_scores
                
                # Check that bowl scores exist but are different
                bowl_scores = scores["bowl"]
                assert bowl_scores["fixation_ratio"] == 1/3  # 1 frame out of 3
                
                # Check that the fixation_scores is updated
                assert mock_detector.fixation_scores["cup"] == cup_scores["final_score"]
                assert mock_detector.fixation_scores["bowl"] == bowl_scores["final_score"]
            
            finally:
                # Restore original thresholds
                mock_detector.confidence_threshold = orig_conf_threshold
                mock_detector.bbox_stability_threshold = orig_stability_threshold
                mock_detector.gaze_proximity_threshold = orig_gaze_threshold
    
    def test_compute_fixation_scores_with_filtering(self, mock_detector):
        # Mock necessary methods with patch.object
        with patch.object(ObjectDetector, "_compute_mean_iou") as mock_iou,\
             patch.object(ObjectDetector, "_compute_geometric_mean") as mock_geo,\
             patch.object(ObjectDetector, "_compute_mean_gaze_distance") as mock_dist:
            
            # Configure mock returns for different objects
            def mock_iou_side_effect(bboxes):
                if bboxes[0][0] == 10:  # cup: Passes stability threshold
                    return 0.7
                else:  # bowl or spoon: Fails stability threshold
                    return 0.3
            
            def mock_geo_side_effect(scores):
                if scores[0] == 0.8:  # cup: Passes confidence threshold
                    return 0.8
                else:  # bowl or spoon: Fails confidence threshold
                    return 0.5
            
            def mock_dist_side_effect(bboxes, gaze_points):
                if bboxes[0][0] == 10:  # cup: Good gaze proximity
                    return 2.0
                else:  # bowl or spoon: Poor gaze proximity
                    return 20.0
            
            mock_iou.side_effect = mock_iou_side_effect
            mock_geo.side_effect = mock_geo_side_effect
            mock_dist.side_effect = mock_dist_side_effect
            
            # Create test data
            mock_detector.all_detections = [
                # Cup: should pass all filters
                Detection(
                    bbox=(10, 10, 20, 20),
                    class_name="cup",
                    score=0.8,
                    class_id=0,
                    frame_idx=0,
                    is_fixated=True
                ),
                Detection(
                    bbox=(12, 12, 20, 20),
                    class_name="cup",
                    score=0.85,
                    class_id=0,
                    frame_idx=1,
                    is_fixated=True
                ),
                # Bowl: should fail stability filter
                Detection(
                    bbox=(30, 30, 20, 20),
                    class_name="bowl",
                    score=0.5,
                    class_id=1,
                    frame_idx=0,
                    is_fixated=True
                ),
                Detection(
                    bbox=(50, 50, 20, 20),
                    class_name="bowl",
                    score=0.55,
                    class_id=1,
                    frame_idx=1,
                    is_fixated=True
                ),
                # Spoon: should fail confidence filter
                Detection(
                    bbox=(70, 70, 20, 20),
                    class_name="spoon",
                    score=0.5,
                    class_id=2,
                    frame_idx=0,
                    is_fixated=True
                ),
                Detection(
                    bbox=(72, 72, 20, 20),
                    class_name="spoon",
                    score=0.55,
                    class_id=2,
                    frame_idx=1,
                    is_fixated=True
                )
            ]
            
            mock_detector.total_frames = 4
            mock_detector.gaze_points = [(15, 15, 0), (18, 18, 1)]
            
            # Run the method
            scores = mock_detector._compute_fixation_scores()
            
            # Check results - only cup should pass
            assert len(scores) == 1
            assert "cup" in scores
            assert "bowl" not in scores
            assert "spoon" not in scores
            
            # Check filter statistics
            assert mock_detector.filtered_stats["total_considered"] == 3
            assert mock_detector.filtered_stats["passed_all"] == 1
            assert mock_detector.filtered_stats["confidence"] == 2  # Both bowl and spoon fail here
            assert mock_detector.filtered_stats["stability"] == 0  # None get filtered here because of confidence filtering
            assert mock_detector.filtered_stats["gaze_proximity"] == 0  # None get filtered here
            
            # Check that final scores match
            assert mock_detector.fixation_scores == {"cup": scores["cup"]["final_score"]}
    
    @patch("gazegraph.graph.object_detection.Image")
    def test_perform_detection(self, mock_image, mock_detector):
        # Mock frame and gaze point
        frame = torch.zeros(3, 100, 100)
        gaze_x, gaze_y = 50, 50
        frame_idx = 0
        
        # Mock detections returned by model
        mock_detector.model.predict.return_value = [
            {"bbox": (40, 40, 20, 20), "class_name": "cup", "score": 0.9, "class_id": 0},  # Intersects with gaze
            {"bbox": (10, 10, 20, 20), "class_name": "bowl", "score": 0.8, "class_id": 1}  # Does not intersect
        ]
        
        # Run detection
        detections = mock_detector._perform_detection(frame, gaze_x, gaze_y, frame_idx)
        
        # Check results
        assert len(detections) == 2
        
        # First object should be fixated (gaze intersects)
        assert detections[0].class_name == "cup"
        assert detections[0].is_fixated == True
        assert detections[0].frame_idx == frame_idx
        
        # Second object should not be fixated
        assert detections[1].class_name == "bowl"
        assert detections[1].is_fixated == False
    
    @patch.object(ObjectDetector, "_perform_detection")
    @patch.object(ObjectDetector, "_compute_fixation_scores")
    def test_detect_objects(self, mock_compute_scores, mock_perform_detection, mock_detector):
        # Setup mocks
        frame = torch.zeros(3, 100, 100)
        gaze_point = GazePoint(
            x=0.5, 
            y=0.5, 
            raw_type=GazeType.FIXATION,
            type=GazeType.FIXATION,
            frame_idx=0
        )
        frame_idx = 0
        
        mock_detector.tracer = MagicMock()
        
        # Mock detections with fixated objects
        fixated_detection = Detection(
            bbox=(40, 40, 20, 20), 
            class_name="cup", 
            score=0.9, 
            class_id=0,
            frame_idx=frame_idx,
            is_fixated=True
        )
        
        non_fixated_detection = Detection(
            bbox=(10, 10, 20, 20), 
            class_name="bowl", 
            score=0.8, 
            class_id=1,
            frame_idx=frame_idx,
            is_fixated=False
        )
        
        mock_perform_detection.return_value = [fixated_detection, non_fixated_detection]
        
        # Mock fixation scores
        mock_compute_scores.return_value = {
            "cup": {
                "final_score": 0.85,
                "confidence": 0.9,
                "stability": 0.7,
                "gaze_proximity": 0.8,
                "fixation_ratio": 0.6,
                "duration_weighted": 0.5,
                "gaze_distance": 5.0
            }
        }
        
        # Run detection
        result = mock_detector.detect_objects(frame, gaze_point, frame_idx)
        
        # Verify results
        assert len(result) == 2
        assert mock_detector.fixated_objects_found == True
        assert len(mock_detector.all_detections) == 2
        assert len(mock_detector.gaze_points) == 1
        assert mock_detector.total_frames == 1
        
        # Check that fixation scores were applied to the detection
        assert result[0].fixation_score == 0.85
        assert result[0].is_top_scoring == True
        assert result[0].components.confidence == 0.9
        assert result[0].components.stability == 0.7
        assert result[0].components.gaze_proximity == 0.8
        
        # Check tracer was called
        mock_detector.tracer.log_yolo_objects_detected.assert_called_once()
        
    def test_detect_objects_error_handling(self, mock_detector):
        # Setup mocks
        frame = torch.zeros(3, 100, 100)
        gaze_point = GazePoint(
            x=0.5, 
            y=0.5, 
            raw_type=GazeType.FIXATION,
            type=GazeType.FIXATION,
            frame_idx=0
        )
        frame_idx = 0
        
        # Make _perform_detection raise an exception
        with patch.object(ObjectDetector, "_perform_detection", side_effect=RuntimeError("Simulated error")):
            # Run detection - should not raise exception
            result = mock_detector.detect_objects(frame, gaze_point, frame_idx)
            
            # Should return empty list
            assert result == []
            # Verify the fixated_objects_found flag was not set
            assert not mock_detector.fixated_objects_found
            # Verify that we didn't add any detections or gaze points
            assert len(mock_detector.all_detections) == 0
            assert len(mock_detector.gaze_points) == 0
            
            # Per the implementation, total_frames is NOT incremented during error handling
            assert mock_detector.total_frames == 0
    
    def test_has_fixated_objects_empty(self, mock_detector):
        mock_detector.fixated_objects_found = False
        assert mock_detector.has_fixated_objects() == False
    
    @patch.object(ObjectDetector, "_compute_fixation_scores")
    def test_has_fixated_objects_with_scores(self, mock_compute_scores, mock_detector):
        mock_detector.fixated_objects_found = True
        mock_detector.fixation_scores = {"cup": 0.8}
        
        assert mock_detector.has_fixated_objects() == True
        mock_compute_scores.assert_not_called()
    
    def test_has_fixated_objects_compute_scores(self, mock_detector):
        # Setup the initial state
        mock_detector.fixated_objects_found = True
        mock_detector.fixation_scores = {}
        
        # Create a real spy for _compute_fixation_scores to track calls
        with patch.object(ObjectDetector, "_compute_fixation_scores", wraps=mock_detector._compute_fixation_scores) as spy:
            # Mock the internal implementation to return scores
            def side_effect():
                mock_detector.fixation_scores = {"cup": 0.8}
                return {"cup": {"final_score": 0.8}}
            
            spy.side_effect = side_effect
            
            # Call the method being tested
            result = mock_detector.has_fixated_objects()
            
            # Verify results
            assert result == True
            assert spy.call_count == 1
    
    @patch.object(ObjectDetector, "_compute_fixation_scores")
    def test_get_fixated_object(self, mock_compute_scores, mock_detector):
        # Setup fixation scores
        mock_detector.fixation_scores = {"cup": 0.8, "bowl": 0.6}
        
        # Get fixated object
        obj, score = mock_detector.get_fixated_object()
        
        # Should return highest scoring object
        assert obj == "cup"
        assert score == 0.8
        mock_compute_scores.assert_not_called()
    
    def test_get_fixated_object_no_scores(self, mock_detector):
        mock_detector.fixation_scores = {}
        mock_detector._compute_fixation_scores = MagicMock(return_value={})
        
        with pytest.raises(ValueError, match="No fixated objects found"):
            mock_detector.get_fixated_object()
    
    def test_get_fixated_object_compute_scores(self, mock_detector):
        # Empty scores initially
        mock_detector.fixation_scores = {}
        
        # Create a spy for _compute_fixation_scores
        with patch.object(ObjectDetector, "_compute_fixation_scores", wraps=mock_detector._compute_fixation_scores) as spy:
            # Mock the implementation behavior
            def side_effect():
                mock_detector.fixation_scores = {"cup": 0.8, "bowl": 0.6}
                return {"cup": {"final_score": 0.8}, "bowl": {"final_score": 0.6}}
            
            spy.side_effect = side_effect
            
            # Get fixated object
            obj, score = mock_detector.get_fixated_object()
            
            # Should compute scores then return highest
            assert obj == "cup"
            assert score == 0.8
            assert spy.call_count == 1 