"""
Unit and integration tests for Phase 2: Food Object Detection module.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.dataset import (
    create_detection_fixture,
    validate_bbox,
    verify_class_mapping,
    xywhn_to_xyxy,
    xyxy_to_xywhn,
)
from src.detection.detector import Detection, FoodDetector
from src.detection.evaluate import box_iou, compute_detection_metrics
from src.detection.visualize import draw_detections, save_annotated_image
from src.utils.schema import FoodInstance


# ---------------------------------------------------------------------------
# 1. Output Schema & Dataclass Tests
# ---------------------------------------------------------------------------

def test_detection_dataclass_and_schema():
    det = Detection(
        class_id=3,
        class_name="apple",
        bbox=[12.5, 24.3, 105.8, 120.4],
        confidence=0.9234,
    )
    d = det.as_dict()
    assert d["class_id"] == 3
    assert d["class_name"] == "apple"
    assert d["bbox"] == [12.5, 24.3, 105.8, 120.4]
    assert d["confidence"] == 0.9234

    # Connects to downstream FoodInstance
    inst = det.to_food_instance(instance_id="food_1")
    assert isinstance(inst, FoodInstance)
    assert inst.class_name == "apple"
    assert inst.class_confidence == 0.9234
    assert inst.bbox == (12.5, 24.3, 105.8, 120.4)


def test_detection_dataclass_validation():
    with pytest.raises(ValueError):
        # Invalid bbox length
        Detection(class_id=0, class_name="x", bbox=[10.0, 20.0, 30.0], confidence=0.5)

    with pytest.raises(ValueError):
        # Invalid confidence
        Detection(class_id=0, class_name="x", bbox=[10.0, 20.0, 30.0, 40.0], confidence=1.5)


# ---------------------------------------------------------------------------
# 2. Bounding Box Transformations & Validation
# ---------------------------------------------------------------------------

def test_xyxy_xywhn_roundtrip():
    img_w, img_h = 640, 480
    orig_box = [100.0, 120.0, 300.0, 360.0]
    xywhn = xyxy_to_xywhn(orig_box, img_w, img_h)

    assert 0.0 <= xywhn[0] <= 1.0
    assert 0.0 <= xywhn[1] <= 1.0
    assert 0.0 <= xywhn[2] <= 1.0
    assert 0.0 <= xywhn[3] <= 1.0

    recovered = xywhn_to_xyxy(xywhn, img_w, img_h)
    np.testing.assert_allclose(recovered, orig_box, atol=1e-3)


def test_validate_bbox_clamping_and_rejection():
    img_w, img_h = 100, 100

    # Normal valid box
    box, valid = validate_bbox([10, 15, 80, 85], img_w, img_h)
    assert valid is True
    assert box == [10.0, 15.0, 80.0, 85.0]

    # Inverted coordinates (x2 <= x1)
    _, valid = validate_bbox([80, 15, 10, 85], img_w, img_h)
    assert valid is False

    # Out of bounds (clamps within [0, 100])
    box, valid = validate_bbox([-10, -5, 120, 110], img_w, img_h)
    assert valid is True
    assert box == [0.0, 0.0, 100.0, 100.0]

    # Degenerate tiny box
    _, valid = validate_bbox([10, 10, 10.5, 10.5], img_w, img_h)
    assert valid is False


# ---------------------------------------------------------------------------
# 3. Class Mapping Verification
# ---------------------------------------------------------------------------

def test_verify_class_mapping():
    classes_list = ["pizza", "burger", "salad"]
    cmap = verify_class_mapping(classes_list)
    assert cmap == {0: "pizza", 1: "burger", 2: "salad"}

    dict_gap = {1: "apple", 5: "banana"}
    cmap2 = verify_class_mapping(dict_gap)
    assert cmap2 == {0: "apple", 1: "banana"}


# ---------------------------------------------------------------------------
# 4. Evaluation Metrics (Precision, Recall, mAP@50, mAP@50:95)
# ---------------------------------------------------------------------------

def test_box_iou_calculation():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [0.0, 0.0, 10.0, 10.0]
    assert np.isclose(box_iou(box1, box2), 1.0)

    # Disjoint
    box3 = [20.0, 20.0, 30.0, 30.0]
    assert np.isclose(box_iou(box1, box3), 0.0)

    # Partial overlap (5x10 overlap = 50 area. Union = 100 + 100 - 50 = 150 -> 50/150 = 1/3)
    box4 = [5.0, 0.0, 15.0, 10.0]
    assert np.isclose(box_iou(box1, box4), 50.0 / 150.0)


def test_compute_detection_metrics():
    gts = [
        [{"class_id": 0, "bbox": [10.0, 10.0, 50.0, 50.0]}],
        [{"class_id": 1, "bbox": [20.0, 20.0, 60.0, 60.0]}],
    ]
    preds = [
        [{"class_id": 0, "bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.95}],
        [{"class_id": 1, "bbox": [20.0, 20.0, 60.0, 60.0], "confidence": 0.90}],
    ]
    metrics = compute_detection_metrics(gts, preds, class_names=["apple", "orange"])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["map50"] == 1.0
    assert metrics["map50_95"] > 0.9


# ---------------------------------------------------------------------------
# 5. Visualization Generation
# ---------------------------------------------------------------------------

def test_visualization_generation(tmp_path):
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    detections = [
        Detection(class_id=0, class_name="apple", bbox=[20.0, 20.0, 80.0, 90.0], confidence=0.88),
        Detection(class_id=1, class_name="sandwich", bbox=[100.0, 110.0, 180.0, 175.0], confidence=0.74),
    ]
    out_file = tmp_path / "annotated.jpg"
    saved_path = save_annotated_image(img, detections, out_file)
    assert saved_path.is_file()
    assert saved_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 6. Detection Fixture Creation
# ---------------------------------------------------------------------------

def test_create_detection_fixture(tmp_path):
    data_yaml = create_detection_fixture(tmp_path / "fixture_dataset", num_train=2, num_val=1, num_test=1)
    assert data_yaml.is_file()
    assert (tmp_path / "fixture_dataset" / "images" / "train" / "sample_000.jpg").is_file()
    assert (tmp_path / "fixture_dataset" / "labels" / "train" / "sample_000.txt").is_file()


# ---------------------------------------------------------------------------
# 7. FoodDetector Smoke Test (End-to-End Inference with YOLOv8n)
# ---------------------------------------------------------------------------

def test_food_detector_smoke(tmp_path):
    weights_path = Path("models/yolov8n.pt")
    if not weights_path.is_file():
        pytest.skip("models/yolov8n.pt not found on disk")

    detector = FoodDetector(weights_path=weights_path, device="cpu")
    detector.load()
    assert detector.is_loaded is True

    # Test inference on a simple image
    test_img = Image.new("RGB", (320, 320), color=(180, 180, 180))
    detections = detector.predict(test_img)
    assert isinstance(detections, list)

    # Output schema compliance for all detections
    for det in detections:
        assert isinstance(det.class_id, int)
        assert isinstance(det.class_name, str)
        assert len(det.bbox) == 4
        assert 0.0 <= det.confidence <= 1.0
        d = det.as_dict()
        assert "class_id" in d and "class_name" in d and "bbox" in d and "confidence" in d
