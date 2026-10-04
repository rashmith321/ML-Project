"""
Unit and integration tests for Phase 3: Food Image Segmentation module.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.detector import Detection
from src.segmentation.dataset import FoodSegmentationDataset, create_segmentation_fixture
from src.segmentation.evaluate import calculate_segmentation_metrics
from src.segmentation.models import DiceBCELoss, FoodUNet
from src.segmentation.sam_utils import MaskType, SAMSegmenter
from src.segmentation.segmenter import FoodSegmenter
from src.segmentation.visualize import (
    create_mask_overlay,
    render_4_panel_comparison,
    save_segmentation_visualization,
)
from src.utils.schema import FoodInstance


# ---------------------------------------------------------------------------
# 1. Mask Provenance & Scientific Rule Tests
# ---------------------------------------------------------------------------

def test_mask_type_provenance():
    assert MaskType.GROUND_TRUTH.value == "ground_truth"
    assert MaskType.PSEUDO.value == "pseudo"
    assert MaskType.PREDICTED.value == "predicted"

    # Pseudo-masks from SAM must be marked explicitly as PSEUDO
    sam = SAMSegmenter(checkpoint_path=None)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        Detection(class_id=0, class_name="apple", bbox=[10.0, 10.0, 50.0, 50.0], confidence=0.9)
    ]
    pseudo_results = sam.generate_pseudo_masks(dummy_img, detections)
    assert len(pseudo_results) == 1
    assert pseudo_results[0]["mask_type"] == MaskType.PSEUDO.value
    assert pseudo_results[0]["mask_type"] != MaskType.GROUND_TRUTH.value


# ---------------------------------------------------------------------------
# 2. Metric Calculation Tests (IoU, Dice, Precision, Recall)
# ---------------------------------------------------------------------------

def test_segmentation_metrics_identical_masks():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True

    m = calculate_segmentation_metrics(mask, mask)
    assert np.isclose(m["iou"], 1.0)
    assert np.isclose(m["dice"], 1.0)
    assert np.isclose(m["precision"], 1.0)
    assert np.isclose(m["recall"], 1.0)


def test_segmentation_metrics_disjoint_masks():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    m1[0:10, 0:10] = True
    m2[20:30, 20:30] = True

    m = calculate_segmentation_metrics(m1, m2)
    assert np.isclose(m["iou"], 0.0, atol=1e-3)
    assert np.isclose(m["dice"], 0.0, atol=1e-3)
    assert np.isclose(m["precision"], 0.0, atol=1e-3)
    assert np.isclose(m["recall"], 0.0, atol=1e-3)


def test_segmentation_metrics_partial_overlap():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    # m1: 10x10 = 100 px
    m1[10:20, 10:20] = True
    # m2: 10x10 = 100 px, overlapping 5x10 = 50 px
    m2[10:20, 15:25] = True

    # intersection = 50, union = 150 -> IoU = 50/150 = 1/3 ~ 0.3333
    # Dice = 2*50 / (100 + 100) = 0.5
    m = calculate_segmentation_metrics(m1, m2)
    assert np.isclose(m["iou"], 1.0 / 3.0, atol=1e-3)
    assert np.isclose(m["dice"], 0.5, atol=1e-3)
    assert np.isclose(m["precision"], 0.5, atol=1e-3)
    assert np.isclose(m["recall"], 0.5, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Model Architecture Tests (FoodUNet & DiceBCELoss)
# ---------------------------------------------------------------------------

def test_food_unet_forward_pass():
    model = FoodUNet(in_channels=3, out_channels=1, base_filters=16)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_dice_bce_loss():
    criterion = DiceBCELoss(bce_weight=0.5)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

    loss = criterion(logits, targets)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0.0

    loss.backward()
    assert logits.grad is not None


# ---------------------------------------------------------------------------
# 4. Dataset & Fixture Tests
# ---------------------------------------------------------------------------

def test_segmentation_dataset_and_fixture(tmp_path):
    manifest_path = create_segmentation_fixture(tmp_path / "seg_fixture", num_train=2, num_val=1, num_test=1)
    assert manifest_path.is_file()

    import json
    with open(manifest_path) as f:
        samples = json.load(f)

    ds = FoodSegmentationDataset(samples, img_size=(128, 128))
    assert len(ds) == 4

    img_tensor, mask_tensor, meta = ds[0]
    assert img_tensor.shape == (3, 128, 128)
    assert mask_tensor.shape == (1, 128, 128)
    assert meta["mask_type"] == MaskType.GROUND_TRUTH.value


# ---------------------------------------------------------------------------
# 5. FoodSegmenter Box-Prompting & Instance Integration
# ---------------------------------------------------------------------------

def test_food_segmenter_from_box():
    segmenter = FoodSegmenter(mode="sam", sam_checkpoint=None)
    segmenter.load()
    assert segmenter.is_loaded is True

    img = Image.new("RGB", (200, 200), color=(200, 200, 200))
    bbox = [20.0, 30.0, 100.0, 120.0]
    mask = segmenter.segment_from_box(img, bbox)

    assert mask.shape == (200, 200)
    assert mask.dtype == bool
    assert np.any(mask)  # contains positive segmented pixels
    assert np.all(mask[0:10, 0:10] == False)  # outside bbox is false


def test_food_segmenter_instance_integration():
    segmenter = FoodSegmenter(mode="sam", sam_checkpoint=None)
    segmenter.load()

    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    detections = [
        Detection(class_id=0, class_name="apple", bbox=[20.0, 20.0, 80.0, 80.0], confidence=0.88),
        Detection(class_id=1, class_name="banana", bbox=[100.0, 100.0, 160.0, 150.0], confidence=0.91),
    ]

    instances = segmenter.segment_instances(img, detections)
    assert len(instances) == 2
    for inst in instances:
        assert isinstance(inst, FoodInstance)
        assert inst.mask is not None
        assert inst.mask.shape == (200, 200)
        assert isinstance(inst.bbox, tuple)


# ---------------------------------------------------------------------------
# 6. 4-Panel Visualization Test
# ---------------------------------------------------------------------------

def test_4_panel_visualization(tmp_path):
    img = Image.new("RGB", (200, 200), color=(240, 240, 240))
    pred_mask = np.zeros((200, 200), dtype=bool)
    pred_mask[40:120, 40:120] = True
    gt_mask = np.zeros((200, 200), dtype=bool)
    gt_mask[50:130, 50:130] = True

    out_file = tmp_path / "4panel_test.png"
    saved = save_segmentation_visualization(img, pred_mask, out_file, ground_truth_mask=gt_mask)

    assert saved.is_file()
    assert saved.stat().st_size > 0
    saved_img = Image.open(saved)
    # Check width is 4 panels
    assert saved_img.size[0] > 200 * 3
