"""
Unit and integration tests for Phase 4: Food Classification module.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.classifier import ClassificationResult, FoodClassifier
from src.classification.dataset import FoodCropDataset, create_classification_fixture
from src.classification.evaluate import compute_classification_metrics
from src.classification.taxonomy import FOOD101_CLASSES, FoodTaxonomy
from src.classification.visualize import draw_classification_badge, plot_confusion_matrix
from src.models.classifier import FoodClassificationModel
from src.utils.schema import FoodInstance


# ---------------------------------------------------------------------------
# 1. FoodTaxonomy & Class Mapping Tests
# ---------------------------------------------------------------------------

def test_taxonomy_food101_classes():
    tax = FoodTaxonomy(primary_dataset="food101")
    assert tax.num_classes == 101
    assert "pizza" in tax.classes
    assert "apple_pie" in tax.classes
    assert "sushi" in tax.classes

    assert tax.id_to_name(0) == "apple_pie"
    assert tax.name_to_id("apple_pie") == 0

    # Normalization handles upper case and spaces
    assert tax.name_to_id("Apple Pie") == 0
    assert tax.name_to_id("pizza") is not None

    # Never fabricate classes outside represented taxonomy
    assert tax.is_valid_class("unicorn_burger") is False
    assert tax.name_to_id("unicorn_burger") is None


def test_taxonomy_export_json(tmp_path):
    tax = FoodTaxonomy(primary_dataset="food101")
    out_file = tmp_path / "tax.json"
    tax.export_json(out_file)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# 2. Output Schema & Dataclass Tests
# ---------------------------------------------------------------------------

def test_classification_result_schema():
    top_k = [
        {"class_id": 76, "food_name": "pizza", "confidence": 0.8924},
        {"class_id": 0, "food_name": "apple_pie", "confidence": 0.0543},
    ]
    res = ClassificationResult(
        food_name="pizza",
        class_id=76,
        confidence=0.8924,
        top_k_predictions=top_k,
    )
    d = res.as_dict()
    assert d["food_name"] == "pizza"
    assert d["class_id"] == 76
    assert d["confidence"] == 0.8924
    assert len(d["top_k_predictions"]) == 2
    assert d["top_k_predictions"][0]["food_name"] == "pizza"


# ---------------------------------------------------------------------------
# 3. Model Architecture Tests (FoodClassificationModel)
# ---------------------------------------------------------------------------

def test_classification_model_forward():
    model = FoodClassificationModel(backbone="efficientnet_b0", num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 10)

    probs = model.predict_probs(x)
    assert probs.shape == (2, 10)
    np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), [1.0, 1.0], atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Preprocessing & Crop Classification Tests
# ---------------------------------------------------------------------------

def test_classifier_preprocess_with_mask():
    classifier = FoodClassifier(weights_path=None, num_classes=10)
    img = Image.new("RGB", (100, 100), color=(200, 50, 50))
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True

    tensor = classifier.preprocess_crop(img, mask=mask)
    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_classifier_predict_and_instance_update():
    tax = FoodTaxonomy(custom_classes=["apple_pie", "pizza", "sushi"])
    classifier = FoodClassifier(weights_path=None, num_classes=3, taxonomy=tax)
    classifier.load()

    test_img = Image.new("RGB", (150, 150), color=(180, 180, 180))
    res = classifier.predict(test_img, top_k=2)
    assert isinstance(res, ClassificationResult)
    assert res.food_name in ["apple_pie", "pizza", "sushi"]
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top_k_predictions) == 2

    # Test instance update
    inst = FoodInstance(
        instance_id="food_1",
        bbox=(10, 10, 80, 80),
        mask=np.ones((150, 150), dtype=bool),
        class_name="initial_guess",
    )
    updated = classifier.classify_instances(test_img, [inst])
    assert updated[0].class_name in ["apple_pie", "pizza", "sushi"]
    assert updated[0].class_confidence is not None


# ---------------------------------------------------------------------------
# 5. Evaluation Metrics & Confusion Matrix Tests
# ---------------------------------------------------------------------------

def test_compute_classification_metrics():
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 2, 0, 0]  # 4/5 correct

    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.1],  # true 1, top-1 pred 0, top-2 pred 1
    ])

    m = compute_classification_metrics(
        y_true, y_pred, y_prob=probs, class_names=["c0", "c1", "c2"], top_k=2
    )

    assert m["accuracy"] == 0.8  # 4/5
    assert m["top1_accuracy"] == 0.8
    assert m["top5_accuracy"] == 1.0  # true class is in top-2 for all samples
    assert "precision_macro" in m
    assert "recall_macro" in m
    assert "f1_macro" in m
    assert len(m["confusion_matrix"]) == 3  # 3x3 matrix


# ---------------------------------------------------------------------------
# 6. Visualization & Confusion Matrix Plotting
# ---------------------------------------------------------------------------

def test_plot_confusion_matrix(tmp_path):
    cm = [[5, 1], [0, 6]]
    names = ["pizza", "burger"]
    out_img = tmp_path / "cm.png"
    saved = plot_confusion_matrix(cm, names, output_path=out_img)
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_draw_classification_badge():
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    res = ClassificationResult(
        food_name="pizza",
        class_id=1,
        confidence=0.912,
        top_k_predictions=[
            {"class_id": 1, "food_name": "pizza", "confidence": 0.912},
            {"class_id": 0, "food_name": "salad", "confidence": 0.045},
        ],
    )
    badged = draw_classification_badge(img, res)
    assert isinstance(badged, Image.Image)
    assert badged.size == (200, 200)


# ---------------------------------------------------------------------------
# 7. Classification Fixture Creation
# ---------------------------------------------------------------------------

def test_create_classification_fixture(tmp_path):
    mpath = create_classification_fixture(tmp_path / "cls_fix", class_names=["pie", "soup"], samples_per_class=2)
    assert mpath.is_file()

    import json
    with open(mpath) as f:
        samples = json.load(f)
    assert len(samples) == 4
    ds = FoodCropDataset(samples)
    assert len(ds) == 4
    tensor, cls_id = ds[0]
    assert tensor.shape == (3, 224, 224)
