"""
Unit and integration tests for Phase 7: Multi-Nutrient Estimation module.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nutrition.aggregate import aggregate_meal
from src.nutrition.dataset import Nutrition5kDataset, create_nutrition_fixture
from src.nutrition.estimator import MultiNutrientEstimator
from src.nutrition.evaluate import calculate_metric_suite
from src.nutrition.model import MultiNutrientModel, NormalizedMultiNutrientLoss
from src.utils.schema import FoodInstance, NutrientSet, Source, Value


# ---------------------------------------------------------------------------
# 1. MultiNutrientModel Architecture & Forward Tests
# ---------------------------------------------------------------------------

def test_multinutrient_model_forward():
    model = MultiNutrientModel(
        backbone="efficientnet_b0",
        num_classes=10,
        num_ingredients=5,
        pretrained=False,
    )
    crops = torch.randn(2, 3, 224, 224)
    c_ids = torch.tensor([1, 2], dtype=torch.long)
    masses = torch.tensor([[150.0], [250.0]], dtype=torch.float32)
    seg = torch.randn(2, 3)
    depth = torch.randn(2, 2)
    ingr = torch.randn(2, 5)

    out = model(crops, class_ids=c_ids, masses=masses, seg_features=seg, depth_features=depth, ingr_features=ingr)
    assert out.shape == (2, 4)
    # Output must be non-negative (ReLU heads)
    assert (out >= 0.0).all()


def test_multinutrient_model_missing_inputs_robustness():
    """Ensure forward pass succeeds with None for optional multimodal features."""
    model = MultiNutrientModel(backbone="efficientnet_b0", pretrained=False)
    crops = torch.randn(1, 3, 224, 224)
    # Pass None for all metadata features
    out = model(crops)
    assert out.shape == (1, 4)
    assert (out >= 0.0).all()


def test_multinutrient_model_checkpoint(tmp_path):
    model = MultiNutrientModel(backbone="efficientnet_b0", pretrained=False)
    ckpt_path = tmp_path / "nut_model.pt"
    model.save_checkpoint(ckpt_path, extra_meta={"loss": 0.42})
    assert ckpt_path.is_file()

    loaded = MultiNutrientModel.load_from_checkpoint(ckpt_path)
    assert isinstance(loaded, MultiNutrientModel)
    out = loaded(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# 2. Normalized Loss Tests
# ---------------------------------------------------------------------------

def test_normalized_multinutrient_loss():
    loss_fn = NormalizedMultiNutrientLoss(scale_factors=(500.0, 30.0, 50.0, 25.0))

    # Perfect prediction -> 0 loss
    preds = torch.tensor([[500.0, 30.0, 50.0, 25.0]], requires_grad=True)
    targets = torch.tensor([[500.0, 30.0, 50.0, 25.0]])
    loss = loss_fn(preds, targets)
    assert loss.item() == 0.0

    # Test balanced gradients: equal relative error of 10% on each target
    preds_err = torch.tensor([[550.0, 33.0, 55.0, 27.5]], requires_grad=True)
    loss_err = loss_fn(preds_err, targets)
    assert loss_err.item() > 0.0
    loss_err.backward()
    assert preds_err.grad is not None
    # Gradients should have identical magnitude because relative error is 10% for all targets
    grads = preds_err.grad.squeeze(0).numpy()
    # Normalized gradients should be approximately uniform
    np.testing.assert_allclose(grads * np.array([500.0, 30.0, 50.0, 25.0]), [0.025, 0.025, 0.025, 0.025], atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Dataset & Synthetic Fixture Tests
# ---------------------------------------------------------------------------

def test_nutrition_dataset_and_fixture(tmp_path):
    mpath, vpath = create_nutrition_fixture(tmp_path / "fix_nut", samples_count=4)
    assert mpath.is_file()
    assert vpath.is_file()

    with open(mpath, "r", encoding="utf-8") as f:
        samples = json.load(f)
    with open(vpath, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    ds = Nutrition5kDataset(samples, ingredient_vocab=vocab, is_training=False)
    assert len(ds) == 4

    batch = ds[0]
    assert batch["visual_crop"].shape == (3, 224, 224)
    assert batch["targets"].shape == (4,)
    # Verify realistic non-negative values
    assert (batch["targets"] > 0).all()


# ---------------------------------------------------------------------------
# 4. Metric Calculations
# ---------------------------------------------------------------------------

def test_calculate_metric_suite():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])  # errors: 10, 10, 30 -> MAE = 16.67

    metrics = calculate_metric_suite(y_true, y_pred, target_name="calories", unit="kcal")
    assert "mae_kcal" in metrics
    assert "rmse_kcal" in metrics
    assert "r2_score" in metrics
    assert "mape_percent" in metrics
    assert round(metrics["mae_kcal"], 1) == 16.7
    assert metrics["r2_score"] > 0.90


# ---------------------------------------------------------------------------
# 5. MultiNutrientEstimator & Aggregation Integration
# ---------------------------------------------------------------------------

def test_multinutrient_estimator_and_aggregation():
    estimator = MultiNutrientEstimator(weights_path=None)
    crop = Image.new("RGB", (100, 100), color=(150, 100, 50))

    nutrients = estimator.estimate_nutrients(crop, mass_g=200.0)
    assert "calories_kcal" in nutrients
    assert "protein_g" in nutrients
    assert "carbohydrates_g" in nutrients
    assert "fat_g" in nutrients
    assert all(v > 0 for v in nutrients.values())

    # Test FoodInstance annotation
    instances = [
        FoodInstance(
            instance_id="inst_1",
            bbox=(10, 10, 80, 80),
            mass=Value(value=150.0, source=Source.PREDICTION),
        ),
        FoodInstance(
            instance_id="inst_2",
            bbox=(90, 90, 160, 160),
            mass=Value(value=220.0, source=Source.PREDICTION),
        ),
    ]
    img = Image.new("RGB", (200, 200), color=(240, 240, 240))
    annotated = estimator.annotate_instances(instances, img)

    assert annotated[0].nutrients is not None
    assert annotated[0].nutrients.calories.value > 0
    assert annotated[0].nutrients.calories.source == Source.PREDICTION

    # Test integration with meal aggregation
    meal_total = aggregate_meal(annotated)
    assert meal_total.calories.value > annotated[0].nutrients.calories.value
    assert meal_total.calories.source == Source.DERIVED
    assert len(meal_total.partial_nutrients) == 0
