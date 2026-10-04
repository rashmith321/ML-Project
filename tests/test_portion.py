"""
Unit and integration tests for Phase 6: Portion / Mass / Volume Estimation module.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.mass_regressor import FoodMassRegressionModel
from src.portion.calibration import ReferenceObjectCalibrator
from src.portion.dataset import Nutrition5kMassDataset, create_mass_fixture
from src.portion.depth_processor import DepthProcessor
from src.portion.estimator import CalibrationObjectEstimator, Nutrition5kMassRegressor
from src.portion.evaluate import calculate_regression_metrics, plot_actual_vs_predicted
from src.portion.inference import PortionInference
from src.portion.mass_estimator import MassEstimator
from src.portion.volume_estimator import EstimationMethod, VolumeEstimator
from src.utils.schema import FoodInstance, Source, Value


# ---------------------------------------------------------------------------
# 1. Model Architecture & Forward Pass
# ---------------------------------------------------------------------------

def test_mass_regression_model_forward():
    model = FoodMassRegressionModel(backbone="efficientnet_b0", in_channels=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1)
    # Mass is strictly non-negative (ReLU)
    assert (out >= 0).all()


def test_mass_regression_model_checkpoint(tmp_path):
    model = FoodMassRegressionModel(backbone="efficientnet_b0", in_channels=3, pretrained=False)
    ckpt_path = tmp_path / "test_model.pt"
    model.save_checkpoint(ckpt_path, extra_meta={"test_val": 42})
    assert ckpt_path.is_file()

    loaded = FoodMassRegressionModel.load_from_checkpoint(ckpt_path)
    assert isinstance(loaded, FoodMassRegressionModel)
    x = torch.randn(1, 3, 224, 224)
    out = loaded(x)
    assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# 2. Nutrition5k Mass Regressor
# ---------------------------------------------------------------------------

def test_nutrition5k_mass_regressor_predict():
    regressor = Nutrition5kMassRegressor(weights_path=None)
    regressor.load()

    img = Image.new("RGB", (150, 150), color=(200, 100, 50))
    val = regressor.predict_mass_g(img)

    assert isinstance(val, Value)
    assert val.source == Source.PREDICTION
    assert val.value is not None
    assert val.value > 0.0


def test_nutrition5k_mass_regressor_estimate_instances():
    regressor = Nutrition5kMassRegressor(weights_path=None)
    regressor.load()

    full_img = Image.new("RGB", (300, 300), color=(220, 220, 220))
    instances = [
        FoodInstance(
            instance_id="food_1",
            bbox=(20, 20, 120, 120),
            mask=np.ones((300, 300), dtype=bool),
            class_name="pizza",
        ),
        FoodInstance(
            instance_id="food_2",
            bbox=(150, 150, 250, 250),
            mask=None,
            class_name="salad",
        ),
    ]

    updated = regressor.estimate_instances(instances, full_img)
    assert updated[0].mass_g is not None
    assert updated[0].mass_g > 0.0
    assert updated[0].mass_source == "prediction"

    assert updated[1].mass_g is not None
    assert updated[1].mass_g > 0.0
    assert updated[1].mass_source == "prediction"


# ---------------------------------------------------------------------------
# 3. Calibration Object Estimator & ReferenceObjectCalibrator
# ---------------------------------------------------------------------------

def test_calibration_object_estimator():
    estimator = CalibrationObjectEstimator(reference_object_real_size_cm=2.5)  # 2.5 cm coin

    # Synthetic image 500x500
    img = Image.new("RGB", (500, 500), color=(200, 200, 200))

    # Reference coin: 50 pixels diameter -> 2.5 cm / 50 px = 0.05 cm/px
    ref_bbox = [10, 10, 60, 60]

    # Food item: 100x100 pixels -> 5 cm x 5 cm
    food_bbox = [100, 100, 200, 200]

    val = estimator.predict_mass_g(
        img,
        reference_bbox=ref_bbox,
        food_bbox=food_bbox,
        food_density_g_per_cm3=1.0,
    )

    assert isinstance(val, Value)
    assert val.source == Source.DERIVED
    assert val.value is not None
    assert val.value > 0.0


def test_reference_calibrator():
    calibrator = ReferenceObjectCalibrator(default_reference_diameter_cm=2.5)
    ref_bbox = [10, 10, 60, 60]  # 50px -> 0.05 cm/px
    scale = calibrator.compute_scale_factor(ref_bbox)
    assert scale == 0.05

    food_bbox = [100, 100, 200, 200]
    vol, conf = calibrator.estimate_volume_from_reference(food_bbox, ref_bbox)
    assert vol > 0.0
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 4. DepthProcessor (3D RGB-D Integration)
# ---------------------------------------------------------------------------

def test_depth_processor():
    processor = DepthProcessor(default_focal_length_px=500.0)

    # 100x100 depth map: plate background at 800mm, food object at 750mm (50mm height)
    depth_map = np.full((100, 100), 800.0, dtype=np.float32)
    depth_map[30:70, 30:70] = 750.0

    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True

    vol, conf = processor.estimate_volume_from_depth(depth_map, mask)
    assert vol > 0.0
    assert conf == 1.0


# ---------------------------------------------------------------------------
# 5. VolumeEstimator Scientific Hierarchy
# ---------------------------------------------------------------------------

def test_volume_estimator_hierarchy():
    estimator = VolumeEstimator()
    food_bbox = [50, 50, 150, 150]
    mask = np.ones((100, 100), dtype=bool)

    # Tier 1: Depth present -> DEPTH_BASED
    depth_map = np.full((100, 100), 800.0, dtype=np.float32)
    depth_map[20:80, 20:80] = 760.0
    vol_d, conf_d, method_d, disc_d = estimator.estimate_volume(
        food_bbox=food_bbox, mask=mask, depth_map=depth_map
    )
    assert method_d == EstimationMethod.DEPTH_BASED
    assert vol_d > 0.0

    # Tier 2: Reference object present (no depth) -> REFERENCE_CALIBRATED
    vol_r, conf_r, method_r, disc_r = estimator.estimate_volume(
        food_bbox=food_bbox, mask=mask, reference_bbox=[0, 0, 50, 50]
    )
    assert method_r == EstimationMethod.REFERENCE_CALIBRATED
    assert vol_r > 0.0

    # Tier 3: Neither present -> RGB_LEARNED (empirical proxy with disclaimer)
    vol_rgb, conf_rgb, method_rgb, disc_rgb = estimator.estimate_volume(
        food_bbox=food_bbox, mask=mask
    )
    assert method_rgb == EstimationMethod.RGB_LEARNED
    assert "Monocular RGB photograph" in disc_rgb
    assert vol_rgb > 0.0


# ---------------------------------------------------------------------------
# 6. MassEstimator & Density Lookup
# ---------------------------------------------------------------------------

def test_mass_estimator():
    estimator = MassEstimator(weights_path=None)
    crop = Image.new("RGB", (100, 100), color=(180, 120, 80))
    food_bbox = [10, 10, 80, 80]

    # Test with reference calibration
    ref_bbox = [0, 0, 40, 40]
    res_calib = estimator.estimate_portion(
        crop=crop,
        food_bbox=food_bbox,
        food_class="apple_pie",
        reference_bbox=ref_bbox,
    )
    assert res_calib["method_used"] == "reference_calibrated"
    assert res_calib["mass_source"] == "derived"
    assert res_calib["estimated_mass_g"] > 0.0
    assert res_calib["density_g_per_cm3"] == 0.80

    # Test RGB learned fallback
    res_rgb = estimator.estimate_portion(
        crop=crop,
        food_bbox=food_bbox,
        food_class="unknown_snack",
    )
    assert res_rgb["method_used"] == "rgb_learned"
    assert res_rgb["estimated_mass_g"] > 0.0


# ---------------------------------------------------------------------------
# 7. Regression Metrics & Actual vs Predicted Plots
# ---------------------------------------------------------------------------

def test_calculate_regression_metrics_r2(tmp_path):
    y_true = [100.0, 200.0, 300.0, 400.0]
    y_pred = [105.0, 195.0, 310.0, 390.0]

    metrics = calculate_regression_metrics(y_true, y_pred, target_name="mass", unit="g")
    assert metrics["mae_g"] == 7.5
    assert metrics["rmse_g"] > 0.0
    assert metrics["r2_score"] > 0.95  # High correlation

    # Test plotting
    plot_path = tmp_path / "actual_vs_pred.png"
    saved = plot_actual_vs_predicted(y_true, y_pred, output_path=plot_path, title="Test Plot")
    assert saved.is_file()
    assert saved.stat().st_size > 0


# ---------------------------------------------------------------------------
# 8. PortionInference Engine
# ---------------------------------------------------------------------------

def test_portion_inference_engine():
    engine = PortionInference(weights_path=None)
    img = Image.new("RGB", (300, 300), color=(240, 240, 240))
    instances = [
        FoodInstance(instance_id="f1", bbox=(20, 20, 120, 120), class_name="pizza"),
        FoodInstance(instance_id="f2", bbox=(140, 140, 220, 220), class_name="salad"),
    ]

    annotated, records = engine.annotate_instances(instances, image=img)
    assert len(annotated) == 2
    assert annotated[0].mass.value is not None
    assert annotated[0].mass.value > 0.0
    assert records[0]["food_name"] == "pizza"
    assert "estimated_volume" in records[0]
    assert "method_used" in records[0]
