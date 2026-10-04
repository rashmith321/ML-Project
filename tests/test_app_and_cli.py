"""
Unit and integration tests for Phase 8: User Application & Unified CLI Inference.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.app import load_demo_presets, plot_macronutrient_donut
from inference.predict import format_detailed_cli_report, predict_meal
from src.nutrition.visualizer import draw_meal_predictions, save_meal_visualization


def test_demo_presets_integrity():
    presets = load_demo_presets()
    assert len(presets) >= 2, "Expected at least 2 demo meal presets"

    for p in presets:
        assert "id" in p
        assert "title" in p
        assert "image_path" in p
        assert Path(p["image_path"]).is_file(), f"Image file missing for preset: {p['id']}"

        if p.get("depth_path"):
            assert Path(p["depth_path"]).is_file(), f"Depth file missing for preset: {p['id']}"

        if p.get("preconfigured_boxes"):
            for b in p["preconfigured_boxes"]:
                assert "bbox" in b
                assert len(b["bbox"]) == 4
                assert "confidence" in b


def test_predict_meal_with_preset_smoke(tmp_path):
    presets = load_demo_presets()
    preset = presets[0]

    out_json = "smoke_predictions.json"
    out_img = "smoke_annotated.png"

    payload = predict_meal(
        image_path=preset["image_path"],
        depth_path=preset.get("depth_path"),
        reference_bbox=preset.get("reference_bbox"),
        reference_real_size_cm=preset.get("reference_size_cm", 2.5),
        output_dir=tmp_path,
        output_json_name=out_json,
        output_img_name=out_img,
        preconfigured_boxes=preset.get("preconfigured_boxes"),
    )

    assert payload is not None
    assert payload["num_foods"] == len(preset["preconfigured_boxes"])
    assert "meal_total" in payload
    assert (tmp_path / out_json).is_file()
    assert (tmp_path / out_img).is_file()

    # Check structure
    assert "calories" in payload["meal_total"]
    assert "protein_g" in payload["meal_total"]
    assert "carbs_g" in payload["meal_total"]
    assert "fat_g" in payload["meal_total"]
    assert payload["meal_total"]["calories"]["value"] > 0.0


def test_format_detailed_cli_report():
    sample_payload = {
        "image_path": "sample.jpg",
        "image_size": [640, 480],
        "num_foods": 1,
        "instances": [
            {
                "instance_id": "food_1",
                "food_name": "pizza",
                "confidence": 0.95,
                "bbox": [50.0, 50.0, 200.0, 200.0],
                "portion": {
                    "estimated_mass_g": 220.0,
                    "estimated_volume": 250.0,
                    "method_used": "reference_calibrated",
                    "mass_source": "derived",
                    "disclaimer": "Calibrated with reference coin.",
                },
                "ingredients": {
                    "candidates": ["dough", "cheese", "tomato sauce"],
                    "source": "recipe_derived",
                },
                "nutrients": {
                    "calories_kcal": 550.0,
                    "protein_g": 22.0,
                    "carbohydrates_g": 65.0,
                    "fat_g": 21.0,
                },
            }
        ],
        "meal_total": {
            "calories": {"value": 550.0, "source": "derived"},
            "protein_g": {"value": 22.0, "source": "derived"},
            "carbs_g": {"value": 65.0, "source": "derived"},
            "fat_g": {"value": 21.0, "source": "derived"},
            "partial_nutrients": [],
        },
    }

    report = format_detailed_cli_report(sample_payload)
    assert "MULTIMODAL FOOD RECOGNITION & MULTI-NUTRIENT ESTIMATION REPORT" in report
    assert "PIZZA" in report
    assert "Portion Mass : 220.0 g" in report
    assert "source: derived, method: reference_calibrated" in report
    assert "dough, cheese, tomato sauce" in report
    assert "recipe_derived" in report
    assert "TOTAL MEAL AGGREGATION" in report
    assert "Calories      : 550.0 kcal" in report
    assert "SCIENTIFIC METHODOLOGY & LIMITATIONS" in report


def test_visualizer_meal_predictions(tmp_path):
    img = Image.new("RGB", (400, 300), color=(200, 200, 200))
    instances = [
        {
            "food_name": "apple_pie",
            "confidence": 0.92,
            "bbox": [50, 50, 180, 180],
            "portion": {"estimated_mass_g": 180.0},
            "nutrients": {"calories_kcal": 320.0, "protein_g": 3.5, "carbohydrates_g": 45.0, "fat_g": 14.0},
        }
    ]
    masks = [np.ones((300, 400), dtype=bool)]
    ref_bbox = [10, 10, 40, 40]

    out_file = tmp_path / "vis_test.png"
    saved = save_meal_visualization(
        image=img,
        instances=instances,
        output_path=out_file,
        masks=masks,
        reference_bbox=ref_bbox,
        reference_size_cm=2.5,
        meal_total={"calories": {"value": 320.0}, "protein_g": {"value": 3.5}, "carbs_g": {"value": 45.0}, "fat_g": {"value": 14.0}},
    )

    assert saved.is_file()
    assert saved.stat().st_size > 0
    loaded = Image.open(saved)
    assert loaded.size == (400, 300)


def test_plot_macronutrient_donut():
    fig = plot_macronutrient_donut(protein_g=30.0, carbs_g=50.0, fat_g=20.0)
    assert fig is not None
    assert len(fig.axes) > 0


def test_zero_detection_cli_report():
    empty_payload = {
        "image_path": "empty.jpg",
        "image_size": [300, 300],
        "num_foods": 0,
        "instances": [],
        "meal_total": {
            "calories": {"value": 0.0, "source": "unavailable"},
            "protein_g": {"value": 0.0, "source": "unavailable"},
            "carbs_g": {"value": 0.0, "source": "unavailable"},
            "fat_g": {"value": 0.0, "source": "unavailable"},
            "partial_nutrients": [],
        },
    }
    report = format_detailed_cli_report(empty_payload)
    assert "No food items detected" in report
