"""
Comprehensive integration tests for the final end-to-end inference pipeline:
  Image -> Preprocessing -> Food Detection -> Food Segmentation -> Food Classification ->
  Ingredient Understanding -> Mass / Portion / Volume -> Multi-Nutrient Estimation ->
  Nutrition Aggregation -> Final Report
"""
from __future__ import annotations

import csv
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

from app.app import load_demo_presets
from inference.predict import format_detailed_cli_report, predict_meal
from src.nutrition.inference import run_complete_nutrition_pipeline
from src.nutrition.visualizer import draw_detected_food_image, draw_segmented_food_image
from src.utils.csv_export import export_predictions_csv
from src.utils.html_report import generate_html_report
from src.utils.preprocessing import ImagePreprocessor, preprocess_meal_input


def test_preprocessing_module(tmp_path):
    # Create test RGB image
    img_path = tmp_path / "test_preproc.jpg"
    img = Image.new("RGB", (320, 240), color=(128, 64, 32))
    img.save(img_path)

    # Preprocess RGB only
    res = preprocess_meal_input(img_path)
    assert res.width == 320
    assert res.height == 240
    assert res.image.mode == "RGB"
    assert res.depth_map is None
    assert res.metadata["depth"]["available"] is False

    # Create test depth map with mismatched dimensions to test automatic spatial alignment
    depth_path = tmp_path / "test_depth.png"
    depth_arr = np.ones((120, 160), dtype=np.uint16) * 1000
    Image.fromarray(depth_arr).save(depth_path)

    res_with_depth = preprocess_meal_input(img_path, depth=depth_path)
    assert res_with_depth.depth_map is not None
    assert res_with_depth.depth_map.shape == (240, 320)
    assert res_with_depth.metadata["depth"]["available"] is True
    assert res_with_depth.metadata["depth"]["was_resized"] is True


def test_per_food_output_schema_and_meal_totals(tmp_path):
    presets = load_demo_presets()
    preset = presets[0]

    out_json = "final_predictions.json"
    out_csv = "final_predictions.csv"
    out_img = "final_annotated.png"
    out_rep = "final_report.txt"

    payload = predict_meal(
        image_path=preset["image_path"],
        depth_path=preset.get("depth_path"),
        reference_bbox=preset.get("reference_bbox"),
        reference_real_size_cm=preset.get("reference_size_cm", 2.5),
        output_dir=tmp_path,
        output_json_name=out_json,
        output_csv_name=out_csv,
        output_img_name=out_img,
        output_report_name=out_rep,
        preconfigured_boxes=preset.get("preconfigured_boxes"),
    )

    # 1. Verify all 4 required artifacts are generated
    assert (tmp_path / out_json).is_file(), "JSON output missing"
    assert (tmp_path / out_csv).is_file(), "CSV output missing"
    assert (tmp_path / out_img).is_file(), "Annotated image output missing"
    assert (tmp_path / out_rep).is_file(), "Report text output missing"

    # 2. Verify PER FOOD OUTPUT schema for every detected food item
    assert payload["num_foods"] > 0
    required_per_food_keys = [
        "food_name",
        "confidence",
        "bounding_box",
        "segmentation",
        "estimated_mass_g",
        "estimated_volume",
        "calories_kcal",
        "protein_g",
        "carbohydrates_g",
        "fat_g",
        "additional_nutrients",
        "nutrition_source",
        "nutrition_confidence",
    ]

    for inst in payload["instances"]:
        for k in required_per_food_keys:
            assert k in inst, f"Missing required per-food key '{k}'"

        # Check types and constraints
        assert isinstance(inst["food_name"], str)
        assert isinstance(inst["confidence"], float) and 0.0 <= inst["confidence"] <= 1.0
        assert isinstance(inst["bounding_box"], list) and len(inst["bounding_box"]) == 4
        assert isinstance(inst["segmentation"], dict)
        assert "mask_area_px" in inst["segmentation"]
        assert "mask_coverage_pct" in inst["segmentation"]
        assert isinstance(inst["estimated_mass_g"], (int, float)) and inst["estimated_mass_g"] >= 0.0
        assert isinstance(inst["estimated_volume"], (int, float)) and inst["estimated_volume"] >= 0.0
        assert isinstance(inst["calories_kcal"], (int, float)) and inst["calories_kcal"] >= 0.0
        assert isinstance(inst["protein_g"], (int, float)) and inst["protein_g"] >= 0.0
        assert isinstance(inst["carbohydrates_g"], (int, float)) and inst["carbohydrates_g"] >= 0.0
        assert isinstance(inst["fat_g"], (int, float)) and inst["fat_g"] >= 0.0
        assert isinstance(inst["additional_nutrients"], dict)
        assert inst["nutrition_source"] in ("MODEL PREDICTION", "DERIVED", "REFERENCE", "NOT AVAILABLE")
        assert isinstance(inst["nutrition_confidence"], float)

    # 3. Verify MEAL OUTPUT calculations
    meal = payload["meal_total"]
    assert "total_mass_g" in meal
    assert "total_calories" in meal
    assert "total_protein" in meal
    assert "total_carbohydrates" in meal
    assert "total_fat" in meal

    assert meal["total_mass_g"] > 0.0
    assert meal["total_calories"] > 0.0
    assert meal["total_protein"] > 0.0
    assert meal["total_carbohydrates"] > 0.0
    assert meal["total_fat"] > 0.0

    # 4. Verify CSV content & headers
    csv_file = tmp_path / out_csv
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "food_name" in header
        assert "confidence" in header
        assert "bounding_box" in header
        assert "estimated_mass_g" in header
        assert "calories_kcal" in header
        assert "protein_g" in header
        assert "carbohydrates_g" in header
        assert "fat_g" in header
        rows = list(reader)
        # Should have rows for instances + 1 TOTAL MEAL row
        assert len(rows) == payload["num_foods"] + 1
        assert rows[-1][0] == "TOTAL MEAL"

    # 5. Verify Report text
    rep_text = (tmp_path / out_rep).read_text(encoding="utf-8")
    assert "MULTIMODAL FOOD RECOGNITION" in rep_text
    assert "TOTAL MEAL AGGREGATION" in rep_text
    assert "Total Mass" in rep_text
    assert "Calories" in rep_text


def test_custom_image_prediction_without_reference(tmp_path):
    # Test on custom image without reference coin or preset
    img_path = tmp_path / "custom_food.jpg"
    img = Image.new("RGB", (300, 300), color=(200, 100, 50))
    img.save(img_path)

    payload = predict_meal(
        image_path=img_path,
        output_dir=tmp_path,
        output_json_name="custom_pred.json",
        output_csv_name="custom_pred.csv",
        output_img_name="custom_img.png",
        output_report_name="custom_rep.txt",
    )

    assert payload is not None
    assert (tmp_path / "custom_pred.json").is_file()
    assert (tmp_path / "custom_pred.csv").is_file()
    assert (tmp_path / "custom_img.png").is_file()
    assert (tmp_path / "custom_rep.txt").is_file()


def test_html_report_generation():
    sample_payload = {
        "image_path": "meal.jpg",
        "num_foods": 1,
        "instances": [
            {
                "food_name": "grilled_chicken_bowl",
                "confidence": 0.94,
                "estimated_mass_g": 350.0,
                "estimated_volume": 420.0,
                "calories_kcal": 450.0,
                "protein_g": 42.0,
                "carbohydrates_g": 25.0,
                "fat_g": 12.0,
                "portion": {"method_used": "reference_calibrated", "mass_source": "derived"},
                "ingredients": {"candidates": ["chicken", "rice"], "source": "recipe_derived"},
                "segmentation": {"mask_area_px": 25000, "mask_coverage_pct": 12.5},
            }
        ],
        "meal_total": {
            "total_mass_g": 350.0,
            "total_calories": 450.0,
            "total_protein": 42.0,
            "total_carbohydrates": 25.0,
            "total_fat": 12.0,
            "total_fiber_g": 3.0,
            "total_sugar_g": 2.0,
            "total_saturated_fat_g": 3.5,
            "total_sodium_mg": 450.0,
            "total_cholesterol_mg": 85.0,
            "additional_nutrients": {
                "potassium_mg": 600.0,
                "calcium_mg": 50.0,
                "iron_mg": 2.5,
                "vitamin_c_mg": 5.0,
            },
        },
    }

    html = generate_html_report(sample_payload)
    assert "<!DOCTYPE html>" in html
    assert "AI-Based Food Image Nutrition Estimation System" in html
    assert "Nutritional values are model estimates" in html
    assert "TOTAL MASS" in html
    assert "TOTAL CALORIES" in html
    assert "TOTAL PROTEIN" in html
    assert "TOTAL CARBS" in html
    assert "TOTAL FAT" in html
    assert "Grilled Chicken Bowl" in html
    assert "450.0" in html


def test_visualizer_detection_and_segmentation_drawings():
    base = Image.new("RGB", (320, 240), color=(100, 150, 200))
    instances = [
        {"food_name": "pizza", "confidence": 0.95, "bounding_box": [30.0, 40.0, 180.0, 190.0]}
    ]
    masks = [np.ones((240, 320), dtype=bool)]

    det_img = draw_detected_food_image(base, instances)
    assert det_img.size == (320, 240)

    seg_img = draw_segmented_food_image(base, masks=masks, instances=instances)
    assert seg_img.size == (320, 240)

