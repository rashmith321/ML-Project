"""
CSV Exporter for Meal Nutrition Predictions.

Converts multi-stage pipeline predictions into standardized tabular CSV files
with item-level rows and meal total aggregation rows.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _to_float_str(item: Any, decimals: int = 1) -> str:
    """Safely extracts a numeric float and formats it to string, returning '' if unavailable."""
    if item is None or item == "":
        return ""
    if isinstance(item, dict):
        item = item.get("value")
    if item is None or item == "":
        return ""
    try:
        val = float(item)
        return f"{val:.{decimals}f}"
    except (ValueError, TypeError):
        return ""


def export_predictions_csv(payload: dict[str, Any], output_path: str | Path) -> Path:
    """
    Exports inference predictions payload to a standard tabular CSV.

    Columns include:
      food_name, confidence, bounding_box, segmentation_area_px, estimated_mass_g,
      estimated_volume, calories_kcal, protein_g, carbohydrates_g, fat_g,
      fiber_g, sugar_g, saturated_fat_g, sodium_mg, cholesterol_mg,
      potassium_mg, calcium_mg, iron_mg, vitamin_c_mg, nutrition_source, nutrition_confidence
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "food_name",
        "confidence",
        "bounding_box",
        "segmentation_area_px",
        "segmentation_coverage_pct",
        "estimated_mass_g",
        "estimated_volume",
        "calories_kcal",
        "protein_g",
        "carbohydrates_g",
        "fat_g",
        "fiber_g",
        "sugar_g",
        "saturated_fat_g",
        "sodium_mg",
        "cholesterol_mg",
        "potassium_mg",
        "calcium_mg",
        "iron_mg",
        "vitamin_c_mg",
        "nutrition_source",
        "nutrition_confidence",
    ]

    instances = payload.get("instances", [])
    meal_total = payload.get("meal_total", {})

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        total_seg_area = 0
        total_seg_cov = 0.0
        total_volume = 0.0

        for inst in instances:
            seg = inst.get("segmentation", {})
            seg_area = seg.get("mask_area_px", 0) if isinstance(seg, dict) else 0
            seg_cov = seg.get("mask_coverage_pct", 0.0) if isinstance(seg, dict) else 0.0
            total_seg_area += seg_area
            total_seg_cov += seg_cov

            vol = inst.get("estimated_volume", 0.0)
            if isinstance(vol, dict):
                vol_num = float(vol.get("volume_cm3", 0.0))
            else:
                try:
                    vol_num = float(vol)
                except (ValueError, TypeError):
                    vol_num = 0.0
            total_volume += vol_num

            add_nuts = inst.get("additional_nutrients", {})

            # Format bounding box as string "[x1, y1, x2, y2]"
            bbox = inst.get("bounding_box") or inst.get("bbox")
            bbox_str = f"[{', '.join(f'{b:.1f}' for b in bbox)}]" if bbox else ""

            row = {
                "food_name": inst.get("food_name", "Unknown"),
                "confidence": _to_float_str(inst.get("confidence", 0.0), decimals=4),
                "bounding_box": bbox_str,
                "segmentation_area_px": seg_area,
                "segmentation_coverage_pct": f"{seg_cov:.2f}",
                "estimated_mass_g": _to_float_str(inst.get("estimated_mass_g", 0.0), decimals=1),
                "estimated_volume": f"{vol_num:.1f}",
                "calories_kcal": _to_float_str(inst.get("calories_kcal", 0.0), decimals=1),
                "protein_g": _to_float_str(inst.get("protein_g", 0.0), decimals=1),
                "carbohydrates_g": _to_float_str(inst.get("carbohydrates_g", 0.0), decimals=1),
                "fat_g": _to_float_str(inst.get("fat_g", 0.0), decimals=1),
                "fiber_g": _to_float_str(add_nuts.get("fiber_g"), decimals=1),
                "sugar_g": _to_float_str(add_nuts.get("sugar_g"), decimals=1),
                "saturated_fat_g": _to_float_str(add_nuts.get("saturated_fat_g"), decimals=1),
                "sodium_mg": _to_float_str(add_nuts.get("sodium_mg"), decimals=1),
                "cholesterol_mg": _to_float_str(add_nuts.get("cholesterol_mg"), decimals=1),
                "potassium_mg": _to_float_str(add_nuts.get("potassium_mg"), decimals=1),
                "calcium_mg": _to_float_str(add_nuts.get("calcium_mg"), decimals=1),
                "iron_mg": _to_float_str(add_nuts.get("iron_mg"), decimals=1),
                "vitamin_c_mg": _to_float_str(add_nuts.get("vitamin_c_mg"), decimals=1),
                "nutrition_source": inst.get("nutrition_source", "MODEL PREDICTION"),
                "nutrition_confidence": _to_float_str(inst.get("nutrition_confidence", inst.get("confidence", 0.0)), decimals=4),
            }
            writer.writerow(row)

        # Summary Row: TOTAL MEAL
        if instances:
            meal_add = meal_total.get("additional_nutrients", {})
            mass_tot = meal_total.get("total_mass_g", meal_total.get("mass_g"))
            cal_tot = meal_total.get("total_calories", meal_total.get("calories"))
            prot_tot = meal_total.get("total_protein", meal_total.get("protein_g"))
            carb_tot = meal_total.get("total_carbohydrates", meal_total.get("carbs_g"))
            fat_tot = meal_total.get("total_fat", meal_total.get("fat_g"))

            total_row = {
                "food_name": "TOTAL MEAL",
                "confidence": "",
                "bounding_box": "",
                "segmentation_area_px": total_seg_area,
                "segmentation_coverage_pct": f"{total_seg_cov:.2f}",
                "estimated_mass_g": _to_float_str(mass_tot, decimals=1),
                "estimated_volume": f"{total_volume:.1f}",
                "calories_kcal": _to_float_str(cal_tot, decimals=1),
                "protein_g": _to_float_str(prot_tot, decimals=1),
                "carbohydrates_g": _to_float_str(carb_tot, decimals=1),
                "fat_g": _to_float_str(fat_tot, decimals=1),
                "fiber_g": _to_float_str(meal_total.get("fiber_g"), decimals=1),
                "sugar_g": _to_float_str(meal_total.get("sugar_g"), decimals=1),
                "saturated_fat_g": _to_float_str(meal_total.get("saturated_fat_g"), decimals=1),
                "sodium_mg": _to_float_str(meal_total.get("sodium_mg"), decimals=1),
                "cholesterol_mg": _to_float_str(meal_total.get("cholesterol_mg"), decimals=1),
                "potassium_mg": _to_float_str(meal_add.get("potassium_mg"), decimals=1),
                "calcium_mg": _to_float_str(meal_add.get("calcium_mg"), decimals=1),
                "iron_mg": _to_float_str(meal_add.get("iron_mg"), decimals=1),
                "vitamin_c_mg": _to_float_str(meal_add.get("vitamin_c_mg"), decimals=1),
                "nutrition_source": "DERIVED",
                "nutrition_confidence": "",
            }
            writer.writerow(total_row)

    return out_file
