"""
End-to-End Food Recognition & Multi-Nutrient Estimation Pipeline CLI.

Unified entry point tying all pipeline stages together:
  Image -> Preprocessing -> Detection -> Segmentation -> Classification ->
  Ingredients -> Portion/Mass/Volume -> Multi-Nutrients -> Aggregation -> Final Report

Outputs generated:
  1. Structured JSON predictions (outputs/nutrition/predictions.json)
  2. Tabular CSV export (outputs/nutrition/predictions.csv)
  3. Visual annotated image with segmentation masks, bounding boxes, and badges (outputs/nutrition/annotated_meal.png)
  4. Human-readable text report (outputs/nutrition/report.txt and console output)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nutrition.aggregate import format_report
from src.nutrition.inference import run_complete_nutrition_pipeline
from src.utils.schema import FoodInstance, NutrientSet, Source, Value


def format_detailed_cli_report(payload: dict[str, Any]) -> str:
    """
    Renders a comprehensive, high-clarity terminal report.
    Includes per-food classification, portion with provenance, ingredients with provenance,
    macronutrients, meal rollup totals, and scientific methodology disclosures.
    """
    lines: list[str] = []
    separator = "=" * 70
    subsep = "-" * 70

    lines.append(separator)
    lines.append("  MULTIMODAL FOOD RECOGNITION & MULTI-NUTRIENT ESTIMATION REPORT")
    lines.append(separator)
    lines.append(f"Image Source   : {payload.get('image_path')}")
    lines.append(f"Image Size     : {payload.get('image_size', [0, 0])[0]} x {payload.get('image_size', [0, 0])[1]} px")
    lines.append(f"Detected Foods : {payload.get('num_foods', 0)}")
    lines.append(separator)

    instances = payload.get("instances", [])
    if not instances:
        lines.append("\nNo food items detected in the provided image.")
        lines.append(separator)
        return "\n".join(lines)

    for i, inst in enumerate(instances, start=1):
        lines.append(f"\n[ITEM #{i}] {inst.get('food_name', 'Unknown').upper()}")
        lines.append(subsep)
        lines.append(f"  Confidence   : {float(inst.get('confidence', 0.0)) * 100:.1f}%")
        
        bbox = inst.get("bounding_box") or inst.get("bbox")
        lines.append(f"  Bounding Box : {bbox}")

        # Segmentation
        seg = inst.get("segmentation", {})
        if isinstance(seg, dict):
            seg_area = seg.get("mask_area_px", 0)
            seg_cov = seg.get("mask_coverage_pct", 0.0)
            lines.append(f"  Segmentation : {seg_area:,} px ({seg_cov:.1f}% image area)")

        # Portion
        portion = inst.get("portion", {})
        mass_g = inst.get("estimated_mass_g", portion.get("estimated_mass_g", 0.0))
        vol_est = inst.get("estimated_volume", portion.get("estimated_volume", {}))
        method = portion.get("method_used", "rgb_learned")
        mass_src = portion.get("mass_source", "prediction")
        lines.append(f"  Portion Mass : {float(mass_g):.1f} g  [source: {mass_src}, method: {method}]")
        if vol_est is not None:
            if isinstance(vol_est, dict):
                vol_val = vol_est.get("volume_cm3", 0.0)
                disc = vol_est.get("disclaimer", "")
            else:
                vol_val = float(vol_est)
                disc = portion.get("disclaimer", "")
            lines.append(f"  Est. Volume  : {vol_val:.1f} cm3  ({disc})")

        # Ingredients
        ingr = inst.get("ingredients", {})
        candidates = ingr.get("candidates", [])
        ingr_src = ingr.get("source", "unavailable")
        ingr_str = ", ".join(candidates) if candidates else "none identified"
        lines.append(f"  Ingredients  : {ingr_str}  [source: {ingr_src}]")

        # Nutrients (Core + Extended with Provenance Labels)
        nuts = inst.get("nutrients", {})
        lines.append("  Macronutrients:")
        def _fmt_n(k: str, default_unit: str = "g", alt_k: str = "") -> str:
            item = nuts.get(k)
            if item is None and alt_k:
                item = nuts.get(alt_k)
            if item is None:
                return "Not available [NOT AVAILABLE]"
            if isinstance(item, dict):
                val = item.get("value")
                lbl = item.get("label") or ("NOT AVAILABLE" if val is None else "MODEL PREDICTION")
            else:
                try:
                    val = float(item)
                    lbl = "MODEL PREDICTION"
                except (ValueError, TypeError):
                    val = None
                    lbl = "NOT AVAILABLE"
            if val is None:
                return f"Not available [{lbl}]"
            unit = "kcal" if ("calor" in k or "calor" in alt_k) else default_unit
            return f"{val:.1f} {unit} [{lbl}]"

        lines.append(f"    - Calories      : {_fmt_n('calories', 'kcal', 'calories_kcal')}")
        lines.append(f"    - Protein       : {_fmt_n('protein_g', 'g')}")
        lines.append(f"    - Carbohydrates : {_fmt_n('carbs_g', 'g', 'carbohydrates_g')}")
        lines.append(f"    - Total Fat     : {_fmt_n('fat_g', 'g')}")

        lines.append("  Extended Nutrients:")
        lines.append(f"    - Dietary Fiber : {_fmt_n('fiber_g', 'g')}")
        lines.append(f"    - Total Sugars  : {_fmt_n('sugar_g', 'g')}")
        lines.append(f"    - Saturated Fat : {_fmt_n('saturated_fat_g', 'g')}")
        lines.append(f"    - Sodium        : {_fmt_n('sodium_mg', 'mg')}")
        lines.append(f"    - Cholesterol   : {_fmt_n('cholesterol_mg', 'mg')}")

        micros = nuts.get("additional_nutrients", {})
        if micros:
            lines.append("  Micronutrients:")
            for mk, mv in micros.items():
                clean_k = mk.replace("_mg", "").replace("_g", "").replace("_", " ").title()
                unit = "mg" if mk.endswith("_mg") else ("g" if mk.endswith("_g") else "")
                if isinstance(mv, dict):
                    val = mv.get("value")
                    lbl = mv.get("label", "REFERENCE")
                else:
                    val = float(mv)
                    lbl = "REFERENCE"
                val_str = f"{val:.1f} {unit}" if val is not None else "Not available"
                lines.append(f"    - {clean_k:<14}: {val_str} [{lbl}]")

    # Meal Totals
    lines.append(f"\n{separator}")
    lines.append("  TOTAL MEAL AGGREGATION")
    lines.append(separator)
    meal = payload.get("meal_total", {})
    partials = meal.get("partial_nutrients", [])

    def _line_total(label: str, key: str, unit: str) -> None:
        item = meal.get(key)
        if not item:
            return
        if isinstance(item, dict):
            val = item.get("value")
            lbl = item.get("label") or ("NOT AVAILABLE" if val is None else "DERIVED")
        else:
            try:
                val = float(item)
                lbl = "DERIVED"
            except (ValueError, TypeError):
                val = None
                lbl = "NOT AVAILABLE"
        flag = " (PARTIAL)" if key in partials else ""
        val_str = f"{val:.1f} {unit}" if val is not None else "Not available"
        lines.append(f"  {label:<18}: {val_str:<14} [{lbl}]{flag}")

    _line_total("Total Mass", "mass_g", "g")
    _line_total("Calories", "calories", "kcal")
    _line_total("Protein", "protein_g", "g")
    _line_total("Carbohydrates", "carbs_g", "g")
    _line_total("Total Fat", "fat_g", "g")
    _line_total("Dietary Fiber", "fiber_g", "g")
    _line_total("Total Sugars", "sugar_g", "g")
    _line_total("Saturated Fat", "saturated_fat_g", "g")
    _line_total("Sodium", "sodium_mg", "mg")
    _line_total("Cholesterol", "cholesterol_mg", "mg")

    add_meal = meal.get("additional_nutrients", {})
    if add_meal:
        lines.append("\n  Additional Micronutrients:")
        for mk, mv in add_meal.items():
            clean_k = mk.replace("_mg", "").replace("_g", "").replace("_", " ").title()
            unit = "mg" if mk.endswith("_mg") else ("g" if mk.endswith("_g") else "")
            val = mv.get("value")
            lbl = mv.get("label", "DERIVED")
            flag = " (PARTIAL)" if mk in partials else ""
            val_str = f"{val:.1f} {unit}" if val is not None else "Not available"
            lines.append(f"  {clean_k:<18}: {val_str:<14} [{lbl}]{flag}")

    # Scientific methodology disclosure
    lines.append(f"\n{subsep}")
    lines.append("  SCIENTIFIC METHODOLOGY & LIMITATIONS")
    lines.append(subsep)
    lines.append("  - Monocular RGB volume estimates are learned visual proxies.")
    lines.append("  - Exact physical volume and density-based mass derivation are")
    lines.append("    reserved for depth-map inputs and reference-calibrated views.")
    lines.append("  - All nutrient and ingredient values maintain explicit provenance")
    lines.append("    (ground_truth, prediction, derived, reference_lookup, unavailable).")
    lines.append(separator)

    return "\n".join(lines)


def predict_meal(
    image_path: str | Path,
    depth_path: str | Path | None = None,
    reference_bbox: Sequence[float] | None = None,
    reference_real_size_cm: float | None = None,
    output_dir: str | Path = "outputs/nutrition",
    output_json_name: str = "predictions.json",
    output_csv_name: str = "predictions.csv",
    output_img_name: str = "annotated_meal.png",
    output_report_name: str = "report.txt",
    detector_weights: str | Path = "models/yolov8n.pt",
    segmenter_weights: str | Path = "models/segmentation/best_unet.pt",
    classifier_weights: str | Path = "models/classification/best_classifier.pt",
    mass_weights: str | Path = "models/portion/best_mass_regressor.pt",
    nutrient_weights: str | Path = "models/nutrition/best_nutrient_model.pt",
    vocab_path: str | Path | None = None,
    device: str = "cpu",
    preconfigured_boxes: Sequence[dict[str, Any]] | None = None,
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Programmatic entrypoint to run full end-to-end food nutrition prediction.
    Generates JSON, CSV, annotated image, and human-readable text report.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / output_json_name
    csv_path = out_dir / output_csv_name if output_csv_name else None
    img_path = out_dir / output_img_name if output_img_name else None
    report_path = out_dir / output_report_name if output_report_name else None

    payload = run_complete_nutrition_pipeline(
        image_path=image_path,
        output_path=json_path,
        output_csv_path=csv_path,
        output_image_path=img_path,
        depth_path=depth_path,
        reference_bbox=reference_bbox,
        reference_real_size_cm=reference_real_size_cm,
        detector_weights=detector_weights,
        segmenter_weights=segmenter_weights,
        classifier_weights=classifier_weights,
        mass_weights=mass_weights,
        nutrient_weights=nutrient_weights,
        vocab_path=vocab_path,
        device=device,
        preconfigured_boxes=preconfigured_boxes,
        conf_threshold=conf_threshold,
    )

    # Generate and save human-readable report
    report_text = format_detailed_cli_report(payload)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        payload["report_path"] = str(report_path.resolve()).replace("\\", "/")

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Unified End-to-End Food Recognition & Multi-Nutrient Estimation Pipeline CLI."
    )
    parser.add_argument("image_pos", nargs="?", default=None, help="Input food image path (positional)")
    parser.add_argument("--image", default=None, help="Input food image path (flag)")
    parser.add_argument("--depth", default=None, help="Optional depth map image path")
    parser.add_argument(
        "--ref-bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional bounding box of reference object [x1, y1, x2, y2]",
    )
    parser.add_argument(
        "--ref-size",
        type=float,
        default=2.5,
        help="Physical reference diameter/dimension in cm (e.g. 2.5 for standard coin)",
    )
    parser.add_argument("--output-dir", default="outputs/nutrition", help="Directory to save output files")
    parser.add_argument("--output-json", default="predictions.json", help="JSON filename")
    parser.add_argument("--output-csv", default="predictions.csv", help="CSV filename")
    parser.add_argument("--output-img", default="annotated_meal.png", help="Annotated image filename")
    parser.add_argument("--output-report", default="report.txt", help="Report text filename")
    parser.add_argument(
        "--format",
        choices=["text", "json", "all"],
        default="all",
        help="Output display format in console",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--preset", choices=["meal_1", "meal_2"], default=None, help="Demo meal preset")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--detector-weights", default="models/yolov8n.pt", help="Detector weights")
    parser.add_argument("--segmenter-weights", default="models/segmentation/best_unet.pt", help="Segmenter weights")
    parser.add_argument("--classifier-weights", default="models/classification/best_classifier.pt", help="Classifier weights")
    parser.add_argument("--mass-weights", default="models/portion/best_mass_regressor.pt", help="Mass weights")
    parser.add_argument("--nutrient-weights", default="models/nutrition/best_nutrient_model.pt", help="Nutrient weights")
    parser.add_argument("--vocab", default=None, help="Ingredient vocab path")

    args = parser.parse_args()

    preconfigured = None
    target_image = args.image or args.image_pos
    depth_target = args.depth
    ref_bbox = args.ref_bbox
    ref_size = args.ref_size

    if args.preset:
        preset_file = Path("data/demo_meals/demo_meals.json")
        if preset_file.is_file():
            with open(preset_file, "r") as f:
                presets = json.load(f)
            chosen = next((p for p in presets if p["id"] == args.preset), None)
            if chosen:
                target_image = chosen["image_path"]
                depth_target = chosen.get("depth_path")
                ref_bbox = chosen.get("reference_bbox")
                ref_size = chosen.get("reference_size_cm", 2.5)
                preconfigured = chosen.get("preconfigured_boxes")
                print(f"[CLI Preset Activated] {chosen['title']}")

    if not target_image:
        parser.error("Must provide an input image path or select a valid --preset.")

    payload = predict_meal(
        image_path=target_image,
        depth_path=depth_target,
        reference_bbox=ref_bbox,
        reference_real_size_cm=ref_size,
        output_dir=args.output_dir,
        output_json_name=args.output_json,
        output_csv_name=args.output_csv,
        output_img_name=args.output_img,
        output_report_name=args.output_report,
        detector_weights=args.detector_weights,
        segmenter_weights=args.segmenter_weights,
        classifier_weights=args.classifier_weights,
        mass_weights=args.mass_weights,
        nutrient_weights=args.nutrient_weights,
        vocab_path=args.vocab,
        device=args.device,
        preconfigured_boxes=preconfigured,
        conf_threshold=args.conf_thresh,
    )

    if args.format in ("text", "all"):
        print(format_detailed_cli_report(payload))

    if args.format in ("json", "all"):
        print(f"\n[Artifacts Generated]")
        print(f"  JSON Results    : {Path(args.output_dir) / args.output_json}")
        print(f"  CSV Results     : {Path(args.output_dir) / args.output_csv}")
        print(f"  Annotated Image : {Path(args.output_dir) / args.output_img}")
        print(f"  Report Text     : {Path(args.output_dir) / args.output_report}")


if __name__ == "__main__":
    main()
