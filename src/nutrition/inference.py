"""
End-to-End Multi-Nutrient Inference Pipeline:
Executes the full final pipeline:
  Image -> Preprocessing -> Detection -> Segmentation -> Classification ->
  Ingredients -> Portion/Mass/Volume -> Multi-Nutrients -> Meal Total Aggregation -> Final Report
Saves complete structured predictions to JSON, CSV, and annotated images.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image

from src.classification.classifier import FoodClassifier
from src.classification.inference import run_full_pipeline
from src.detection.detector import FoodDetector
from src.ingredients.inference import IngredientPredictor
from src.nutrition.aggregate import aggregate_meal
from src.nutrition.estimator import MultiNutrientEstimator
from src.portion.inference import PortionInference
from src.segmentation.segmenter import FoodSegmenter
from src.utils.csv_export import export_predictions_csv
from src.utils.preprocessing import preprocess_meal_input
from src.utils.schema import FoodInstance, NutrientSet, Source, UnifiedFoodNutrition, Value


def extract_mask_polygon(mask: np.ndarray, max_points: int = 32) -> list[list[int]]:
    """
    Extract simplified polygon coordinates from a 2D boolean segmentation mask.
    Falls back gracefully to the bounding outline if contour extraction is unavailable.
    """
    if not np.any(mask):
        return []
    try:
        import cv2
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.015 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            pts = approx.reshape(-1, 2).tolist()
            return pts[:max_points]
    except Exception:
        pass

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def run_complete_nutrition_pipeline(
    image_path: str | Path,
    output_path: str | Path = "outputs/nutrition/predictions.json",
    output_csv_path: str | Path | None = "outputs/nutrition/predictions.csv",
    output_image_path: str | Path | None = "outputs/nutrition/annotated_meal.png",
    depth_path: str | Path | None = None,
    reference_bbox: Sequence[float] | None = None,
    reference_real_size_cm: float | None = None,
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
    Run complete final end-to-end food nutrition pipeline:
      Image -> Preprocessing -> Detection -> Segmentation -> Classification ->
      Ingredients -> Portion/Mass/Volume -> Multi-Nutrients -> Aggregation -> Output
    """
    # 1. Preprocessing Stage (validation, EXIF orientation correction, RGB enforcement, depth alignment)
    preproc = preprocess_meal_input(image=image_path, depth=depth_path)
    img = preproc.image
    w, h = preproc.width, preproc.height
    depth_map = preproc.depth_map

    # Compute the scale factor used during preprocessing so we can map bboxes from
    # original-image pixel space into preprocessed-image pixel space.
    # preproc.metadata contains 'original_width' / 'original_height' if set.
    orig_w = preproc.metadata.get("original_width", w) if preproc.metadata else w
    orig_h = preproc.metadata.get("original_height", h) if preproc.metadata else h
    scale_x = w / max(1, orig_w)
    scale_y = h / max(1, orig_h)

    # 2. Food Detection + Segmentation + Classification models
    detector = FoodDetector(weights_path=detector_weights, conf_threshold=conf_threshold, device=device)
    detector.load()

    segmenter = FoodSegmenter(mode="auto", unet_checkpoint=segmenter_weights, device=device)
    segmenter.load()

    classifier = FoodClassifier(weights_path=classifier_weights, device=device)
    classifier.load()

    instance_masks: list[np.ndarray] = []
    base_instances: list[dict[str, Any]] = []

    if preconfigured_boxes:
        from src.detection.detector import Detection
        preset_dets = []
        for i, b in enumerate(preconfigured_boxes):
            bbox = b["bbox"]
            fn = b.get("food_name", "food")
            conf = b.get("confidence", 0.90)
            preset_dets.append(Detection(class_id=i, class_name=fn, bbox=bbox, confidence=conf))

        seg_insts = segmenter.segment_instances(img, preset_dets)
        for i, inst in enumerate(seg_insts):
            b_info = preconfigured_boxes[i]
            fn = b_info.get("food_name", "food")
            conf = b_info.get("confidence", 0.90)
            cid = b_info.get("class_id", 0)

            # If class_id not specified, look up in taxonomy or predict
            if fn and fn != "food":
                cid = classifier.taxonomy.name_to_id(fn) or 0
            else:
                c_res = classifier.predict(img.crop(inst.bbox))
                fn = c_res.food_name
                cid = c_res.class_id
                conf = c_res.confidence

            if inst.mask is not None:
                instance_masks.append(inst.mask)
            else:
                instance_masks.append(np.zeros((h, w), dtype=bool))

            base_instances.append({
                "instance_id": f"food_{i + 1}",
                "food_name": fn,
                "class_id": cid,
                "confidence": conf,
                "bbox": inst.bbox,
            })
    else:
        # Standard automated pipeline.
        # NOTE: run_full_pipeline receives image_path (original resolution) so all
        # bounding boxes it returns are in ORIGINAL image pixel coordinates.
        # We must scale them into preprocessed-image pixel coordinates before
        # passing them to PortionInference, which operates on the preprocessed image.
        base_results = run_full_pipeline(
            image=image_path,
            detector=detector,
            segmenter=segmenter,
            classifier=classifier,
            output_json=None,
        )
        raw_instances = base_results.get("instances", [])
        for inst in raw_instances:
            bbox = inst.get("bbox")
            if bbox and (scale_x != 1.0 or scale_y != 1.0):
                ox1, oy1, ox2, oy2 = bbox
                bbox = [
                    ox1 * scale_x,
                    oy1 * scale_y,
                    ox2 * scale_x,
                    oy2 * scale_y,
                ]
                inst = dict(inst)
                inst["bbox"] = bbox
            base_instances.append(inst)
            if bbox:
                # Segment directly to get mask for visualization (preprocessed image coords)
                m = segmenter.segment_from_box(img, bbox)
                instance_masks.append(m)
            else:
                instance_masks.append(np.zeros((h, w), dtype=bool))

    # 3. Ingredient Understanding
    ingr_predictor = IngredientPredictor(device=device)

    # 4. Portion / Mass / Volume Estimation
    portion_engine = PortionInference(weights_path=mass_weights, device=device)

    # 5. Multi-Nutrient Estimator
    nutrient_estimator = MultiNutrientEstimator(weights_path=nutrient_weights, vocab_path=vocab_path, device=device)

    instance_objects: list[FoodInstance] = []
    detailed_instances: list[dict[str, Any]] = []

    for idx, inst_dict in enumerate(base_instances):
        inst_id = inst_dict.get("instance_id", f"food_{idx + 1}")
        bbox = inst_dict.get("bbox")
        food_name = inst_dict.get("food_name", "food")
        cls_id = inst_dict.get("class_id", 0)
        conf = float(inst_dict.get("confidence", 0.8))

        crop = None
        if bbox:
            fx1, fy1, fx2, fy2 = [int(round(v)) for v in bbox]
            crop = img.crop((max(0, fx1), max(0, fy1), min(w, fx2), min(h, fy2)))

        # A. Predict Ingredients
        ingr_res = ingr_predictor.predict(crop=crop, food_name=food_name)

        # B. Estimate Portion / Mass / Volume
        portion_res = portion_engine.estimate_instance(
            image=img,
            food_bbox=bbox,
            food_class=food_name,
            depth_map=depth_map,
            reference_bbox=reference_bbox,
            reference_real_size_cm=reference_real_size_cm,
        )
        mass_val = float(portion_res["estimated_mass_g"])
        mass_src = Source.DERIVED if portion_res["mass_source"] == "derived" else Source.PREDICTION

        # Sanity check: warn (do NOT silently clip) if mass is implausible
        if mass_val > 2000.0:
            print(
                f"[Pipeline] WARNING: Implausible mass estimate for '{food_name}' "
                f"({mass_val:.1f} g) — bbox={bbox}. "
                f"Check detection quality and volume estimation inputs."
            )

        vol_data = portion_res["estimated_volume"]
        if isinstance(vol_data, dict):
            vol_val = float(vol_data.get("volume_cm3", 0.0))
        else:
            try:
                vol_val = float(vol_data)
            except (ValueError, TypeError):
                vol_val = 0.0

        # C. Estimate Multi-Nutrients (Calories, Protein, Carbs, Fat)
        nutrients_dict = nutrient_estimator.estimate_nutrients(
            crop=crop if crop else img,
            class_id=cls_id,
            mass_g=mass_val,
            bbox=bbox,
            img_size=(w, h),
            depth_map=depth_map,
            ingredients=ingr_res.ingredient_candidates,
        )

        # D. Derive extended nutrients from verified USDA reference profiles scaled by mass
        n_set = nutrient_estimator.ref_db.derive_nutrients_for_instance(
            food_name=food_name,
            mass_g=mass_val,
            predicted_calories=nutrients_dict["calories_kcal"],
            predicted_protein=nutrients_dict["protein_g"],
            predicted_carbs=nutrients_dict["carbohydrates_g"],
            predicted_fat=nutrients_dict["fat_g"],
            primary_source=Source.PREDICTION,
        )

        unified_nut = UnifiedFoodNutrition(
            food_name=food_name,
            mass_g=Value(value=mass_val, source=mass_src),
            calories_kcal=n_set.calories,
            protein_g=n_set.protein_g,
            carbohydrates_g=n_set.carbs_g,
            fat_g=n_set.fat_g,
            fiber_g=n_set.fiber_g,
            sugar_g=n_set.sugar_g,
            saturated_fat_g=n_set.saturated_fat_g,
            sodium_mg=n_set.sodium_mg,
            cholesterol_mg=n_set.cholesterol_mg,
            additional_nutrients=n_set.additional_nutrients,
            source=Source.PREDICTION,
            confidence=conf,
        )

        # Build segmentation details
        inst_mask = instance_masks[idx] if idx < len(instance_masks) else np.zeros((h, w), dtype=bool)
        mask_area_px = int(np.sum(inst_mask))
        mask_coverage_pct = round(float(mask_area_px) / float(w * h) * 100.0, 2) if (w * h) > 0 else 0.0
        polygon_pts = extract_mask_polygon(inst_mask)

        seg_dict = {
            "mask_area_px": mask_area_px,
            "mask_coverage_pct": mask_coverage_pct,
            "mask_type": "predicted",
            "polygon": polygon_pts,
        }

        # Build additional nutrients dict for flat access
        additional_flat: dict[str, float | None] = {
            "fiber_g": round(float(n_set.fiber_g.value), 2) if n_set.fiber_g.value is not None else None,
            "sugar_g": round(float(n_set.sugar_g.value), 2) if n_set.sugar_g.value is not None else None,
            "saturated_fat_g": round(float(n_set.saturated_fat_g.value), 2) if n_set.saturated_fat_g.value is not None else None,
            "sodium_mg": round(float(n_set.sodium_mg.value), 2) if n_set.sodium_mg.value is not None else None,
            "cholesterol_mg": round(float(n_set.cholesterol_mg.value), 2) if n_set.cholesterol_mg.value is not None else None,
        }
        for mk, mv in n_set.additional_nutrients.items():
            additional_flat[mk] = round(float(mv.value), 2) if mv.value is not None else None

        # Create FoodInstance object
        inst_obj = FoodInstance(
            instance_id=inst_id,
            bbox=tuple(bbox) if bbox else None,
            class_name=food_name,
            class_confidence=conf,
            ingredients=ingr_res.ingredient_candidates,
            mass=Value(value=mass_val, source=mass_src),
            nutrients=n_set,
            unified_nutrition=unified_nut,
        )
        instance_objects.append(inst_obj)

        cal_val = round(float(n_set.calories.value), 2) if n_set.calories.value is not None else 0.0
        prot_val = round(float(n_set.protein_g.value), 2) if n_set.protein_g.value is not None else 0.0
        carb_val = round(float(n_set.carbs_g.value), 2) if n_set.carbs_g.value is not None else 0.0
        fat_val = round(float(n_set.fat_g.value), 2) if n_set.fat_g.value is not None else 0.0

        # Construct instance record containing all required PER FOOD OUTPUT fields
        detailed_instances.append({
            # Explicit Per-Food Output requirements:
            "food_name": food_name,
            "confidence": round(float(conf), 4),
            "bounding_box": [round(float(b), 2) for b in bbox] if bbox else [],
            "segmentation": seg_dict,
            "estimated_mass_g": round(float(mass_val), 1),
            "estimated_volume": round(float(vol_val), 1),
            "calories_kcal": cal_val,
            "protein_g": prot_val,
            "carbohydrates_g": carb_val,
            "fat_g": fat_val,
            "additional_nutrients": additional_flat,
            "nutrition_source": "MODEL PREDICTION",
            "nutrition_confidence": round(float(conf), 4),

            # Backward-compatible fields:
            "instance_id": inst_id,
            "class_id": cls_id,
            "bbox": bbox,
            "portion": {
                "estimated_mass_g": round(float(mass_val), 1),
                "estimated_volume": portion_res["estimated_volume"],
                "method_used": portion_res["method_used"],
                "mass_source": portion_res["mass_source"],
                "disclaimer": portion_res["disclaimer"],
            },
            "ingredients": {
                "candidates": ingr_res.ingredient_candidates,
                "source": ingr_res.source.value,
            },
            "nutrients": n_set.as_dict(),
            "unified_nutrition": unified_nut.as_dict(),
        })

    # 6. Meal-Level Nutrition Aggregation with Provenance Rollup
    if instance_objects:
        meal_total_obj = aggregate_meal(instance_objects)
        meal_total_dict = meal_total_obj.as_dict()
    else:
        meal_total_dict = {
            "calories": {"value": 0.0, "source": "unavailable", "label": "NOT AVAILABLE"},
            "protein_g": {"value": 0.0, "source": "unavailable", "label": "NOT AVAILABLE"},
            "carbs_g": {"value": 0.0, "source": "unavailable", "label": "NOT AVAILABLE"},
            "fat_g": {"value": 0.0, "source": "unavailable", "label": "NOT AVAILABLE"},
            "partial_nutrients": [],
            "total_mass_g": 0.0,
            "total_calories": 0.0,
            "total_protein": 0.0,
            "total_carbohydrates": 0.0,
            "total_fat": 0.0,
        }

    annotated_img_path = None
    detected_img_path = None
    segmented_img_path = None
    if output_image_path:
        out_parent = Path(output_image_path).parent
        from src.nutrition.visualizer import (
            save_detected_food_image,
            save_meal_visualization,
            save_segmented_food_image,
        )
        det_out = out_parent / "detected_food.png"
        seg_out = out_parent / "segmented_food.png"
        save_detected_food_image(image=img, instances=detailed_instances, output_path=det_out)
        save_segmented_food_image(image=img, masks=instance_masks, instances=detailed_instances, output_path=seg_out)
        save_meal_visualization(
            image=img,
            instances=detailed_instances,
            masks=instance_masks,
            output_path=output_image_path,
            reference_bbox=reference_bbox,
            reference_size_cm=reference_real_size_cm,
            meal_total=meal_total_dict,
        )
        detected_img_path = str(det_out.resolve()).replace("\\", "/")
        segmented_img_path = str(seg_out.resolve()).replace("\\", "/")
        annotated_img_path = str(Path(output_image_path).resolve()).replace("\\", "/")

    csv_path_str = None
    if output_csv_path:
        out_csv = export_predictions_csv(
            payload={"instances": detailed_instances, "meal_total": meal_total_dict},
            output_path=output_csv_path,
        )
        csv_path_str = str(out_csv.resolve()).replace("\\", "/")

    payload = {
        "image_path": str(Path(image_path).resolve()).replace("\\", "/"),
        "image_size": [w, h],
        "preprocessing": preproc.metadata,
        "num_foods": len(detailed_instances),
        "instances": detailed_instances,
        "meal_total": meal_total_dict,
        "annotated_image_path": annotated_img_path,
        "detected_image_path": detected_img_path,
        "segmented_image_path": segmented_img_path,
        "csv_path": csv_path_str,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved complete meal nutrition predictions to: {out}")
    if csv_path_str:
        print(f"Saved meal nutrition tabular CSV to: {csv_path_str}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run complete multi-nutrient nutrition inference.")
    parser.add_argument("--image", required=True, help="Input food image path")
    parser.add_argument("--output", default="outputs/nutrition/predictions.json", help="Output JSON path")
    parser.add_argument("--output-csv", default="outputs/nutrition/predictions.csv", help="Output CSV path")
    parser.add_argument("--output-img", default="outputs/nutrition/annotated_meal.png", help="Output annotated image path")
    parser.add_argument("--depth", default=None, help="Optional depth image path")
    parser.add_argument("--ref-bbox", nargs=4, type=float, default=None, help="Reference bbox [x1, y1, x2, y2]")
    parser.add_argument("--ref-size", type=float, default=2.5, help="Reference size in cm")
    parser.add_argument("--detector-weights", default="models/yolov8n.pt", help="Detector weights")
    parser.add_argument("--segmenter-weights", default="models/segmentation/best_unet.pt", help="Segmenter weights")
    parser.add_argument("--classifier-weights", default="models/classification/best_classifier.pt", help="Classifier weights")
    parser.add_argument("--mass-weights", default="models/portion/best_mass_regressor.pt", help="Mass weights")
    parser.add_argument("--nutrient-weights", default="models/nutrition/best_nutrient_model.pt", help="Nutrient weights")
    parser.add_argument("--vocab", default=None, help="Ingredient vocab JSON path")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_complete_nutrition_pipeline(
        image_path=args.image,
        output_path=args.output,
        output_csv_path=args.output_csv,
        output_image_path=args.output_img,
        depth_path=args.depth,
        reference_bbox=args.ref_bbox,
        reference_real_size_cm=args.ref_size,
        detector_weights=args.detector_weights,
        segmenter_weights=args.segmenter_weights,
        classifier_weights=args.classifier_weights,
        mass_weights=args.mass_weights,
        nutrient_weights=args.nutrient_weights,
        vocab_path=args.vocab,
        device=args.device,
    )


if __name__ == "__main__":
    main()
