"""
End-to-End Inference Pipeline:
Image -> Detection -> Segmentation -> Food Classification -> Food Name + Confidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence, Union

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image

from src.classification.classifier import FoodClassifier
from src.classification.visualize import draw_classification_badge
from src.detection.detector import FoodDetector
from src.segmentation.segmenter import FoodSegmenter
from src.utils.schema import FoodInstance


def run_full_pipeline(
    image: Union[str, Path, np.ndarray, Image.Image],
    detector: FoodDetector,
    segmenter: FoodSegmenter,
    classifier: FoodClassifier,
    top_k: int = 5,
    output_json: str | Path | None = "outputs/classification/predictions.json",
    visualization_dir: str | Path | None = "outputs/classification/visualizations",
) -> dict[str, Any]:
    """
    Execute complete Detection -> Segmentation -> Classification pipeline.
    """
    if isinstance(image, (str, Path)):
        img_path_str = str(Path(image).resolve()).replace("\\", "/")
        pil_img = Image.open(str(image)).convert("RGB")
        stem = Path(image).stem
    else:
        img_path_str = "in_memory"
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        stem = "sample"

    w, h = pil_img.size

    # 1. Detection
    detections = detector.predict(pil_img)

    # 2. Segmentation
    instances = segmenter.segment_instances(pil_img, detections)

    # 3. Classification
    classified_instances = classifier.classify_instances(pil_img, instances, top_k=top_k)

    # Compile detailed results
    detailed_instances: list[dict[str, Any]] = []
    vdir = Path(visualization_dir) if visualization_dir else None
    if vdir:
        vdir.mkdir(parents=True, exist_ok=True)

    for i, inst in enumerate(classified_instances):
        # Extract top-k for detailed report
        x1, y1, x2, y2 = [int(round(c)) for c in inst.bbox] if inst.bbox else [0, 0, w, h]
        crop = pil_img.crop((x1, y1, x2, y2))
        crop_mask = inst.mask[y1:y2, x1:x2] if inst.mask is not None else None
        res = classifier.predict(crop, mask=crop_mask, top_k=top_k)

        # Save annotated crop if vis requested
        vis_crop_path = None
        if vdir:
            badged = draw_classification_badge(crop, res)
            crop_out = vdir / f"{stem}_inst_{i + 1}_{res.food_name}.jpg"
            badged.save(crop_out, quality=95)
            vis_crop_path = str(crop_out.resolve()).replace("\\", "/")

        detailed_instances.append({
            "instance_id": inst.instance_id,
            "food_name": res.food_name,
            "class_id": res.class_id,
            "confidence": res.confidence,
            "bbox": [round(c, 2) for c in inst.bbox] if inst.bbox else None,
            "mask_area_pixels": int(np.sum(inst.mask)) if inst.mask is not None else None,
            "top_k_predictions": res.top_k_predictions,
            "crop_visualization": vis_crop_path,
        })

    result = {
        "image_path": img_path_str,
        "image_size": [w, h],
        "num_classified_foods": len(detailed_instances),
        "instances": detailed_instances,
    }

    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved full pipeline predictions to: {out}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Detection -> Segmentation -> Classification pipeline.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--detector-weights", default="models/yolov8n.pt", help="YOLO detector weights")
    parser.add_argument("--classifier-weights", default="models/classification/best_classifier.pt", help="Classifier weights")
    parser.add_argument("--backbone", default="efficientnet_b0", help="Classifier backbone")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K predictions")
    parser.add_argument("--output", default="outputs/classification/predictions.json", help="Predictions JSON output")
    parser.add_argument("--vis-dir", default="outputs/classification/visualizations", help="Visualizations directory")
    args = parser.parse_args()

    detector = FoodDetector(weights_path=args.detector_weights, conf_threshold=0.15, device=args.device)
    detector.load()

    segmenter = FoodSegmenter(mode="auto", device=args.device)
    segmenter.load()

    classifier = FoodClassifier(
        weights_path=args.classifier_weights,
        backbone=args.backbone,
        device=args.device,
    )
    classifier.load()

    result = run_full_pipeline(
        image=args.image,
        detector=detector,
        segmenter=segmenter,
        classifier=classifier,
        top_k=args.top_k,
        output_json=args.output,
        visualization_dir=args.vis_dir,
    )

    print("\n--- Full Pipeline Inference Result ---")
    print(f"Identified {result['num_classified_foods']} food instances:")
    for inst in result["instances"]:
        print(f"  * {inst['food_name'].upper()} ({inst['confidence']*100:.1f}%) [class_id: {inst['class_id']}]")
        top_alts = [f"{p['food_name']}: {p['confidence']*100:.1f}%" for p in inst['top_k_predictions'][:3]]
        print(f"    Top-k: {', '.join(top_alts)}")


if __name__ == "__main__":
    main()
