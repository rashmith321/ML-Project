"""
Portion / Mass / Volume Inference Pipeline.
Integrates image + mask + food class + optional depth + optional reference object.
Produces:
  - estimated_mass_g
  - estimated_volume
  - confidence
  - method_used (depth_based | reference_calibrated | rgb_learned)
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

from src.portion.mass_estimator import MassEstimator
from src.portion.volume_estimator import EstimationMethod, VolumeEstimator
from src.utils.schema import FoodInstance, Source, Value


class PortionInference:
    """
    Portion estimation inference engine.
    """

    def __init__(
        self,
        weights_path: str | Path | None = "models/portion/best_mass_regressor.pt",
        device: str = "cpu",
    ):
        self.volume_estimator = VolumeEstimator()
        self.mass_estimator = MassEstimator(
            weights_path=weights_path,
            volume_estimator=self.volume_estimator,
            device=device,
        )

    def estimate_instance(
        self,
        image: Image.Image | np.ndarray,
        food_bbox: Sequence[float],
        food_class: str | None = None,
        mask: np.ndarray | None = None,
        depth_map: np.ndarray | None = None,
        reference_bbox: Sequence[float] | None = None,
        reference_real_size_cm: float | None = None,
    ) -> dict[str, Any]:
        """
        Estimate quantity for a single detected and segmented food instance.
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        w, h = pil_img.size
        fx1, fy1, fx2, fy2 = [int(round(v)) for v in food_bbox]
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(w, max(fx1 + 1, fx2)), min(h, max(fy1 + 1, fy2))

        crop = pil_img.crop((fx1, fy1, fx2, fy2))

        crop_mask = None
        if mask is not None:
            m_h, m_w = mask.shape[:2]
            if (m_w, m_h) == (w, h):
                crop_mask = mask[fy1:fy2, fx1:fx2]
            else:
                crop_mask = mask

        res = self.mass_estimator.estimate_portion(
            crop=crop,
            food_bbox=[fx1, fy1, fx2, fy2],
            food_class=food_class,
            mask=crop_mask,
            image_shape=(h, w),
            depth_map=depth_map,
            reference_bbox=reference_bbox,
            reference_real_size_cm=reference_real_size_cm,
        )

        return res

    def annotate_instances(
        self,
        instances: list[FoodInstance],
        image: Image.Image | np.ndarray,
        depth_map: np.ndarray | None = None,
        reference_bbox: Sequence[float] | None = None,
        reference_real_size_cm: float | None = None,
    ) -> tuple[list[FoodInstance], list[dict[str, Any]]]:
        """
        Annotate FoodInstance records with estimated mass and volume.
        """
        detailed_records: list[dict[str, Any]] = []

        for inst in instances:
            if inst.bbox is None:
                continue

            est = self.estimate_instance(
                image=image,
                food_bbox=inst.bbox,
                food_class=inst.class_name,
                mask=inst.mask,
                depth_map=depth_map,
                reference_bbox=reference_bbox,
                reference_real_size_cm=reference_real_size_cm,
            )

            # Update FoodInstance
            source_enum = Source.DERIVED if est["mass_source"] == "derived" else Source.PREDICTION
            inst.mass = Value(value=float(est["estimated_mass_g"]), source=source_enum)

            record = {
                "instance_id": inst.instance_id,
                **est,
                "bbox": list(inst.bbox),
            }
            detailed_records.append(record)

        return instances, detailed_records


def run_portion_pipeline(
    image_path: str | Path,
    output_path: str | Path = "outputs/portion/predictions.json",
    depth_path: str | Path | None = None,
    reference_bbox: Sequence[float] | None = None,
    reference_real_size_cm: float | None = None,
    detector_weights: str | Path = "models/yolov8n.pt",
    segmenter_weights: str | Path = "models/segmentation/best_unet.pt",
    classifier_weights: str | Path = "models/classification/best_classifier.pt",
    mass_weights: str | Path = "models/portion/best_mass_regressor.pt",
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Execute complete pipeline:
      Image -> Detection -> Segmentation -> Classification -> Portion Estimation
    """
    from src.classification.classifier import FoodClassifier
    from src.classification.inference import run_full_pipeline
    from src.detection.detector import FoodDetector
    from src.segmentation.segmenter import FoodSegmenter

    img = Image.open(image_path).convert("RGB")

    # Load depth if provided
    depth_map = None
    if depth_path and Path(depth_path).is_file():
        depth_img = Image.open(depth_path)
        depth_map = np.array(depth_img)

    # 1. Detection + Segmentation + Classification
    detector = FoodDetector(weights_path=detector_weights, device=device)
    detector.load()

    segmenter = FoodSegmenter(mode="auto", unet_checkpoint=segmenter_weights, device=device)
    segmenter.load()

    classifier = FoodClassifier(weights_path=classifier_weights, device=device)
    classifier.load()

    base_results = run_full_pipeline(
        image=image_path,
        detector=detector,
        segmenter=segmenter,
        classifier=classifier,
        output_json=None,
    )

    # 2. Portion Estimation
    engine = PortionInference(weights_path=mass_weights, device=device)
    portion_instances: list[dict[str, Any]] = []

    for inst_dict in base_results.get("instances", []):
        bbox = inst_dict.get("bbox")
        food_name = inst_dict.get("food_name")

        est = engine.estimate_instance(
            image=img,
            food_bbox=bbox,
            food_class=food_name,
            depth_map=depth_map,
            reference_bbox=reference_bbox,
            reference_real_size_cm=reference_real_size_cm,
        )

        item = {
            "instance_id": inst_dict.get("instance_id"),
            "food_name": food_name,
            "confidence": inst_dict.get("confidence"),
            "bbox": bbox,
            "estimated_mass_g": est["estimated_mass_g"],
            "estimated_volume": est["estimated_volume"],
            "portion_confidence": est["confidence"],
            "method_used": est["method_used"],
            "mass_source": est["mass_source"],
            "disclaimer": est["disclaimer"],
        }
        portion_instances.append(item)

    payload = {
        "image_path": str(Path(image_path).resolve()).replace("\\", "/"),
        "image_size": list(img.size),
        "num_foods": len(portion_instances),
        "has_depth_input": depth_map is not None,
        "has_reference_object": reference_bbox is not None,
        "instances": portion_instances,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved portion predictions to: {out}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run portion, mass, and volume estimation inference.")
    parser.add_argument("--image", required=True, help="Input food image path")
    parser.add_argument("--output", default="outputs/portion/predictions.json", help="Output JSON path")
    parser.add_argument("--depth", default=None, help="Optional depth image path (RGB-D)")
    parser.add_argument("--ref-bbox", nargs=4, type=float, default=None, help="Optional reference bbox [x1, y1, x2, y2]")
    parser.add_argument("--ref-size", type=float, default=2.5, help="Reference real size in cm")
    parser.add_argument("--detector-weights", default="models/yolov8n.pt", help="Detector weights")
    parser.add_argument("--segmenter-weights", default="models/segmentation/best_unet.pt", help="Segmenter weights")
    parser.add_argument("--classifier-weights", default="models/classification/best_classifier.pt", help="Classifier weights")
    parser.add_argument("--mass-weights", default="models/portion/best_mass_regressor.pt", help="Mass regressor weights")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_portion_pipeline(
        image_path=args.image,
        output_path=args.output,
        depth_path=args.depth,
        reference_bbox=args.ref_bbox,
        reference_real_size_cm=args.ref_size,
        detector_weights=args.detector_weights,
        segmenter_weights=args.segmenter_weights,
        classifier_weights=args.classifier_weights,
        mass_weights=args.mass_weights,
        device=args.device,
    )


if __name__ == "__main__":
    main()
