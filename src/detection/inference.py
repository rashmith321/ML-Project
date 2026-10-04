"""
Inference pipeline for food object detection.
Runs single-image or batch prediction and writes structured JSON outputs
and annotated visual output images.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.detection.detector import FoodDetector
from src.detection.visualize import save_annotated_image


def run_detection_inference(
    images: Sequence[str | Path],
    weights_path: str | Path = "models/yolov8n.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "cpu",
    output_json: str | Path | None = "outputs/detection/predictions.json",
    annotated_dir: str | Path | None = "outputs/detection/annotated",
) -> list[dict[str, Any]]:
    """
    Run detection on a list of images.
    Returns and optionally writes structured detection results.
    """
    detector = FoodDetector(
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )
    detector.load()

    all_results: list[dict[str, Any]] = []
    ann_dir = Path(annotated_dir) if annotated_dir else None
    if ann_dir:
        ann_dir.mkdir(parents=True, exist_ok=True)

    for img_path_raw in images:
        img_path = Path(img_path_raw)
        if not img_path.is_file():
            print(f"[warning] Image not found: {img_path}")
            continue

        detections = detector.predict(img_path)
        det_dicts = [d.as_dict() for d in detections]

        entry = {
            "image_path": str(img_path.resolve()).replace("\\", "/"),
            "num_detections": len(detections),
            "detections": det_dicts,
        }
        all_results.append(entry)

        if ann_dir:
            out_img_path = ann_dir / f"annotated_{img_path.name}"
            save_annotated_image(img_path, detections, out_img_path)

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved {len(all_results)} detection predictions to: {out_p}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run food detection inference.")
    parser.add_argument("--image", nargs="+", required=True, help="One or more image paths or glob pattern")
    parser.add_argument("--weights", default="models/yolov8n.pt", help="YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output", default="outputs/detection/predictions.json", help="Output predictions JSON")
    parser.add_argument("--annotated-dir", default="outputs/detection/annotated", help="Directory for annotated images")
    args = parser.parse_args()

    # Collect images
    image_paths: list[Path] = []
    for item in args.image:
        p = Path(item)
        if p.is_file():
            image_paths.append(p)
        elif p.is_dir():
            image_paths.extend([f for f in p.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        else:
            # Try glob
            image_paths.extend([f for f in Path().glob(item) if f.is_file()])

    if not image_paths:
        print("No valid images found for inference.")
        return

    results = run_detection_inference(
        images=image_paths,
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        output_json=args.output,
        annotated_dir=args.annotated_dir,
    )
    print(f"Processed {len(results)} images.")


if __name__ == "__main__":
    main()
