"""
End-to-end integration: Food Detection -> Food Segmentation.
Takes an image, detects food regions, prompts the segmentation model,
and outputs pixel masks and visualizations.
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

from src.detection.detector import FoodDetector
from src.segmentation.sam_utils import MaskType
from src.segmentation.segmenter import FoodSegmenter
from src.segmentation.visualize import save_segmentation_visualization


def run_detection_and_segmentation(
    image: Union[str, Path, np.ndarray, Image.Image],
    detector: FoodDetector,
    segmenter: FoodSegmenter,
    output_json: str | Path | None = "outputs/segmentation/predictions.json",
    visualization_dir: str | Path | None = "outputs/segmentation/visualizations",
) -> dict[str, Any]:
    """
    Execute full Detection -> Segmentation pipeline on a single image.
    Returns structured results with mask properties.
    """
    if isinstance(image, (str, Path)):
        img_path_str = str(Path(image).resolve()).replace("\\", "/")
        pil_img = Image.open(str(image)).convert("RGB")
        stem = Path(image).stem
    else:
        img_path_str = "in_memory"
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        stem = "sample"

    img_w, img_h = pil_img.size
    total_pixels = img_w * img_h

    # Step 1: Detect food objects
    detections = detector.predict(pil_img)

    # Step 2: Segment each detected food region
    instances: list[dict[str, Any]] = []
    combined_mask = np.zeros((img_h, img_w), dtype=bool)

    for idx, det in enumerate(detections):
        mask = segmenter.segment_from_box(pil_img, det.bbox)
        mask_area = int(np.sum(mask))
        combined_mask = np.logical_or(combined_mask, mask)

        mask_type = MaskType.PREDICTED.value if segmenter.mode == "unet" else MaskType.PSEUDO.value

        inst_entry = {
            "instance_id": f"food_instance_{idx + 1}",
            "class_id": det.class_id,
            "class_name": det.class_name,
            "confidence": round(det.confidence, 4),
            "bbox": [round(c, 2) for c in det.bbox],
            "mask_type": mask_type,
            "mask_area_pixels": mask_area,
            "area_percentage": round((mask_area / max(1, total_pixels)) * 100.0, 2),
        }
        instances.append(inst_entry)

    # Step 3: Save visualization if directory specified
    vis_path_str = None
    if visualization_dir:
        vdir = Path(visualization_dir)
        vdir.mkdir(parents=True, exist_ok=True)
        vis_out = vdir / f"seg_4panel_{stem}.png"
        save_segmentation_visualization(
            image=pil_img,
            predicted_mask=combined_mask,
            output_path=vis_out,
            ground_truth_mask=None,  # Live inference has no GT mask
        )
        vis_path_str = str(vis_out.resolve()).replace("\\", "/")

    result = {
        "image_path": img_path_str,
        "image_size": [img_w, img_h],
        "num_food_instances": len(instances),
        "instances": instances,
        "visualization_path": vis_path_str,
    }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved segmentation predictions to: {out_p}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run integrated Detection -> Segmentation.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--detector-weights", default="models/yolov8n.pt", help="YOLO detector weights")
    parser.add_argument("--conf", type=float, default=0.20, help="Detection confidence threshold")
    parser.add_argument("--mode", default="auto", choices=["sam", "unet", "auto"], help="Segmentation mode")
    parser.add_argument("--sam-checkpoint", default="models/sam_vit_b_01ec64.pth", help="SAM checkpoint")
    parser.add_argument("--unet-checkpoint", default="models/segmentation/best_unet.pt", help="U-Net checkpoint")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output", default="outputs/segmentation/predictions.json", help="Output predictions JSON")
    parser.add_argument("--vis-dir", default="outputs/segmentation/visualizations", help="Output visualizations dir")
    args = parser.parse_args()

    detector = FoodDetector(weights_path=args.detector_weights, conf_threshold=args.conf, device=args.device)
    detector.load()

    segmenter = FoodSegmenter(
        mode=args.mode,
        sam_checkpoint=args.sam_checkpoint,
        unet_checkpoint=args.unet_checkpoint,
        device=args.device,
    )
    segmenter.load()

    result = run_detection_and_segmentation(
        image=args.image,
        detector=detector,
        segmenter=segmenter,
        output_json=args.output,
        visualization_dir=args.vis_dir,
    )

    print(f"\n--- Segmentation Inference Complete ---")
    print(f"Detected {result['num_food_instances']} food instances:")
    for inst in result["instances"]:
        print(f"  - {inst['class_name']} ({inst['confidence']*100:.1f}%) | Area: {inst['mask_area_pixels']} px ({inst['area_percentage']}%) | Type: {inst['mask_type']}")


if __name__ == "__main__":
    main()
