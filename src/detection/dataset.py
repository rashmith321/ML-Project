"""
YOLO dataset utilities, bounding box validation, class mapping verification,
and split leakage checking for food object detection.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw
import yaml

from src.data.splits import find_split_leakage


def xyxy_to_xywhn(bbox: Sequence[float], img_w: int, img_h: int) -> list[float]:
    """Convert [x1, y1, x2, y2] pixel coordinates to normalized [x_center, y_center, w, h]."""
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    x_center = x1 + (w / 2.0)
    y_center = y1 + (h / 2.0)
    return [
        float(np.clip(x_center / img_w, 0.0, 1.0)),
        float(np.clip(y_center / img_h, 0.0, 1.0)),
        float(np.clip(w / img_w, 0.0, 1.0)),
        float(np.clip(h / img_h, 0.0, 1.0)),
    ]


def xywhn_to_xyxy(xywhn: Sequence[float], img_w: int, img_h: int) -> list[float]:
    """Convert normalized [x_center, y_center, w, h] to [x1, y1, x2, y2] in pixels."""
    xc, yc, w, h = xywhn
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return [
        float(np.clip(x1, 0.0, float(img_w))),
        float(np.clip(y1, 0.0, float(img_h))),
        float(np.clip(x2, 0.0, float(img_w))),
        float(np.clip(y2, 0.0, float(img_h))),
    ]


def validate_bbox(
    bbox: Sequence[float], img_w: int, img_h: int, min_size: float = 2.0
) -> tuple[list[float], bool]:
    """
    Validate and clamp bounding box [x1, y1, x2, y2].
    Returns (clamped_bbox, is_valid).
    """
    if len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0], False

    x1, y1, x2, y2 = bbox
    # Ensure min < max
    if x2 <= x1 or y2 <= y1:
        return [0.0, 0.0, 0.0, 0.0], False

    clamped_x1 = max(0.0, min(float(x1), float(img_w)))
    clamped_y1 = max(0.0, min(float(y1), float(img_h)))
    clamped_x2 = max(0.0, min(float(x2), float(img_w)))
    clamped_y2 = max(0.0, min(float(y2), float(img_h)))

    w = clamped_x2 - clamped_x1
    h = clamped_y2 - clamped_y1
    if w < min_size or h < min_size:
        return [clamped_x1, clamped_y1, clamped_x2, clamped_y2], False

    return [clamped_x1, clamped_y1, clamped_x2, clamped_y2], True


def verify_class_mapping(classes: Sequence[str] | dict[int, str]) -> dict[int, str]:
    """
    Verify and build a contiguous zero-indexed mapping: 0 -> class_0, 1 -> class_1.
    """
    if isinstance(classes, dict):
        keys = sorted(classes.keys())
        if keys != list(range(len(keys))):
            # Remap to 0-indexed contiguous
            return {i: classes[k] for i, k in enumerate(keys)}
        return {int(k): str(v) for k, v in classes.items()}

    return {i: str(cls_name) for i, cls_name in enumerate(classes)}


def write_yolo_data_yaml(
    output_path: Path,
    dataset_root: Path,
    class_mapping: dict[int, str],
    train_rel: str = "images/train",
    val_rel: str = "images/val",
    test_rel: str = "images/test",
) -> Path:
    """Create data.yaml for Ultralytics YOLO training."""
    data_dict = {
        "path": str(dataset_root.resolve()).replace("\\", "/"),
        "train": train_rel,
        "val": val_rel,
        "test": test_rel,
        "names": {int(k): str(v) for k, v in class_mapping.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)
    return output_path


def create_detection_fixture(
    fixture_dir: Path,
    class_names: list[str] | None = None,
    num_train: int = 4,
    num_val: int = 2,
    num_test: int = 2,
    img_size: tuple[int, int] = (320, 320),
) -> Path:
    """
    Generate a format-accurate mini synthetic YOLO dataset for testing training,
    evaluation, and inference loops end-to-end without fabricating real benchmark scores.
    """
    classes = class_names or ["apple", "bread", "bowl"]
    class_map = {i: name for i, name in enumerate(classes)}
    fixture_dir = Path(fixture_dir)

    for split in ("train", "val", "test"):
        (fixture_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (fixture_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": num_train, "val": num_val, "test": num_test}
    colors = [(220, 50, 50), (200, 160, 60), (50, 120, 220)]

    for split, count in counts.items():
        for idx in range(count):
            img = Image.new("RGB", img_size, color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            labels: list[str] = []

            # Add 1 to 2 distinct objects per image
            cls1 = idx % len(classes)
            box1 = [30.0, 40.0, 120.0, 140.0]
            draw.rectangle(box1, fill=colors[cls1 % len(colors)], outline=(0, 0, 0))
            xywhn1 = xyxy_to_xywhn(box1, img_size[0], img_size[1])
            labels.append(f"{cls1} " + " ".join(f"{v:.6f}" for v in xywhn1))

            if idx % 2 == 1:
                cls2 = (cls1 + 1) % len(classes)
                box2 = [160.0, 150.0, 280.0, 270.0]
                draw.rectangle(box2, fill=colors[cls2 % len(colors)], outline=(0, 0, 0))
                xywhn2 = xyxy_to_xywhn(box2, img_size[0], img_size[1])
                labels.append(f"{cls2} " + " ".join(f"{v:.6f}" for v in xywhn2))

            img_path = fixture_dir / "images" / split / f"sample_{idx:03d}.jpg"
            lbl_path = fixture_dir / "labels" / split / f"sample_{idx:03d}.txt"
            img.save(img_path, quality=90)
            lbl_path.write_text("\n".join(labels) + "\n")

    yaml_path = fixture_dir / "data.yaml"
    write_yolo_data_yaml(yaml_path, fixture_dir, class_map)
    return yaml_path
