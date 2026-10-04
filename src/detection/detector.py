"""
Food object detection stage (ARCHITECTURE.md: YOLO-family detector).

Supports detecting multiple food items per image and outputs structured Detection
objects matching the required schema:
{
    "class_id": int,
    "class_name": str,
    "bbox": [x1, y1, x2, y2],
    "confidence": float
}
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Union

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image

from src.utils.schema import FoodInstance


@dataclass
class Detection:
    """Standardized detection output for a single detected food object."""
    class_id: int
    class_name: str
    bbox: list[float]  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 coordinates [x1, y1, x2, y2], got {self.bbox}")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    def as_dict(self) -> dict[str, Any]:
        """Serialize detection to the required project output schema."""
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "bbox": [float(round(coord, 2)) for coord in self.bbox],
            "confidence": float(round(self.confidence, 4)),
        }

    def to_food_instance(self, instance_id: str) -> FoodInstance:
        """Convert this detection into a FoodInstance for downstream stages."""
        return FoodInstance(
            instance_id=instance_id,
            bbox=tuple(float(round(c, 2)) for c in self.bbox),
            class_name=self.class_name,
            class_confidence=float(round(self.confidence, 4)),
        )


# ---------------------------------------------------------------------------
# BLACKLIST: COCO class IDs that are definitively NOT food.
# Strategy: block known non-food classes rather than whitelisting food classes.
# This is safer because:
#  - Fine-tuned / custom models may use class IDs outside the COCO food range.
#  - Any detection not in this blacklist is treated as a potential food item.
#  - Only explicitly identified non-food COCO classes are blocked.
#
# MS-COCO 2017 non-food class IDs (0-indexed, as YOLOv8 uses internally):
# ---------------------------------------------------------------------------
COCO_NON_FOOD_CLASS_IDS: frozenset[int] = frozenset({
    # People
    0,   # person
    # Vehicles
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    4,   # airplane
    5,   # bus
    6,   # train
    7,   # truck
    8,   # boat
    # Outdoor objects
    9,   # traffic light
    10,  # fire hydrant
    11,  # stop sign
    12,  # parking meter
    13,  # bench
    # Animals
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
    # Accessories
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    27,  # tie
    28,  # suitcase
    # Sports
    29,  # frisbee
    30,  # skis
    31,  # snowboard
    32,  # sports ball
    33,  # kite
    34,  # baseball bat
    35,  # baseball glove
    36,  # skateboard
    37,  # surfboard
    38,  # tennis racket
    # Electronics / indoor objects (bottle/cup/utensils are intentionally NOT blacklisted)
    56,  # chair
    57,  # couch
    58,  # potted plant
    59,  # bed
    60,  # dining table  ← THE ROOT CAUSE of the mass explosion
    61,  # toilet
    62,  # tv
    63,  # laptop
    64,  # mouse
    65,  # remote
    66,  # keyboard
    67,  # cell phone
    68,  # microwave
    69,  # oven
    70,  # toaster
    71,  # sink
    72,  # refrigerator
    73,  # book
    74,  # clock
    75,  # vase
    76,  # scissors
    77,  # teddy bear
    78,  # hair drier
    79,  # toothbrush
})

# Name substrings that conclusively identify a non-food class
_NON_FOOD_NAME_KEYWORDS: frozenset[str] = frozenset({
    "dining table", "table", "chair", "bench", "couch", "sofa", "bed",
    "toilet", "sink", "refrigerator", "oven", "microwave", "toaster",
    "person", "people", "human",
    "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "train", "boat",
    "dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear",
    "tv", "laptop", "phone", "keyboard", "remote", "monitor",
    "book", "clock", "vase", "scissors",
})


def _is_non_food_class(cls_id: int, cls_name: str) -> bool:
    """Return True if the detection is definitively a non-food object to exclude."""
    if cls_id in COCO_NON_FOOD_CLASS_IDS:
        return True
    name_lower = cls_name.lower()
    return any(kw in name_lower for kw in _NON_FOOD_NAME_KEYWORDS)


class FoodDetector:
    """
    Modern YOLO-family food detector supporting transfer learning and multi-food detection.
    Default backbone: YOLOv8 (configurable via weights_path or model_name).
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        class_mapping: dict[int, str] | None = None,
        filter_non_food: bool = True,
        max_bbox_fraction: float = 0.70,
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_mapping = class_mapping or {}
        self._model = None
        # When True, detections matching the non-food blacklist are filtered out.
        self.filter_non_food = filter_non_food
        # Detections whose bbox covers more than this fraction of image area are dropped
        # with an explicit WARNING (not silently clipped).
        self.max_bbox_fraction = max_bbox_fraction

    def load(self) -> None:
        """Load YOLO model weights. Prefers weights_path if existing, otherwise model_name."""
        from ultralytics import YOLO

        target_path: str | Path
        if self.weights_path and self.weights_path.is_file():
            target_path = self.weights_path
        elif Path("models") / self.model_name and (Path("models") / self.model_name).is_file():
            target_path = Path("models") / self.model_name
        else:
            target_path = self.model_name

        self._model = YOLO(str(target_path))
        if hasattr(self._model, "names") and isinstance(self._model.names, dict):
            # Populate class mapping from model names if not explicitly overridden
            for k, v in self._model.names.items():
                if int(k) not in self.class_mapping:
                    self.class_mapping[int(k)] = str(v)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[Detection]:
        """
        Run food object detection on a single image.
        Supports file paths, numpy arrays, or PIL Images.
        Returns a list of Detection instances (one per detected food item).

        Known non-food COCO classes (e.g. 'dining table', 'chair', 'person') are
        filtered out when filter_non_food=True (the default).  All other classes
        — including custom fine-tuned model classes — are passed through.
        Detections whose bounding box covers more than max_bbox_fraction of image
        area are also dropped with a WARNING (not silently clipped).
        """
        if self._model is None:
            self.load()

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        # Run inference using ultralytics YOLO
        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        # Determine image dimensions for bbox-fraction guard
        img_h, img_w = first_res.orig_shape  # (H, W)
        img_area = max(1.0, float(img_w * img_h))

        for box in first_res.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf_score = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.class_mapping.get(cls_id, str(first_res.names.get(cls_id, f"class_{cls_id}")))

            # --- Non-food blacklist filter ---
            if self.filter_non_food and _is_non_food_class(cls_id, cls_name):
                print(
                    f"[FoodDetector] INFO: Skipping non-food class "
                    f"'{cls_name}' (id={cls_id}, conf={conf_score:.3f})."
                )
                continue

            # --- Oversized bbox guard ---
            x1, y1, x2, y2 = xyxy
            bbox_area = max(0.0, (x2 - x1) * (y2 - y1))
            bbox_fraction = bbox_area / img_area
            if bbox_fraction > self.max_bbox_fraction:
                print(
                    f"[FoodDetector] WARNING: Skipping '{cls_name}' (id={cls_id}) — "
                    f"bbox covers {bbox_fraction * 100:.1f}% of image area "
                    f"(limit={self.max_bbox_fraction * 100:.0f}%). "
                    f"Likely a background/container detection."
                )
                continue

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=conf_score,
                )
            )

        return detections

    def to_food_instances(self, detections: Sequence[Detection], prefix: str = "food") -> list[FoodInstance]:
        """Convert a sequence of Detections into FoodInstance objects."""
        return [
            det.to_food_instance(instance_id=f"{prefix}_{i + 1}")
            for i, det in enumerate(detections)
        ]
