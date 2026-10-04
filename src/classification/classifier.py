"""
Food classification module.
Predicts food categories on detected and segmented food regions using transfer learning.
Outputs:
{
    "food_name": "...",
    "class_id": int,
    "confidence": float,
    "top_k_predictions": [...]
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
import torch
import torch.nn.functional as F
from PIL import Image

from src.classification.taxonomy import FoodTaxonomy
from src.models.classifier import FoodClassificationModel
from src.utils.schema import FoodInstance


@dataclass
class ClassificationResult:
    """Standardized classification result matching project requirements."""
    food_name: str
    class_id: int
    confidence: float
    top_k_predictions: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "food_name": str(self.food_name),
            "class_id": int(self.class_id),
            "confidence": float(round(self.confidence, 4)),
            "top_k_predictions": [
                {
                    "class_id": int(p["class_id"]),
                    "food_name": str(p["food_name"]),
                    "confidence": float(round(p["confidence"], 4)),
                }
                for p in self.top_k_predictions
            ],
        }


class FoodClassifier:
    """
    Food classifier utilizing transfer learning (EfficientNet-B0 default).
    Takes masked crops and outputs top-k predictions.
    """

    def __init__(
        self,
        weights_path: str | Path | None = "models/classification/best_classifier.pt",
        backbone: str = "efficientnet_b0",
        num_classes: int = 101,
        device: str = "cpu",
        taxonomy: FoodTaxonomy | None = None,
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self.backbone = backbone
        self.num_classes = num_classes
        self.device = device
        self.taxonomy = taxonomy or FoodTaxonomy(primary_dataset="food101")
        self._model: FoodClassificationModel | None = None
        self._is_loaded = False

    def load(self) -> None:
        """Load classification model weights."""
        if self.weights_path and self.weights_path.is_file():
            try:
                self._model = FoodClassificationModel.load_from_checkpoint(
                    self.weights_path, device=self.device
                )
                print(f"[FoodClassifier] Loaded checkpoint from {self.weights_path}")
            except Exception as e:
                print(f"[FoodClassifier] Warning: Could not load checkpoint ({e}). Initializing backbone.")
                self._model = FoodClassificationModel(
                    backbone=self.backbone, num_classes=self.num_classes, pretrained=True
                ).to(self.device)
        else:
            self._model = FoodClassificationModel(
                backbone=self.backbone, num_classes=self.num_classes, pretrained=True
            ).to(self.device)

        self._model.eval()
        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def preprocess_crop(
        self,
        image_crop: Union[str, Path, np.ndarray, Image.Image],
        mask: np.ndarray | None = None,
        target_size: tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """
        Preprocess a food crop. If a segmentation mask is provided,
        applies it to exclude background clutter before normalizing.
        """
        if isinstance(image_crop, (str, Path)):
            pil_img = Image.open(str(image_crop)).convert("RGB")
        elif isinstance(image_crop, Image.Image):
            pil_img = image_crop.convert("RGB")
        else:
            pil_img = Image.fromarray(image_crop).convert("RGB")

        # If a segmentation mask is provided, apply as foreground mask
        if mask is not None:
            w, h = pil_img.size
            if mask.shape != (h, w):
                mask_resized = np.array(
                    Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
                ) > 0
            else:
                mask_resized = mask.astype(bool)

            arr = np.array(pil_img)
            # Set background pixels outside mask to neutral gray
            arr[~mask_resized] = [235, 235, 235]
            pil_img = Image.fromarray(arr)

        # Resize to standard classification input resolution
        pil_img = pil_img.resize(target_size, Image.BILINEAR)
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        # HWC -> 1CHW
        tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def predict(
        self,
        image_crop: Union[str, Path, np.ndarray, Image.Image],
        mask: np.ndarray | None = None,
        top_k: int = 5,
    ) -> ClassificationResult:
        """
        Classify a single food image crop with optional segmentation mask.
        Returns ClassificationResult with food_name, class_id, confidence, and top_k_predictions.
        """
        if not self._is_loaded or self._model is None:
            self.load()

        tensor = self.preprocess_crop(image_crop, mask=mask).to(self.device)

        with torch.no_grad():
            probs = self._model.predict_probs(tensor).squeeze(0)

        k = min(top_k, self.taxonomy.num_classes, probs.shape[0])
        top_probs, top_indices = torch.topk(probs, k=k)

        top_probs_list = top_probs.cpu().numpy().tolist()
        top_indices_list = top_indices.cpu().numpy().tolist()

        top_k_list: list[dict[str, Any]] = []
        for idx, prob in zip(top_indices_list, top_probs_list):
            name = self.taxonomy.id_to_name(idx)
            top_k_list.append({
                "class_id": idx,
                "food_name": name,
                "confidence": prob,
            })

        best_cls_id = top_indices_list[0]
        best_name = self.taxonomy.id_to_name(best_cls_id)
        best_conf = top_probs_list[0]

        return ClassificationResult(
            food_name=best_name,
            class_id=best_cls_id,
            confidence=best_conf,
            top_k_predictions=top_k_list,
        )

    def classify_instances(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        instances: Sequence[FoodInstance],
        top_k: int = 5,
    ) -> list[FoodInstance]:
        """
        Classify each detected and segmented FoodInstance.
        Crops each bounding box, applies its instance mask, and updates
        class_name and class_confidence in place.
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.fromarray(image).convert("RGB")

        w, h = pil_img.size

        for inst in instances:
            if inst.bbox is None:
                continue

            x1, y1, x2, y2 = [int(round(c)) for c in inst.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            crop_mask = None
            if inst.mask is not None:
                crop_mask = inst.mask[y1:y2, x1:x2]

            res = self.predict(crop, mask=crop_mask, top_k=top_k)
            inst.class_name = res.food_name
            inst.class_confidence = round(res.confidence, 4)

        return list(instances)
