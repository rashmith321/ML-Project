"""
Unified FoodSegmenter module connecting object detection boxes to segmentation masks.
Supports both SAM box-prompted segmentation and supervised U-Net models.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, Union

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from PIL import Image

from src.detection.detector import Detection
from src.segmentation.models import FoodUNet
from src.segmentation.sam_utils import MaskType, SAMSegmenter
from src.utils.schema import FoodInstance


class FoodSegmenter:
    """
    Food instance segmenter. Takes an image and detection boxes, producing
    pixel-accurate boolean segmentation masks.
    """

    def __init__(
        self,
        mode: str = "sam",  # "sam" | "unet" | "auto"
        sam_checkpoint: str | Path | None = "models/sam_vit_b_01ec64.pth",
        unet_checkpoint: str | Path | None = "models/segmentation/best_unet.pt",
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        self.mode = mode
        self.sam_checkpoint = Path(sam_checkpoint) if sam_checkpoint else None
        self.unet_checkpoint = Path(unet_checkpoint) if unet_checkpoint else None
        self.device = device
        self.threshold = threshold

        self.sam_segmenter = SAMSegmenter(checkpoint_path=self.sam_checkpoint, device=self.device)
        self.unet_model: FoodUNet | None = None
        self._is_loaded = False

    def load(self) -> None:
        """Load segmentation backend weights."""
        if self.mode in ("unet", "auto"):
            if self.unet_checkpoint and self.unet_checkpoint.is_file():
                try:
                    self.unet_model = FoodUNet(in_channels=3, out_channels=1)
                    ckpt = torch.load(self.unet_checkpoint, map_location=self.device)
                    state_dict = ckpt.get("model_state_dict", ckpt)
                    self.unet_model.load_state_dict(state_dict)
                    self.unet_model.to(self.device)
                    self.unet_model.eval()
                    self._is_loaded = True
                    print(f"[FoodSegmenter] Loaded U-Net from {self.unet_checkpoint}")
                    return
                except Exception as e:
                    print(f"[FoodSegmenter] Warning: Failed loading U-Net ({e}).")

        # Fallback or SAM mode
        self.sam_segmenter.load()
        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def segment_from_box(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        bbox: Sequence[float],
    ) -> np.ndarray:
        """
        Generate a boolean segmentation mask for one bounding box [x1, y1, x2, y2].
        Returns a boolean 2D numpy array of shape (H, W).
        """
        if not self._is_loaded:
            self.load()

        if self.mode in ("unet", "auto") and self.unet_model is not None:
            return self._segment_with_unet(image, bbox)

        # SAM mode or fallback
        return self.sam_segmenter.segment_from_box(image, bbox)

    def _segment_with_unet(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        bbox: Sequence[float],
    ) -> np.ndarray:
        """Run supervised U-Net on cropped detection box region."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.fromarray(image).convert("RGB")

        w, h = pil_img.size
        x1, y1, x2, y2 = [int(round(c)) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        full_mask = np.zeros((h, w), dtype=bool)
        if x2 <= x1 or y2 <= y1:
            return full_mask

        crop = pil_img.crop((x1, y1, x2, y2)).resize((256, 256), Image.BILINEAR)
        crop_arr = np.array(crop, dtype=np.float32) / 255.0
        crop_tensor = torch.from_numpy(crop_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.unet_model(crop_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        crop_mask_small = probs > self.threshold
        crop_mask = Image.fromarray(crop_mask_small.astype(np.uint8) * 255).resize((x2 - x1, y2 - y1), Image.NEAREST)
        full_mask[y1:y2, x1:x2] = np.array(crop_mask) > 127
        return full_mask

    def segment_instances(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        detections: Sequence[Detection],
        prefix: str = "food",
    ) -> list[FoodInstance]:
        """
        Given an image and a sequence of Detections from FoodDetector,
        generate a boolean mask for each detection and return a list of FoodInstance objects.
        """
        instances: list[FoodInstance] = []
        for i, det in enumerate(detections):
            mask = self.segment_from_box(image, det.bbox)
            instance = FoodInstance(
                instance_id=f"{prefix}_{i + 1}",
                bbox=tuple(float(round(c, 2)) for c in det.bbox),
                mask=mask,
                class_name=det.class_name,
                class_confidence=float(round(det.confidence, 4)),
            )
            instances.append(instance)
        return instances
