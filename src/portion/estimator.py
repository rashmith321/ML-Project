"""
Portion / mass / volume estimation (ARCHITECTURE.md: primary path is a
Nutrition5k-trained regressor on the masked RGB(-D) crop; ECUSTFD's
calibration-object method is kept as an independent comparison estimator,
not folded into the same model -- see ARCHITECTURE.md for why).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence, Union

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.mass_regressor import FoodMassRegressionModel
from src.utils.schema import FoodInstance, Source, Value

# Common food densities in g/cm^3 for ECUSTFD metric derivation
FOOD_DENSITY_TABLE: dict[str, float] = {
    "apple": 0.85,
    "apple_pie": 0.80,
    "banana": 0.95,
    "bread": 0.35,
    "burger": 0.85,
    "cake": 0.55,
    "carrot": 0.64,
    "cheese": 1.10,
    "chicken": 1.05,
    "curry": 1.02,
    "doughnut": 0.45,
    "egg": 1.03,
    "fish": 1.02,
    "french_fries": 0.65,
    "meat": 1.05,
    "pasta": 0.90,
    "pizza": 0.75,
    "rice": 0.85,
    "salad": 0.40,
    "soup": 1.00,
    "steak": 1.05,
    "sushi": 0.92,
    "tomato": 0.98,
}
DEFAULT_DENSITY_G_PER_CM3 = 0.85


class Nutrition5kMassRegressor:
    """
    Primary estimator: learned neural regression on masked RGB(-D) food crop.
    Predicts mass in grams with provenance Source.PREDICTION.
    """

    def __init__(
        self,
        weights_path: str | Path | None = "models/portion/best_mass_regressor.pt",
        device: str = "cpu",
        backbone: str = "efficientnet_b0",
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = device
        self.backbone = backbone
        self.model: FoodMassRegressionModel | None = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load(self, weights_path: str | Path | None = None) -> None:
        """Load trained mass regressor model checkpoint."""
        wpath = Path(weights_path) if weights_path else self.weights_path
        if wpath and wpath.is_file():
            self.model = FoodMassRegressionModel.load_from_checkpoint(wpath, device=self.device)
        else:
            # Fallback to initialized model (e.g. for testing / smoke checks)
            self.model = FoodMassRegressionModel(
                backbone=self.backbone,
                in_channels=3,
                pretrained=False,
            ).to(self.device)
            self.model.eval()

    def preprocess_crop(
        self,
        crop: Image.Image | np.ndarray,
        mask: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Mask background pixels and transform to normalized tensor."""
        if isinstance(crop, np.ndarray):
            pil_img = Image.fromarray(crop).convert("RGB")
        else:
            pil_img = crop.convert("RGB")

        if mask is not None:
            mask_arr = np.array(mask)
            if mask_arr.shape[:2] != (pil_img.height, pil_img.width):
                m_img = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize(
                    (pil_img.width, pil_img.height), Image.NEAREST
                )
                mask_bool = np.array(m_img) > 127
            else:
                mask_bool = mask_arr.astype(bool)

            img_arr = np.array(pil_img)
            img_arr[~mask_bool] = 0
            pil_img = Image.fromarray(img_arr)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def predict_mass_g(
        self,
        masked_crop: Image.Image | np.ndarray,
        depth_crop: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> Value:
        """
        Forward pass returning estimated mass as a Value with Source.PREDICTION.
        """
        if self.model is None:
            self.load()

        tensor = self.preprocess_crop(masked_crop, mask=mask)
        with torch.no_grad():
            raw_mass = self.model(tensor).item()

        mass_val = max(1.0, round(float(raw_mass), 1))
        return Value(value=mass_val, source=Source.PREDICTION)

    def estimate_instances(
        self,
        instances: list[FoodInstance],
        image: Image.Image | np.ndarray,
    ) -> list[FoodInstance]:
        """
        Iterate over FoodInstance objects, crop each food instance,
        predict portion mass, and update inst.mass_g and inst.mass_source.
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        w, h = pil_img.size

        for inst in instances:
            if inst.bbox is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in inst.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))

            crop = pil_img.crop((x1, y1, x2, y2))

            # Crop instance mask if available
            crop_mask = None
            if inst.mask is not None:
                m_h, m_w = inst.mask.shape[:2]
                if (m_w, m_h) == (w, h):
                    crop_mask = inst.mask[y1:y2, x1:x2]
                else:
                    crop_mask = inst.mask

            val = self.predict_mass_g(crop, mask=crop_mask)
            inst.mass_g = val.value
            inst.mass_source = "prediction"

        return instances


class CalibrationObjectEstimator:
    """
    Comparison-only estimator: ECUSTFD-style metric calibration.
    Requires a known reference object in frame (e.g. coin of known diameter).
    Derives food mass from calibrated volume and density with Source.DERIVED.
    """

    def __init__(self, reference_object_real_size_cm: float = 2.5):
        """
        reference_object_real_size_cm: physical width/diameter in cm (e.g. 2.5 cm for a coin).
        """
        self.reference_object_real_size_cm = float(reference_object_real_size_cm)

    @staticmethod
    def get_food_density(class_name: str | None) -> float:
        """Lookup typical food density in g/cm3."""
        if not class_name:
            return DEFAULT_DENSITY_G_PER_CM3
        norm = class_name.lower().strip().replace(" ", "_")
        return FOOD_DENSITY_TABLE.get(norm, DEFAULT_DENSITY_G_PER_CM3)

    def predict_mass_g(
        self,
        image: Image.Image | np.ndarray,
        reference_bbox: Sequence[float],
        food_bbox: Sequence[float],
        food_density_g_per_cm3: float | None = None,
        mask: np.ndarray | None = None,
    ) -> Value:
        """
        Derive mass from reference object scaling and food geometry:
        1. scale_factor = real_size_cm / max(ref_w_px, ref_h_px)
        2. Compute estimated volume V in cm3
        3. mass = volume * density -> Value(mass, Source.DERIVED)
        """
        rx1, ry1, rx2, ry2 = reference_bbox
        ref_w_px = max(1.0, rx2 - rx1)
        ref_h_px = max(1.0, ry2 - ry1)
        ref_size_px = max(ref_w_px, ref_h_px)

        # Scale: centimeters per pixel
        scale_cm_per_px = self.reference_object_real_size_cm / ref_size_px

        fx1, fy1, fx2, fy2 = food_bbox
        food_w_px = max(1.0, fx2 - fx1)
        food_h_px = max(1.0, fy2 - fy1)

        food_w_cm = food_w_px * scale_cm_per_px
        food_h_cm = food_h_px * scale_cm_per_px

        density = food_density_g_per_cm3 if food_density_g_per_cm3 is not None else DEFAULT_DENSITY_G_PER_CM3

        if mask is not None:
            # Calibrated area from segmentation mask
            mask_px = float(np.sum(mask > 0))
            area_cm2 = mask_px * (scale_cm_per_px ** 2)
            # Estimate food thickness/depth as proportional to the minor axis
            depth_cm = 0.5 * min(food_w_cm, food_h_cm)
            volume_cm3 = area_cm2 * depth_cm
        else:
            # Tri-axial ellipsoid volume approximation
            depth_cm = 0.5 * min(food_w_cm, food_h_cm)
            volume_cm3 = (4.0 / 3.0) * math.pi * (food_w_cm / 2.0) * (food_h_cm / 2.0) * (depth_cm / 2.0)

        mass_g = round(max(0.5, volume_cm3 * density), 1)
        return Value(value=mass_g, source=Source.DERIVED)
