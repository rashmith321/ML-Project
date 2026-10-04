"""
Mass Estimation Module.
Derives portion mass from volume and density, or predicts via neural regression.
Attributing provenance: 'prediction' vs 'derived'.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.mass_regressor import FoodMassRegressionModel
from src.portion.volume_estimator import EstimationMethod, VolumeEstimator
from src.utils.schema import FoodInstance, Source, Value

# Verified food density values in g/cm^3
FOOD_DENSITY_TABLE: dict[str, float] = {
    "apple": 0.85,
    "apple_pie": 0.80,
    "banana": 0.95,
    "beef_carpaccio": 1.05,
    "bread": 0.35,
    "burger": 0.85,
    "caesar_salad": 0.38,
    "cake": 0.55,
    "carrot": 0.64,
    "cheese": 1.10,
    "chicken": 1.05,
    "chicken_curry": 1.02,
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
DEFAULT_DENSITY = 0.85


class MassEstimator:
    """
    Estimates mass in grams from physical volume or neural regression.
    """

    def __init__(
        self,
        weights_path: str | Path | None = "models/portion/best_mass_regressor.pt",
        volume_estimator: VolumeEstimator | None = None,
        device: str = "cpu",
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = device
        self.volume_estimator = volume_estimator or VolumeEstimator()
        self.model: FoodMassRegressionModel | None = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.weights_path and self.weights_path.is_file():
            self.load_model(self.weights_path)

    def load_model(self, checkpoint_path: str | Path) -> None:
        """Load trained mass regression model."""
        self.model = FoodMassRegressionModel.load_from_checkpoint(checkpoint_path, device=self.device)
        print(f"[MassEstimator] Loaded checkpoint from {checkpoint_path}")

    @staticmethod
    def get_density(food_class: str | None) -> float:
        """Retrieve food density in g/cm3."""
        if not food_class:
            return DEFAULT_DENSITY
        norm = food_class.lower().strip().replace(" ", "_").replace("-", "_")
        return FOOD_DENSITY_TABLE.get(norm, DEFAULT_DENSITY)

    def estimate_portion(
        self,
        crop: Image.Image | np.ndarray,
        food_bbox: Sequence[float],
        food_class: str | None = None,
        mask: np.ndarray | None = None,
        image_shape: tuple[int, int] | None = None,
        depth_map: np.ndarray | None = None,
        reference_bbox: Sequence[float] | None = None,
        reference_real_size_cm: float | None = None,
    ) -> dict[str, Any]:
        """
        Estimate quantity: mass, volume, confidence, and methodology.
        """
        vol, vol_conf, method, disclaimer = self.volume_estimator.estimate_volume(
            food_bbox=food_bbox,
            mask=mask,
            image_shape=image_shape,
            depth_map=depth_map,
            reference_bbox=reference_bbox,
            reference_real_size_cm=reference_real_size_cm,
        )

        density = self.get_density(food_class)

        # If Depth or Reference calibration was available, derive mass from physics
        if method in (EstimationMethod.DEPTH_BASED, EstimationMethod.REFERENCE_CALIBRATED):
            mass_g = round(vol * density, 1)
            confidence = vol_conf
            mass_source = "derived"
        elif self.model is not None:
            # Monocular RGB: Use learned neural regression
            if isinstance(crop, np.ndarray):
                pil_crop = Image.fromarray(crop).convert("RGB")
            else:
                pil_crop = crop.convert("RGB")

            tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_mass = self.model(tensor).item()

            # Plausibility guard: the mass model is trained on limited fixture data and
            # can return nonsensical values for real images (e.g. 2.7 g or 50000 g).
            # If the prediction is outside the physiologically plausible range for a
            # single food item [10 g, 2000 g], fall back to the physics-based estimate.
            if 10.0 <= float(pred_mass) <= 2000.0:
                mass_g = round(float(pred_mass), 1)
                confidence = 0.65
                mass_source = "prediction"
            else:
                print(
                    f"[MassEstimator] WARNING: Neural prediction {pred_mass:.1f} g is outside "
                    f"plausible range [10, 2000] g — falling back to vol×density estimate."
                )
                mass_g = round(vol * density, 1)
                confidence = vol_conf * 0.8  # slightly lower confidence than depth-based
                mass_source = "derived"

        else:
            # Fallback visual proxy
            mass_g = round(vol * density, 1)
            confidence = vol_conf
            mass_source = "derived"

        return {
            "food_name": food_class or "unknown",
            "estimated_mass_g": mass_g,
            "estimated_volume": vol,
            "confidence": confidence,
            "method_used": method.value,
            "mass_source": mass_source,
            "density_g_per_cm3": density,
            "disclaimer": disclaimer,
        }
