"""
Reference object metric calibration module (ECUSTFD-style).
Derives real-world scale factor (cm/pixel) from an in-frame reference object (coin/marker)
to reconstruct physical surface area and geometric 3D volume.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


class ReferenceObjectCalibrator:
    """
    Calibrates physical real-world scale and estimates volume using an in-frame reference object.
    Method: REFERENCE_CALIBRATED
    """

    def __init__(self, default_reference_diameter_cm: float = 2.5):
        """
        default_reference_diameter_cm: Nominal real-world size in cm (e.g., 2.5 cm for 1 Yuan coin or US Quarter).
        """
        self.default_reference_diameter_cm = float(default_reference_diameter_cm)

    def compute_scale_factor(
        self,
        reference_bbox: Sequence[float],
        reference_real_size_cm: float | None = None,
    ) -> float:
        """
        Compute real-world metric scale factor in centimeters per pixel.
        Args:
            reference_bbox: [rx1, ry1, rx2, ry2] bounding box coordinates.
            reference_real_size_cm: known diameter or dimension in cm.
        Returns:
            scale_cm_per_px: cm per pixel.
        """
        real_size = reference_real_size_cm if reference_real_size_cm is not None else self.default_reference_diameter_cm
        rx1, ry1, rx2, ry2 = reference_bbox
        ref_w = max(1.0, rx2 - rx1)
        ref_h = max(1.0, ry2 - ry1)
        ref_size_px = max(ref_w, ref_h)

        scale = real_size / ref_size_px
        return float(scale)

    def estimate_volume_from_reference(
        self,
        food_bbox: Sequence[float],
        reference_bbox: Sequence[float],
        mask: np.ndarray | None = None,
        reference_real_size_cm: float | None = None,
        model_type: str = "ellipsoid",  # "ellipsoid" | "cylindrical" | "auto"
    ) -> tuple[float, float]:
        """
        Estimate 3D volume from calibrated geometric modeling.
        Returns:
            volume_cm3: Volume in cubic centimeters.
            confidence: Reliability score [0.0, 1.0].
        """
        scale = self.compute_scale_factor(reference_bbox, reference_real_size_cm)

        fx1, fy1, fx2, fy2 = food_bbox
        w_px = max(1.0, fx2 - fx1)
        h_px = max(1.0, fy2 - fy1)

        w_cm = w_px * scale
        h_cm = h_px * scale

        # Estimated depth / thickness (proportional to minor axis in monocular overhead/oblique views)
        d_cm = 0.5 * min(w_cm, h_cm)

        if mask is not None and np.any(mask):
            mask_px = float(np.sum(mask > 0))
            area_cm2 = mask_px * (scale ** 2)

            if model_type == "cylindrical" or (model_type == "auto" and (w_cm / max(0.1, h_cm)) < 1.3):
                # Cylindrical or flat dish approximation (pizza slice, pancake, steak)
                volume_cm3 = area_cm2 * d_cm
            else:
                # Elliptical disk / paraboloid approximation
                volume_cm3 = area_cm2 * d_cm * 0.6
        else:
            # Tri-axial ellipsoid approximation: V = (pi / 6) * w * h * d
            volume_cm3 = (math.pi / 6.0) * w_cm * h_cm * d_cm

        # Confidence: quality of reference bounding box circularity and food scale
        rx1, ry1, rx2, ry2 = reference_bbox
        ref_ratio = min(rx2 - rx1, ry2 - ry1) / max(1.0, max(rx2 - rx1, ry2 - ry1))
        confidence = round(float(np.clip(ref_ratio * 0.9, 0.5, 0.95)), 2)

        return max(0.5, round(float(volume_cm3), 2)), confidence
