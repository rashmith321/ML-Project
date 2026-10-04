"""
Volume Estimation Module.
Adheres strictly to the scientific hierarchy:
  1. Depth-based -> when depth map is available (Nutrition5k RealSense)
  2. Reference object -> when calibration object is present (ECUSTFD)
  3. RGB-based learned estimate -> monocular visual volume proxy when neither is available

Monocular RGB photographs cannot provide exact physical scale without depth or reference objects;
this limitation is explicitly recorded in every output.
"""
from __future__ import annotations

from enum import Enum
import math
from typing import Any, Sequence

import numpy as np

from src.portion.calibration import ReferenceObjectCalibrator
from src.portion.depth_processor import DepthProcessor


class EstimationMethod(str, Enum):
    DEPTH_BASED = "depth_based"
    REFERENCE_CALIBRATED = "reference_calibrated"
    RGB_LEARNED = "rgb_learned"


class VolumeEstimator:
    """
    Unified volume estimator supporting Depth, Reference Object, and RGB-Learned fallback.
    """

    def __init__(
        self,
        depth_processor: DepthProcessor | None = None,
        calibrator: ReferenceObjectCalibrator | None = None,
    ):
        self.depth_processor = depth_processor or DepthProcessor()
        self.calibrator = calibrator or ReferenceObjectCalibrator()

    def estimate_volume(
        self,
        food_bbox: Sequence[float],
        mask: np.ndarray | None = None,
        image_shape: tuple[int, int] | None = None,
        depth_map: np.ndarray | None = None,
        reference_bbox: Sequence[float] | None = None,
        reference_real_size_cm: float | None = None,
    ) -> tuple[float, float, EstimationMethod, str]:
        """
        Estimate food volume adhering to the strict scientific priority hierarchy.
        Returns:
            volume_cm3: Estimated volume in cubic centimeters.
            confidence: Reliability score [0.0, 1.0].
            method: EstimationMethod enum member.
            disclaimer: Scientific methodology notes.
        """
        fx1, fy1, fx2, fy2 = food_bbox
        w_px = max(1.0, fx2 - fx1)
        h_px = max(1.0, fy2 - fy1)

        # -------------------------------------------------------------
        # Tier 1: 3D Depth Map Available
        # -------------------------------------------------------------
        if depth_map is not None and mask is not None and np.any(mask):
            try:
                vol, conf = self.depth_processor.estimate_volume_from_depth(depth_map, mask)
                if vol > 0.0:
                    disclaimer = "Volume integrated from 3D RGB-D depth map relative to plate plane."
                    return vol, conf, EstimationMethod.DEPTH_BASED, disclaimer
            except Exception as e:
                # If depth parsing fails, proceed to next tier
                pass

        # -------------------------------------------------------------
        # Tier 2: Reference Calibration Object Available
        # -------------------------------------------------------------
        if reference_bbox is not None and len(reference_bbox) == 4:
            try:
                vol, conf = self.calibrator.estimate_volume_from_reference(
                    food_bbox=food_bbox,
                    reference_bbox=reference_bbox,
                    mask=mask,
                    reference_real_size_cm=reference_real_size_cm,
                )
                if vol > 0.0:
                    disclaimer = "Volume reconstructed via metric reference object calibration."
                    return vol, conf, EstimationMethod.REFERENCE_CALIBRATED, disclaimer
            except Exception:
                pass

        # -------------------------------------------------------------
        # Tier 3: RGB-Based Monocular Learned Fallback
        # -------------------------------------------------------------
        # Guard: if the bounding box covers an unreasonably large fraction of the
        # image, the detection is almost certainly a background/container artefact.
        # We cap the effective pixel dimensions rather than skipping entirely, and
        # emit a WARNING so the issue is visible in logs.
        if image_shape is not None:
            img_h, img_w = image_shape[0], image_shape[1]
            img_area = max(1.0, float(img_w * img_h))
            bbox_area = max(0.0, w_px * h_px)
            bbox_fraction = bbox_area / img_area

            if bbox_fraction > 0.40:
                print(
                    f"[VolumeEstimator] WARNING: bbox covers {bbox_fraction * 100:.1f}% of "
                    f"image area — likely a background/container detection. "
                    f"Capping effective dimensions to 40% of image to prevent mass explosion."
                )
                # Cap effective width/height to what a single food item could realistically occupy
                max_side_px = math.sqrt(0.40 * img_area)
                w_px = min(w_px, max_side_px)
                h_px = min(h_px, max_side_px)

        # In the absence of physical scale reference, estimate apparent volume proxy
        # based on standard dinner plate assumed scale (~25 cm diameter meal frame)
        assumed_frame_width_cm = 25.0
        frame_width_px = float(image_shape[1]) if image_shape else 640.0
        empirical_scale = assumed_frame_width_cm / max(100.0, frame_width_px)

        w_cm = w_px * empirical_scale
        h_cm = h_px * empirical_scale
        d_cm = 0.45 * min(w_cm, h_cm)

        if mask is not None and np.any(mask):
            area_px = float(np.sum(mask > 0))
            area_cm2 = area_px * (empirical_scale ** 2)
            vol = round(area_cm2 * d_cm * 0.7, 2)
        else:
            vol = round((math.pi / 6.0) * w_cm * h_cm * d_cm, 2)

        conf = 0.50  # Lower confidence due to uncalibrated monocular projection
        disclaimer = (
            "Monocular RGB photograph: volume is an empirical visual proxy without direct "
            "3D physical ground truth."
        )
        return max(1.0, vol), conf, EstimationMethod.RGB_LEARNED, disclaimer
