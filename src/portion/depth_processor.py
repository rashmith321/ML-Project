"""
Depth processing module for 3D food volume calculation from RGB-D / depth maps (Nutrition5k RealSense overhead).
Computes exact 3D volume by integrating height above the supporting plate plane.
"""
from __future__ import annotations

import os
from typing import Any

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np


class DepthProcessor:
    """
    Processes depth images to compute food volume via differential height integration.
    Method: DEPTH_BASED
    """

    def __init__(
        self,
        default_focal_length_px: float = 615.0,  # Intel RealSense D435 nominal focal length
        depth_scale_mm: float = 1.0,            # 1.0 if depth map is in millimeters
    ):
        self.focal_length_px = float(default_focal_length_px)
        self.depth_scale_mm = float(depth_scale_mm)

    def estimate_volume_from_depth(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        intrinsics: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """
        Integrate 3D voxel volume under the food mask relative to the estimated table/plate plane.
        Args:
            depth_map: 2D numpy array of depth values (in mm).
            mask: 2D boolean array of the food region.
            intrinsics: optional dict with 'fx', 'fy'.
        Returns:
            volume_cm3: Estimated food volume in cubic centimeters.
            confidence: Reliability score based on valid depth pixels [0.0, 1.0].
        """
        if depth_map.ndim != 2 or mask.ndim != 2:
            raise ValueError(f"depth_map and mask must be 2D arrays, got {depth_map.shape} and {mask.shape}")

        d_map = depth_map.astype(np.float64) * self.depth_scale_mm
        bool_mask = mask.astype(bool)

        if not np.any(bool_mask):
            return 0.0, 0.0

        fx = intrinsics.get("fx", self.focal_length_px) if intrinsics else self.focal_length_px
        fy = intrinsics.get("fy", self.focal_length_px) if intrinsics else self.focal_length_px

        # Extract food depth values (excluding invalid zeros/sensor noise)
        food_depths = d_map[bool_mask]
        valid_food_depths = food_depths[food_depths > 0]

        if len(valid_food_depths) == 0:
            return 0.0, 0.0

        # Estimate supporting plate surface depth Z0 from the immediate ring surrounding the mask
        # Dilate mask slightly to capture plate background
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(bool_mask, iterations=5)
        border_mask = dilated & (~bool_mask)
        border_depths = d_map[border_mask]
        valid_border_depths = border_depths[border_depths > 0]

        if len(valid_border_depths) >= 10:
            plate_depth_z0 = float(np.median(valid_border_depths))
        else:
            # Fallback: maximum depth within the food bbox as background approximation
            plate_depth_z0 = float(np.percentile(valid_food_depths, 95))

        # Height of each food voxel above the supporting plate plane (in mm)
        # Food surface is closer to camera -> depth Z is smaller than plate Z0
        heights_mm = np.maximum(0.0, plate_depth_z0 - food_depths)

        # Differential pixel area at depth Z in mm^2: dA = (Z / fx) * (Z / fy)
        # Using food depth for projection scale
        pixel_area_mm2 = (valid_food_depths / fx) * (valid_food_depths / fy)

        # Differential volume dV = height * dA in mm^3
        valid_heights = heights_mm[food_depths > 0]
        volume_mm3 = float(np.sum(valid_heights * pixel_area_mm2))

        # Convert mm^3 to cm^3 (1 cm^3 = 1000 mm^3)
        volume_cm3 = round(volume_mm3 / 1000.0, 2)

        # Confidence: ratio of valid non-zero depth readings in food mask
        confidence = round(float(len(valid_food_depths) / len(food_depths)), 2)

        return max(0.5, volume_cm3), confidence
