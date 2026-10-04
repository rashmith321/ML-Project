"""
Image and Sensor Preprocessing Pipeline.

Provides robust validation, orientation correction, spatial alignment,
and tensor/array preparation for meal images and optional depth maps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image, ImageOps


@dataclass
class PreprocessedInput:
    """
    Standard container holding validated and preprocessed meal data.
    """
    image: Image.Image
    image_array: np.ndarray
    width: int
    height: int
    depth_map: np.ndarray | None
    metadata: dict[str, Any]


class ImagePreprocessor:
    """
    Validates, standardizes, and aligns RGB meal images and optional depth maps.
    """

    def __init__(
        self,
        max_dimension: int | None = 2048,
        min_dimension: int = 64,
        enforce_rgb: bool = True,
    ):
        self.max_dimension = max_dimension
        self.min_dimension = min_dimension
        self.enforce_rgb = enforce_rgb

    def preprocess(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        depth: Union[str, Path, Image.Image, np.ndarray, None] = None,
    ) -> PreprocessedInput:
        """
        Executes full preprocessing pipeline:
          1. Validates input existence and type.
          2. Applies EXIF orientation correction (e.g. from mobile devices).
          3. Converts color space to RGB.
          4. Validates minimum/maximum spatial dimensions.
          5. If depth is provided, aligns spatial resolution and validates values.
          6. Assembles metadata dictionary.
        """
        raw_format = None
        orig_size = None

        # 1. Load / standardize PIL Image
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.is_file():
                raise FileNotFoundError(f"Input meal image not found: {p}")
            pil_img = Image.open(p)
            raw_format = pil_img.format
            orig_size = pil_img.size
        elif isinstance(image, Image.Image):
            pil_img = image
            raw_format = pil_img.format
            orig_size = pil_img.size
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.ndim == 3:
                pil_img = Image.fromarray(image.astype(np.uint8))
            else:
                raise ValueError(f"Unsupported image array dimensions: {image.shape}")
            orig_size = pil_img.size
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 2. Correct EXIF orientation (handles phone camera rotation)
        try:
            transposed = ImageOps.exif_transpose(pil_img)
            if transposed is not None:
                pil_img = transposed
        except Exception:
            pass  # If EXIF reading fails, continue with original

        # 3. Enforce RGB
        if self.enforce_rgb and pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        w, h = pil_img.size

        # 4. Dimension validation & optional downscaling
        if w < self.min_dimension or h < self.min_dimension:
            raise ValueError(f"Image dimensions ({w}x{h}) are below minimum ({self.min_dimension}px).")

        scale_factor = 1.0
        if self.max_dimension and max(w, h) > self.max_dimension:
            scale_factor = self.max_dimension / float(max(w, h))
            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            w, h = pil_img.size

        image_arr = np.array(pil_img, dtype=np.uint8)

        # 5. Process and spatially align optional Depth Map
        depth_map: np.ndarray | None = None
        depth_meta: dict[str, Any] = {"available": False}

        if depth is not None:
            depth_map, depth_meta = self._process_depth(depth, target_size=(w, h))

        metadata: dict[str, Any] = {
            "original_format": raw_format,
            "original_size": orig_size,
            "processed_size": [w, h],
            "channels": 3,
            "scale_factor": scale_factor,
            "depth": depth_meta,
        }

        return PreprocessedInput(
            image=pil_img,
            image_array=image_arr,
            width=w,
            height=h,
            depth_map=depth_map,
            metadata=metadata,
        )

    def _process_depth(
        self,
        depth: Union[str, Path, Image.Image, np.ndarray],
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """
        Loads and verifies spatial alignment of depth map with RGB image.
        Uses nearest-neighbor interpolation if resizing is needed to preserve metric values.
        """
        tw, th = target_size
        raw_depth_arr: np.ndarray | None = None

        if isinstance(depth, (str, Path)):
            dp = Path(depth)
            if not dp.is_file():
                return None, {"available": False, "error": f"Depth file not found: {dp}"}
            d_img = Image.open(dp)
            raw_depth_arr = np.array(d_img)
        elif isinstance(depth, Image.Image):
            raw_depth_arr = np.array(depth)
        elif isinstance(depth, np.ndarray):
            raw_depth_arr = depth
        else:
            return None, {"available": False, "error": f"Invalid depth type: {type(depth)}"}

        if raw_depth_arr is None or raw_depth_arr.size == 0:
            return None, {"available": False, "error": "Empty depth array"}

        # Reduce 3D depth image to single channel 2D
        if raw_depth_arr.ndim == 3:
            raw_depth_arr = raw_depth_arr[:, :, 0]

        dh, dw = raw_depth_arr.shape[:2]
        resized = False

        if (dw, dh) != (tw, th):
            # Spatial resize with nearest neighbor to preserve metric depth
            depth_pil = Image.fromarray(raw_depth_arr)
            depth_pil = depth_pil.resize((tw, th), Image.NEAREST)
            raw_depth_arr = np.array(depth_pil)
            resized = True

        depth_float = raw_depth_arr.astype(np.float32)
        valid_mask = depth_float > 0
        min_depth = float(depth_float[valid_mask].min()) if np.any(valid_mask) else 0.0
        max_depth = float(depth_float.max())

        meta = {
            "available": True,
            "original_shape": [dh, dw],
            "aligned_shape": [th, tw],
            "was_resized": resized,
            "min_valid_depth": min_depth,
            "max_depth": max_depth,
        }
        return depth_float, meta


def preprocess_meal_input(
    image: Union[str, Path, Image.Image, np.ndarray],
    depth: Union[str, Path, Image.Image, np.ndarray, None] = None,
    max_dimension: int | None = 2048,
) -> PreprocessedInput:
    """Convenience function executing standard image and depth preprocessing."""
    processor = ImagePreprocessor(max_dimension=max_dimension)
    return processor.preprocess(image, depth)
