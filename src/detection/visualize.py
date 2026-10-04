"""
Visualization utilities for food object detection.
Renders high-contrast bounding boxes, class names, and confidence scores onto images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from src.detection.detector import Detection

# Distinct high-visibility palette for food detection bounding boxes
DEFAULT_PALETTE = [
    (230, 25, 75),    # Red
    (60, 180, 75),    # Green
    (255, 225, 25),   # Yellow
    (0, 130, 200),    # Blue
    (245, 130, 48),   # Orange
    (145, 30, 180),   # Purple
    (70, 240, 240),   # Cyan
    (240, 50, 230),   # Magenta
    (210, 245, 60),   # Lime
    (250, 190, 212),  # Pink
    (0, 128, 128),    # Teal
    (220, 190, 255),  # Lavender
    (170, 110, 40),   # Brown
    (255, 250, 200),  # Beige
    (128, 0, 0),      # Maroon
]


def _to_pil_image(image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
    """Ensure image is an RGB PIL Image."""
    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.shape[2] == 4:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(image)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def draw_detections(
    image: Union[str, Path, np.ndarray, Image.Image],
    detections: Sequence[Detection],
    line_width: int = 3,
) -> Image.Image:
    """
    Draw bounding boxes, class labels, and confidence percentages on the image.
    Returns the annotated PIL Image.
    """
    img = _to_pil_image(image).copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        color = DEFAULT_PALETTE[det.class_id % len(DEFAULT_PALETTE)]
        x1, y1, x2, y2 = det.bbox

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Label text
        label = f"{det.class_name} {det.confidence * 100:.1f}%"

        # Text bounding box for badge background
        if font and hasattr(draw, "textbbox"):
            tb = draw.textbbox((x1, max(0.0, y1 - 15.0)), label, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
        else:
            text_w = len(label) * 6
            text_h = 12

        badge_y1 = max(0.0, y1 - text_h - 4)
        badge_y2 = badge_y1 + text_h + 4
        badge_x2 = x1 + text_w + 6

        # Draw background badge and text
        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=color)
        draw.text((x1 + 3, badge_y1 + 2), label, fill=(255, 255, 255), font=font)

    return img


def save_annotated_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    detections: Sequence[Detection],
    output_path: str | Path,
) -> Path:
    """Draw detections and save the annotated image to output_path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(image, detections)
    annotated.save(out, quality=95)
    return out
