"""
Visualization generator for food image segmentation.
Generates 4-panel comparative figures:
[ Original Image | Ground Truth Mask | Predicted Mask | Overlay ]
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_pil(image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def create_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (60, 180, 240),
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay a semi-transparent color mask onto the RGB image.
    """
    img = image.convert("RGBA")
    w, h = img.size

    # Ensure mask matches image size
    if mask.shape != (h, w):
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0

    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
    overlay_arr[mask] = [*color, int(alpha * 255)]
    overlay_img = Image.fromarray(overlay_arr, "RGBA")

    blended = Image.alpha_composite(img, overlay_img)
    return blended.convert("RGB")


def render_4_panel_comparison(
    image: Union[str, Path, np.ndarray, Image.Image],
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray | None = None,
    panel_size: tuple[int, int] = (256, 256),
    title_height: int = 30,
) -> Image.Image:
    """
    Render a 4-panel horizontal comparison figure:
    1. Original Image
    2. Ground Truth Mask (or 'No GT' if None)
    3. Predicted Mask
    4. Overlay
    """
    orig_img = _to_pil(image).resize(panel_size, Image.BILINEAR)
    pw, ph = panel_size

    # 1. Original Image panel
    panel1 = orig_img

    # 2. Ground Truth Mask panel
    if ground_truth_mask is not None:
        gt_bool = (ground_truth_mask > 0.5) if ground_truth_mask.dtype != bool else ground_truth_mask
        gt_resized = Image.fromarray((gt_bool.astype(np.uint8) * 255)).resize(panel_size, Image.NEAREST)
        panel2 = gt_resized.convert("RGB")
    else:
        panel2 = Image.new("RGB", panel_size, color=(60, 60, 60))
        d = ImageDraw.Draw(panel2)
        d.text((pw // 4, ph // 2 - 10), "Ground Truth: N/A", fill=(200, 200, 200))

    # 3. Predicted Mask panel
    pred_bool = (predicted_mask > 0.5) if predicted_mask.dtype != bool else predicted_mask
    pred_resized = Image.fromarray((pred_bool.astype(np.uint8) * 255)).resize(panel_size, Image.NEAREST)
    panel3 = pred_resized.convert("RGB")

    # 4. Overlay panel
    panel4 = create_mask_overlay(orig_img, np.array(pred_resized.convert("L")) > 127)

    # Combine into single canvas (4 columns + title headers)
    total_w = pw * 4 + 15
    total_h = ph + title_height + 10
    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    headers = ["1. Original Image", "2. Ground Truth Mask", "3. Predicted Mask", "4. Overlay"]
    for i, (panel, header) in enumerate(zip([panel1, panel2, panel3, panel4], headers)):
        x_offset = i * (pw + 5)
        # Header text
        draw.text((x_offset + 10, 8), header, fill=(240, 240, 240))
        # Paste panel
        canvas.paste(panel, (x_offset, title_height))

    return canvas


def save_segmentation_visualization(
    image: Union[str, Path, np.ndarray, Image.Image],
    predicted_mask: np.ndarray,
    output_path: str | Path,
    ground_truth_mask: np.ndarray | None = None,
) -> Path:
    """Save 4-panel comparison visualization to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas = render_4_panel_comparison(image, predicted_mask, ground_truth_mask)
    canvas.save(out, quality=95)
    print(f"Saved 4-panel segmentation visualization to: {out}")
    return out
