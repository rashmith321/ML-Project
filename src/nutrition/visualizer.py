"""
Meal-level visualization utility for food detection, segmentation, and multi-nutrient estimation.
Renders segmentation masks, bounding boxes, food classification labels, portion estimates,
macronutrients, and meal total banners onto the input image.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, Union

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Distinct palette for multiple food instances
COLOR_PALETTE = [
    (230, 25, 75),    # Red
    (60, 180, 75),    # Green
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
]


def _to_pil_image(image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
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


def draw_meal_predictions(
    image: Union[str, Path, np.ndarray, Image.Image],
    instances: Sequence[dict[str, Any]],
    masks: Sequence[np.ndarray] | None = None,
    reference_bbox: Sequence[float] | None = None,
    reference_size_cm: float | None = None,
    meal_total: dict[str, Any] | None = None,
    alpha: float = 0.35,
    line_width: int = 3,
) -> Image.Image:
    """
    Renders rich bounding boxes, segmentation masks, portion, and nutrition badges on the meal image.
    """
    base_img = _to_pil_image(image).copy()
    w, h = base_img.size

    # Prepare RGBA canvas for mask blending
    rgba_base = base_img.convert("RGBA")
    mask_layer = np.zeros((h, w, 4), dtype=np.uint8)

    # 1. Overlay segmentation masks if provided
    if masks is not None:
        for idx, mask in enumerate(masks):
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            if mask.shape != (h, w):
                m_pil = Image.fromarray((mask > 0).astype(np.uint8)).resize((w, h), Image.NEAREST)
                m_arr = np.array(m_pil) > 0
            else:
                m_arr = mask > 0

            mask_layer[m_arr] = [*color, int(alpha * 255)]

    mask_overlay = Image.fromarray(mask_layer, mode="RGBA")
    composite = Image.alpha_composite(rgba_base, mask_overlay).convert("RGB")

    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # 2. Draw each food instance
    for idx, inst in enumerate(instances):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        bbox = inst.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Build informative labels
        name = inst.get("food_name", "Food")
        conf = inst.get("confidence", 1.0)
        portion = inst.get("portion", {})
        mass_g = portion.get("estimated_mass_g", 0.0)
        nutrients = inst.get("nutrients", {})
        def _get_num(k: str, alt_k: str = "") -> float:
            v = nutrients.get(k)
            if v is None and alt_k:
                v = nutrients.get(alt_k)
            if isinstance(v, dict):
                v = v.get("value")
            try:
                return float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                return 0.0

        cals = _get_num("calories_kcal", "calories")
        prot = _get_num("protein_g")
        carb = _get_num("carbohydrates_g", "carbs_g")
        fat = _get_num("fat_g")

        line1 = f"{name} ({conf * 100:.0f}%)"
        line2 = f"{mass_g:.0f}g | {cals:.0f} kcal"
        line3 = f"P:{prot:.1f}g C:{carb:.1f}g F:{fat:.1f}g"

        lines = [line1, line2, line3]

        # Calculate badge dimensions
        max_line_len = max(len(l) for l in lines)
        badge_w = max_line_len * 7 + 10
        badge_h = len(lines) * 14 + 6

        # Position badge neatly above or inside box
        badge_y1 = max(0, y1 - badge_h - 2) if y1 >= badge_h + 4 else y1 + 4
        badge_y2 = badge_y1 + badge_h
        badge_x2 = min(w, x1 + badge_w)

        # Badge background (solid color header)
        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=(20, 20, 20, 220), outline=color, width=1)
        draw.rectangle([x1, badge_y1, badge_x2, badge_y1 + 14], fill=color)

        # Text inside badge
        draw.text((x1 + 4, badge_y1 + 1), line1, fill=(255, 255, 255), font=font)
        draw.text((x1 + 4, badge_y1 + 16), line2, fill=(255, 220, 100), font=font)
        draw.text((x1 + 4, badge_y1 + 30), line3, fill=(200, 230, 255), font=font)

    # 3. Draw Reference Object if provided
    if reference_bbox:
        rx1, ry1, rx2, ry2 = [int(round(v)) for v in reference_bbox]
        ref_color = (255, 215, 0)  # Gold
        draw.rectangle([rx1, ry1, rx2, ry2], outline=ref_color, width=line_width)
        ref_label = f"Ref Coin ({reference_size_cm:.1f}cm)" if reference_size_cm else "Reference Object"
        draw.rectangle([rx1, max(0, ry1 - 16), rx1 + len(ref_label) * 7 + 8, max(16, ry1)], fill=ref_color)
        draw.text((rx1 + 4, max(0, ry1 - 14)), ref_label, fill=(0, 0, 0), font=font)

    # 4. Optional Top Meal Banner
    if meal_total:
        cals_val = meal_total.get("calories", {}).get("value", 0.0)
        prot_val = meal_total.get("protein_g", {}).get("value", 0.0)
        carb_val = meal_total.get("carbs_g", {}).get("value", 0.0)
        fat_val = meal_total.get("fat_g", {}).get("value", 0.0)

        banner_text = (
            f"MEAL TOTAL:  {cals_val:.0f} kcal  |  "
            f"Protein: {prot_val:.1f}g  |  Carbs: {carb_val:.1f}g  |  Fat: {fat_val:.1f}g"
        )
        banner_h = 24
        # Semi-transparent dark banner bar across the bottom
        banner_layer = composite.copy()
        b_draw = ImageDraw.Draw(banner_layer)
        b_draw.rectangle([0, h - banner_h, w, h], fill=(15, 23, 42))
        b_draw.text((10, h - banner_h + 5), banner_text, fill=(255, 255, 255), font=font)
        composite = banner_layer

    return composite


def save_meal_visualization(
    image: Union[str, Path, np.ndarray, Image.Image],
    instances: Sequence[dict[str, Any]],
    output_path: str | Path,
    masks: Sequence[np.ndarray] | None = None,
    reference_bbox: Sequence[float] | None = None,
    reference_size_cm: float | None = None,
    meal_total: dict[str, Any] | None = None,
) -> Path:
    """Renders annotated meal image and saves to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_meal_predictions(
        image=image,
        instances=instances,
        masks=masks,
        reference_bbox=reference_bbox,
        reference_size_cm=reference_size_cm,
        meal_total=meal_total,
    )
    annotated.save(out, quality=95)
    print(f"Saved annotated meal visualization to: {out}")
    return out


def draw_detected_food_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    instances: Sequence[dict[str, Any]],
    line_width: int = 3,
) -> Image.Image:
    """
    Renders pure object detection view: bounding boxes, food names, and detection confidence.
    """
    base_img = _to_pil_image(image).copy()
    draw = ImageDraw.Draw(base_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, inst in enumerate(instances):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        bbox = inst.get("bounding_box") or inst.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        name = inst.get("food_name", "Food").replace("_", " ").title()
        conf = float(inst.get("confidence", 1.0)) * 100
        label_text = f"{name} ({conf:.1f}%)"
        tw = int(draw.textlength(label_text, font=font)) if font and hasattr(draw, "textlength") else len(label_text) * 7
        th = 14
        by1 = max(0, y1 - th - 4)
        draw.rectangle([x1, by1, x1 + tw + 8, by1 + th + 4], fill=color)
        draw.text((x1 + 4, by1 + 2), label_text, fill=(255, 255, 255), font=font)

    return base_img


def draw_segmented_food_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    masks: Sequence[np.ndarray] | None = None,
    instances: Sequence[dict[str, Any]] | None = None,
    alpha: float = 0.50,
) -> Image.Image:
    """
    Renders pure instance segmentation view: colored translucent masks overlaying food regions.
    """
    base_img = _to_pil_image(image).copy()
    w, h = base_img.size
    rgba_base = base_img.convert("RGBA")
    mask_layer = np.zeros((h, w, 4), dtype=np.uint8)

    if masks is not None:
        for idx, mask in enumerate(masks):
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            if mask.shape != (h, w):
                m_pil = Image.fromarray((mask > 0).astype(np.uint8)).resize((w, h), Image.NEAREST)
                m_arr = np.array(m_pil) > 0
            else:
                m_arr = mask > 0

            mask_layer[m_arr] = [*color, int(alpha * 255)]

    mask_overlay = Image.fromarray(mask_layer, mode="RGBA")
    blended = Image.alpha_composite(rgba_base, mask_overlay).convert("RGB")

    if instances and masks:
        draw = ImageDraw.Draw(blended)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for idx, inst in enumerate(instances):
            if idx >= len(masks):
                break
            mask = masks[idx]
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                name = inst.get("food_name", f"Food {idx+1}").replace("_", " ").title()
                badge_text = f"Mask #{idx+1}: {name}"
                color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
                tw = int(draw.textlength(badge_text, font=font)) if font and hasattr(draw, "textlength") else len(badge_text) * 7
                th = 14
                draw.rectangle([cx - tw//2 - 4, cy - th//2 - 2, cx + tw//2 + 4, cy + th//2 + 2], fill=color)
                draw.text((cx - tw//2, cy - th//2), badge_text, fill=(255, 255, 255), font=font)

    return blended


def save_detected_food_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    instances: Sequence[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Saves detected food image with bounding boxes to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = draw_detected_food_image(image, instances)
    img.save(out, quality=95)
    return out


def save_segmented_food_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    masks: Sequence[np.ndarray],
    output_path: str | Path,
    instances: Sequence[dict[str, Any]] | None = None,
) -> Path:
    """Saves segmented food image with translucent masks to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = draw_segmented_food_image(image, masks=masks, instances=instances)
    img.save(out, quality=95)
    return out

