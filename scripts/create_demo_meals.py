"""
Generates rich demo meal images with multiple food items, reference calibration coins,
and depth maps for the interactive Streamlit user dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_demo_meals(output_dir: str | Path = "data/demo_meals") -> list[dict]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meals = []

    # -------------------------------------------------------------------------
    # Meal 1: Chicken Bowl + Caesar Salad + Reference Coin + Depth Map
    # -------------------------------------------------------------------------
    w, h = 640, 480
    img1 = Image.new("RGB", (w, h), color=(228, 224, 216))  # Wooden table / tablecloth
    draw1 = ImageDraw.Draw(img1)

    # Plate 1 (Chicken bowl): [80, 100, 340, 400]
    draw1.ellipse([70, 90, 350, 410], fill=(245, 245, 245), outline=(190, 190, 190), width=3)
    # Chicken pieces & rice
    draw1.ellipse([90, 110, 330, 390], fill=(230, 215, 175))
    for i in range(5):
        draw1.rectangle([110 + i * 40, 160 + (i % 2) * 20, 160 + i * 40, 210 + (i % 2) * 20], fill=(160, 95, 45))
        # Grill marks
        draw1.line([(115 + i * 40, 165), (155 + i * 40, 205)], fill=(80, 40, 15), width=2)
    # Broccoli / greens
    for cx, cy in [(130, 310), (190, 330), (250, 310), (290, 270)]:
        draw1.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], fill=(45, 130, 50))

    # Plate 2 (Caesar salad): [360, 120, 580, 380]
    draw1.ellipse([350, 110, 590, 390], fill=(240, 240, 245), outline=(190, 190, 190), width=3)
    # Lettuce
    for cx, cy in [(420, 200), (510, 210), (450, 280), (520, 290), (470, 240)]:
        draw1.ellipse([cx - 45, cy - 40, cx + 45, cy + 40], fill=(65, 165, 60))
    # Croutons
    for cx, cy in [(410, 220), (490, 180), (460, 310), (540, 260)]:
        draw1.rectangle([cx - 10, cy - 10, cx + 10, cy + 10], fill=(210, 170, 100), outline=(150, 110, 50))
    # Tomato slices
    for cx, cy in [(450, 170), (530, 220), (420, 270)]:
        draw1.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(225, 40, 40))

    # Reference Coin (US Quarter / 2.5cm): [30, 30, 80, 80] (50x50 px)
    draw1.ellipse([30, 30, 80, 80], fill=(215, 180, 60), outline=(150, 120, 30), width=2)
    draw1.text((45, 48), "2.5cm", fill=(60, 40, 10))

    img1_path = out_dir / "meal_1_chicken_salad.jpg"
    img1.save(img1_path, quality=95)

    # Corresponding Depth Map (uint8 grayscale, brighter = closer to camera / taller)
    depth_arr = np.full((h, w), 80, dtype=np.uint8)  # Table surface background
    # Chicken bowl mound
    y, x = np.ogrid[:h, :w]
    bowl1_dist = np.sqrt((x - 210) ** 2 + (y - 250) ** 2)
    mask1 = bowl1_dist < 130
    depth_arr[mask1] = np.clip(80 + (130 - bowl1_dist[mask1]) * 0.9, 80, 220).astype(np.uint8)

    # Salad bowl mound
    bowl2_dist = np.sqrt((x - 470) ** 2 + (y - 250) ** 2)
    mask2 = bowl2_dist < 120
    depth_arr[mask2] = np.clip(80 + (120 - bowl2_dist[mask2]) * 0.8, 80, 210).astype(np.uint8)

    # Coin (flat surface, slight height)
    coin_dist = np.sqrt((x - 55) ** 2 + (y - 55) ** 2)
    depth_arr[coin_dist < 25] = 95

    depth1_path = out_dir / "meal_1_depth.png"
    Image.fromarray(depth_arr).save(depth1_path)

    meals.append({
        "id": "meal_1",
        "title": "Healthy Chicken Bowl with Caesar Salad & Reference Coin",
        "image_path": str(img1_path.resolve()).replace("\\", "/"),
        "depth_path": str(depth1_path.resolve()).replace("\\", "/"),
        "reference_bbox": [30, 30, 80, 80],
        "reference_size_cm": 2.5,
        "preconfigured_boxes": [
            {"food_name": "grilled_chicken_bowl", "bbox": [80, 100, 340, 400], "confidence": 0.94},
            {"food_name": "caesar_salad", "bbox": [360, 120, 580, 380], "confidence": 0.91},
        ],
    })

    # -------------------------------------------------------------------------
    # Meal 2: Pasta Primavera & Apple Pastry
    # -------------------------------------------------------------------------
    img2 = Image.new("RGB", (w, h), color=(235, 230, 220))
    draw2 = ImageDraw.Draw(img2)

    # Pasta bowl: [80, 90, 360, 410]
    draw2.ellipse([70, 80, 370, 420], fill=(250, 250, 250), outline=(200, 200, 200), width=3)
    draw2.ellipse([90, 100, 350, 400], fill=(245, 220, 160))
    # Penne shapes / spirals
    for _ in range(35):
        rx, ry = np.random.randint(110, 330), np.random.randint(120, 380)
        draw2.arc([rx - 15, ry - 8, rx + 15, ry + 8], 0, 180, fill=(215, 175, 80), width=3)
    # Bell peppers & herbs
    for _ in range(12):
        rx, ry = np.random.randint(120, 320), np.random.randint(130, 370)
        draw2.rectangle([rx, ry, rx + 12, ry + 6], fill=(230, 50, 40))
        draw2.rectangle([rx + 8, ry - 8, rx + 16, ry - 2], fill=(40, 160, 50))

    # Pastry plate: [380, 130, 580, 370]
    draw2.ellipse([370, 120, 590, 380], fill=(245, 245, 245), outline=(200, 200, 200), width=2)
    # Golden crust pastry
    draw2.ellipse([395, 145, 565, 355], fill=(210, 140, 50), outline=(140, 80, 20), width=2)
    draw2.ellipse([415, 165, 545, 335], fill=(240, 190, 90))
    draw2.text((455, 240), "Apple Pie", fill=(120, 60, 10))

    img2_path = out_dir / "meal_2_pasta_pastry.jpg"
    img2.save(img2_path, quality=95)

    meals.append({
        "id": "meal_2",
        "title": "Italian Dinner (Pasta Primavera & Apple Pastry)",
        "image_path": str(img2_path.resolve()).replace("\\", "/"),
        "depth_path": None,
        "reference_bbox": None,
        "reference_size_cm": None,
        "preconfigured_boxes": [
            {"food_name": "pasta_primavera", "bbox": [80, 90, 360, 410], "confidence": 0.92},
            {"food_name": "apple_pastry", "bbox": [380, 130, 580, 370], "confidence": 0.88},
        ],
    })

    # Save metadata index
    index_path = out_dir / "demo_meals.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(meals, f, indent=2)

    print(f"Generated {len(meals)} demo meal packages in {out_dir}")
    return meals


if __name__ == "__main__":
    create_demo_meals()
