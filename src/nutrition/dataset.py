"""
Dataset loaders and synthetic fixture generator for Nutrition5k multi-nutrient estimation.
Loads visual crops alongside multimodal features: class, mass, mask metrics, depth, ingredients,
and continuous ground truth targets: [calories_kcal, protein_g, carbohydrates_g, fat_g].
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms


class Nutrition5kDataset(Dataset):
    """
    Multimodal dataset for food nutrition regression.
    """

    def __init__(
        self,
        samples: Sequence[dict[str, Any]],
        ingredient_vocab: Sequence[str] | None = None,
        img_size: tuple[int, int] = (224, 224),
        is_training: bool = False,
    ):
        self.samples = list(samples)
        self.img_size = img_size
        self.is_training = is_training
        self.vocab = list(ingredient_vocab) if ingredient_vocab else []
        self.vocab_to_idx = {ingr: i for i, ingr in enumerate(self.vocab)}

        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_path = sample["image_path"]

        img = Image.open(img_path).convert("RGB")

        # Apply optional mask if path provided
        mask_path = sample.get("mask_path")
        if mask_path and Path(mask_path).is_file():
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != img.size:
                mask_img = mask_img.resize(img.size, Image.NEAREST)
            mask_arr = np.array(mask_img) > 127
            img_arr = np.array(img)
            img_arr[~mask_arr] = 0
            img = Image.fromarray(img_arr)

        visual_tensor = self.transform(img)

        class_id = torch.tensor(sample.get("class_id", 0), dtype=torch.long)
        mass = torch.tensor([float(sample.get("mass_g", 100.0))], dtype=torch.float32)

        # Seg features: [area_ratio, aspect_ratio, compactness]
        seg_feats = torch.tensor(
            sample.get("mask_features", [0.25, 1.0, 0.5]),
            dtype=torch.float32,
        )

        # Depth features: [mean_depth, height_estimate]
        depth_feats = torch.tensor(
            sample.get("depth_features", [700.0, 35.0]),
            dtype=torch.float32,
        )

        # Ingredients multi-hot vector
        num_ingrs = max(1, len(self.vocab))
        ingr_vec = torch.zeros(num_ingrs, dtype=torch.float32)
        sample_ingrs = sample.get("ingredients", [])
        for ingr in sample_ingrs:
            norm_ingr = ingr.lower().strip()
            if norm_ingr in self.vocab_to_idx:
                ingr_vec[self.vocab_to_idx[norm_ingr]] = 1.0

        # Targets: [calories_kcal, protein_g, carbohydrates_g, fat_g]
        targets = torch.tensor([
            float(sample.get("calories_kcal", 250.0)),
            float(sample.get("protein_g", 15.0)),
            float(sample.get("carbohydrates_g", 30.0)),
            float(sample.get("fat_g", 10.0)),
        ], dtype=torch.float32)

        return {
            "visual_crop": visual_tensor,
            "class_id": class_id,
            "mass": mass,
            "seg_features": seg_feats,
            "depth_features": depth_feats,
            "ingr_features": ingr_vec,
            "targets": targets,
        }


def create_nutrition_fixture(
    fixture_dir: str | Path = "data/fixture_nutrition",
    samples_count: int = 16,
    img_size: tuple[int, int] = (224, 224),
) -> tuple[Path, Path]:
    """
    Generate a format-accurate synthetic multi-nutrient dataset adhering to Nutrition5k labels
    with physically consistent macronutrient-calorie relationships:
    Calories = 4 * protein + 4 * carbs + 9 * fat
    """
    fdir = Path(fixture_dir)
    images_dir = fdir / "crops"
    masks_dir = fdir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    vocab = [
        "cheese", "tomato", "dough", "lettuce", "chicken",
        "rice", "onion", "garlic", "apple", "butter",
    ]

    base_dishes = [
        ("grilled_chicken_bowl", 0, 220.0, 35.0, 15.0, 8.0, ["chicken", "rice", "garlic"]),
        ("pasta_primavera", 1, 300.0, 12.0, 65.0, 14.0, ["dough", "tomato", "cheese"]),
        ("caesar_salad", 2, 180.0, 8.0, 10.0, 18.0, ["lettuce", "cheese", "garlic"]),
        ("apple_pastry", 3, 140.0, 4.0, 45.0, 16.0, ["apple", "dough", "butter"]),
    ]

    manifest_entries: list[dict[str, Any]] = []

    for idx in range(samples_count):
        name, cls_id, base_mass, base_prot, base_carb, base_fat, ings = base_dishes[idx % len(base_dishes)]

        # Scale portion slightly per sample
        scale = 0.8 + (idx / max(1, samples_count - 1)) * 0.5
        mass_g = round(base_mass * scale, 1)
        prot_g = round(base_prot * scale, 1)
        carb_g = round(base_carb * scale, 1)
        fat_g = round(base_fat * scale, 1)

        # Consistent calorie calculation: 4 kcal/g for protein & carb, 9 kcal/g for fat
        calories = round(4.0 * prot_g + 4.0 * carb_g + 9.0 * fat_g, 1)

        img = Image.new("RGB", img_size, color=(245, 245, 245))
        mask = Image.new("L", img_size, color=0)

        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        # Draw visual food region
        r = int(img_size[0] * (0.25 + 0.15 * scale))
        cx, cy = img_size[0] // 2, img_size[1] // 2
        bbox = [cx - r, cy - r, cx + r, cy + r]

        color = (120 + idx * 8 % 100, 70 + idx * 10 % 120, 40 + idx * 12 % 90)
        draw_img.ellipse(bbox, fill=color, outline=(20, 20, 20))
        draw_mask.ellipse(bbox, fill=255)

        img_path = images_dir / f"crop_{idx:03d}_{name}.jpg"
        mask_path = masks_dir / f"mask_{idx:03d}_{name}.png"

        img.save(img_path, quality=90)
        mask.save(mask_path)

        split = "train" if idx < int(samples_count * 0.75) else "val"
        manifest_entries.append({
            "image_path": str(img_path.resolve()).replace("\\", "/"),
            "mask_path": str(mask_path.resolve()).replace("\\", "/"),
            "food_name": name,
            "class_id": cls_id,
            "mass_g": mass_g,
            "mask_features": [round(float((2 * r) ** 2 / (img_size[0] * img_size[1])), 3), 1.0, 0.785],
            "depth_features": [round(750.0 - r * 1.5, 1), round(r * 0.8, 1)],
            "ingredients": ings,
            "calories_kcal": calories,
            "protein_g": prot_g,
            "carbohydrates_g": carb_g,
            "fat_g": fat_g,
            "split": split,
        })

    manifest_path = fdir / "nutrition_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    vocab_path = fdir / "ingredient_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    return manifest_path, vocab_path
