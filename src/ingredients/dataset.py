"""
Dataset loaders and synthetic fixture generator for multi-label ingredient understanding.
Supports Recipe1M+, Vireo Food-172, and Nutrition5k ingredient annotations.
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


class MultiLabelIngredientDataset(Dataset):
    """
    Dataset of food image crops paired with multi-hot binary ingredient target vectors.
    """

    def __init__(
        self,
        samples: Sequence[dict[str, Any]],
        ingredient_vocab: Sequence[str],
        img_size: tuple[int, int] = (224, 224),
        is_training: bool = False,
    ):
        self.samples = list(samples)
        self.vocab = list(ingredient_vocab)
        self.vocab_to_idx = {ingr: idx for idx, ingr in enumerate(self.vocab)}
        self.img_size = img_size
        self.is_training = is_training

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = sample["image_path"]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)

        # Multi-hot vector for ingredients
        target = torch.zeros(len(self.vocab), dtype=torch.float32)
        sample_ingredients = sample.get("ingredients", [])
        for ingr in sample_ingredients:
            norm_ingr = ingr.lower().strip()
            if norm_ingr in self.vocab_to_idx:
                target[self.vocab_to_idx[norm_ingr]] = 1.0

        return tensor, target


def create_ingredient_fixture(
    fixture_dir: str | Path = "data/fixture_ingredients",
    samples_count: int = 12,
    img_size: tuple[int, int] = (224, 224),
) -> tuple[Path, Path]:
    """
    Generate a format-accurate synthetic multi-label ingredient dataset
    for testing without requiring external dataset downloads.
    Returns:
        (manifest_path, vocab_path)
    """
    fdir = Path(fixture_dir)
    images_dir = fdir / "crops"
    images_dir.mkdir(parents=True, exist_ok=True)

    vocab = [
        "cheese", "tomato", "dough", "lettuce", "chicken",
        "rice", "onion", "garlic", "apple", "butter",
    ]

    sample_dishes = [
        ("pizza", ["cheese", "tomato", "dough"]),
        ("caesar_salad", ["lettuce", "cheese", "garlic"]),
        ("chicken_curry", ["chicken", "rice", "onion", "garlic"]),
        ("apple_pie", ["apple", "dough", "butter"]),
    ]

    manifest_entries: list[dict[str, Any]] = []

    for s_idx in range(samples_count):
        dish_name, ings = sample_dishes[s_idx % len(sample_dishes)]

        img = Image.new("RGB", img_size, color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Draw representative geometric patterns
        color = (50 + s_idx * 20 % 180, 80 + s_idx * 30 % 150, 100 + s_idx * 15 % 140)
        draw.rectangle([30, 30, 194, 194], fill=color, outline=(0, 0, 0))

        img_path = images_dir / f"crop_{s_idx:03d}_{dish_name}.jpg"
        img.save(img_path, quality=90)

        split = "train" if s_idx < int(samples_count * 0.75) else "val"
        manifest_entries.append({
            "image_path": str(img_path.resolve()).replace("\\", "/"),
            "food_name": dish_name,
            "ingredients": ings,
            "split": split,
        })

    manifest_path = fdir / "ingredient_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    vocab_path = fdir / "ingredient_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    return manifest_path, vocab_path
