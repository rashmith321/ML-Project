"""
Dataset loaders and fixture generator for portion and mass estimation on Nutrition5k.
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


class Nutrition5kMassDataset(Dataset):
    """
    Dataset of food image crops paired with ground-truth mass values in grams.
    """

    def __init__(
        self,
        samples: Sequence[dict[str, Any]],
        img_size: tuple[int, int] = (224, 224),
        is_training: bool = False,
    ):
        self.samples = list(samples)
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        sample = self.samples[idx]
        img_path = sample["image_path"]
        mass_g = float(sample.get("mass_g", 0.0))

        img = Image.open(img_path).convert("RGB")

        # Apply optional mask if path is present
        mask_path = sample.get("mask_path")
        if mask_path and Path(mask_path).is_file():
            mask_img = Image.open(mask_path).convert("L")
            if mask_img.size != img.size:
                mask_img = mask_img.resize(img.size, Image.NEAREST)
            mask_arr = np.array(mask_img) > 127
            img_arr = np.array(img)
            img_arr[~mask_arr] = 0
            img = Image.fromarray(img_arr)

        tensor = self.transform(img)
        return tensor, mass_g


def create_mass_fixture(
    fixture_dir: str | Path = "data/fixture_portion",
    samples_count: int = 16,
    img_size: tuple[int, int] = (224, 224),
) -> Path:
    """
    Generate a format-accurate synthetic mass regression dataset
    with correlated visual sizes and ground-truth mass values in grams.
    """
    fdir = Path(fixture_dir)
    images_dir = fdir / "images"
    masks_dir = fdir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []

    # Ground truth masses correlated with visual food size
    for idx in range(samples_count):
        # Target radius correlates with mass (50g to 400g)
        radius = 30 + int((idx / max(1, samples_count - 1)) * 60)
        mass_g = round(50.0 + (radius / 90.0) ** 2 * 350.0, 1)

        img = Image.new("RGB", img_size, color=(235, 235, 235))
        mask = Image.new("L", img_size, color=0)

        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        cx, cy = img_size[0] // 2, img_size[1] // 2
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]

        # Draw food object
        color = (180 + idx * 4 % 60, 120 + idx * 3 % 80, 50 + idx * 5 % 100)
        draw_img.ellipse(bbox, fill=color, outline=(40, 40, 40))
        draw_mask.ellipse(bbox, fill=255)

        img_path = images_dir / f"mass_sample_{idx:03d}.jpg"
        mask_path = masks_dir / f"mass_mask_{idx:03d}.png"

        img.save(img_path, quality=90)
        mask.save(mask_path)

        split = "train" if idx < int(samples_count * 0.75) else "val"
        manifest_entries.append({
            "image_path": str(img_path.resolve()).replace("\\", "/"),
            "mask_path": str(mask_path.resolve()).replace("\\", "/"),
            "mass_g": mass_g,
            "split": split,
        })

    manifest_path = fdir / "mass_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    return manifest_path
