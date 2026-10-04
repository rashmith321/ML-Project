"""
Dataset loaders and fixture generation for food classification crops.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class FoodCropDataset(Dataset):
    """
    PyTorch Dataset for cropped food image classification.
    Supports optional foreground mask application and data augmentations.
    """

    def __init__(
        self,
        samples: Sequence[dict[str, Any]],
        img_size: tuple[int, int] = (224, 224),
        is_training: bool = False,
    ):
        """
        samples: list of dicts with {"image_path": str, "class_id": int, (optional) "mask_path": str}
        """
        self.samples = samples
        self.img_size = img_size
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img_p = Path(sample["image_path"])
        class_id = int(sample["class_id"])

        pil_img = Image.open(img_p).convert("RGB")

        # Apply mask if present
        mask_path = sample.get("mask_path")
        if mask_path and Path(mask_path).is_file():
            pil_mask = Image.open(mask_path).convert("L")
            w, h = pil_img.size
            mask_arr = np.array(pil_mask.resize((w, h), Image.NEAREST)) > 127
            img_arr = np.array(pil_img)
            img_arr[~mask_arr] = [235, 235, 235]
            pil_img = Image.fromarray(img_arr)

        pil_img = pil_img.resize(self.img_size, Image.BILINEAR)

        # Simple data augmentations during training
        if self.is_training:
            if np.random.rand() > 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        # HWC -> CHW
        tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, class_id


def create_classification_fixture(
    fixture_dir: str | Path,
    class_names: Sequence[str] | None = None,
    samples_per_class: int = 4,
    img_size: tuple[int, int] = (224, 224),
) -> Path:
    """
    Generate a format-accurate mini synthetic classification dataset
    with distinct visual categories for test runs.
    """
    fdir = Path(fixture_dir)
    classes = list(class_names) if class_names else ["apple_pie", "caesar_salad", "pizza", "sushi"]
    colors = [
        (220, 50, 50),    # red (apple_pie)
        (60, 180, 75),    # green (salad)
        (255, 160, 50),   # orange (pizza)
        (70, 70, 70),     # dark (sushi)
    ]

    manifest_entries: list[dict[str, Any]] = []

    for cls_id, cls_name in enumerate(classes):
        cls_dir = fdir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        color = colors[cls_id % len(colors)]

        for s_idx in range(samples_per_class):
            img = Image.new("RGB", img_size, color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Draw distinctive patterns per class
            cx, cy = img_size[0] // 2, img_size[1] // 2
            r = int(img_size[0] * 0.35)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0))

            split = "train" if s_idx < int(samples_per_class * 0.75) else "val"
            img_path = cls_dir / f"crop_{s_idx:03d}.jpg"
            img.save(img_path, quality=90)

            manifest_entries.append({
                "image_path": str(img_path.resolve()).replace("\\", "/"),
                "class_id": cls_id,
                "food_class": cls_name,
                "split": split,
            })

    manifest_path = fdir / "classification_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)

    return manifest_path
