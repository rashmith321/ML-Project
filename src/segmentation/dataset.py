"""
Dataset loaders and utilities for food image segmentation.
Supports loading ground truth masks (UNIMIB2016) and pseudo-masks with provenance tracking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence, Union

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from src.segmentation.sam_utils import MaskType


class FoodSegmentationDataset(Dataset):
    """
    PyTorch Dataset for supervised food image segmentation.
    Explicitly tracks mask_type (GROUND_TRUTH vs PSEUDO).
    """

    def __init__(
        self,
        samples: Sequence[dict[str, Any]],
        img_size: tuple[int, int] = (256, 256),
        transform: Callable | None = None,
        default_mask_type: MaskType = MaskType.GROUND_TRUTH,
    ):
        """
        samples: list of dicts with {"image_path": str, "mask_path": str, (optional) "mask_type": str}
        """
        self.samples = samples
        self.img_size = img_size
        self.transform = transform
        self.default_mask_type = default_mask_type

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sample = self.samples[idx]
        img_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])
        mask_type = sample.get("mask_type", self.default_mask_type.value)

        # Load RGB image
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = pil_img.resize(self.img_size, Image.BILINEAR)
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        # HWC -> CHW
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        # Normalize with standard ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Load binary mask
        pil_mask = Image.open(mask_path).convert("L")
        pil_mask = pil_mask.resize(self.img_size, Image.NEAREST)
        mask_arr = (np.array(pil_mask, dtype=np.float32) > 127).astype(np.float32)
        # 1HW
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        meta = {
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "mask_type": mask_type,
            "orig_size": Image.open(img_path).size,
        }

        return img_tensor, mask_tensor, meta


def create_segmentation_fixture(
    fixture_dir: str | Path,
    num_train: int = 4,
    num_val: int = 2,
    num_test: int = 2,
    img_size: tuple[int, int] = (256, 256),
) -> Path:
    """
    Generate a format-accurate synthetic segmentation fixture with paired
    images and ground-truth binary masks for automated testing.
    """
    fdir = Path(fixture_dir)
    counts = {"train": num_train, "val": num_val, "test": num_test}
    colors = [(220, 50, 50), (60, 180, 75), (255, 160, 50)]

    manifest_entries: list[dict[str, Any]] = []

    for split, count in counts.items():
        img_dir = fdir / "images" / split
        mask_dir = fdir / "masks" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            img = Image.new("RGB", img_size, color=(240, 240, 240))
            mask = Image.new("L", img_size, color=0)
            draw_img = ImageDraw.Draw(img)
            draw_mask = ImageDraw.Draw(mask)

            # Draw a simulated food item (circle/ellipse)
            cx, cy = img_size[0] // 2, img_size[1] // 2
            rx = int(img_size[0] * 0.3)
            ry = int(img_size[1] * 0.25)
            box = [cx - rx, cy - ry, cx + rx, cy + ry]

            draw_img.ellipse(box, fill=colors[idx % len(colors)], outline=(0, 0, 0))
            draw_mask.ellipse(box, fill=255)

            stem = f"sample_{idx:03d}"
            img_path = img_dir / f"{stem}.jpg"
            mask_path = mask_dir / f"{stem}.png"

            img.save(img_path, quality=90)
            mask.save(mask_path)

            manifest_entries.append({
                "image_path": str(img_path.resolve()).replace("\\", "/"),
                "mask_path": str(mask_path.resolve()).replace("\\", "/"),
                "split": split,
                "mask_type": MaskType.GROUND_TRUTH.value,
            })

    manifest_path = fdir / "segmentation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)

    return manifest_path
