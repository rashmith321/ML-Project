"""
Supervised training pipeline for food image segmentation (U-Net).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader

from src.segmentation.dataset import FoodSegmentationDataset
from src.segmentation.models import DiceBCELoss, FoodUNet


def train_segmenter(
    manifest_path: str | Path,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    save_dir: str | Path = "models/segmentation",
    checkpoint_name: str = "best_unet.pt",
    img_size: tuple[int, int] = (256, 256),
) -> dict[str, Any]:
    """
    Train FoodUNet on paired images and masks defined in manifest_path.
    """
    mpath = Path(manifest_path).resolve()
    if not mpath.is_file():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    with open(mpath) as f:
        samples = json.load(f)

    train_samples = [s for s in samples if s.get("split") == "train"] or samples
    val_samples = [s for s in samples if s.get("split") == "val"] or samples

    train_ds = FoodSegmentationDataset(train_samples, img_size=img_size)
    val_ds = FoodSegmentationDataset(val_samples, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FoodUNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    save_path = Path(save_dir) / checkpoint_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting FoodUNet supervised segmentation training...")
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}, Device: {device}")

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= max(1, len(train_ds))

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= max(1, len(val_ds))
        print(f"  Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                save_path,
            )
            print(f"  -> Saved new best checkpoint to {save_path}")

    return {
        "status": "success",
        "best_loss": round(best_loss, 4),
        "checkpoint": str(save_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train FoodUNet segmentation model.")
    parser.add_argument("--manifest", required=True, help="Path to segmentation manifest JSON")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--out", default="models/segmentation/best_unet.pt", help="Checkpoint output path")
    args = parser.parse_args()

    train_segmenter(
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        save_dir=Path(args.out).parent,
        checkpoint_name=Path(args.out).name,
    )


if __name__ == "__main__":
    main()
