"""
Supervised transfer learning training pipeline for food classification.
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
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classification.dataset import FoodCropDataset
from src.models.classifier import FoodClassificationModel


def train_classifier(
    manifest_path: str | Path,
    backbone: str = "efficientnet_b0",
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    device: str = "cpu",
    save_dir: str | Path = "models/classification",
    checkpoint_name: str = "best_classifier.pt",
) -> dict[str, Any]:
    """
    Train FoodClassificationModel using transfer learning.
    """
    mpath = Path(manifest_path).resolve()
    if not mpath.is_file():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    with open(mpath) as f:
        samples = json.load(f)

    # Determine number of classes
    all_class_ids = set(s["class_id"] for s in samples)
    num_classes = max(all_class_ids) + 1 if all_class_ids else 101

    train_samples = [s for s in samples if s.get("split") == "train"] or samples
    val_samples = [s for s in samples if s.get("split") == "val"] or samples

    train_ds = FoodCropDataset(train_samples, is_training=True)
    val_ds = FoodCropDataset(val_samples, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FoodClassificationModel(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    save_path = Path(save_dir) / checkpoint_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting Food Classification transfer learning...")
    print(f"  Backbone: {backbone}, Classes: {num_classes}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Epochs: {epochs}, Batch size: {batch_size}")

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum())
            total += targets.size(0)

        train_acc = correct / max(1, total)

        # Validation step
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                val_correct += int((preds == targets).sum())
                val_total += targets.size(0)

        val_acc = val_correct / max(1, val_total)
        print(f"  Epoch {epoch:02d}/{epochs:02d} - Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            model.save_checkpoint(
                save_path,
                extra_meta={"epoch": epoch, "val_acc": val_acc, "num_classes": num_classes},
            )
            print(f"  -> Saved best classifier checkpoint to {save_path}")

    return {
        "status": "success",
        "best_accuracy": round(best_acc, 4),
        "checkpoint": str(save_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train food classification model.")
    parser.add_argument("--manifest", required=True, help="Path to classification manifest JSON")
    parser.add_argument("--backbone", default="efficientnet_b0", help="Model backbone")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--out", default="models/classification/best_classifier.pt", help="Checkpoint output path")
    args = parser.parse_args()

    train_classifier(
        manifest_path=args.manifest,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        save_dir=Path(args.out).parent,
        checkpoint_name=Path(args.out).name,
    )


if __name__ == "__main__":
    main()
