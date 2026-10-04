"""
Supervised training pipeline for Multi-Nutrient Estimation on Nutrition5k.
Trains MultiNutrientModel using Target-Normalized Multi-Output Loss.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader

from src.nutrition.dataset import Nutrition5kDataset
from src.nutrition.model import MultiNutrientModel, NormalizedMultiNutrientLoss


def train_nutrient_model(
    manifest_path: str | Path,
    vocab_path: str | Path | None = None,
    backbone: str = "efficientnet_b0",
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    scale_factors: Sequence[float] = (500.0, 30.0, 50.0, 25.0),
    device: str = "cpu",
    save_dir: str | Path = "models/nutrition",
    checkpoint_name: str = "best_nutrient_model.pt",
) -> dict[str, Any]:
    """
    Train MultiNutrientModel to predict [calories, protein, carbs, fat].
    """
    mpath = Path(manifest_path).resolve()
    if not mpath.is_file():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    with open(mpath, "r", encoding="utf-8") as f:
        samples = json.load(f)

    vocab: list[str] = []
    if vocab_path and Path(vocab_path).is_file():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

    train_samples = [s for s in samples if s.get("split") == "train"] or samples
    val_samples = [s for s in samples if s.get("split") == "val"] or samples

    train_ds = Nutrition5kDataset(train_samples, ingredient_vocab=vocab, is_training=True)
    val_ds = Nutrition5kDataset(val_samples, ingredient_vocab=vocab, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Determine num_classes and num_ingredients
    all_class_ids = set(s.get("class_id", 0) for s in samples)
    num_classes = max(105, max(all_class_ids) + 1 if all_class_ids else 105)
    num_ingredients = max(1, len(vocab))

    model = MultiNutrientModel(
        backbone=backbone,
        num_classes=num_classes,
        num_ingredients=num_ingredients,
        pretrained=True,
    ).to(device)

    criterion = NormalizedMultiNutrientLoss(scale_factors=scale_factors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    save_path = Path(save_dir) / checkpoint_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting Multi-Nutrient Estimation training...")
    print(f"  Backbone: {backbone}, Classes: {num_classes}, Ingredients: {num_ingredients}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  Targets: [calories_kcal, protein_g, carbohydrates_g, fat_g]")
    print(f"  Target Scales: {list(scale_factors)}")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch in train_loader:
            crops = batch["visual_crop"].to(device)
            c_ids = batch["class_id"].to(device)
            mass = batch["mass"].to(device)
            seg = batch["seg_features"].to(device)
            depth = batch["depth_features"].to(device)
            ingr = batch["ingr_features"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            preds = model(crops, class_ids=c_ids, masses=mass, seg_features=seg, depth_features=depth, ingr_features=ingr)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * crops.size(0)
            train_count += crops.size(0)

        avg_train_loss = train_loss / max(1, train_count)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_count = 0
        total_abs_errors = torch.zeros(4, device=device)

        with torch.no_grad():
            for batch in val_loader:
                crops = batch["visual_crop"].to(device)
                c_ids = batch["class_id"].to(device)
                mass = batch["mass"].to(device)
                seg = batch["seg_features"].to(device)
                depth = batch["depth_features"].to(device)
                ingr = batch["ingr_features"].to(device)
                targets = batch["targets"].to(device)

                preds = model(crops, class_ids=c_ids, masses=mass, seg_features=seg, depth_features=depth, ingr_features=ingr)
                loss = criterion(preds, targets)
                val_loss += loss.item() * crops.size(0)
                val_count += crops.size(0)

                total_abs_errors += torch.abs(preds - targets).sum(dim=0)

        avg_val_loss = val_loss / max(1, val_count)
        mae_per_target = (total_abs_errors / max(1, val_count)).cpu().numpy()

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
            f"| MAE [Cal: {mae_per_target[0]:.1f}kcal, Prot: {mae_per_target[1]:.1f}g, "
            f"Carb: {mae_per_target[2]:.1f}g, Fat: {mae_per_target[3]:.1f}g]"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_checkpoint(save_path, extra_meta={"best_val_loss": best_val_loss, "epoch": epoch})
            print(f"  -> Saved best multi-nutrient model checkpoint to {save_path}")

    return {
        "status": "success",
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": str(save_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train multi-nutrient estimation model.")
    parser.add_argument("--manifest", required=True, help="Path to nutrition manifest JSON")
    parser.add_argument("--vocab", default=None, help="Path to ingredient vocabulary JSON")
    parser.add_argument("--backbone", default="efficientnet_b0", help="Model backbone")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--out", default="models/nutrition/best_nutrient_model.pt", help="Checkpoint output path")
    args = parser.parse_args()

    train_nutrient_model(
        manifest_path=args.manifest,
        vocab_path=args.vocab,
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
