"""
Multi-label ingredient classification model architecture.
Predicts multiple simultaneous ingredient candidates and confidence scores from food crops.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import timm


class MultiLabelIngredientModel(nn.Module):
    """
    Multi-label convolutional/transformer classifier predicting ingredient probabilities.
    Uses independent Sigmoid activations per ingredient candidate.
    """

    def __init__(
        self,
        num_ingredients: int = 50,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_ingredients = num_ingredients
        self.backbone_name = backbone
        self.pretrained = pretrained

        try:
            self.encoder = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
            )
            feat_dim = self.encoder.num_features
        except Exception:
            # Fallback to uninitialized backbone if offline / network error
            self.encoder = timm.create_model(
                backbone,
                pretrained=False,
                num_classes=0,
            )
            feat_dim = self.encoder.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_ingredients),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw unnormalized logits for BCEWithLogitsLoss.
        Input: (B, 3, H, W)
        Output: (B, num_ingredients)
        """
        features = self.encoder(x)
        logits = self.head(features)
        return logits

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return Sigmoid-normalized probabilities in [0.0, 1.0] for each ingredient.
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict_ingredients(
        self,
        x: torch.Tensor,
        ingredient_vocab: Sequence[str],
        threshold: float = 0.3,
    ) -> list[list[tuple[str, float]]]:
        """
        Predict list of (ingredient_name, confidence) for each sample in batch.
        Filters candidates below threshold.
        """
        probs = self.predict_probs(x).cpu().detach().numpy()
        batch_results: list[list[tuple[str, float]]] = []

        for sample_probs in probs:
            candidates: list[tuple[str, float]] = []
            for idx, prob in enumerate(sample_probs):
                if prob >= threshold and idx < len(ingredient_vocab):
                    candidates.append((ingredient_vocab[idx], float(round(prob, 4))))
            # Sort by descending confidence
            candidates.sort(key=lambda item: item[1], reverse=True)
            batch_results.append(candidates)

        return batch_results

    def save_checkpoint(
        self,
        path: str | Path,
        ingredient_vocab: Sequence[str] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Save model checkpoint with ingredient vocabulary."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "num_ingredients": self.num_ingredients,
            "backbone": self.backbone_name,
            "ingredient_vocab": list(ingredient_vocab) if ingredient_vocab else [],
            "metadata": extra_meta or {},
        }
        torch.save(payload, out)
        return out

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> tuple[MultiLabelIngredientModel, list[str]]:
        """Load trained multi-label ingredient model and vocabulary."""
        p = Path(path)
        checkpoint = torch.load(p, map_location=device)
        num_ingredients = checkpoint.get("num_ingredients", 50)
        backbone = checkpoint.get("backbone", "efficientnet_b0")
        vocab = checkpoint.get("ingredient_vocab", [])

        model = cls(num_ingredients=num_ingredients, backbone=backbone, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model, vocab
