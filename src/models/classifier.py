"""
Transfer learning neural network model wrapper for food classification.
Default backbone: EfficientNet-B0 via timm.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class FoodClassificationModel(nn.Module):
    """
    Food classification neural network supporting transfer learning.
    Default backbone: EfficientNet-B0 (optimal accuracy-per-FLOP).
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Create model via timm with custom classification head
        self.net = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw unnormalized class logits."""
        return self.net(x)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized class probabilities using Softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def save_checkpoint(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> Path:
        """Save model checkpoint with configuration metadata."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = extra_meta or {}
        payload = {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "state_dict": self.state_dict(),
            "meta": meta,
        }
        torch.save(payload, out)
        return out

    @classmethod
    def load_from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "FoodClassificationModel":
        """Reconstruct model from a saved checkpoint."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        ckpt = torch.load(p, map_location=device)
        model = cls(
            backbone=ckpt.get("backbone", "efficientnet_b0"),
            num_classes=ckpt.get("num_classes", 101),
            pretrained=False,
        )
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        model.to(device)
        model.eval()
        return model
