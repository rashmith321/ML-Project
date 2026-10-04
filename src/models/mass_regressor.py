"""
Neural network regression model for food mass prediction from masked image crops (+ optional depth).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import timm


class FoodMassRegressionModel(nn.Module):
    """
    Continuous mass regression model predicting food portion weight in grams.
    Operates on masked RGB crops (3 channels) or RGB-D crops (4 channels).
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.pretrained = pretrained

        try:
            self.encoder = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_channels,
            )
            feat_dim = self.encoder.num_features
        except Exception as e:
            # Fallback to non-pretrained if offline
            self.encoder = timm.create_model(
                backbone,
                pretrained=False,
                num_classes=0,
                in_chans=in_channels,
            )
            feat_dim = self.encoder.num_features

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
            nn.ReLU(),  # Enforces non-negative mass in grams
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning estimated mass in grams.
        Input shape: (B, C, H, W)
        Output shape: (B, 1)
        """
        features = self.encoder(x)
        mass = self.regressor(features)
        return mass

    def predict_mass(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward returning squeezed mass tensor (B,)."""
        return self.forward(x).squeeze(-1)

    def save_checkpoint(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "backbone": self.backbone_name,
            "in_channels": self.in_channels,
            "metadata": extra_meta or {},
        }
        torch.save(payload, out)
        return out

    @classmethod
    def load_from_checkpoint(cls, path: str | Path, device: str = "cpu") -> FoodMassRegressionModel:
        p = Path(path)
        checkpoint = torch.load(p, map_location=device)
        backbone = checkpoint.get("backbone", "efficientnet_b0")
        in_channels = checkpoint.get("in_channels", 3)
        model = cls(backbone=backbone, in_channels=in_channels, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
