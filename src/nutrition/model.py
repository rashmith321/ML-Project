"""
Multimodal Feature Fusion Neural Network for Multi-Nutrient Estimation:
Simultaneously estimates Calories (kcal), Protein (g), Carbohydrates (g), and Fat (g).
Uses Target-Normalized Multi-Output Loss to balance gradient dynamics across targets.
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


class NormalizedMultiNutrientLoss(nn.Module):
    """
    Target-Normalized Multi-Output Regression Loss.
    Scales targets so calorie magnitude (~100-1500) does not dominate
    protein/carb/fat learning (~5-60g).
    """

    def __init__(
        self,
        scale_factors: Sequence[float] = (500.0, 30.0, 50.0, 25.0),
        weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        beta: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scale_factors, dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.criterion = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes weighted sum of normalized Smooth L1 errors across all 4 targets.
        preds, targets: (B, 4)
        """
        scales = self.scales.to(preds.device)
        weights = self.weights.to(preds.device)

        norm_preds = preds / scales
        norm_targets = targets / scales

        per_target_loss = self.criterion(norm_preds, norm_targets)  # (B, 4)
        weighted_loss = per_target_loss * weights
        return weighted_loss.mean()


class MultiNutrientModel(nn.Module):
    """
    Multimodal fusion architecture for multi-nutrient estimation.
    Fuses:
      - Visual features (EfficientNet-B0)
      - Food classification embedding
      - Segmentation mask geometry
      - Estimated mass (log-scaled)
      - Depth features
      - Ingredient bag-of-words
    Outputs:
      [calories_kcal, protein_g, carbohydrates_g, fat_g]
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 105,
        num_ingredients: int = 50,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.num_ingredients = num_ingredients

        # 1. Visual Backbone Encoder
        try:
            self.visual_encoder = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
            )
            visual_dim = self.visual_encoder.num_features
        except Exception:
            self.visual_encoder = timm.create_model(
                backbone,
                pretrained=False,
                num_classes=0,
            )
            visual_dim = self.visual_encoder.num_features

        # 2. Metadata Feature Encoders
        self.class_embedding = nn.Embedding(num_classes, 32)
        self.mass_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
        )
        self.seg_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
        )
        self.depth_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
        )
        self.ingr_encoder = nn.Sequential(
            nn.Linear(num_ingredients, 32),
            nn.ReLU(inplace=True),
        )

        total_fused_dim = visual_dim + 32 + 16 + 16 + 16 + 32

        # 3. Multimodal Fusion Trunk
        self.fusion_trunk = nn.Sequential(
            nn.Linear(total_fused_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # 4. Multi-Output Regression Heads (enforcing non-negative outputs via ReLU)
        self.calories_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.ReLU(),
        )
        self.protein_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.ReLU(),
        )
        self.carbs_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.ReLU(),
        )
        self.fat_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        # Initialize biases to positive base values to avoid dead ReLU units during early training
        nn.init.constant_(self.calories_head[2].bias, 150.0)
        nn.init.constant_(self.protein_head[2].bias, 15.0)
        nn.init.constant_(self.carbs_head[2].bias, 20.0)
        nn.init.constant_(self.fat_head[2].bias, 8.0)

    def forward(
        self,
        visual_crops: torch.Tensor,
        class_ids: torch.Tensor | None = None,
        masses: torch.Tensor | None = None,
        seg_features: torch.Tensor | None = None,
        depth_features: torch.Tensor | None = None,
        ingr_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Multimodal forward pass.
        Returns:
            targets: (B, 4) containing [calories_kcal, protein_g, carbohydrates_g, fat_g]
        """
        B = visual_crops.size(0)
        device = visual_crops.device

        # 1. Extract visual features
        f_visual = self.visual_encoder(visual_crops)  # (B, visual_dim)

        # 2. Impute / encode class embeddings
        if class_ids is None:
            class_ids = torch.zeros(B, dtype=torch.long, device=device)
        else:
            class_ids = torch.clamp(class_ids.to(device), 0, self.num_classes - 1)
        f_class = self.class_embedding(class_ids)

        # 3. Impute / encode mass (log-scaled: log(1 + mass_g))
        if masses is None:
            masses = torch.zeros(B, 1, dtype=torch.float32, device=device)
        else:
            masses = masses.view(B, 1).to(device)
        f_mass = self.mass_encoder(torch.log1p(torch.clamp(masses, min=0.0)))

        # 4. Impute / encode segmentation mask metrics: [area_ratio, aspect_ratio, compactness]
        if seg_features is None:
            seg_features = torch.zeros(B, 3, dtype=torch.float32, device=device)
        else:
            seg_features = seg_features.view(B, 3).to(device)
        f_seg = self.seg_encoder(seg_features)

        # 5. Impute / encode depth features: [mean_depth, height_estimate]
        if depth_features is None:
            depth_features = torch.zeros(B, 2, dtype=torch.float32, device=device)
        else:
            depth_features = depth_features.view(B, 2).to(device)
        f_depth = self.depth_encoder(depth_features)

        # 6. Impute / encode ingredients
        if ingr_features is None:
            ingr_features = torch.zeros(B, self.num_ingredients, dtype=torch.float32, device=device)
        else:
            ingr_features = ingr_features.view(B, self.num_ingredients).to(device)
        f_ingr = self.ingr_encoder(ingr_features)

        # 7. Concatenate and fuse
        f_fused = torch.cat([f_visual, f_class, f_mass, f_seg, f_depth, f_ingr], dim=-1)
        latent = self.fusion_trunk(f_fused)

        # 8. Predict 4 targets
        cal = self.calories_head(latent)
        prot = self.protein_head(latent)
        carbs = self.carbs_head(latent)
        fat = self.fat_head(latent)

        out = torch.cat([cal, prot, carbs, fat], dim=-1)  # (B, 4)
        return out

    def save_checkpoint(
        self,
        path: str | Path,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Save model checkpoint."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "num_ingredients": self.num_ingredients,
            "metadata": extra_meta or {},
        }
        torch.save(payload, out)
        return out

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> MultiNutrientModel:
        """Load trained multi-nutrient model from checkpoint."""
        p = Path(path)
        checkpoint = torch.load(p, map_location=device)
        backbone = checkpoint.get("backbone", "efficientnet_b0")
        num_classes = checkpoint.get("num_classes", 105)
        num_ingredients = checkpoint.get("num_ingredients", 50)

        model = cls(
            backbone=backbone,
            num_classes=num_classes,
            num_ingredients=num_ingredients,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
