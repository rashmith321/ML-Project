"""
Multi-Nutrient Estimation Engine.
Connects multimodal inputs (visual crop, classification, mask metrics, mass, depth, ingredients)
into the trained MultiNutrientModel to predict:
  - calories_kcal
  - protein_g
  - carbohydrates_g
  - fat_g
and populates FoodInstance.nutrients with explicit provenance Source.PREDICTION.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.nutrition.model import MultiNutrientModel
from src.nutrition.reference_database import FoodNutritionReferenceDatabase
from src.utils.schema import FoodInstance, NutrientSet, Source, UnifiedFoodNutrition, Value


class MultiNutrientEstimator:
    """
    Multimodal nutrition estimation engine.
    """

    def __init__(
        self,
        weights_path: str | Path | None = "models/nutrition/best_nutrient_model.pt",
        vocab_path: str | Path | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.weights_path = Path(weights_path) if weights_path else None
        self.model: MultiNutrientModel | None = None
        self.ref_db = FoodNutritionReferenceDatabase()

        self.vocab: list[str] = []
        if vocab_path and Path(vocab_path).is_file():
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        self.vocab_to_idx = {ingr: i for i, ingr in enumerate(self.vocab)}

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.weights_path and self.weights_path.is_file():
            self.load_model(self.weights_path)

    def load_model(self, checkpoint_path: str | Path) -> None:
        """Load trained multimodal multi-nutrient model."""
        self.model = MultiNutrientModel.load_from_checkpoint(checkpoint_path, device=self.device)
        print(f"[MultiNutrientEstimator] Loaded checkpoint from {checkpoint_path}")

    def extract_mask_features(self, mask: np.ndarray | None, bbox: Sequence[float] | None, img_size: tuple[int, int]) -> torch.Tensor:
        """Extract [area_ratio, aspect_ratio, compactness]."""
        w, h = img_size
        img_area = max(1.0, float(w * h))

        if mask is not None and np.any(mask):
            area_px = float(np.sum(mask > 0))
            area_ratio = area_px / img_area
            # Bounding box of mask
            y_indices, x_indices = np.where(mask > 0)
            bw = max(1.0, float(np.ptp(x_indices)))
            bh = max(1.0, float(np.ptp(y_indices)))
            aspect_ratio = bw / bh
            compactness = area_px / (bw * bh)
        elif bbox is not None:
            bx1, by1, bx2, by2 = bbox
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)
            area_ratio = (bw * bh) / img_area
            aspect_ratio = bw / bh
            compactness = 0.785
        else:
            area_ratio, aspect_ratio, compactness = 0.25, 1.0, 0.5

        return torch.tensor([[area_ratio, aspect_ratio, compactness]], dtype=torch.float32, device=self.device)

    def extract_depth_features(self, depth_map: np.ndarray | None, mask: np.ndarray | None) -> torch.Tensor:
        """Extract [mean_depth, height_estimate]."""
        if depth_map is not None and mask is not None and np.any(mask):
            food_depths = depth_map[mask > 0]
            valid = food_depths[food_depths > 0]
            if len(valid) > 0:
                mean_d = float(np.mean(valid))
                height_est = float(np.ptp(valid))
                return torch.tensor([[mean_d, height_est]], dtype=torch.float32, device=self.device)

        # Default zero-imputed
        return torch.zeros((1, 2), dtype=torch.float32, device=self.device)

    def encode_ingredients(self, ingredients: Sequence[str] | None) -> torch.Tensor:
        """Encode bag-of-ingredients vector."""
        num_ingrs = max(1, self.model.num_ingredients if self.model else len(self.vocab))
        vec = torch.zeros((1, num_ingrs), dtype=torch.float32, device=self.device)
        if ingredients:
            for ingr in ingredients:
                norm = ingr.lower().strip()
                if norm in self.vocab_to_idx and self.vocab_to_idx[norm] < num_ingrs:
                    vec[0, self.vocab_to_idx[norm]] = 1.0
        return vec

    def estimate_nutrients(
        self,
        crop: Image.Image | np.ndarray,
        class_id: int = 0,
        mass_g: float = 150.0,
        mask: np.ndarray | None = None,
        bbox: Sequence[float] | None = None,
        img_size: tuple[int, int] = (640, 640),
        depth_map: np.ndarray | None = None,
        ingredients: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """
        Estimate 4 target nutrients: calories_kcal, protein_g, carbohydrates_g, fat_g.
        """
        if isinstance(crop, np.ndarray):
            pil_crop = Image.fromarray(crop).convert("RGB")
        else:
            pil_crop = crop.convert("RGB")

        # Fallback if model not trained/loaded: use verified culinary macro density heuristics.
        # Canonical per-100g values for a generic mixed meal:
        #   150 kcal / 10 g protein / 18 g carbs / 5 g fat
        # Formula: (density_per_100g / 100) * mass_g
        # We clamp mass_g to a physiologically plausible single-food range [30, 600] g
        # ONLY for the purpose of this heuristic calculation.  We do NOT mutate mass_val
        # so the downstream mass figure reported to the user is unchanged.
        if self.model is None:
            if mass_g > 600.0:
                print(
                    f"[MultiNutrientEstimator] WARNING: mass_g={mass_g:.1f} g exceeds 600 g — "
                    f"clamping to 600 g for heuristic nutrient calculation only. "
                    f"Upstream detection or volume estimation may be incorrect."
                )
            effective_mass = max(30.0, min(600.0, mass_g))
            return {
                "calories_kcal": round(150.0 * effective_mass / 100.0, 1),
                "protein_g": round(10.0 * effective_mass / 100.0, 1),
                "carbohydrates_g": round(18.0 * effective_mass / 100.0, 1),
                "fat_g": round(5.0 * effective_mass / 100.0, 1),
            }

        crop_tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
        class_tensor = torch.tensor([class_id], dtype=torch.long, device=self.device)
        mass_tensor = torch.tensor([[mass_g]], dtype=torch.float32, device=self.device)
        seg_tensor = self.extract_mask_features(mask, bbox, img_size)
        depth_tensor = self.extract_depth_features(depth_map, mask)
        ingr_tensor = self.encode_ingredients(ingredients)

        with torch.no_grad():
            preds = self.model(
                crop_tensor,
                class_ids=class_tensor,
                masses=mass_tensor,
                seg_features=seg_tensor,
                depth_features=depth_tensor,
                ingr_features=ingr_tensor,
            ).squeeze(0).cpu().numpy()

        return {
            "calories_kcal": round(float(preds[0]), 1),
            "protein_g": round(float(preds[1]), 1),
            "carbohydrates_g": round(float(preds[2]), 1),
            "fat_g": round(float(preds[3]), 1),
        }


    def annotate_instances(
        self,
        instances: list[FoodInstance],
        image: Image.Image | np.ndarray,
        depth_map: np.ndarray | None = None,
    ) -> list[FoodInstance]:
        """
        Annotate FoodInstance records with predicted NutrientSet.
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        w, h = pil_img.size

        for inst in instances:
            if inst.bbox is None:
                continue

            fx1, fy1, fx2, fy2 = [int(round(v)) for v in inst.bbox]
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(w, max(fx1 + 1, fx2)), min(h, max(fy1 + 1, fy2))

            crop = pil_img.crop((fx1, fy1, fx2, fy2))

            # Retrieve mass
            mass_val = inst.mass.value if (inst.mass and inst.mass.value is not None) else 150.0

            nutrients_dict = self.estimate_nutrients(
                crop=crop,
                class_id=getattr(inst, "class_id", 0) or 0,
                mass_g=float(mass_val),
                mask=inst.mask,
                bbox=inst.bbox,
                img_size=(w, h),
                depth_map=depth_map,
                ingredients=inst.ingredients,
            )

            # Combine predicted core macronutrients with verified reference-derived extended nutrients
            inst.nutrients = self.ref_db.derive_nutrients_for_instance(
                food_name=inst.class_name or "unknown",
                mass_g=float(mass_val),
                predicted_calories=nutrients_dict["calories_kcal"],
                predicted_protein=nutrients_dict["protein_g"],
                predicted_carbs=nutrients_dict["carbohydrates_g"],
                predicted_fat=nutrients_dict["fat_g"],
                primary_source=Source.PREDICTION,
            )
            inst.unified_nutrition = UnifiedFoodNutrition.from_food_instance(inst)

        return instances
