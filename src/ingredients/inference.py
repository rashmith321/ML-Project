"""
Inference engine for Ingredient Understanding.
Chains with Food Classification to output:
  - food_name
  - ingredient_candidates
  - ingredient_confidence
  - source provenance
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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

from src.ingredients.mapping import IngredientMapping, IngredientSource
from src.ingredients.model import MultiLabelIngredientModel
from src.utils.schema import FoodInstance


@dataclass
class IngredientResult:
    """
    Ingredient candidates and confidence for a detected food item.
    """
    food_name: str
    ingredient_candidates: list[str] = field(default_factory=list)
    ingredient_confidence: dict[str, float] = field(default_factory=dict)
    source: IngredientSource = IngredientSource.UNAVAILABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "food_name": self.food_name,
            "ingredient_candidates": self.ingredient_candidates,
            "ingredient_confidence": self.ingredient_confidence,
            "source": self.source.value,
        }


class IngredientPredictor:
    """
    Modular ingredient understanding predictor.
    Prioritizes:
      1. Model Prediction (if multi-label model checkpoint is available)
      2. Dataset Ground Truth (if annotated in dataset)
      3. Recipe-Derived (Recipe1M+)
      4. Reference Lookup (Vireo Food-172)
      5. Unavailable (unrepresented/unknown dish; never fabricates)
    """

    def __init__(
        self,
        model_checkpoint: str | Path | None = None,
        device: str = "cpu",
        mapping: IngredientMapping | None = None,
        confidence_threshold: float = 0.3,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.mapping = mapping or IngredientMapping()
        self.model: MultiLabelIngredientModel | None = None
        self.ingredient_vocab: list[str] = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if model_checkpoint and Path(model_checkpoint).is_file():
            self.load_model(model_checkpoint)

    def load_model(self, checkpoint_path: str | Path) -> None:
        """Load trained multi-label ingredient model."""
        self.model, self.ingredient_vocab = MultiLabelIngredientModel.load_from_checkpoint(
            checkpoint_path, device=self.device
        )
        print(f"[IngredientPredictor] Loaded model from {checkpoint_path} with {len(self.ingredient_vocab)} ingredient classes.")

    def predict(
        self,
        crop: Image.Image | np.ndarray | None = None,
        food_name: str | None = None,
    ) -> IngredientResult:
        """
        Predict ingredient candidates and confidence for a food item.
        """
        # Strategy 1: Trained multi-label neural network
        if self.model is not None and crop is not None and len(self.ingredient_vocab) > 0:
            if isinstance(crop, np.ndarray):
                pil_crop = Image.fromarray(crop).convert("RGB")
            else:
                pil_crop = crop.convert("RGB")

            tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                batch_preds = self.model.predict_ingredients(
                    tensor,
                    ingredient_vocab=self.ingredient_vocab,
                    threshold=self.confidence_threshold,
                )

            if batch_preds and len(batch_preds[0]) > 0:
                candidates = [ingr for ingr, _ in batch_preds[0]]
                conf_dict = {ingr: round(conf, 4) for ingr, conf in batch_preds[0]}
                return IngredientResult(
                    food_name=food_name or "unknown",
                    ingredient_candidates=candidates,
                    ingredient_confidence=conf_dict,
                    source=IngredientSource.MODEL_PREDICTION,
                )

        # Strategy 2-5: Mapping lookup (Ground Truth -> Recipe-Derived -> Reference Lookup -> Unavailable)
        candidates, conf_dict, source = self.mapping.lookup(food_name)
        return IngredientResult(
            food_name=food_name or "unknown",
            ingredient_candidates=candidates,
            ingredient_confidence=conf_dict,
            source=source,
        )

    def annotate_instances(
        self,
        instances: list[FoodInstance],
        image: Image.Image | np.ndarray | None = None,
    ) -> tuple[list[FoodInstance], list[IngredientResult]]:
        """
        Annotate FoodInstance objects with predicted ingredient tags.
        Returns updated instances and list of structured IngredientResult objects.
        """
        results: list[IngredientResult] = []

        pil_img = None
        if image is not None:
            if isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image).convert("RGB")
            else:
                pil_img = image.convert("RGB")

        for inst in instances:
            crop = None
            if pil_img is not None and inst.bbox is not None:
                w, h = pil_img.size
                x1, y1, x2, y2 = [int(v) for v in inst.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
                crop = pil_img.crop((x1, y1, x2, y2))

            res = self.predict(crop=crop, food_name=inst.class_name)
            inst.ingredients = list(res.ingredient_candidates)
            results.append(res)

        return instances, results


def run_ingredients_pipeline(
    image_path: str | Path,
    output_path: str | Path = "outputs/ingredients/predictions.json",
    detector_weights: str | Path = "models/yolov8n.pt",
    segmenter_weights: str | Path = "models/segmentation/best_unet.pt",
    classifier_weights: str | Path = "models/classification/best_classifier.pt",
    ingredient_model_weights: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Run end-to-end chained pipeline:
      Image -> Detection -> Segmentation -> Classification -> Ingredients
    """
    from src.classification.classifier import FoodClassifier
    from src.classification.inference import run_full_pipeline
    from src.detection.detector import FoodDetector
    from src.segmentation.segmenter import FoodSegmenter

    detector = FoodDetector(weights_path=detector_weights, device=device)
    detector.load()

    segmenter = FoodSegmenter(mode="auto", unet_checkpoint=segmenter_weights, device=device)
    segmenter.load()

    classifier = FoodClassifier(weights_path=classifier_weights, device=device)
    classifier.load()

    base_results = run_full_pipeline(
        image=image_path,
        detector=detector,
        segmenter=segmenter,
        classifier=classifier,
        output_json=None,
    )

    # Instantiate ingredient predictor
    predictor = IngredientPredictor(
        model_checkpoint=ingredient_model_weights,
        device=device,
    )

    img = Image.open(image_path).convert("RGB")
    annotated_instances: list[dict[str, Any]] = []

    for inst_dict in base_results.get("instances", []):
        food_name = inst_dict.get("food_name")
        bbox = inst_dict.get("bbox")

        crop = None
        if bbox:
            w, h = img.size
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))

        ingr_res = predictor.predict(crop=crop, food_name=food_name)

        item = dict(inst_dict)
        item["ingredient_candidates"] = ingr_res.ingredient_candidates
        item["ingredient_confidence"] = ingr_res.ingredient_confidence
        item["ingredient_source"] = ingr_res.source.value
        annotated_instances.append(item)

    payload = {
        "image_path": str(Path(image_path).resolve()).replace("\\", "/"),
        "image_size": list(img.size),
        "num_foods": len(annotated_instances),
        "instances": annotated_instances,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved ingredient predictions to: {out}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run food ingredient understanding inference.")
    parser.add_argument("--image", required=True, help="Input food image path")
    parser.add_argument("--output", default="outputs/ingredients/predictions.json", help="Output JSON path")
    parser.add_argument("--detector-weights", default="models/yolov8n.pt", help="Detector weights")
    parser.add_argument("--segmenter-weights", default="models/segmentation/best_unet.pt", help="Segmenter weights")
    parser.add_argument("--classifier-weights", default="models/classification/best_classifier.pt", help="Classifier weights")
    parser.add_argument("--ingredient-model", default=None, help="Optional multi-label ingredient model weights")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_ingredients_pipeline(
        image_path=args.image,
        output_path=args.output,
        detector_weights=args.detector_weights,
        segmenter_weights=args.segmenter_weights,
        classifier_weights=args.classifier_weights,
        ingredient_model_weights=args.ingredient_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
