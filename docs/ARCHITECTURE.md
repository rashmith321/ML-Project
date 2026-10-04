# ARCHITECTURE.md

## Pipeline

```
Image
 → Food Detection            (YOLO-family; food-vs-background + rough boxes)
 → Food Instance Segmentation (SAM/SAM2 prompted by detection boxes, + supervised head fine-tuned on UNIMIB2016)
 → Food Classification        (EfficientNet/ConvNeXt on Food-101, applied per detected/segmented instance)
 → Ingredient Understanding    (auxiliary, Recipe1M+ / Vireo-172 — text/tag association per class, not per-pixel)
 → Portion / Mass / Volume     (Nutrition5k-trained regressor on masked RGB(-D) crop; ECUSTFD calibration-object method as a fallback/comparison path)
 → Multi-Nutrient Estimation   (Nutrition5k-trained multi-output regressor: calories, protein, carbs, fat; per-instance)
 → Nutrition Aggregation       (sum per-instance outputs into a meal total, tracking ground-truth/prediction/derived/reference/unavailable provenance per value)
 → Nutrition Report            (per-food breakdown + meal total)
```

## Why each model family (not chosen for popularity)

- **Detection — YOLO-family:** real-time, well-supported transfer learning from COCO, and directly comparable to the reference repo's own choice (useful for an apples-to-apples "did we improve on the baseline" comparison in the final report).
- **Segmentation — SAM/SAM2 + supervised fine-tune:** SAM gives strong zero-shot instance masks from a box prompt (no food-specific segmentation training needed to get a usable mask), and UNIMIB2016 provides real multi-food-tray ground truth to fine-tune/evaluate against — this is the only dataset in the plan with actual segmentation masks, so it's the only one that can drive supervised segmentation metrics.
- **Classification — EfficientNet (default) / ConvNeXt (comparison):** EfficientNet gives strong accuracy-per-FLOP for Food-101-scale transfer learning and is practical for CPU inference; ConvNeXt is kept as a documented comparison point if GPU budget allows, per the brief's transfer-learning-first instruction. ViT is not the default because Food-101's ~75k training images is on the small side for training a ViT from scratch and would depend heavily on a strong pretrained checkpoint — worth a documented experiment, not the baseline.
- **Portion/mass/volume — regression on Nutrition5k:** Nutrition5k is the *only* dataset in the plan with real mass ground truth tied to RGB(-D) images, so it's the only defensible source for this stage; ECUSTFD's calibration-object method is kept as an independent, non-learned comparison method rather than folded into the same model, since its assumptions (single reference object in frame) don't generalize to arbitrary meal photos.
- **Nutrition regression — multi-output head on Nutrition5k:** one shared trunk with four output heads (calories, protein, carbs, fat) trained jointly, since Nutrition5k provides all four for the same image — avoids training four separate models on the same input.

## Module responsibilities (`src/`)

| Module | Responsibility |
|---|---|
| `src/data` | Dataset loaders/adapters, one per dataset, each returning a common internal schema (see below) — no cross-dataset merging logic lives here. |
| `src/preprocessing` | Image resize/normalize, RGB-D alignment (Nutrition5k), augmentation. |
| `src/detection` | YOLO wrapper: train/infer, box output schema. |
| `src/segmentation` | SAM prompting from boxes + supervised fine-tune head, trained/evaluated on UNIMIB2016. |
| `src/classification` | Food-101/Vireo-172-backed classifier, per detected instance. |
| `src/ingredients` | Recipe1M+/Vireo-172 ingredient-tag lookup per predicted class (auxiliary, not per-pixel). |
| `src/portion` | Nutrition5k mass/volume regressor + ECUSTFD calibration-object method as an alternate estimator. |
| `src/nutrition` | Multi-output nutrient regression head + the ground-truth/prediction/derived/reference/unavailable tagging logic + meal aggregation. |
| `src/models` | Shared model-building blocks (backbones, heads) used by the above. |
| `src/evaluation` | Per-stage metrics (detection mAP, segmentation IoU, classification top-1/5, mass MAE, nutrient MAE/MAPE). |
| `src/utils` | Config loading, path resolution (no hardcoded paths), logging. |

## Common internal schema

Every stage passes a list of `FoodInstance` records downstream:

```python
FoodInstance = {
    "instance_id": str,
    "bbox": [x, y, w, h] | None,
    "mask": np.ndarray | None,
    "class_name": str | None,
    "class_confidence": float | None,
    "ingredients": list[str] | None,          # auxiliary, may be empty
    "mass_g": float | None,
    "mass_source": "prediction" | "derived" | "unavailable",
    "nutrients": {
        "calories": {"value": float | None, "source": "ground_truth"|"prediction"|"derived"|"reference_lookup"|"unavailable"},
        "protein_g": {...}, "carbs_g": {...}, "fat_g": {...},
    },
}
```

The `source` tagging is mandatory at every stage output — this is what lets the final report distinguish measured, predicted, derived, looked-up, and missing values, per the brief's scientific rule.

## Cross-platform / environment support

- All paths come from `configs/*.yaml` + environment variables (`FOOD_PROJECT_DATA_ROOT`, etc.) — nothing hardcoded.
- `configs/env.windows.yaml` / `configs/env.colab.yaml` set only the root paths and device (`cpu`/`cuda`); everything else (model configs, dataset roles) is environment-independent.
- Training assumed on Colab GPU; inference designed to run on CPU (lighter classification/regression heads, SAM inference is the heaviest step and is the one place GPU is recommended even for inference).

## Final application (`app/`)

A single-image or single-meal-image upload endpoint that runs the full pipeline and returns/display a report exactly matching the brief's example format (per-food mass/calories/protein/carbs/fat, then a meal total). Framework choice deferred to `IMPLEMENTATION_PLAN.md` (kept lightweight — Streamlit/FastAPI — not decided here since it doesn't affect the ML architecture).
