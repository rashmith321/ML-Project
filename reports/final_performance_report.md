# Final Performance Report

> [!IMPORTANT]
> All metrics are computed on **synthetic fixture data** only (2–80 programmatically generated images).
> Metrics reflect code correctness and architecture capability, **not real-world food recognition performance**.
> Real-world performance requires real food datasets (currently absent from `data/raw/`).

---

## Experiment Summary (7 experiments across 4 modules)

| Exp | Stage | Dataset | Key Metric | Baseline | Result | Change |
|-----|-------|---------|-----------|---------|--------|--------|
| N1 | Nutrition | 16 synthetic | Cal MAE | 400 kcal | 233 kcal | −167 kcal |
| N2 | Nutrition | 80 synthetic | Cal MAE | 400 kcal | **111 kcal** | −289 kcal ✓ |
| S1 | Segmentation | 2 synthetic | IoU | 0.000 | 0.964 | +0.964 ✓ |
| S2 | Segmentation | 24 synthetic | IoU | 0.000 | **0.992** | +0.992 ✓ |
| C1 | Classification | 8 synthetic | Acc | 0.50 | 0.75 | +0.25 ✓ |
| C2 | Classification | 40 synthetic | Acc | 0.50 | **0.70** | +0.20 ✓ |
| M1 | Mass | 40 synthetic | MAE | 354 g | **58 g** | −296 g ✓ |

---

## Ablation Study Results — Nutrition Model (N2, 20 val samples)

| Config | Image | +Class | +Seg | +Mass | +Depth | Cal MAE (kcal) | Cal R² |
|--------|:-----:|:------:|:----:|:-----:|:------:|----------------|--------|
| A — Image only | ✓ | | | | | 192.9 | 0.011 |
| B — + Class | ✓ | ✓ | | | | 194.4 | −0.034 |
| C — + Seg features | ✓ | ✓ | ✓ | | | 194.5 | −0.034 |
| D — + Mass | ✓ | ✓ | ✓ | ✓ | | 193.0 | −0.022 |
| **E — Full multimodal** | ✓ | ✓ | ✓ | ✓ | ✓ | **111.0** | **0.681** |

> [!NOTE]
> The dramatic jump from Config D → E shows that **depth features** carry the most information in the fixture dataset (depth is synthetically correlated with portion size in fixture data). This is consistent with the research hypothesis. On real Nutrition5k data with actual depth maps, this advantage should be even more pronounced.

---

## Best Models Selected (for each stage)

| Stage | Model | Checkpoint | Selection Criterion |
|-------|-------|-----------|---------------------|
| Detection | YOLOv8n (pretrained) | `models/yolov8n.pt` | Pretrained MS-COCO (no fine-tuning data) |
| Segmentation | FoodUNet S2 | `models/segmentation/best_unet.pt` | Best val IoU (0.992) |
| Classification | EfficientNet-B0 C2 | `models/classification/best_classifier.pt` | Best val accuracy (0.70) |
| Mass Estimation | EfficientNet-B0 M1 | `models/portion/best_mass_regressor.pt` | Best val MAE (54.9 g) |
| Nutrition | MultiNutrientModel N2 | `models/nutrition/best_nutrient_model.pt` | Best val loss + Cal R²=0.68 |

---

## What Improved

| Component | Improvement |
|-----------|------------|
| Segmentation | IoU 0.000 → 0.992 (24× more fixture training data, cosine LR, adjusted DiceBCE weight) |
| Mass estimation | MAE 354 g → 58 g (10× more fixture data, SmoothL1 loss, cosine LR) |
| Mass estimation | R² −118.7 → −3.9 (still negative, but massive improvement) |
| Nutrition | Cal MAE 400 → 111 kcal (5× more data, cosine LR, early stopping, gradient clipping) |
| Nutrition | Cal R² −17.6 → +0.68 (now positive — model better than mean prediction) |
| Classification | Top-1 50% → 70% (label smoothing prevents overfit; cosine LR) |
| Inference | Plausibility guard prevents 2.7 g → vol×density fallback |
| Inference | Blacklist filter correctly blocks `dining table`, `person`, `vehicles` |

---

## What Did NOT Improve

| Component | Status | Reason |
|-----------|--------|--------|
| Detection mAP | Unchanged (0.1555) | No food-specific training data; pretrained YOLOv8n only |
| Detection precision | Unchanged (0.0062) | Same reason |
| Mass R² | Still negative (−3.9) | 40 synthetic samples insufficient for image→mass regression |
| Real-image segmentation | Still all-zero | Fixture segmentation doesn't generalize to real photos |
| Nutrition on real images | Heuristic fallback | Real Nutrition5k data needed for genuine supervision |

---

## Remaining Limitations

1. **No real data** — All improvements are within synthetic fixture distribution. Real-world performance unknown.
2. **Segmentation still fails on real images** — FoodUNet trained on ellipse fixtures does not transfer.
3. **Mass R² negative** — Even 40 fixture samples are insufficient for the backbone to learn size-mass correlation.
4. **Detection not fine-tuned** — YOLOv8n with blacklist is functional but will miss food items not in COCO-80.
5. **Classification at 70%** — With Food101 (101,000 images), EfficientNet-B0 should reach 80%+ with fine-tuning.
6. **Nutrition ablation** — Full multimodal advantage (config E) driven by fixture depth features, not real depth maps.
7. **No test set** — Test set evaluation impossible without real data. Val = test in current fixture setup (same fixture).

---

## Recommended Next Steps (Priority Order)

1. **Download Nutrition5k** → enables real depth + nutrition supervision in one step
2. **Download Food101** → enables real classification and detection fine-tuning
3. Re-run all training pipelines (code is ready, manifests just need real image paths)
4. With real data, increase epochs: Classification 30→100, Segmentation 50→200, Nutrition 50→100
5. Consider SAM (Segment Anything) for segmentation prompting once detection is reliable
6. Add `ReduceLROnPlateau` scheduler variant for nutrition model
