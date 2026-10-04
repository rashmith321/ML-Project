# Baseline Metrics Report

> [!IMPORTANT]
> **All metrics are computed on SYNTHETIC FIXTURE DATA only.**
> Real datasets (Food101, Nutrition5k, ECUSTFD, UEC256) are absent from `data/raw/`.
> These metrics establish the code-correctness baseline, NOT real-world ML performance.

---

## Detection

| Metric | Baseline Value | Data Source |
|--------|---------------|-------------|
| Precision | 0.0062 | YOLO predictions vs 2 fixture bboxes |
| Recall | 1.000 | — |
| mAP@50 | 0.1555 | — |
| mAP@50:95 | 0.1428 | — |
| Samples | 2 | Synthetic fixture |

**Notes:** Using pretrained YOLOv8n (80 COCO classes). Non-food blacklist filter now active (dining table, person, vehicles, etc. are blocked).

---

## Segmentation

| Metric | Baseline Value | Data Source |
|--------|---------------|-------------|
| Mean IoU | 0.000 | FoodUNet on 2 fixture samples |
| Mean Dice | 0.000 | — |
| Mean Precision | 0.000 | — |
| Mean Recall | 0.000 | — |

**Notes:** FoodUNet trained on 2 synthetic fixture images. Produces all-zero masks on all real images.

---

## Classification

| Metric | Baseline Value | Data Source |
|--------|---------------|-------------|
| Top-1 Accuracy | 0.50 | 8 fixture samples, 4 classes |
| Top-5 Accuracy | 1.00 | — |
| Precision (macro) | 0.375 | — |
| Recall (macro) | 0.500 | — |
| F1 (macro) | 0.417 | — |
| Samples | 8 | Synthetic fixture |

---

## Mass / Volume Estimation

| Metric | Baseline Value | Data Source |
|--------|---------------|-------------|
| MAE (g) | 353.89 | 4 fixture samples |
| RMSE (g) | 355.33 | — |
| R² | −118.72 | — |
| Mean actual (g) | 355.75 | — |
| Mean predicted (g) | 1.86 | Model outputs ~2g for all inputs |

**Notes:** Plausibility guard [10–2000 g] added — neural predictions outside this range fall back to vol×density.

---

## Nutrition Estimation

### Calories

| Metric | Baseline |
|--------|---------|
| MAE (kcal) | 400.12 |
| RMSE (kcal) | 411.32 |
| R² | −17.63 |
| Mean actual (kcal) | 400.12 |
| Mean predicted (kcal) | 0.00 |

### Protein

| Metric | Baseline |
|--------|---------|
| MAE (g) | 18.03 |
| RMSE (g) | 22.98 |
| R² | −1.60 |

### Carbohydrates

| Metric | Baseline |
|--------|---------|
| MAE (g) | 40.99 |
| RMSE (g) | 49.71 |
| R² | −2.12 |

### Fat

| Metric | Baseline |
|--------|---------|
| MAE (g) | 15.01 |
| RMSE (g) | 15.83 |
| R² | −8.89 |

**Notes:** All R² < 0 — model is worse than predicting the mean. Root cause: trained on 4 synthetic fixture samples.
