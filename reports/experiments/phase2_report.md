# ML Performance Improvement — Phase II Report

**Date:** 2026-09-30  
**Data split:** 60 / 20 / 20 train / val / test (fixed seed=42)  
**Rule enforced:** Validation used for model selection only. Test set evaluated once, after selection.  
**Experiment log:** [`reports/experiments/experiment_log.csv`](file:///c:/Users/rashm/OneDrive/Desktop/ML/reports/experiments/experiment_log.csv)

> [!IMPORTANT]
> All results are on **synthetic fixture data only** (120–80 programmatically generated images with solid-colour ellipses). These metrics confirm architectural correctness and training methodology, not real-world food recognition performance.

---

## 1. Data Setup

| Task | Total | Train (60%) | Val (20%) | Test (20%) |
|------|-------|------------|-----------|-----------|
| Nutrition | 120 | 72 | 24 | 24 |
| Mass | 80 | 48 | 16 | 16 |

Split is **deterministic** (seed=42 shuffle by index). Test split was not inspected until all model selection was complete.

---

## 2. Baseline Preservation

Before any experiment, current production checkpoints were copied:

| Stage | Baseline Saved As |
|-------|------------------|
| Nutrition | `models/nutrition/baseline_p2_nutrition_best_nutrient_model.pt` |
| Segmentation | `models/segmentation/baseline_p2_segmentation_best_unet.pt` |
| Classification | `models/classification/baseline_p2_classification_best_classifier.pt` |
| Mass | `models/portion/baseline_p2_mass_best_mass_regressor.pt` |

> [!NOTE]
> Production models are **not overwritten** unless the Phase II best model beats the Phase I model evaluated on the **same Phase II validation set** (fair comparison — same data distribution).

---

## 3. Nutrition Experiments (3 experiments)

Architecture: `MultiNutrientModel` (EfficientNet-B0 + multimodal fusion)  
Loss: `NormalizedMultiNutrientLoss` (scale factors: 500, 30, 50, 25)  
Optimizer: `AdamW(lr, weight_decay=1e-4)` + gradient clipping (norm=1.0)  
Patience: 12 epochs early stopping

| Exp | Key Change | Val CalMAE | Val R² | **Test CalMAE** | **Test R²** |
|-----|-----------|-----------|--------|----------------|------------|
| N-A (baseline) | CosineAnnealing, standard training | 92.3 kcal | 0.766 | 86.6 kcal | 0.803 |
| **N-B** | `ReduceLROnPlateau(patience=5, factor=0.5)` | **85.5 kcal** | **0.801** | **80.1 kcal** | **0.838** |
| N-C | Two-stage (freeze backbone ep 1–20, finetune all ep 21–60) | 87.1 kcal | 0.796 | 85.7 kcal | 0.796 |

### Winner: **N-B** (ReduceLROnPlateau)
- Best validation CalMAE = **85.5 kcal** → selected as production model
- Best test CalMAE = **80.1 kcal**, R² = **0.838**
- Compared to Phase I best (N2): Phase I scored CalMAE=138.3 on Phase II val set → Phase II N-B improves by **52.8 kcal (38%)**

### Interpretation
- `ReduceLROnPlateau` adapts the learning rate to actual validation loss plateaus — more effective than fixed cosine schedule on small fixture data where the loss landscape is noisy
- Two-stage training (N-C) converged more slowly due to early frozen backbone, and did not beat N-B once the backbone was unfrozen

---

## 4. Mass Estimation Experiments (2 experiments)

Architecture: `FoodMassRegressionModel` (EfficientNet-B0 → 256d head → scalar)  
Loss: `SmoothL1`  
Optimizer: `AdamW(lr=5e-4, weight_decay=1e-4)` + CosineAnnealing  
Patience: 10 epochs early stopping

| Exp | Key Change | Val MAE (g) | Val R² | **Test MAE (g)** | **Test R²** |
|-----|-----------|------------|--------|-----------------|------------|
| M-A (baseline) | Linear-space target `mass_g` | 44.1 g | −0.029 | 53.4 g | −0.140 |
| **M-B** | Log-space target `log(mass_g)`, eval `exp(pred)` | **36.7 g** | **0.209** | **41.6 g** | **0.174** |

### Winner: **M-B** (Log-space prediction)
- Best validation MAE = **36.7 g** → selected as production model
- Best test MAE = **41.6 g**, R² = **0.174**
- Compared to Phase I M1 model: Phase I scored MAE=58.8 g on Phase II val set → Phase II M-B improves by **22.1 g (38%)**

### Interpretation
- Mass values span a wide range (low-density salads ~50g to dense steaks ~500g). Predicting in log-space compresses this range and allows the network to learn relative differences more easily
- Log-space prediction also makes the loss function more symmetric (percentage-error based), which is appropriate when both 10g and 200g masses are in the dataset
- R² crossed positive territory (0.174) — first time mass regression R² > 0 in this project

---

## 5. Production Model Updates

Phase I models were evaluated on Phase II validation data for a fair baseline:

| Stage | Phase I Val MAE | Phase II Best Val MAE | Improvement | Production Updated? |
|-------|----------------|----------------------|-------------|---------------------|
| Nutrition (Cal) | 138.3 kcal | **85.5 kcal** | −38% | ✅ Yes — N-B |
| Mass | 58.8 g | **36.7 g** | −38% | ✅ Yes — M-B |

Production checkpoints:
- `models/nutrition/best_nutrient_model.pt` ← N-B weights
- `models/portion/best_mass_regressor.pt` ← M-B weights

---

## 6. Cumulative Progress (Phase I + Phase II)

### Nutrition Model — Calorie MAE Progression

| Stage | Samples | Val CalMAE | Val R² | Note |
|-------|---------|-----------|--------|------|
| Original baseline | 4 | 400.1 kcal | −17.6 | Untrained fixture |
| Phase I N1 | 16 | 232.5 kcal | <0 | No scheduler |
| Phase I N2 | 80 | 111.0 kcal | 0.68 | Cosine LR |
| **Phase II N-B** | 120 | **85.5 kcal** | **0.80** | ReduceLROnPlateau |
| Phase II N-B test | 120 | **80.1 kcal** | **0.84** | ← held-out test result |

### Mass MAE Progression

| Stage | Samples | Val MAE | Val R² | Note |
|-------|---------|---------|--------|------|
| Original baseline | 4 | 353.9 g | −118.7 | Untrained fixture |
| Phase I M1 | 40 | 57.9 g | −3.9 | SmoothL1 linear |
| **Phase II M-B** | 80 | **36.7 g** | **0.21** | Log-space prediction |
| Phase II M-B test | 80 | **41.6 g** | **0.17** | ← held-out test result |

---

## 7. Experiment Tracking

All experiments logged to [`reports/experiments/experiment_log.csv`](file:///c:/Users/rashm/OneDrive/Desktop/ML/reports/experiments/experiment_log.csv):

| ID | Stage | Key Change | Val Metric | Test Metric |
|----|-------|-----------|-----------|------------|
| EXP-001 | Nutrition | N-A Cosine baseline | 92.3 kcal | 86.6 kcal |
| EXP-002 | Nutrition | N-B ReduceLROnPlateau ✓ | **85.5 kcal** | **80.1 kcal** |
| EXP-003 | Nutrition | N-C Two-stage training | 87.1 kcal | 85.7 kcal |
| EXP-004 | Mass | M-A Linear SmoothL1 | 44.1 g | 53.4 g |
| EXP-005 | Mass | M-B Log-space ✓ | **36.7 g** | **41.6 g** |

---

## 8. Remaining Limitations

1. **All results on synthetic fixture data.** Fixture metrics do not predict real-world performance.
2. **Mass R² = 0.17** — positive but weak. Real-world size/mass correlation requires real food images.
3. **Segmentation and classification** experiments deferred (Phase I results already good on fixture; real improvement requires real data).
4. **Detection** not fine-tuned — no food-labeled detection dataset available.

> [!CAUTION]
> **The single most impactful action remaining:** Download Nutrition5k (free, academic). It provides RGB-D images, per-dish mass, and all four macro-nutrients in a single dataset. Every training script is ready to accept it.
