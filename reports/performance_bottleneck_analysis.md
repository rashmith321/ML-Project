# Performance Bottleneck Analysis

> [!NOTE]
> Analysis is based on both the baseline metric results and the root-cause investigation of the mass/calorie explosion bug (previously fixed). All conclusions are evidence-based, not assumptions.

---

## Bottleneck Ranking (Most → Least Critical)

### Rank 1 — SEGMENTATION (CRITICAL)

**Evidence:**
- Baseline IoU = **0.000**, Dice = **0.000**
- `FoodUNet` was trained on **2 synthetic fixture images** — insufficient by any ML standard
- Produces **all-zero masks** on every real food photograph
- Confirmed empirically: running the model on `20151130_114733.jpg` and demo meals shows `Segmentation: 0 px (0.0% image area)`

**Cascade Impact:**
- Volume estimation falls back to **Tier 3 RGB monocular** (bbox-based ellipse approximation)
- No segmentation → no pixel-level food area → poorer volume estimate → poorer mass
- Multi-nutrient model receives all-zero `seg_features` for every real image

**Fix applied:** Retrained with 24 samples + cosine LR → IoU=0.992, Dice=0.996 (on fixture only)

---

### Rank 2 — MASS / VOLUME ESTIMATION (CRITICAL)

**Evidence:**
- Baseline MAE = **353.9 g**, R² = **−118.7**
- Neural regressor predicted **~2.7 g** for all real images (fixture-trained only)
- This triggered the 6626 g / 9940 kcal bug reported by the user

**Root cause confirmed:**
- Model trained on 4 fixture samples → completely out-of-distribution on real images
- No depth map available for real images → Tier 1 (depth-based) never engaged
- No reference object in most real images → Tier 2 (calibrated) rarely engaged
- Tier 3 (RGB monocular) was using the **full dining-table bbox** (94% of image) → 4652 cm³

**Fixes applied:**
1. `dining table` blacklisted from detector
2. Bbox-fraction cap at 40% in volume estimator
3. Plausibility guard [10–2000 g] in mass estimator → falls back to vol×density
4. Coordinate scaling in nutrition/inference.py

**After fix:** Retrained M1 on 40 samples → MAE=57.9 g (fixture), R²=−3.93

---

### Rank 3 — NUTRITION REGRESSION (HIGH)

**Evidence:**
- Baseline R² < 0 for all four targets (Calories, Protein, Carbs, Fat)
- Model predicted **near-zero** for all inputs on the 4-sample fixture
- Heuristic fallback now operative: `150 kcal/100g` (now clamped)

**Root cause:** `MultiNutrientModel` trained on 4 synthetic samples with visually simple ellipses — backbone features meaningless

**After improvement (N2 — 80 samples, cosine LR, early stopping):**
- Calorie MAE = **111 kcal**, R² = **0.68** (on 20-sample fixture val set)
- Full multimodal (config E) clearly outperforms image-only (config A): R² 0.68 vs 0.011

---

### Rank 4 — DETECTION (MEDIUM)

**Evidence:**
- Precision = 0.006 (very low — many false positives on non-food COCO classes)
- Recall = 1.0 (pretrained YOLOv8n picks up most objects, but not all food-specific items)
- Detection is **pretrained** (MS-COCO), not food-fine-tuned
- After blacklist fix: correctly skips `dining table`, passes `bowl`, `cup`, `fork`, `knife`

**Impact:** Foods not in COCO-80 vocabulary (e.g., sushi, biryani, dal) may not be detected

**Fix applied:** Blacklist of 30+ non-food COCO classes. No fine-tuning (no labeled food detection data available).

---

### Rank 5 — CLASSIFICATION (MEDIUM)

**Evidence:**
- Baseline Top-1 = 50% (8 samples, 4 classes)
- EfficientNet-B0 backbone with ImageNet weights is strong — the bottleneck is training data, not architecture
- After C2 (40 samples, label smoothing): Top-1 = 70%

**Impact:** Wrong class → wrong ingredient lookup → wrong reference nutritional profile

---

## Primary Bottleneck Conclusion

> **The primary bottleneck is the absence of real training data.**
>
> Every model in this project was trained on 2–80 synthetic fixture images with solid-color ellipses on uniform backgrounds. No real food photographs are in `data/raw/`.
>
> **Even a single real dataset (Nutrition5k, Food101) with 100+ images would produce dramatically better generalization than any hyperparameter optimization can achieve on fixture data.**

---

## Recommended Priority Actions

1. **Acquire Nutrition5k** (free academic dataset) → enables supervised nutrition regression with ground-truth mass, depth, and macros
2. **Acquire Food101** → enables real food classification and detection fine-tuning
3. **Re-run all training pipelines** with real data → all pipeline code is ready
4. **Then revisit** backbone selection, augmentation, and hyperparameter tuning
