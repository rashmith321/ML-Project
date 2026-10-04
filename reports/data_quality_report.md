# Data Quality Report

> [!CAUTION]
> **No real food dataset images are present in `data/raw/`.**
> All training data is synthetic fixture data generated programmatically.
> This report documents both the current synthetic fixture quality and what to verify when real datasets are acquired.

---

## 1. Dataset Availability

| Dataset | Status | Expected Size | Path |
|---------|--------|---------------|------|
| Food101 | **UNAVAILABLE** | 101,000 images / 101 classes | `data/raw/food101/` |
| Nutrition5k | **UNAVAILABLE** | ~4,000+ dishes w/ RGB-D + nutrition labels | `data/raw/nutrition5k/` |
| ECUSTFD | **UNAVAILABLE** | 2,978 images w/ reference object | `data/raw/ecustfd/` |
| UEC-Food256 | **UNAVAILABLE** | 31,651 images / 256 classes | `data/raw/uec256/` |
| VireoFood-172 | **UNAVAILABLE** | ~110,000 images / 172 classes | `data/raw/vireofood172/` |
| FoodSeg103 | **UNAVAILABLE** | ~7,000 images w/ segmentation | `data/raw/foodseg103/` |
| UNIMIB2016 | **UNAVAILABLE** | ~1,000+ images | `data/raw/unimib2016/` |

---

## 2. Synthetic Fixture Data Quality

### 2a. Fixture v1 (original — used for baseline)

| Task | Samples | Issues |
|------|---------|--------|
| Classification | 8 | Far too few; 4 classes; solid ellipses ≠ real food |
| Segmentation | 2 | Grossly insufficient; all-zero output on real images |
| Nutrition | 4 (val), 12 (train) | Calorie targets physically consistent (4P+4C+9F); features meaningless |
| Mass | 4 | Size-correlated fixture; totally out-of-distribution for real images |

**Known fixture issues:**
- All images are solid-color ellipses on uniform white/grey backgrounds
- No texture, shadow, occlusion, or lighting variation
- Labels are deterministically generated — no annotation errors, but also no representativeness
- Class distribution: perfectly balanced (by construction)

### 2b. Fixture v2 (enlarged — used for improved training)

| Task | Samples | Train / Val |
|------|---------|-------------|
| Nutrition | 80 | 60 / 20 |
| Classification | 40 | 30 / 10 |
| Segmentation | 24 | 18 / 6 |
| Mass | 40 | 30 / 10 |

**Improvements over v1:**
- Varied food ellipse sizes (radius 25%–45% of image)
- Varied background colors (random ±20 per channel)
- Varied food colors (random ±30 per channel)
- Random center offset (±15 px)
- Optional Gaussian blur (50% of samples)
- Physically consistent nutrition labels: `calories = 4×protein + 4×carbs + 9×fat`
- 8 food classes vs 4
- Correct train/val split (no leakage)

---

## 3. Data Leakage Check

| Check | Result |
|-------|--------|
| Fixture train/val split | **CLEAN** — deterministic 75/25 split, no overlap |
| Demo meal presets | **CLEAN** — used at inference only, never in training |
| Real datasets | **N/A** — no real data present |
| Nutrition5k official split | **N/A** — dataset not available |

> [!TIP]
> When Nutrition5k is acquired, verify: no dish ID appears in both train and val (use `src/data/leakage_check.py` via `test_data_layer.py`).

---

## 4. Class Imbalance Assessment

**Fixture data:** Perfectly balanced by construction (not representative of real-world distribution).

**Expected real-world distribution (from literature):**
- Food101: ~1,000 images/class (balanced)
- Nutrition5k: Some dish types over-represented (rice dishes common in dataset)
- UEC256: Significant imbalance (some classes have <100 images)

**Recommendation:** If class imbalance > 10:1, apply:
1. `WeightedRandomSampler` for training
2. `focal_loss` with γ=2 for classification
3. Monitor per-class metrics in addition to macro averages

---

## 5. What to Check When Real Data Arrives

### Image Quality Checks
- [ ] Corrupted/unreadable images (try `PIL.Image.open` on all images, catch exceptions)
- [ ] Near-duplicate images (use perceptual hash; check within same split only)
- [ ] Incorrect image resolution (<64px → discard; >4096px → resize on load)

### Label Quality Checks
- [ ] Missing `calories_kcal`, `protein_g`, `carbohydrates_g`, `fat_g` fields → fill with NaN, exclude from nutrition training
- [ ] Unrealistic nutrition values: calories < 0 or > 5000 → flag as outlier
- [ ] Calorie-macro consistency: check if `|calories - (4P + 4C + 9F)| > 50 kcal` → flag
- [ ] Bounding box sanity: x1 < x2, y1 < y2, area > 0.1% of image
- [ ] Segmentation mask sanity: mask area > 0 and < 95% of image
- [ ] Mass values: < 5 g or > 3000 g → flag as outlier

### Nutrition5k Specific
- [ ] Use official train/test split CSV from dataset repository
- [ ] Check for multi-dish images (Nutrition5k has per-dish and per-ingredient labels)
- [ ] Verify depth map alignment with RGB image
