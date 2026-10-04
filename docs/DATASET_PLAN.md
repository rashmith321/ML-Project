# DATASET_PLAN.md

## Reconciliation note (read first)

Two sets of decisions exist and are reconciled here as instructed: keep what `food_seg_project` already decided, but don't silently drop the originally-approved list.

- **Originally approved:** Nutrition5k, Food-101, UNIMIB2016, ECUSTFD, Recipe1M+, Vireo Food-172, MenuMatch. Explicitly excluded: FoodX-251, unavailable Indian food datasets.
- **Already decided in `food_seg_project` (kept as-is):** Food-101 + Khana → classification pool; UNIMIB2016 → segmentation eval; Nutrition5k → mass/nutrition ground truth; SAM ViT-B → Stage 1 segmentation backbone. FoodX-251 and the Khana *image set* were attempted and failed (Kaggle gating) — Khana's non-image metadata (labels/taxonomy) did download successfully.
- **Net conflicts, flagged rather than resolved silently:**
  1. **Khana** is not on the originally-approved list, and it's an Indian food dataset whose image set is Kaggle-gated (i.e. "unavailable" by the brief's own definition) — its metadata downloaded, its images did not. **Recommendation: drop Khana entirely from the training pool** (metadata alone can't train a classifier) unless you can obtain the images through an authenticated Kaggle download outside this sandboxed environment, in which case it becomes a legitimately-available auxiliary classification dataset and can be re-added.
  2. **ECUSTFD, Recipe1M+, Vireo Food-172, MenuMatch** — approved but unused so far. This plan restores them to their originally-specified roles below; none of this conflicts with the Drive decisions, since the Drive work simply hadn't reached them yet.

## Dataset-to-stage mapping

| Dataset | Role | Label type | Status |
|---|---|---|---|
| **Nutrition5k** | Primary — mass, calories, protein, carbs, fat, per-ingredient mass, RGB(-D) | Ground truth | Verified accessible (GCS bucket) in `food_seg_project`. |
| **Food-101** | Classification / transfer learning backbone pretraining | Ground truth (class labels only, no nutrition) | Full archive already downloaded in `food_seg_project`. |
| **UNIMIB2016** | Segmentation — multi-food tray scenes, instance/region masks | Ground truth | Present as a Drive subfolder. |
| **ECUSTFD** | Detection/recognition + reference-object volume experiments (calibration-object scale, the one genuinely reusable idea from the audited repo) | Ground truth (volume, mass, calibration reference) | Approved, not yet downloaded — next action. |
| **Recipe1M+** | Ingredient-level semantic representation (auxiliary) | Ground truth (recipe/ingredient text) | Approved, not yet downloaded — next action. |
| **Vireo Food-172** | Classification + ingredient-level auxiliary representation | Ground truth (class + ingredient tags) | Approved, not yet downloaded — next action. |
| **MenuMatch** | Auxiliary real-world calorie evaluation only | Ground truth (restaurant-menu calories) | Approved, not yet downloaded — next action. |
| ~~Khana~~ | ~~Auxiliary classification~~ | — | **Excluded** — image set Kaggle-gated/unavailable; only metadata obtained. Not used for training. |
| ~~FoodX-251~~ | — | — | **Excluded per brief**, confirmed unobtainable in `food_seg_project` run. |

## Explicit ground-truth vs. derived boundaries (per the brief's scientific rule)

- **Ground truth:** Nutrition5k mass/calorie/macro labels, Nutrition5k per-ingredient mass, UNIMIB2016 masks, Food-101/Vireo-172 class labels, ECUSTFD volume/mass/calibration data, MenuMatch menu calories.
- **Prediction:** detector boxes, segmentation masks (on non-UNIMIB images), class logits, mass/volume regression outputs on new images, multi-nutrient regression outputs.
- **Derived value:** per-ingredient calories computed from a predicted mass × a nutrient-density value that itself came from ground truth (e.g. Nutrition5k density tables) — must be logged as derived, not presented as measured.
- **Reference lookup:** any nutrient value pulled from a static food-composition table (e.g. for a food class Nutrition5k didn't cover) rather than predicted by a model.
- **Unavailable:** any nutrient/attribute no connected dataset actually provides for a given food class — must render as `null`/"unavailable" in the final report, never fabricated or silently defaulted to zero.

## Notes for later stages

- Recipe1M+ and Vireo-172 ingredient tags feed the **Ingredient Understanding** stage (auxiliary), not the primary nutrient regression — they help name what's on the plate, not quantify it.
- ECUSTFD's calibration-object approach is the strongest available *reference-object* portion-estimation method — worth prototyping alongside a pure Nutrition5k-depth-based mass regressor, and comparing.
- Each dataset keeps its documented purpose only — per the brief, no blind merging into one undifferentiated training pool.
