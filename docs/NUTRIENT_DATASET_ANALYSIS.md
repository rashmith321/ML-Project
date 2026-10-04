# Dataset Nutritional Fields Inspection Report

## 1. Dataset Annotation Inspection Matrix

| Dataset | Mass Ground Truth | Core Macros (Cal/P/C/F) | Fiber/Sugar/SatFat | Sodium/Cholesterol | Micronutrients |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nutrition5k** | ✅ Yes | ✅ Yes (All 4) | ❌ None | ❌ None | ❌ None |
| **Food-101** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **UNIMIB2016** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **ECUSTFD** | ✅ Yes | ❌ None | ❌ None | ❌ None | ❌ None |
| **Recipe1M+** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **Vireo Food-172** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **MenuMatch** | ❌ None | ⚠️ Cal Only (Coarse) | ❌ None | ❌ None | ❌ None |

## 2. Nutrient Feasibility & Provenance Determination

| Nutrient | Ground Truth Available? | Which Dataset? | Predictable? | Derivable? | Reference Lookup? | Strategy |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Calories** (`kcal`) | ✅ Yes | Nutrition5k, MenuMatch (coarse) | ✅ Yes | ✅ Yes | ✅ Yes | PRIMARY: Model Prediction (Nutrition5k); FALLBACK: Reference lookup scaled by mass. |
| **Protein** (`g`) | ✅ Yes | Nutrition5k | ✅ Yes | ✅ Yes | ✅ Yes | PRIMARY: Model Prediction (Nutrition5k); FALLBACK: Reference lookup scaled by mass. |
| **Carbohydrates** (`g`) | ✅ Yes | Nutrition5k | ✅ Yes | ✅ Yes | ✅ Yes | PRIMARY: Model Prediction (Nutrition5k); FALLBACK: Reference lookup scaled by mass. |
| **Total Fat** (`g`) | ✅ Yes | Nutrition5k | ✅ Yes | ✅ Yes | ✅ Yes | PRIMARY: Model Prediction (Nutrition5k); FALLBACK: Reference lookup scaled by mass. |
| **Dietary Fiber** (`g`) | ❌ No | None | ❌ No | ✅ Yes | ✅ Yes | DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE. |
| **Total Sugars** (`g`) | ❌ No | None | ❌ No | ✅ Yes | ✅ Yes | DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE. |
| **Saturated Fat** (`g`) | ❌ No | None | ❌ No | ✅ Yes | ✅ Yes | DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE. |
| **Sodium** (`mg`) | ❌ No | None | ❌ No | ✅ Yes | ✅ Yes | DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE. |
| **Cholesterol** (`mg`) | ❌ No | None | ❌ No | ✅ Yes | ✅ Yes | DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE. |
| **Potassium** (`mg`) | ❌ No | None | ❌ No | ✅ Yes | ✅ Yes | DERIVED from reference profile * mass where reference exists; else NOT AVAILABLE. |
| **Calcium** (`mg`) | ❌ No | None | ❌ No | ✅ Yes | ✅ Yes | DERIVED from reference profile * mass where reference exists; else NOT AVAILABLE. |
| **Iron** (`mg`) | ❌ No | None | ❌ No | ✅ Yes | ✅ Yes | DERIVED from reference profile * mass where reference exists; else NOT AVAILABLE. |
| **Vitamin C** (`mg`) | ❌ No | None | ❌ No | ✅ Yes | ✅ Yes | DERIVED from reference profile * mass where reference exists; else NOT AVAILABLE. |

## 3. Scientific Methodology Rules

1. **Never Invent Values:** When a food item is unrepresented in verified reference profiles, unsupported fields return `null` / `'Not available'` with status `NOT AVAILABLE`.
2. **No Groundless Neural Regression:** Fiber, Sugar, Saturated Fat, Sodium, and Cholesterol CANNOT be directly regressed by neural networks because zero vision training datasets provide ground truth for them.
3. **Verified Compositional Derivation:** Where standard food categories are classified, reference composition per 100g from USDA FoodData Central is multiplied by estimated mass to produce mathematically grounded values labeled `REFERENCE` / `DERIVED`.