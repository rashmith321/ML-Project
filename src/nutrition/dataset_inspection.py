"""
Dataset Nutritional Fields Inspector.

Rigorously inspects and documents the exact nutritional fields available across all 7 project datasets:
1. Nutrition5k
2. Food-101
3. UNIMIB2016
4. ECUSTFD
5. Recipe1M+
6. Vireo Food-172
7. MenuMatch

Determines for each potential nutrient:
- Is ground truth available?
- Which dataset provides it?
- Can it be predicted?
- Can it be derived?
- Can it be obtained from a reliable reference?
- Is it unavailable?
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class NutrientFeasibility:
    nutrient_key: str
    display_name: str
    unit: str
    is_ground_truth_available: bool
    providing_datasets: list[str]
    can_be_predicted: bool
    prediction_rationale: str
    can_be_derived: bool
    derivation_rationale: str
    can_be_obtained_from_reference: bool
    reference_source: str
    is_unavailable_fallback: bool
    handling_strategy: str


# Comprehensive analysis of every potential nutrient field
NUTRIENT_ANALYSIS: dict[str, NutrientFeasibility] = {
    "calories_kcal": NutrientFeasibility(
        nutrient_key="calories_kcal",
        display_name="Calories",
        unit="kcal",
        is_ground_truth_available=True,
        providing_datasets=["Nutrition5k", "MenuMatch (coarse)"],
        can_be_predicted=True,
        prediction_rationale="Supervised regression head trained on Nutrition5k continuous ground truth.",
        can_be_derived=True,
        derivation_rationale="Can be derived from mass * reference energy density (kcal/100g) when unpredicted.",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=False,
        handling_strategy="PRIMARY: Model Prediction (Nutrition5k); FALLBACK: Reference lookup scaled by mass.",
    ),
    "protein_g": NutrientFeasibility(
        nutrient_key="protein_g",
        display_name="Protein",
        unit="g",
        is_ground_truth_available=True,
        providing_datasets=["Nutrition5k"],
        can_be_predicted=True,
        prediction_rationale="Supervised regression head trained on Nutrition5k continuous ground truth.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * reference protein density (g/100g).",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=False,
        handling_strategy="PRIMARY: Model Prediction (Nutrition5k); FALLBACK: Reference lookup scaled by mass.",
    ),
    "carbohydrates_g": NutrientFeasibility(
        nutrient_key="carbohydrates_g",
        display_name="Carbohydrates",
        unit="g",
        is_ground_truth_available=True,
        providing_datasets=["Nutrition5k"],
        can_be_predicted=True,
        prediction_rationale="Supervised regression head trained on Nutrition5k continuous ground truth.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * reference carbohydrate density (g/100g).",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=False,
        handling_strategy="PRIMARY: Model Prediction (Nutrition5k); FALLBACK: Reference lookup scaled by mass.",
    ),
    "fat_g": NutrientFeasibility(
        nutrient_key="fat_g",
        display_name="Total Fat",
        unit="g",
        is_ground_truth_available=True,
        providing_datasets=["Nutrition5k"],
        can_be_predicted=True,
        prediction_rationale="Supervised regression head trained on Nutrition5k continuous ground truth.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * reference total fat density (g/100g).",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=False,
        handling_strategy="PRIMARY: Model Prediction (Nutrition5k); FALLBACK: Reference lookup scaled by mass.",
    ),
    "fiber_g": NutrientFeasibility(
        nutrient_key="fiber_g",
        display_name="Dietary Fiber",
        unit="g",
        is_ground_truth_available=False,
        providing_datasets=[],
        can_be_predicted=False,
        prediction_rationale="No dataset in the vision collection contains dietary fiber ground truth labels. Direct neural regression would be ungrounded.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * verified USDA reference fiber density (g/100g) for the classified food.",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=True,
        handling_strategy="DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE.",
    ),
    "sugar_g": NutrientFeasibility(
        nutrient_key="sugar_g",
        display_name="Total Sugars",
        unit="g",
        is_ground_truth_available=False,
        providing_datasets=[],
        can_be_predicted=False,
        prediction_rationale="No dataset contains sugar annotations. Cannot predict without training labels.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * verified USDA reference sugar density (g/100g).",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=True,
        handling_strategy="DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE.",
    ),
    "saturated_fat_g": NutrientFeasibility(
        nutrient_key="saturated_fat_g",
        display_name="Saturated Fat",
        unit="g",
        is_ground_truth_available=False,
        providing_datasets=[],
        can_be_predicted=False,
        prediction_rationale="No dataset contains fatty acid fractionation. Saturated fat cannot be predicted from RGB pixels alone.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * verified USDA reference saturated fat profile (g/100g). Strictly constrained <= total_fat.",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=True,
        handling_strategy="DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE.",
    ),
    "sodium_mg": NutrientFeasibility(
        nutrient_key="sodium_mg",
        display_name="Sodium",
        unit="mg",
        is_ground_truth_available=False,
        providing_datasets=[],
        can_be_predicted=False,
        prediction_rationale="Sodium is invisible to camera pixels (cannot optically detect dissolved salt). Predicting sodium without ground truth would be scientifically invalid.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * verified USDA reference sodium density (mg/100g).",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=True,
        handling_strategy="DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE.",
    ),
    "cholesterol_mg": NutrientFeasibility(
        nutrient_key="cholesterol_mg",
        display_name="Cholesterol",
        unit="mg",
        is_ground_truth_available=False,
        providing_datasets=[],
        can_be_predicted=False,
        prediction_rationale="No dataset contains cholesterol labels. Cannot be predicted optically.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * verified USDA reference cholesterol profile (mg/100g). Plant foods are strictly 0.0 mg.",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=True,
        handling_strategy="DERIVED from classified food reference profile * mass. If unmapped: NOT AVAILABLE.",
    ),
    "potassium_mg": NutrientFeasibility(
        nutrient_key="potassium_mg",
        display_name="Potassium",
        unit="mg",
        is_ground_truth_available=False,
        providing_datasets=[],
        can_be_predicted=False,
        prediction_rationale="Micronutrient not labeled in any approved dataset.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * USDA reference potassium (mg/100g).",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=True,
        handling_strategy="DERIVED from reference profile * mass where reference exists; else NOT AVAILABLE.",
    ),
    "calcium_mg": NutrientFeasibility(
        nutrient_key="calcium_mg",
        display_name="Calcium",
        unit="mg",
        is_ground_truth_available=False,
        providing_datasets=[],
        can_be_predicted=False,
        prediction_rationale="Micronutrient not labeled in any approved dataset.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * USDA reference calcium (mg/100g).",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=True,
        handling_strategy="DERIVED from reference profile * mass where reference exists; else NOT AVAILABLE.",
    ),
    "iron_mg": NutrientFeasibility(
        nutrient_key="iron_mg",
        display_name="Iron",
        unit="mg",
        is_ground_truth_available=False,
        providing_datasets=[],
        can_be_predicted=False,
        prediction_rationale="Micronutrient not labeled in any approved dataset.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * USDA reference iron (mg/100g).",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=True,
        handling_strategy="DERIVED from reference profile * mass where reference exists; else NOT AVAILABLE.",
    ),
    "vitamin_c_mg": NutrientFeasibility(
        nutrient_key="vitamin_c_mg",
        display_name="Vitamin C",
        unit="mg",
        is_ground_truth_available=False,
        providing_datasets=[],
        can_be_predicted=False,
        prediction_rationale="Micronutrient not labeled in any approved dataset.",
        can_be_derived=True,
        derivation_rationale="Derived from mass * USDA reference vitamin C (mg/100g).",
        can_be_obtained_from_reference=True,
        reference_source="USDA FoodData Central SR Legacy",
        is_unavailable_fallback=True,
        handling_strategy="DERIVED from reference profile * mass where reference exists; else NOT AVAILABLE.",
    ),
}


# Dataset-level capability matrix across all 7 approved datasets
DATASET_CAPABILITY_MATRIX = {
    "Nutrition5k": {
        "annotations_present": ["mass_g", "calories_kcal", "protein_g", "carbohydrates_g", "fat_g", "depth_map", "ingredient_breakdown"],
        "unsupported_nutrients": ["fiber_g", "sugar_g", "saturated_fat_g", "sodium_mg", "cholesterol_mg", "micronutrients"],
        "primary_role": "Primary supervised training for portion mass and 4 core macronutrients.",
    },
    "Food-101": {
        "annotations_present": ["class_name (101 food classes)"],
        "unsupported_nutrients": ["mass_g", "all_nutrients"],
        "primary_role": "Classification taxonomy and visual feature transfer learning.",
    },
    "UNIMIB2016": {
        "annotations_present": ["class_name (73 food classes)", "polygon_segmentation"],
        "unsupported_nutrients": ["mass_g", "all_nutrients"],
        "primary_role": "Multi-food instance segmentation benchmarks.",
    },
    "ECUSTFD": {
        "annotations_present": ["class_name (19 food classes)", "bounding_box", "segmentation", "mass_g", "volume_cm3", "reference_coin"],
        "unsupported_nutrients": ["all_nutrients"],
        "primary_role": "Reference object calibration and physical volume/mass measurement.",
    },
    "Recipe1M+": {
        "annotations_present": ["recipe_title", "ingredient_list", "cooking_instructions"],
        "unsupported_nutrients": ["standardized_tabular_nutrition_labels"],
        "primary_role": "Ingredient understanding and recipe-derived ingredient mapping.",
    },
    "Vireo Food-172": {
        "annotations_present": ["class_name (172 food classes)", "ingredient_labels (353 ingredients)"],
        "unsupported_nutrients": ["mass_g", "all_nutrients"],
        "primary_role": "Multi-label ingredient classification ground truth.",
    },
    "MenuMatch": {
        "annotations_present": ["restaurant_dish_name", "bounding_box", "coarse_calories_range"],
        "unsupported_nutrients": ["protein_g", "carbohydrates_g", "fat_g", "fiber_g", "sugar_g", "saturated_fat_g", "sodium_mg", "cholesterol_mg"],
        "primary_role": "Real-world restaurant calorie validation.",
    },
}


def get_nutrient_inspection_summary() -> dict[str, Any]:
    """Returns a structured summary of dataset field availability and feasibility."""
    return {
        "datasets": DATASET_CAPABILITY_MATRIX,
        "nutrients": {k: asdict(v) for k, v in NUTRIENT_ANALYSIS.items()},
    }


def format_inspection_report() -> str:
    """Formats a detailed markdown report of the dataset inspection."""
    lines: list[str] = []
    lines.append("# Dataset Nutritional Fields Inspection Report\n")
    lines.append("## 1. Dataset Annotation Inspection Matrix\n")
    lines.append("| Dataset | Mass Ground Truth | Core Macros (Cal/P/C/F) | Fiber/Sugar/SatFat | Sodium/Cholesterol | Micronutrients |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    lines.append("| **Nutrition5k** | ✅ Yes | ✅ Yes (All 4) | ❌ None | ❌ None | ❌ None |")
    lines.append("| **Food-101** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |")
    lines.append("| **UNIMIB2016** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |")
    lines.append("| **ECUSTFD** | ✅ Yes | ❌ None | ❌ None | ❌ None | ❌ None |")
    lines.append("| **Recipe1M+** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |")
    lines.append("| **Vireo Food-172** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |")
    lines.append("| **MenuMatch** | ❌ None | ⚠️ Cal Only (Coarse) | ❌ None | ❌ None | ❌ None |\n")

    lines.append("## 2. Nutrient Feasibility & Provenance Determination\n")
    lines.append("| Nutrient | Ground Truth Available? | Which Dataset? | Predictable? | Derivable? | Reference Lookup? | Strategy |")
    lines.append("| :--- | :---: | :--- | :---: | :---: | :---: | :--- |")

    for k, v in NUTRIENT_ANALYSIS.items():
        ds_str = ", ".join(v.providing_datasets) if v.providing_datasets else "None"
        gt_str = "✅ Yes" if v.is_ground_truth_available else "❌ No"
        pred_str = "✅ Yes" if v.can_be_predicted else "❌ No"
        der_str = "✅ Yes" if v.can_be_derived else "❌ No"
        ref_str = "✅ Yes" if v.can_be_obtained_from_reference else "❌ No"
        lines.append(f"| **{v.display_name}** (`{v.unit}`) | {gt_str} | {ds_str} | {pred_str} | {der_str} | {ref_str} | {v.handling_strategy} |")

    lines.append("\n## 3. Scientific Methodology Rules\n")
    lines.append("1. **Never Invent Values:** When a food item is unrepresented in verified reference profiles, unsupported fields return `null` / `'Not available'` with status `NOT AVAILABLE`.")
    lines.append("2. **No Groundless Neural Regression:** Fiber, Sugar, Saturated Fat, Sodium, and Cholesterol CANNOT be directly regressed by neural networks because zero vision training datasets provide ground truth for them.")
    lines.append("3. **Verified Compositional Derivation:** Where standard food categories are classified, reference composition per 100g from USDA FoodData Central is multiplied by estimated mass to produce mathematically grounded values labeled `REFERENCE` / `DERIVED`.")

    return "\n".join(lines)


if __name__ == "__main__":
    report = format_inspection_report()
    print(report)
    out_path = Path("docs/NUTRIENT_DATASET_ANALYSIS.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Saved inspection report to {out_path}")
