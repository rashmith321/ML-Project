"""
Unit and integration tests for Unified Nutrition Representation,
Reference Database Derivation, and Provenance Enforcement.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nutrition.aggregate import aggregate_meal, format_unified_nutrition_report
from src.nutrition.dataset_inspection import DATASET_CAPABILITY_MATRIX, NUTRIENT_ANALYSIS
from src.nutrition.reference_database import FoodNutritionReferenceDatabase, USDA_REFERENCE_DATABASE
from src.utils.schema import FoodInstance, NutrientSet, Source, UnifiedFoodNutrition, Value


# ---------------------------------------------------------------------------
# 1. Source Provenance Labels & Value Tests
# ---------------------------------------------------------------------------

def test_source_provenance_labels():
    assert Source.PREDICTION.label == "MODEL PREDICTION"
    assert Source.GROUND_TRUTH.label == "GROUND TRUTH"
    assert Source.DERIVED.label == "DERIVED"
    assert Source.REFERENCE_LOOKUP.label == "REFERENCE"
    assert Source.UNAVAILABLE.label == "NOT AVAILABLE"

    assert Source.from_label("MODEL PREDICTION") == Source.PREDICTION
    assert Source.from_label("GROUND TRUTH") == Source.GROUND_TRUTH
    assert Source.from_label("DERIVED") == Source.DERIVED
    assert Source.from_label("REFERENCE") == Source.REFERENCE_LOOKUP
    assert Source.from_label("NOT AVAILABLE") == Source.UNAVAILABLE


def test_value_display_str():
    val_num = Value(value=42.5, source=Source.PREDICTION)
    assert val_num.display_str("g") == "42.5 g"
    assert val_num.display_str("kcal") == "42.5 kcal"

    val_unavail = Value.unavailable()
    assert val_unavail.display_str("g") == "Not available"
    assert val_unavail.value is None
    assert val_unavail.source.label == "NOT AVAILABLE"


# ---------------------------------------------------------------------------
# 2. UnifiedFoodNutrition Schema Tests
# ---------------------------------------------------------------------------

def test_unified_food_nutrition_schema_complete():
    n_set = NutrientSet(
        calories=Value(value=250.0, source=Source.PREDICTION),
        protein_g=Value(value=15.0, source=Source.PREDICTION),
        carbs_g=Value(value=30.0, source=Source.PREDICTION),
        fat_g=Value(value=8.0, source=Source.PREDICTION),
        fiber_g=Value(value=3.5, source=Source.DERIVED),
        sugar_g=Value(value=5.0, source=Source.DERIVED),
        saturated_fat_g=Value(value=2.0, source=Source.DERIVED),
        sodium_mg=Value(value=400.0, source=Source.DERIVED),
        cholesterol_mg=Value(value=25.0, source=Source.DERIVED),
        additional_nutrients={"potassium_mg": Value(value=250.0, source=Source.DERIVED)},
    )

    inst = FoodInstance(
        instance_id="item_1",
        class_name="pizza",
        class_confidence=0.95,
        mass=Value(value=200.0, source=Source.DERIVED),
        nutrients=n_set,
    )

    unified = UnifiedFoodNutrition.from_food_instance(inst)
    assert unified.food_name == "pizza"
    assert unified.calories_kcal.value == 250.0
    assert unified.fiber_g.value == 3.5
    assert unified.fiber_g.source.label == "DERIVED"

    # Test serialization
    d = unified.as_dict()
    assert d["food_name"] == "pizza"
    assert d["calories_kcal"]["value"] == 250.0
    assert d["calories_kcal"]["label"] == "MODEL PREDICTION"
    assert d["fiber_g"]["value"] == 3.5
    assert d["fiber_g"]["label"] == "DERIVED"
    assert d["sodium_mg"]["value"] == 400.0
    assert "potassium_mg" in d["additional_nutrients"]

    # Test display table format
    display = unified.as_display_dict()
    assert display["Food Name"] == "pizza"
    assert display["Calories"] == "250.0 kcal"
    assert display["Dietary Fiber"] == "3.5 g"
    assert display["Sodium"] == "400.0 mg"
    assert display["Primary Source"] == "MODEL PREDICTION"


def test_unified_food_nutrition_unsupported_not_available():
    """Unsupported values must return null and 'Not available'."""
    inst = FoodInstance(
        instance_id="item_unsupported",
        class_name="exotic_foraged_root",
        mass=Value(value=100.0, source=Source.PREDICTION),
        nutrients=NutrientSet(
            calories=Value(value=80.0, source=Source.PREDICTION),
            protein_g=Value(value=2.0, source=Source.PREDICTION),
            carbs_g=Value(value=18.0, source=Source.PREDICTION),
            fat_g=Value(value=0.5, source=Source.PREDICTION),
            # Extended nutrients left as default Value.unavailable()
        ),
    )

    unified = UnifiedFoodNutrition.from_food_instance(inst)
    assert unified.fiber_g.value is None
    assert unified.fiber_g.source == Source.UNAVAILABLE
    assert unified.fiber_g.source.label == "NOT AVAILABLE"

    d = unified.as_dict()
    assert d["fiber_g"]["value"] is None
    assert d["fiber_g"]["label"] == "NOT AVAILABLE"
    assert d["sodium_mg"]["value"] is None
    assert d["sodium_mg"]["label"] == "NOT AVAILABLE"

    display = unified.as_display_dict()
    assert display["Dietary Fiber"] == "Not available"
    assert display["Sodium"] == "Not available"


# ---------------------------------------------------------------------------
# 3. Verified Reference Database & Non-Fabrication Tests
# ---------------------------------------------------------------------------

def test_reference_database_lookup():
    ref_db = FoodNutritionReferenceDatabase()

    # Mapped foods exist
    profile = ref_db.lookup("pizza")
    assert profile is not None
    assert profile.calories_kcal == 266.0
    assert profile.sodium_mg == 598.0
    assert profile.fiber_g == 2.3

    profile_salad = ref_db.lookup("Caesar Salad")
    assert profile_salad is not None
    assert profile_salad.food_name == "caesar_salad"

    # Unmapped food returns None
    assert ref_db.lookup("unobtainium_berry") is None


def test_reference_database_derivation_scaling():
    ref_db = FoodNutritionReferenceDatabase()

    # Derive for 200g of pizza (double the 100g reference values)
    nutrients = ref_db.derive_nutrients_for_instance(
        food_name="pizza",
        mass_g=200.0,
        predicted_calories=500.0,  # model predicted 500 kcal
        predicted_protein=22.0,
        predicted_carbs=64.0,
        predicted_fat=20.0,
    )

    # Core macronutrients retain MODEL PREDICTION
    assert nutrients.calories.value == 500.0
    assert nutrients.calories.source == Source.PREDICTION
    assert nutrients.calories.source.label == "MODEL PREDICTION"

    # Extended nutrients scaled by mass (200g / 100g = 2.0x)
    assert nutrients.fiber_g.value == pytest.approx(2.3 * 2.0, abs=0.1)
    assert nutrients.fiber_g.source == Source.DERIVED
    assert nutrients.fiber_g.source.label == "DERIVED"

    assert nutrients.sodium_mg.value == pytest.approx(598.0 * 2.0, abs=0.1)
    assert nutrients.sodium_mg.source == Source.DERIVED

    assert nutrients.cholesterol_mg.value == pytest.approx(17.0 * 2.0, abs=0.1)


def test_reference_database_zero_fabrication_on_unmapped():
    """Never invent values for unmapped food classes."""
    ref_db = FoodNutritionReferenceDatabase()

    nutrients = ref_db.derive_nutrients_for_instance(
        food_name="alien_mushroom",
        mass_g=150.0,
        predicted_calories=75.0,
        predicted_protein=3.0,
        predicted_carbs=12.0,
        predicted_fat=1.0,
    )

    # Core predictions preserved
    assert nutrients.calories.value == 75.0
    assert nutrients.calories.source == Source.PREDICTION

    # Extended nutrients strictly NOT AVAILABLE (None)
    assert nutrients.fiber_g.value is None
    assert nutrients.fiber_g.source == Source.UNAVAILABLE
    assert nutrients.fiber_g.source.label == "NOT AVAILABLE"

    assert nutrients.sugar_g.value is None
    assert nutrients.sodium_mg.value is None
    assert nutrients.cholesterol_mg.value is None


# ---------------------------------------------------------------------------
# 4. Extended Meal-Level Aggregation Tests
# ---------------------------------------------------------------------------

def test_aggregate_meal_with_extended_nutrients():
    ref_db = FoodNutritionReferenceDatabase()

    inst1 = FoodInstance(
        instance_id="food_1",
        class_name="pizza",
        mass=Value(value=100.0, source=Source.PREDICTION),
        nutrients=ref_db.derive_nutrients_for_instance("pizza", mass_g=100.0, predicted_calories=266.0, predicted_protein=11.4, predicted_carbs=33.3, predicted_fat=9.7),
    )

    inst2 = FoodInstance(
        instance_id="food_2",
        class_name="caesar_salad",
        mass=Value(value=100.0, source=Source.PREDICTION),
        nutrients=ref_db.derive_nutrients_for_instance("caesar_salad", mass_g=100.0, predicted_calories=127.0, predicted_protein=5.4, predicted_carbs=8.0, predicted_fat=9.0),
    )

    meal_total = aggregate_meal([inst1, inst2])

    # Core sums
    assert meal_total.calories.value == pytest.approx(266.0 + 127.0, abs=0.2)
    assert meal_total.protein_g.value == pytest.approx(11.4 + 5.4, abs=0.2)
    assert meal_total.carbs_g.value == pytest.approx(33.3 + 8.0, abs=0.2)
    assert meal_total.fat_g.value == pytest.approx(9.7 + 9.0, abs=0.2)

    # Extended sums
    assert meal_total.fiber_g.value == pytest.approx(2.3 + 2.1, abs=0.2)
    assert meal_total.fiber_g.source.label == "DERIVED"
    assert meal_total.sodium_mg.value == pytest.approx(598.0 + 340.0, abs=0.2)
    assert meal_total.cholesterol_mg.value == pytest.approx(17.0 + 12.0, abs=0.2)
    assert len(meal_total.partial_nutrients) == 0


def test_aggregate_meal_partial_extended_nutrients():
    """If one food is missing an extended nutrient, total is marked PARTIAL."""
    ref_db = FoodNutritionReferenceDatabase()

    inst1 = FoodInstance(
        instance_id="food_1",
        class_name="pizza",
        mass=Value(value=100.0, source=Source.PREDICTION),
        nutrients=ref_db.derive_nutrients_for_instance("pizza", mass_g=100.0, predicted_calories=266.0, predicted_protein=11.4, predicted_carbs=33.3, predicted_fat=9.7),
    )

    # Food 2 is unmapped: fiber, sodium, etc. are unavailable
    inst2 = FoodInstance(
        instance_id="food_2",
        class_name="alien_mushroom",
        mass=Value(value=100.0, source=Source.PREDICTION),
        nutrients=ref_db.derive_nutrients_for_instance("alien_mushroom", mass_g=100.0, predicted_calories=50.0, predicted_protein=2.0, predicted_carbs=10.0, predicted_fat=0.5),
    )

    meal_total = aggregate_meal([inst1, inst2])
    assert meal_total.calories.value == pytest.approx(316.0, abs=0.2)
    assert "calories" not in meal_total.partial_nutrients

    # Extended nutrients must be flagged as PARTIAL because inst2 was unavailable
    assert "fiber_g" in meal_total.partial_nutrients
    assert "sodium_mg" in meal_total.partial_nutrients
    assert "cholesterol_mg" in meal_total.partial_nutrients


def test_format_unified_nutrition_report():
    ref_db = FoodNutritionReferenceDatabase()
    inst = FoodInstance(
        instance_id="food_1",
        class_name="pizza",
        class_confidence=0.92,
        mass=Value(value=150.0, source=Source.PREDICTION),
        nutrients=ref_db.derive_nutrients_for_instance("pizza", mass_g=150.0, predicted_calories=400.0, predicted_protein=17.0, predicted_carbs=50.0, predicted_fat=14.5),
    )
    meal_total = aggregate_meal([inst])
    report = format_unified_nutrition_report([inst], meal_total)

    assert "UNIFIED MULTI-NUTRIENT INTELLIGENCE REPORT" in report
    assert "Pizza" in report
    assert "Dietary Fiber" in report
    assert "Total Sugars" in report
    assert "Saturated Fat" in report
    assert "Sodium" in report
    assert "Cholesterol" in report
    assert "MODEL PREDICTION" in report
    assert "DERIVED" in report
    assert "TOTAL MEAL ROLLUP" in report


# ---------------------------------------------------------------------------
# 5. Dataset Inspection Feasibility Matrix Tests
# ---------------------------------------------------------------------------

def test_dataset_feasibility_analysis():
    assert "Nutrition5k" in DATASET_CAPABILITY_MATRIX
    assert DATASET_CAPABILITY_MATRIX["Nutrition5k"]["annotations_present"] == [
        "mass_g", "calories_kcal", "protein_g", "carbohydrates_g", "fat_g", "depth_map", "ingredient_breakdown"
    ]

    # Verify that fiber, sugar, sodium are confirmed unavailable in raw vision datasets
    assert not NUTRIENT_ANALYSIS["fiber_g"].is_ground_truth_available
    assert not NUTRIENT_ANALYSIS["sugar_g"].is_ground_truth_available
    assert not NUTRIENT_ANALYSIS["sodium_mg"].is_ground_truth_available
    assert not NUTRIENT_ANALYSIS["cholesterol_mg"].is_ground_truth_available
    assert not NUTRIENT_ANALYSIS["saturated_fat_g"].is_ground_truth_available

    # But confirmed derivable via verified reference lookup
    assert NUTRIENT_ANALYSIS["fiber_g"].can_be_derived
    assert NUTRIENT_ANALYSIS["sodium_mg"].can_be_derived
    assert NUTRIENT_ANALYSIS["fiber_g"].can_be_obtained_from_reference
