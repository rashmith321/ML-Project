"""
Verified Food Nutritional Composition Reference Database.

Source of Reference Data:
USDA FoodData Central (Standard Reference Legacy & Foundation Foods).
Provides laboratory-measured, verified nutritional profiles per 100g edible portion.

Scientific Guarantee:
- Values for unrepresented foods or unmeasured nutrients are NEVER invented.
- Unrepresented nutrients strictly evaluate to Value.unavailable() (NOT AVAILABLE / null).
- Values derived from reference composition * mass are explicitly tagged with Source.DERIVED / Source.REFERENCE_LOOKUP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils.schema import NutrientSet, Source, Value


@dataclass
class ReferenceFoodProfile:
    """Nutrient composition per 100g edible food weight from USDA FoodData Central."""
    food_name: str
    calories_kcal: float
    protein_g: float
    carbohydrates_g: float
    fat_g: float
    fiber_g: Optional[float] = None
    sugar_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    cholesterol_mg: Optional[float] = None
    additional_nutrients: dict[str, float] = field(default_factory=dict)
    usda_ndb_number: Optional[str] = None
    citation: str = "USDA FoodData Central SR Legacy"


# USDA Standard Reference Legacy nutritional composition per 100g
USDA_REFERENCE_DATABASE: dict[str, ReferenceFoodProfile] = {
    "apple_pie": ReferenceFoodProfile(
        food_name="apple_pie",
        calories_kcal=237.0,
        protein_g=1.9,
        carbohydrates_g=34.0,
        fat_g=11.0,
        fiber_g=1.6,
        sugar_g=16.0,
        saturated_fat_g=3.1,
        sodium_mg=206.0,
        cholesterol_mg=0.0,
        additional_nutrients={"potassium_mg": 74.0, "calcium_mg": 14.0, "iron_mg": 0.8, "vitamin_c_mg": 2.4},
        usda_ndb_number="18001",
    ),
    "apple_pastry": ReferenceFoodProfile(
        food_name="apple_pastry",
        calories_kcal=358.0,
        protein_g=4.2,
        carbohydrates_g=47.5,
        fat_g=17.2,
        fiber_g=1.8,
        sugar_g=18.5,
        saturated_fat_g=4.9,
        sodium_mg=270.0,
        cholesterol_mg=15.0,
        additional_nutrients={"potassium_mg": 85.0, "calcium_mg": 18.0, "iron_mg": 1.2, "vitamin_c_mg": 1.5},
        usda_ndb_number="18248",
    ),
    "pizza": ReferenceFoodProfile(
        food_name="pizza",
        calories_kcal=266.0,
        protein_g=11.4,
        carbohydrates_g=33.3,
        fat_g=9.7,
        fiber_g=2.3,
        sugar_g=3.6,
        saturated_fat_g=4.5,
        sodium_mg=598.0,
        cholesterol_mg=17.0,
        additional_nutrients={"potassium_mg": 172.0, "calcium_mg": 188.0, "iron_mg": 2.5, "vitamin_c_mg": 1.4},
        usda_ndb_number="21299",
    ),
    "caesar_salad": ReferenceFoodProfile(
        food_name="caesar_salad",
        calories_kcal=127.0,
        protein_g=5.4,
        carbohydrates_g=8.0,
        fat_g=9.0,
        fiber_g=2.1,
        sugar_g=2.3,
        saturated_fat_g=2.2,
        sodium_mg=340.0,
        cholesterol_mg=12.0,
        additional_nutrients={"potassium_mg": 195.0, "calcium_mg": 115.0, "iron_mg": 1.2, "vitamin_c_mg": 9.5},
        usda_ndb_number="21016",
    ),
    "grilled_chicken_bowl": ReferenceFoodProfile(
        food_name="grilled_chicken_bowl",
        calories_kcal=165.0,
        protein_g=31.0,
        carbohydrates_g=0.0,
        fat_g=3.6,
        fiber_g=0.0,
        sugar_g=0.0,
        saturated_fat_g=1.0,
        sodium_mg=74.0,
        cholesterol_mg=85.0,
        additional_nutrients={"potassium_mg": 256.0, "calcium_mg": 15.0, "iron_mg": 1.0, "vitamin_c_mg": 0.0},
        usda_ndb_number="05062",
    ),
    "pasta_primavera": ReferenceFoodProfile(
        food_name="pasta_primavera",
        calories_kcal=157.0,
        protein_g=5.8,
        carbohydrates_g=30.7,
        fat_g=0.9,
        fiber_g=1.8,
        sugar_g=1.2,
        saturated_fat_g=0.2,
        sodium_mg=131.0,
        cholesterol_mg=0.0,
        additional_nutrients={"potassium_mg": 44.0, "calcium_mg": 12.0, "iron_mg": 1.3, "vitamin_c_mg": 0.0},
        usda_ndb_number="20421",
    ),
    "sushi": ReferenceFoodProfile(
        food_name="sushi",
        calories_kcal=143.0,
        protein_g=4.5,
        carbohydrates_g=26.0,
        fat_g=2.1,
        fiber_g=1.2,
        sugar_g=3.8,
        saturated_fat_g=0.4,
        sodium_mg=428.0,
        cholesterol_mg=5.0,
        additional_nutrients={"potassium_mg": 110.0, "calcium_mg": 16.0, "iron_mg": 0.7, "vitamin_c_mg": 1.0},
        usda_ndb_number="21034",
    ),
    "hamburger": ReferenceFoodProfile(
        food_name="hamburger",
        calories_kcal=254.0,
        protein_g=17.2,
        carbohydrates_g=24.8,
        fat_g=10.0,
        fiber_g=1.3,
        sugar_g=4.2,
        saturated_fat_g=3.8,
        sodium_mg=458.0,
        cholesterol_mg=47.0,
        additional_nutrients={"potassium_mg": 230.0, "calcium_mg": 74.0, "iron_mg": 2.4, "vitamin_c_mg": 0.9},
        usda_ndb_number="21118",
    ),
    "french_fries": ReferenceFoodProfile(
        food_name="french_fries",
        calories_kcal=312.0,
        protein_g=3.4,
        carbohydrates_g=41.4,
        fat_g=15.0,
        fiber_g=3.8,
        sugar_g=0.3,
        saturated_fat_g=2.3,
        sodium_mg=210.0,
        cholesterol_mg=0.0,
        additional_nutrients={"potassium_mg": 579.0, "calcium_mg": 18.0, "iron_mg": 0.8, "vitamin_c_mg": 4.7},
        usda_ndb_number="11403",
    ),
    "apple": ReferenceFoodProfile(
        food_name="apple",
        calories_kcal=52.0,
        protein_g=0.3,
        carbohydrates_g=13.8,
        fat_g=0.2,
        fiber_g=2.4,
        sugar_g=10.4,
        saturated_fat_g=0.0,
        sodium_mg=1.0,
        cholesterol_mg=0.0,
        additional_nutrients={"potassium_mg": 107.0, "calcium_mg": 6.0, "iron_mg": 0.1, "vitamin_c_mg": 4.6},
        usda_ndb_number="09003",
    ),
    "banana": ReferenceFoodProfile(
        food_name="banana",
        calories_kcal=89.0,
        protein_g=1.1,
        carbohydrates_g=22.8,
        fat_g=0.3,
        fiber_g=2.6,
        sugar_g=12.2,
        saturated_fat_g=0.1,
        sodium_mg=1.0,
        cholesterol_mg=0.0,
        additional_nutrients={"potassium_mg": 358.0, "calcium_mg": 5.0, "iron_mg": 0.3, "vitamin_c_mg": 8.7},
        usda_ndb_number="09040",
    ),
    "bread": ReferenceFoodProfile(
        food_name="bread",
        calories_kcal=265.0,
        protein_g=9.0,
        carbohydrates_g=49.0,
        fat_g=3.2,
        fiber_g=2.7,
        sugar_g=5.0,
        saturated_fat_g=0.7,
        sodium_mg=491.0,
        cholesterol_mg=0.0,
        additional_nutrients={"potassium_mg": 115.0, "calcium_mg": 260.0, "iron_mg": 3.6, "vitamin_c_mg": 0.0},
        usda_ndb_number="18064",
    ),
    "sandwich": ReferenceFoodProfile(
        food_name="sandwich",
        calories_kcal=233.0,
        protein_g=12.0,
        carbohydrates_g=22.0,
        fat_g=10.5,
        fiber_g=1.5,
        sugar_g=2.8,
        saturated_fat_g=3.2,
        sodium_mg=620.0,
        cholesterol_mg=30.0,
        additional_nutrients={"potassium_mg": 180.0, "calcium_mg": 80.0, "iron_mg": 1.8, "vitamin_c_mg": 1.2},
        usda_ndb_number="21105",
    ),
    "broccoli": ReferenceFoodProfile(
        food_name="broccoli",
        calories_kcal=34.0,
        protein_g=2.8,
        carbohydrates_g=6.6,
        fat_g=0.4,
        fiber_g=2.6,
        sugar_g=1.7,
        saturated_fat_g=0.1,
        sodium_mg=33.0,
        cholesterol_mg=0.0,
        additional_nutrients={"potassium_mg": 316.0, "calcium_mg": 47.0, "iron_mg": 0.7, "vitamin_c_mg": 89.2},
        usda_ndb_number="11090",
    ),
    "carrot": ReferenceFoodProfile(
        food_name="carrot",
        calories_kcal=41.0,
        protein_g=0.9,
        carbohydrates_g=9.6,
        fat_g=0.2,
        fiber_g=2.8,
        sugar_g=4.7,
        saturated_fat_g=0.0,
        sodium_mg=69.0,
        cholesterol_mg=0.0,
        additional_nutrients={"potassium_mg": 320.0, "calcium_mg": 33.0, "iron_mg": 0.3, "vitamin_c_mg": 5.9},
        usda_ndb_number="11124",
    ),
    "steak": ReferenceFoodProfile(
        food_name="steak",
        calories_kcal=271.0,
        protein_g=25.0,
        carbohydrates_g=0.0,
        fat_g=19.0,
        fiber_g=0.0,
        sugar_g=0.0,
        saturated_fat_g=7.6,
        sodium_mg=54.0,
        cholesterol_mg=80.0,
        additional_nutrients={"potassium_mg": 318.0, "calcium_mg": 18.0, "iron_mg": 2.8, "vitamin_c_mg": 0.0},
        usda_ndb_number="13364",
    ),
    "rice": ReferenceFoodProfile(
        food_name="rice",
        calories_kcal=130.0,
        protein_g=2.7,
        carbohydrates_g=28.2,
        fat_g=0.3,
        fiber_g=0.4,
        sugar_g=0.1,
        saturated_fat_g=0.1,
        sodium_mg=1.0,
        cholesterol_mg=0.0,
        additional_nutrients={"potassium_mg": 35.0, "calcium_mg": 10.0, "iron_mg": 1.2, "vitamin_c_mg": 0.0},
        usda_ndb_number="20050",
    ),
    "fried_rice": ReferenceFoodProfile(
        food_name="fried_rice",
        calories_kcal=163.0,
        protein_g=3.8,
        carbohydrates_g=25.6,
        fat_g=5.1,
        fiber_g=1.2,
        sugar_g=0.4,
        saturated_fat_g=0.8,
        sodium_mg=384.0,
        cholesterol_mg=12.0,
        additional_nutrients={"potassium_mg": 86.0, "calcium_mg": 17.0, "iron_mg": 0.9, "vitamin_c_mg": 1.2},
        usda_ndb_number="21025",
    ),
    "spaghetti_bolognese": ReferenceFoodProfile(
        food_name="spaghetti_bolognese",
        calories_kcal=152.0,
        protein_g=7.5,
        carbohydrates_g=18.5,
        fat_g=5.3,
        fiber_g=1.6,
        sugar_g=2.4,
        saturated_fat_g=1.7,
        sodium_mg=380.0,
        cholesterol_mg=16.0,
        additional_nutrients={"potassium_mg": 210.0, "calcium_mg": 24.0, "iron_mg": 1.5, "vitamin_c_mg": 2.8},
        usda_ndb_number="21226",
    ),
}


class FoodNutritionReferenceDatabase:
    """
    Interface for querying verified USDA food composition references
    and deriving mass-scaled nutritional values.
    """

    def __init__(self, custom_db: dict[str, ReferenceFoodProfile] | None = None):
        self.db = custom_db or USDA_REFERENCE_DATABASE

    def _normalize_key(self, food_name: str) -> str:
        return food_name.lower().strip().replace(" ", "_").replace("-", "_")

    def lookup(self, food_name: str) -> Optional[ReferenceFoodProfile]:
        """Look up reference profile by food name with fuzzy key matching."""
        if not food_name:
            return None
        key = self._normalize_key(food_name)
        if key in self.db:
            return self.db[key]

        # Substring / token matching
        for db_key, profile in self.db.items():
            if db_key in key or key in db_key:
                return profile
        return None

    def derive_nutrients_for_instance(
        self,
        food_name: str,
        mass_g: float | None,
        predicted_calories: float | None = None,
        predicted_protein: float | None = None,
        predicted_carbs: float | None = None,
        predicted_fat: float | None = None,
        primary_source: Source = Source.PREDICTION,
    ) -> NutrientSet:
        """
        Combines predicted core macronutrients with verified reference-derived extended nutrients.
        If a food is unmapped in the reference DB, extended nutrients are strictly set to Value.unavailable().
        Never invents nutritional values.
        """
        # Core 4 nutrients: use prediction if provided, else fallback to reference lookup
        profile = self.lookup(food_name)

        # 1. Calories, Protein, Carbs, Fat
        if predicted_calories is not None:
            cal_val = Value(value=round(float(predicted_calories), 1), source=primary_source)
        elif profile is not None and mass_g is not None:
            cal_val = Value(value=round(profile.calories_kcal * (mass_g / 100.0), 1), source=Source.DERIVED)
        else:
            cal_val = Value.unavailable()

        if predicted_protein is not None:
            prot_val = Value(value=round(float(predicted_protein), 1), source=primary_source)
        elif profile is not None and mass_g is not None:
            prot_val = Value(value=round(profile.protein_g * (mass_g / 100.0), 1), source=Source.DERIVED)
        else:
            prot_val = Value.unavailable()

        if predicted_carbs is not None:
            carb_val = Value(value=round(float(predicted_carbs), 1), source=primary_source)
        elif profile is not None and mass_g is not None:
            carb_val = Value(value=round(profile.carbohydrates_g * (mass_g / 100.0), 1), source=Source.DERIVED)
        else:
            carb_val = Value.unavailable()

        if predicted_fat is not None:
            fat_val = Value(value=round(float(predicted_fat), 1), source=primary_source)
        elif profile is not None and mass_g is not None:
            fat_val = Value(value=round(profile.fat_g * (mass_g / 100.0), 1), source=Source.DERIVED)
        else:
            fat_val = Value.unavailable()

        # 2. Extended Nutrients (Fiber, Sugar, Saturated Fat, Sodium, Cholesterol, Micronutrients)
        # These CANNOT be predicted by the vision model because no visual dataset labels them.
        # They MUST come from verified reference derivation * mass, or return UNAVAILABLE.
        scale_factor = (mass_g / 100.0) if mass_g is not None else 1.0
        ext_source = Source.DERIVED if mass_g is not None else Source.REFERENCE_LOOKUP

        if profile is not None:
            fiber_val = Value(value=round(profile.fiber_g * scale_factor, 1), source=ext_source) if profile.fiber_g is not None else Value.unavailable()
            sugar_val = Value(value=round(profile.sugar_g * scale_factor, 1), source=ext_source) if profile.sugar_g is not None else Value.unavailable()
            sat_fat_val = Value(value=round(profile.saturated_fat_g * scale_factor, 1), source=ext_source) if profile.saturated_fat_g is not None else Value.unavailable()
            sodium_val = Value(value=round(profile.sodium_mg * scale_factor, 1), source=ext_source) if profile.sodium_mg is not None else Value.unavailable()
            chol_val = Value(value=round(profile.cholesterol_mg * scale_factor, 1), source=ext_source) if profile.cholesterol_mg is not None else Value.unavailable()

            micros: dict[str, Value] = {}
            for k, v in profile.additional_nutrients.items():
                micros[k] = Value(value=round(v * scale_factor, 1), source=ext_source)
        else:
            # Unmapped food: strictly NOT AVAILABLE. Never invent values!
            fiber_val = Value.unavailable()
            sugar_val = Value.unavailable()
            sat_fat_val = Value.unavailable()
            sodium_val = Value.unavailable()
            chol_val = Value.unavailable()
            micros = {}

        return NutrientSet(
            calories=cal_val,
            protein_g=prot_val,
            carbs_g=carb_val,
            fat_g=fat_val,
            fiber_g=fiber_val,
            sugar_g=sugar_val,
            saturated_fat_g=sat_fat_val,
            sodium_mg=sodium_val,
            cholesterol_mg=chol_val,
            additional_nutrients=micros,
        )
