"""
Common internal schema passed between pipeline stages (see ARCHITECTURE.md).

Every numeric value that could come from more than one kind of source
(ground truth, a model prediction, something derived from a prediction, a
static reference-table lookup, or genuinely not available) is wrapped in a
`Value` that records which of those five it is. This is what lets the final
report show real provenance instead of a single flat number, per the
project's "never fabricate nutrition labels" rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class Source(str, Enum):
    GROUND_TRUTH = "ground_truth"
    PREDICTION = "prediction"
    DERIVED = "derived"
    REFERENCE_LOOKUP = "reference_lookup"
    UNAVAILABLE = "unavailable"

    @property
    def label(self) -> str:
        """Returns the formal uppercase provenance label required by project specification."""
        mapping = {
            Source.GROUND_TRUTH: "GROUND TRUTH",
            Source.PREDICTION: "MODEL PREDICTION",
            Source.DERIVED: "DERIVED",
            Source.REFERENCE_LOOKUP: "REFERENCE",
            Source.UNAVAILABLE: "NOT AVAILABLE",
        }
        return mapping[self]

    @classmethod
    def from_label(cls, label: str) -> "Source":
        norm = label.upper().strip()
        if norm in ("MODEL PREDICTION", "PREDICTION"):
            return cls.PREDICTION
        if norm in ("GROUND TRUTH", "GROUND_TRUTH"):
            return cls.GROUND_TRUTH
        if norm in ("DERIVED",):
            return cls.DERIVED
        if norm in ("REFERENCE", "REFERENCE_LOOKUP"):
            return cls.REFERENCE_LOOKUP
        if norm in ("NOT AVAILABLE", "UNAVAILABLE"):
            return cls.UNAVAILABLE
        raise ValueError(f"Unknown provenance label: '{label}'")


@dataclass
class Value:
    """A single scalar with explicit provenance. value is None iff source is UNAVAILABLE."""
    value: Optional[float]
    source: Source

    def __post_init__(self) -> None:
        if self.source == Source.UNAVAILABLE and self.value is not None:
            raise ValueError("A Value marked UNAVAILABLE must have value=None.")
        if self.source != Source.UNAVAILABLE and self.value is None:
            raise ValueError(f"A Value with source={self.source} must have a numeric value.")

    @staticmethod
    def unavailable() -> "Value":
        return Value(value=None, source=Source.UNAVAILABLE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source.value,
            "label": self.source.label,
        }

    def display_str(self, unit: str = "") -> str:
        if self.source == Source.UNAVAILABLE or self.value is None:
            return "Not available"
        unit_str = f" {unit}".rstrip()
        return f"{self.value:.1f}{unit_str}"


@dataclass
class NutrientSet:
    calories: Value
    protein_g: Value
    carbs_g: Value
    fat_g: Value
    fiber_g: Value = field(default_factory=Value.unavailable)
    sugar_g: Value = field(default_factory=Value.unavailable)
    saturated_fat_g: Value = field(default_factory=Value.unavailable)
    sodium_mg: Value = field(default_factory=Value.unavailable)
    cholesterol_mg: Value = field(default_factory=Value.unavailable)
    additional_nutrients: dict[str, Value] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "calories": {"value": self.calories.value, "source": self.calories.source.value, "label": self.calories.source.label},
            "protein_g": {"value": self.protein_g.value, "source": self.protein_g.source.value, "label": self.protein_g.source.label},
            "carbs_g": {"value": self.carbs_g.value, "source": self.carbs_g.source.value, "label": self.carbs_g.source.label},
            "fat_g": {"value": self.fat_g.value, "source": self.fat_g.source.value, "label": self.fat_g.source.label},
        }
        if self.fiber_g.source != Source.UNAVAILABLE:
            data["fiber_g"] = {"value": self.fiber_g.value, "source": self.fiber_g.source.value, "label": self.fiber_g.source.label}
        if self.sugar_g.source != Source.UNAVAILABLE:
            data["sugar_g"] = {"value": self.sugar_g.value, "source": self.sugar_g.source.value, "label": self.sugar_g.source.label}
        if self.saturated_fat_g.source != Source.UNAVAILABLE:
            data["saturated_fat_g"] = {"value": self.saturated_fat_g.value, "source": self.saturated_fat_g.source.value, "label": self.saturated_fat_g.source.label}
        if self.sodium_mg.source != Source.UNAVAILABLE:
            data["sodium_mg"] = {"value": self.sodium_mg.value, "source": self.sodium_mg.source.value, "label": self.sodium_mg.source.label}
        if self.cholesterol_mg.source != Source.UNAVAILABLE:
            data["cholesterol_mg"] = {"value": self.cholesterol_mg.value, "source": self.cholesterol_mg.source.value, "label": self.cholesterol_mg.source.label}
        if self.additional_nutrients:
            data["additional_nutrients"] = {
                k: {"value": v.value, "source": v.source.value, "label": v.source.label}
                for k, v in self.additional_nutrients.items()
            }
        return data


@dataclass
class UnifiedFoodNutrition:
    """
    Unified nutrition representation compliant with project specification:
      food_name: str
      mass_g: Value
      calories_kcal: Value
      protein_g: Value
      carbohydrates_g: Value
      fat_g: Value
      fiber_g: Value
      sugar_g: Value
      saturated_fat_g: Value
      sodium_mg: Value
      cholesterol_mg: Value
      additional_nutrients: dict[str, Value]
      source: Source
      confidence: Optional[float]
    """
    food_name: str
    mass_g: Value
    calories_kcal: Value
    protein_g: Value
    carbohydrates_g: Value
    fat_g: Value
    fiber_g: Value = field(default_factory=Value.unavailable)
    sugar_g: Value = field(default_factory=Value.unavailable)
    saturated_fat_g: Value = field(default_factory=Value.unavailable)
    sodium_mg: Value = field(default_factory=Value.unavailable)
    cholesterol_mg: Value = field(default_factory=Value.unavailable)
    additional_nutrients: dict[str, Value] = field(default_factory=dict)
    source: Source = Source.PREDICTION
    confidence: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        """
        Serializes unified nutrition representation.
        If unsupported, value is null and labeled 'NOT AVAILABLE'.
        """
        def _fmt(val: Value) -> dict[str, Any]:
            return {
                "value": val.value,
                "label": val.source.label,
                "source": val.source.value,
            }

        return {
            "food_name": self.food_name,
            "mass_g": _fmt(self.mass_g),
            "calories_kcal": _fmt(self.calories_kcal),
            "protein_g": _fmt(self.protein_g),
            "carbohydrates_g": _fmt(self.carbohydrates_g),
            "fat_g": _fmt(self.fat_g),
            "fiber_g": _fmt(self.fiber_g),
            "sugar_g": _fmt(self.sugar_g),
            "saturated_fat_g": _fmt(self.saturated_fat_g),
            "sodium_mg": _fmt(self.sodium_mg),
            "cholesterol_mg": _fmt(self.cholesterol_mg),
            "additional_nutrients": {k: _fmt(v) for k, v in self.additional_nutrients.items()},
            "source": self.source.label,
            "confidence": self.confidence,
        }

    def as_display_dict(self) -> dict[str, str]:
        """Human-readable table presentation where unsupported values are 'Not available'."""
        d = {
            "Food Name": self.food_name,
            "Mass": self.mass_g.display_str("g"),
            "Calories": self.calories_kcal.display_str("kcal"),
            "Protein": self.protein_g.display_str("g"),
            "Carbohydrates": self.carbohydrates_g.display_str("g"),
            "Total Fat": self.fat_g.display_str("g"),
            "Dietary Fiber": self.fiber_g.display_str("g"),
            "Total Sugars": self.sugar_g.display_str("g"),
            "Saturated Fat": self.saturated_fat_g.display_str("g"),
            "Sodium": self.sodium_mg.display_str("mg"),
            "Cholesterol": self.cholesterol_mg.display_str("mg"),
            "Primary Source": self.source.label,
            "Confidence": f"{self.confidence * 100:.1f}%" if self.confidence is not None else "Not available",
        }
        for k, v in self.additional_nutrients.items():
            clean_name = k.replace("_mg", "").replace("_g", "").replace("_", " ").title()
            unit = "mg" if k.endswith("_mg") else ("g" if k.endswith("_g") else "")
            d[clean_name] = v.display_str(unit)
        return d

    @classmethod
    def from_food_instance(cls, inst: FoodInstance) -> "UnifiedFoodNutrition":
        """Build UnifiedFoodNutrition from a FoodInstance record."""
        nutrients = inst.nutrients or NutrientSet(
            calories=Value.unavailable(),
            protein_g=Value.unavailable(),
            carbs_g=Value.unavailable(),
            fat_g=Value.unavailable(),
        )
        return cls(
            food_name=inst.class_name or "unknown",
            mass_g=inst.mass,
            calories_kcal=nutrients.calories,
            protein_g=nutrients.protein_g,
            carbohydrates_g=nutrients.carbs_g,
            fat_g=nutrients.fat_g,
            fiber_g=nutrients.fiber_g,
            sugar_g=nutrients.sugar_g,
            saturated_fat_g=nutrients.saturated_fat_g,
            sodium_mg=nutrients.sodium_mg,
            cholesterol_mg=nutrients.cholesterol_mg,
            additional_nutrients=nutrients.additional_nutrients,
            source=nutrients.calories.source if nutrients.calories.source != Source.UNAVAILABLE else inst.mass.source,
            confidence=inst.class_confidence,
        )


@dataclass
class FoodInstance:
    instance_id: str
    bbox: Optional[tuple] = None                 # (x, y, w, h)
    mask: Optional[np.ndarray] = None
    class_name: Optional[str] = None
    class_confidence: Optional[float] = None
    ingredients: list[str] = field(default_factory=list)   # auxiliary, may be empty
    mass: Value = field(default_factory=Value.unavailable)
    nutrients: Optional[NutrientSet] = None
    unified_nutrition: Optional[UnifiedFoodNutrition] = None

    def as_report_dict(self) -> dict:
        return {
            "food": self.class_name or "unknown",
            "confidence": self.class_confidence,
            "mass_g": self.mass.value,
            "mass_source": self.mass.source.value,
            "nutrients": self.nutrients.as_dict() if self.nutrients else None,
            "unified_nutrition": self.unified_nutrition.as_dict() if self.unified_nutrition else None,
        }
