"""
Meal-level nutrition aggregation.

Sums per-instance nutrient Values into a meal total, and rolls up their
provenance: a meal total is reported as "derived" if contributing instance
values were present; if any contributing food's nutrient value was unavailable,
the total for that nutrient is flagged as PARTIAL rather than presented as a
complete number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils.schema import FoodInstance, NutrientSet, Source, Value

CORE_NUTRIENT_FIELDS = ("calories", "protein_g", "carbs_g", "fat_g")
NUTRIENT_FIELDS = CORE_NUTRIENT_FIELDS  # Backward compatibility alias
EXTENDED_NUTRIENT_FIELDS = ("fiber_g", "sugar_g", "saturated_fat_g", "sodium_mg", "cholesterol_mg")
ALL_NUTRIENT_FIELDS = CORE_NUTRIENT_FIELDS + EXTENDED_NUTRIENT_FIELDS


@dataclass
class MealTotal:
    calories: Value
    protein_g: Value
    carbs_g: Value
    fat_g: Value
    partial_nutrients: list[str] = field(default_factory=list)   # nutrient names where >=1 contributing food was unavailable
    fiber_g: Value = field(default_factory=Value.unavailable)
    sugar_g: Value = field(default_factory=Value.unavailable)
    saturated_fat_g: Value = field(default_factory=Value.unavailable)
    sodium_mg: Value = field(default_factory=Value.unavailable)
    cholesterol_mg: Value = field(default_factory=Value.unavailable)
    additional_nutrients: dict[str, Value] = field(default_factory=dict)
    mass_g: Value = field(default_factory=Value.unavailable)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "calories": {"value": self.calories.value, "source": self.calories.source.value, "label": self.calories.source.label},
            "protein_g": {"value": self.protein_g.value, "source": self.protein_g.source.value, "label": self.protein_g.source.label},
            "carbs_g": {"value": self.carbs_g.value, "source": self.carbs_g.source.value, "label": self.carbs_g.source.label},
            "fat_g": {"value": self.fat_g.value, "source": self.fat_g.source.value, "label": self.fat_g.source.label},
            "partial_nutrients": self.partial_nutrients,
            "total_mass_g": self.mass_g.value if self.mass_g.value is not None else 0.0,
            "total_calories": self.calories.value if self.calories.value is not None else 0.0,
            "total_protein": self.protein_g.value if self.protein_g.value is not None else 0.0,
            "total_carbohydrates": self.carbs_g.value if self.carbs_g.value is not None else 0.0,
            "total_fat": self.fat_g.value if self.fat_g.value is not None else 0.0,
        }
        if self.mass_g.source != Source.UNAVAILABLE:
            d["mass_g"] = {"value": self.mass_g.value, "source": self.mass_g.source.value, "label": self.mass_g.source.label}
        if self.fiber_g.source != Source.UNAVAILABLE:
            d["fiber_g"] = {"value": self.fiber_g.value, "source": self.fiber_g.source.value, "label": self.fiber_g.source.label}
            d["total_fiber_g"] = self.fiber_g.value
        if self.sugar_g.source != Source.UNAVAILABLE:
            d["sugar_g"] = {"value": self.sugar_g.value, "source": self.sugar_g.source.value, "label": self.sugar_g.source.label}
            d["total_sugar_g"] = self.sugar_g.value
        if self.saturated_fat_g.source != Source.UNAVAILABLE:
            d["saturated_fat_g"] = {"value": self.saturated_fat_g.value, "source": self.saturated_fat_g.source.value, "label": self.saturated_fat_g.source.label}
            d["total_saturated_fat_g"] = self.saturated_fat_g.value
        if self.sodium_mg.source != Source.UNAVAILABLE:
            d["sodium_mg"] = {"value": self.sodium_mg.value, "source": self.sodium_mg.source.value, "label": self.sodium_mg.source.label}
            d["total_sodium_mg"] = self.sodium_mg.value
        if self.cholesterol_mg.source != Source.UNAVAILABLE:
            d["cholesterol_mg"] = {"value": self.cholesterol_mg.value, "source": self.cholesterol_mg.source.value, "label": self.cholesterol_mg.source.label}
            d["total_cholesterol_mg"] = self.cholesterol_mg.value
        if self.additional_nutrients:
            d["additional_nutrients"] = {
                k: {"value": v.value, "source": v.source.value, "label": v.source.label}
                for k, v in self.additional_nutrients.items()
            }
        return d


def _sum_nutrient(instances: list[FoodInstance], field_name: str) -> tuple[Value, bool]:
    """Sum one nutrient field across instances. Returns (Value, is_partial)."""
    total = 0.0
    any_present = False
    any_missing = False

    for inst in instances:
        if inst.nutrients is None:
            any_missing = True
            continue
        v: Value = getattr(inst.nutrients, field_name, Value.unavailable())
        if v.source == Source.UNAVAILABLE or v.value is None:
            any_missing = True
        else:
            total += v.value
            any_present = True

    if not any_present:
        return Value.unavailable(), True
    # Total is a DERIVED value (a sum), regardless of the sources it was built from.
    return Value(value=round(total, 2), source=Source.DERIVED), any_missing


def _sum_additional_nutrients(instances: list[FoodInstance]) -> tuple[dict[str, Value], list[str]]:
    """Sums dynamic additional nutrients (e.g. potassium_mg, iron_mg) across instances."""
    keys: set[str] = set()
    for inst in instances:
        if inst.nutrients and inst.nutrients.additional_nutrients:
            keys.update(inst.nutrients.additional_nutrients.keys())

    totals: dict[str, Value] = {}
    partial_keys: list[str] = []

    for k in keys:
        total = 0.0
        any_present = False
        any_missing = False
        for inst in instances:
            if not inst.nutrients or k not in inst.nutrients.additional_nutrients:
                any_missing = True
                continue
            v = inst.nutrients.additional_nutrients[k]
            if v.source == Source.UNAVAILABLE or v.value is None:
                any_missing = True
            else:
                total += v.value
                any_present = True

        if any_present:
            totals[k] = Value(value=round(total, 2), source=Source.DERIVED)
            if any_missing:
                partial_keys.append(k)

    return totals, partial_keys


def aggregate_meal(instances: list[FoodInstance]) -> MealTotal:
    """
    Build a MealTotal from a list of per-food FoodInstance records.

    Raises ValueError on an empty instance list — an empty meal isn't a
    meaningful total and callers should handle that case explicitly rather
    than silently getting a zero-calorie report.
    """
    if not instances:
        raise ValueError("Cannot aggregate an empty list of FoodInstance records.")

    totals: dict[str, Value] = {}
    partial: list[str] = []

    # Sum core and extended fields
    for field_name in ALL_NUTRIENT_FIELDS:
        value, is_partial = _sum_nutrient(instances, field_name)
        totals[field_name] = value
        if is_partial and value.source != Source.UNAVAILABLE:
            partial.append(field_name)

    # Sum mass across instances
    total_mass = 0.0
    any_mass_present = False
    any_mass_missing = False
    for inst in instances:
        if inst.mass and inst.mass.source != Source.UNAVAILABLE and inst.mass.value is not None:
            total_mass += inst.mass.value
            any_mass_present = True
        else:
            any_mass_missing = True
    if any_mass_present:
        mass_val = Value(value=round(total_mass, 2), source=Source.DERIVED)
        if any_mass_missing:
            partial.append("mass_g")
    else:
        mass_val = Value.unavailable()

    # Sum additional micronutrients
    add_totals, add_partials = _sum_additional_nutrients(instances)
    partial.extend(add_partials)

    return MealTotal(
        calories=totals["calories"],
        protein_g=totals["protein_g"],
        carbs_g=totals["carbs_g"],
        fat_g=totals["fat_g"],
        fiber_g=totals["fiber_g"],
        sugar_g=totals["sugar_g"],
        saturated_fat_g=totals["saturated_fat_g"],
        sodium_mg=totals["sodium_mg"],
        cholesterol_mg=totals["cholesterol_mg"],
        additional_nutrients=add_totals,
        partial_nutrients=partial,
        mass_g=mass_val,
    )


def format_report(instances: list[FoodInstance], meal_total: MealTotal) -> str:
    """Render a plain-text report matching the format in the project brief."""
    lines: list[str] = []
    for inst in instances:
        lines.append(f"Detected Food:\n{inst.class_name or 'Unknown'}\n")
        mass_str = f"{inst.mass.value:.0f} g" if inst.mass.value is not None else "unavailable"
        lines.append(f"Estimated Mass:\n{mass_str}\n")
        if inst.nutrients:
            for label, val in (
                ("Calories", inst.nutrients.calories),
                ("Protein", inst.nutrients.protein_g),
                ("Carbohydrates", inst.nutrients.carbs_g),
                ("Fat", inst.nutrients.fat_g),
            ):
                unit = " kcal" if label == "Calories" else " g"
                shown = f"{val.value:.1f}{unit}" if val.value is not None else "unavailable"
                lines.append(f"{label}:\n{shown}\n")
        lines.append("")

    lines.append("TOTAL MEAL\n")
    for label, val in (
        ("Calories", meal_total.calories),
        ("Protein", meal_total.protein_g),
        ("Carbohydrates", meal_total.carbs_g),
        ("Fat", meal_total.fat_g),
    ):
        unit = " kcal" if label == "Calories" else " g"
        shown = f"{val.value:.1f}{unit}" if val.value is not None else "unavailable"
        flag = "  (partial — one or more foods missing this nutrient)" if label.lower().replace(" ", "_") in [f.replace("_g", "") for f in meal_total.partial_nutrients] else ""
        lines.append(f"{label}:\n{shown}{flag}\n")

    return "\n".join(lines)


def format_unified_nutrition_report(instances: list[FoodInstance], meal_total: MealTotal) -> str:
    """
    Renders an exhaustive multi-nutrient terminal report with explicit uppercase provenance labels:
    MODEL PREDICTION, GROUND TRUTH, DERIVED, REFERENCE, NOT AVAILABLE.
    """
    lines: list[str] = []
    sep = "=" * 76
    subsep = "-" * 76

    lines.append(sep)
    lines.append("  UNIFIED MULTI-NUTRIENT INTELLIGENCE REPORT")
    lines.append(sep)

    for i, inst in enumerate(instances, start=1):
        name = (inst.class_name or "Unknown").replace("_", " ").title()
        conf_str = f"{inst.class_confidence * 100:.1f}%" if inst.class_confidence is not None else "N/A"
        lines.append(f"\n[FOOD #{i}] {name}  (Confidence: {conf_str})")
        lines.append(subsep)
        mass_str = inst.mass.display_str("g")
        lines.append(f"  Portion Mass  : {mass_str:<18} [{inst.mass.source.label}]")

        if inst.nutrients:
            n = inst.nutrients
            lines.append("  Macronutrients:")
            lines.append(f"    - Calories      : {n.calories.display_str('kcal'):<14} [{n.calories.source.label}]")
            lines.append(f"    - Protein       : {n.protein_g.display_str('g'):<14} [{n.protein_g.source.label}]")
            lines.append(f"    - Carbohydrates : {n.carbs_g.display_str('g'):<14} [{n.carbs_g.source.label}]")
            lines.append(f"    - Total Fat     : {n.fat_g.display_str('g'):<14} [{n.fat_g.source.label}]")

            lines.append("  Extended Nutrients:")
            lines.append(f"    - Dietary Fiber : {n.fiber_g.display_str('g'):<14} [{n.fiber_g.source.label}]")
            lines.append(f"    - Total Sugars  : {n.sugar_g.display_str('g'):<14} [{n.sugar_g.source.label}]")
            lines.append(f"    - Saturated Fat : {n.saturated_fat_g.display_str('g'):<14} [{n.saturated_fat_g.source.label}]")
            lines.append(f"    - Sodium        : {n.sodium_mg.display_str('mg'):<14} [{n.sodium_mg.source.label}]")
            lines.append(f"    - Cholesterol   : {n.cholesterol_mg.display_str('mg'):<14} [{n.cholesterol_mg.source.label}]")

            if n.additional_nutrients:
                lines.append("  Micronutrients:")
                for k, v in n.additional_nutrients.items():
                    clean_k = k.replace("_mg", "").replace("_g", "").replace("_", " ").title()
                    unit = "mg" if k.endswith("_mg") else ("g" if k.endswith("_g") else "")
                    lines.append(f"    - {clean_k:<14}: {v.display_str(unit):<14} [{v.source.label}]")

    # Meal Total Rollup
    lines.append(f"\n{sep}")
    lines.append("  TOTAL MEAL ROLLUP")
    lines.append(sep)

    def _line(label: str, val: Value, unit: str, field_key: str) -> str:
        part_flag = "  (PARTIAL)" if field_key in meal_total.partial_nutrients else ""
        return f"  {label:<18}: {val.display_str(unit):<14} [{val.source.label}]{part_flag}"

    lines.append(_line("Calories", meal_total.calories, "kcal", "calories"))
    lines.append(_line("Protein", meal_total.protein_g, "g", "protein_g"))
    lines.append(_line("Carbohydrates", meal_total.carbs_g, "g", "carbs_g"))
    lines.append(_line("Total Fat", meal_total.fat_g, "g", "fat_g"))
    lines.append(_line("Dietary Fiber", meal_total.fiber_g, "g", "fiber_g"))
    lines.append(_line("Total Sugars", meal_total.sugar_g, "g", "sugar_g"))
    lines.append(_line("Saturated Fat", meal_total.saturated_fat_g, "g", "saturated_fat_g"))
    lines.append(_line("Sodium", meal_total.sodium_mg, "mg", "sodium_mg"))
    lines.append(_line("Cholesterol", meal_total.cholesterol_mg, "mg", "cholesterol_mg"))

    if meal_total.additional_nutrients:
        lines.append("\n  Additional Micronutrients:")
        for k, v in meal_total.additional_nutrients.items():
            clean_k = k.replace("_mg", "").replace("_g", "").replace("_", " ").title()
            unit = "mg" if k.endswith("_mg") else ("g" if k.endswith("_g") else "")
            lines.append(_line(clean_k, v, unit, k))

    lines.append(f"\n{subsep}")
    lines.append("  PROVENANCE METHODOLOGY DISCLOSURE")
    lines.append(subsep)
    lines.append("  - MODEL PREDICTION : Multimodal neural regression trained on Nutrition5k ground truth.")
    lines.append("  - GROUND TRUTH     : Direct verified sensor or chemical lab measurement.")
    lines.append("  - DERIVED          : Mass-scaled computation from verified reference profiles.")
    lines.append("  - REFERENCE        : Standard USDA FoodData Central reference profile.")
    lines.append("  - NOT AVAILABLE    : Value unmeasured and unmapped (strictly never fabricated).")
    lines.append(sep)

    return "\n".join(lines)
