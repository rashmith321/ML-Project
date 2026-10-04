"""
Sanity tests for src.nutrition.aggregate. Run with: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nutrition.aggregate import aggregate_meal, format_report
from src.utils.schema import FoodInstance, NutrientSet, Source, Value


def _nutrients(cal, prot, carb, fat, source=Source.PREDICTION):
    return NutrientSet(
        calories=Value(cal, source),
        protein_g=Value(prot, source),
        carbs_g=Value(carb, source),
        fat_g=Value(fat, source),
    )


def test_aggregate_two_complete_foods():
    rice = FoodInstance(
        instance_id="1", class_name="Rice",
        mass=Value(180.0, Source.PREDICTION),
        nutrients=_nutrients(233, 4.3, 50.4, 0.4),
    )
    chicken = FoodInstance(
        instance_id="2", class_name="Chicken",
        mass=Value(150.0, Source.PREDICTION),
        nutrients=_nutrients(247, 46.5, 0.0, 5.4),
    )
    total = aggregate_meal([rice, chicken])
    assert total.calories.value == 480.0
    assert total.protein_g.value == 50.8
    assert total.partial_nutrients == []


def test_aggregate_flags_partial_nutrient():
    rice = FoodInstance(
        instance_id="1", class_name="Rice",
        mass=Value(180.0, Source.PREDICTION),
        nutrients=_nutrients(233, 4.3, 50.4, 0.4),
    )
    mystery = FoodInstance(
        instance_id="2", class_name="Unidentified item",
        mass=Value.unavailable(),
        nutrients=NutrientSet(
            calories=Value.unavailable(),
            protein_g=Value.unavailable(),
            carbs_g=Value.unavailable(),
            fat_g=Value.unavailable(),
        ),
    )
    total = aggregate_meal([rice, mystery])
    # rice's calories still count -- total is not itself unavailable
    assert total.calories.value == 233.0
    assert "calories" in total.partial_nutrients


def test_aggregate_empty_raises():
    try:
        aggregate_meal([])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_format_report_runs():
    rice = FoodInstance(
        instance_id="1", class_name="Rice",
        mass=Value(180.0, Source.PREDICTION),
        nutrients=_nutrients(233, 4.3, 50.4, 0.4),
    )
    total = aggregate_meal([rice])
    text = format_report([rice], total)
    assert "Rice" in text
    assert "TOTAL MEAL" in text


if __name__ == "__main__":
    test_aggregate_two_complete_foods()
    test_aggregate_flags_partial_nutrient()
    test_aggregate_empty_raises()
    test_format_report_runs()
    print("All tests passed.")
