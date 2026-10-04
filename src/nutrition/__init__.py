"""
Multi-Nutrient Estimation package export.
"""
from src.nutrition.aggregate import NUTRIENT_FIELDS, MealTotal, aggregate_meal
from src.nutrition.dataset import Nutrition5kDataset, create_nutrition_fixture
from src.nutrition.estimator import MultiNutrientEstimator
from src.nutrition.inference import run_complete_nutrition_pipeline
from src.nutrition.model import MultiNutrientModel, NormalizedMultiNutrientLoss

__all__ = [
    "MultiNutrientModel",
    "NormalizedMultiNutrientLoss",
    "MultiNutrientEstimator",
    "Nutrition5kDataset",
    "create_nutrition_fixture",
    "run_complete_nutrition_pipeline",
    "aggregate_meal",
    "MealTotal",
    "NUTRIENT_FIELDS",
]
