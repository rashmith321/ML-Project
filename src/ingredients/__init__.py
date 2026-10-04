"""
Ingredients package export.
"""
from src.ingredients.dataset import MultiLabelIngredientDataset, create_ingredient_fixture
from src.ingredients.inference import IngredientPredictor, IngredientResult
from src.ingredients.ingredient_lookup import IngredientLookup
from src.ingredients.mapping import IngredientMapping, IngredientSource
from src.ingredients.model import MultiLabelIngredientModel
from src.ingredients.taxonomy import FOOD101_INGREDIENTS, VIREO172_INGREDIENTS, get_canonical_ingredients

__all__ = [
    "IngredientMapping",
    "IngredientSource",
    "IngredientPredictor",
    "IngredientResult",
    "MultiLabelIngredientModel",
    "MultiLabelIngredientDataset",
    "create_ingredient_fixture",
    "IngredientLookup",
    "FOOD101_INGREDIENTS",
    "VIREO172_INGREDIENTS",
    "get_canonical_ingredients",
]
