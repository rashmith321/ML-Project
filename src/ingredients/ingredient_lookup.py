"""
Auxiliary ingredient understanding (ARCHITECTURE.md: Recipe1M+ / Vireo-172
tag association per predicted class -- not per-pixel, not a primary
nutrient input).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.ingredients.taxonomy import FOOD101_INGREDIENTS, VIREO172_INGREDIENTS, get_canonical_ingredients
from src.utils.schema import FoodInstance


class IngredientLookup:
    """
    Ingredient tag lookup per predicted food class.
    Associates predicted dish names with verified ingredient sets.
    Returns an empty list for unknown classes -- ingredients are never fabricated.
    """

    def __init__(
        self,
        recipe1m_index_path: str | Path | None = None,
        vireo172_index_path: str | Path | None = None,
        custom_index: dict[str, list[str]] | None = None,
    ):
        self.recipe1m_index_path = Path(recipe1m_index_path) if recipe1m_index_path else None
        self.vireo172_index_path = Path(vireo172_index_path) if vireo172_index_path else None
        self._index: dict[str, list[str]] = {}
        if custom_index:
            self._index.update(custom_index)

    def load(self) -> None:
        """
        Load ingredient mappings from canonical taxonomies and optional external JSON files.
        """
        # Load canonical base mappings
        self._index.update(FOOD101_INGREDIENTS)
        self._index.update(VIREO172_INGREDIENTS)

        # Load external Recipe1M+ index if provided
        if self.recipe1m_index_path and self.recipe1m_index_path.is_file():
            with open(self.recipe1m_index_path, "r", encoding="utf-8") as f:
                r1m_data = json.load(f)
                if isinstance(r1m_data, dict):
                    for k, v in r1m_data.items():
                        norm_k = self.normalize_class_name(k)
                        self._index[norm_k] = v

        # Load external Vireo172 index if provided
        if self.vireo172_index_path and self.vireo172_index_path.is_file():
            with open(self.vireo172_index_path, "r", encoding="utf-8") as f:
                v172_data = json.load(f)
                if isinstance(v172_data, dict):
                    for k, v in v172_data.items():
                        norm_k = self.normalize_class_name(k)
                        self._index[norm_k] = v

    @staticmethod
    def normalize_class_name(name: str) -> str:
        """Normalize class name for key lookup (e.g. 'Apple Pie' -> 'apple_pie')."""
        return name.lower().strip().replace(" ", "_").replace("-", "_")

    def lookup(self, class_name: str) -> list[str]:
        """
        Best-effort ingredient list for a predicted class.
        Returns an empty list if unknown -- never guessed or fabricated.
        """
        if not class_name:
            return []
        norm = self.normalize_class_name(class_name)
        if norm in self._index:
            return list(self._index[norm])
        # Fallback to module-level canonical lookup if load() was not explicitly called
        canonical = get_canonical_ingredients(norm)
        if canonical:
            return canonical
        return []

    def annotate_instances(self, instances: list[FoodInstance]) -> list[FoodInstance]:
        """
        Annotate a list of FoodInstance objects with ingredient tags.
        """
        for inst in instances:
            if inst.class_name:
                inst.ingredients = self.lookup(inst.class_name)
            else:
                inst.ingredients = []
        return instances

    def export_index(self, output_path: str | Path) -> Path:
        """Export current ingredient index to a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)
        return out
