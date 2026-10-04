"""
Food classification taxonomy and class mapping system.
Maintains dataset-specific labels (Food-101, Vireo-172, Nutrition5k) without blind merging.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

# Canonical list of 101 food classes in the official Food-101 benchmark
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles"
]

# Representative sample of Vireo Food-172 classes
VIREO172_COMMON_CLASSES = [
    "kung_pao_chicken", "mapo_tofu", "peking_duck", "sweet_and_sour_pork",
    "fried_dumplings", "wonton_soup", "fried_rice", "hot_and_sour_soup",
    "chow_mein", "steamed_buns", "spring_rolls", "dim_sum", "braised_pork_belly"
]

# Common Nutrition5k ingredient labels
NUTRITION5K_INGREDIENT_LABELS = [
    "rice", "chicken_breast", "broccoli", "salmon", "pasta",
    "egg", "potato", "carrot", "tomato", "beef", "tofu", "cheese"
]


class FoodTaxonomy:
    """
    Manages taxonomy definitions, class mappings, and safe cross-dataset lookups
    without fabricating labels or blindly merging incompatible categories.
    """

    def __init__(self, primary_dataset: str = "food101", custom_classes: Sequence[str] | None = None):
        self.primary_dataset = primary_dataset.lower()
        if custom_classes:
            self._classes = list(custom_classes)
        elif self.primary_dataset == "food101":
            self._classes = list(FOOD101_CLASSES)
        elif self.primary_dataset == "vireo172":
            self._classes = list(VIREO172_COMMON_CLASSES)
        elif self.primary_dataset == "nutrition5k":
            self._classes = list(NUTRITION5K_INGREDIENT_LABELS)
        else:
            raise ValueError(f"Unknown dataset taxonomy: {primary_dataset}")

        self._id_to_name = {i: name for i, name in enumerate(self._classes)}
        self._name_to_id = {name: i for i, name in enumerate(self._classes)}

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def id_to_name(self, class_id: int) -> str:
        """Map integer class_id to canonical string name."""
        return self._id_to_name.get(class_id, f"unknown_class_{class_id}")

    def name_to_id(self, class_name: str) -> int | None:
        """Map canonical string name to integer class_id (None if unrepresented)."""
        clean = self.normalize_name(class_name)
        return self._name_to_id.get(clean)

    @staticmethod
    def normalize_name(name: str) -> str:
        """Harmonize string representation (lowercased, spaces/hyphens to underscores)."""
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    def is_valid_class(self, class_name: str) -> bool:
        """Check if class is genuinely represented in the taxonomy."""
        return self.normalize_name(class_name) in self._name_to_id

    def export_json(self, output_path: str | Path) -> Path:
        """Save taxonomy to JSON for full traceability."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dataset": self.primary_dataset,
            "num_classes": self.num_classes,
            "classes": self._classes,
            "id_to_name": {str(k): v for k, v in self._id_to_name.items()},
        }
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        return out
