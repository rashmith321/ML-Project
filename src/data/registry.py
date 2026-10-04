"""
Dataset registry: one adapter per dataset, per DATASET_PLAN.md. Each
adapter is responsible ONLY for reading its own dataset's native format and
exposing it through a small common interface -- it does NOT merge across
datasets. Cross-dataset combination only ever happens at the stage/model
level (e.g. a classifier trained on Food-101 alone, or Food-101 + Vireo-172
alone), never as an undifferentiated blended pool.

Explicitly NOT implemented: foodx251, khana -- excluded per DATASET_PLAN.md.
"""
from __future__ import annotations

from src.data.base import BaseDatasetAdapter
from src.data.ecustfd import ECUSTFDAdapter
from src.data.food101 import Food101Adapter
from src.data.menumatch import MenuMatchAdapter
from src.data.nutrition5k import Nutrition5kAdapter
from src.data.recipe1m import Recipe1MAdapter
from src.data.unimib2016 import UNIMIB2016Adapter
from src.data.vireo172 import Vireo172Adapter

REGISTRY: dict[str, type[BaseDatasetAdapter]] = {
    "nutrition5k": Nutrition5kAdapter,
    "food101": Food101Adapter,
    "unimib2016": UNIMIB2016Adapter,
    "ecustfd": ECUSTFDAdapter,
    "recipe1m": Recipe1MAdapter,
    "vireo172": Vireo172Adapter,
    "menumatch": MenuMatchAdapter,
}
