"""
Nutrition5k adapter.

Expected layout (Google's official release):
  <root>/
    metadata/dish_metadata_cafe1.csv
    metadata/dish_metadata_cafe2.csv
    imagery/realsense_overhead/<dish_id>/rgb.png
    imagery/realsense_overhead/<dish_id>/depth_raw.png
    dish_ids/splits/rgb_train_ids.txt
    dish_ids/splits/rgb_test_ids.txt

The metadata CSV has no header row and a variable-length repeating group per
row: dish_id, total_calories, total_mass, total_fat, total_carb,
total_protein, then repeats of (ingr_id, ingr_name, ingr_grams,
ingr_calories, ingr_fat, ingr_carb, ingr_protein) for each ingredient in
the dish. This parser reads it positionally, exactly as documented in the
dataset's own README, and does not guess at columns that aren't there.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.data.base import BaseDatasetAdapter
from src.data.schema import DatasetStatus, ManifestSample

_METADATA_FILES = ["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"]
_HEADER_FIELDS = ("dish_id", "total_calories", "total_mass", "total_fat", "total_carb", "total_protein")
_INGR_STRIDE = 7  # ingr_id, ingr_name, ingr_grams, ingr_calories, ingr_fat, ingr_carb, ingr_protein


class Nutrition5kAdapter(BaseDatasetAdapter):
    name = "nutrition5k"
    display_name = "Nutrition5k"

    def _metadata_paths(self) -> list[Path]:
        if self.root is None:
            return []
        return [self.root / "metadata" / f for f in _METADATA_FILES if (self.root / "metadata" / f).is_file()]

    def _split_dir(self) -> Path | None:
        if self.root is None:
            return None
        d = self.root / "dish_ids" / "splits"
        return d if d.is_dir() else None

    def inspect(self) -> DatasetStatus:
        if self.root is None:
            return self._not_found_status()

        meta_paths = self._metadata_paths()
        overhead_dir = self.root / "imagery" / "realsense_overhead"
        split_dir = self._split_dir()

        num_dishes = None
        if meta_paths:
            dish_ids = set()
            for p in meta_paths:
                with open(p, newline="") as f:
                    for row in csv.reader(f):
                        if row:
                            dish_ids.add(row[0])
            num_dishes = len(dish_ids)

        num_images = None
        has_depth = None
        if overhead_dir.is_dir():
            rgb_files = list(overhead_dir.glob("*/rgb.png"))
            depth_files = list(overhead_dir.glob("*/depth_raw.png"))
            num_images = len(rgb_files)
            has_depth = len(depth_files) > 0

        notes = []
        if not meta_paths:
            notes.append("metadata/dish_metadata_cafe{1,2}.csv not found -- cannot report nutrition/mass counts.")
        if not overhead_dir.is_dir():
            notes.append("imagery/realsense_overhead/ not found -- cannot report image counts or depth availability.")

        return DatasetStatus(
            name=self.name,
            display_name=self.display_name,
            available=bool(meta_paths) or overhead_dir.is_dir(),
            location=str(self.root),
            num_images=num_images,
            num_classes=None,  # Nutrition5k is per-dish nutrition, not a fixed class taxonomy
            annotation_format="csv" if meta_paths else None,
            has_nutrition_labels=bool(meta_paths) or None,
            has_segmentation_labels=False,
            has_detection_labels=False,
            has_mass_labels=bool(meta_paths) or None,
            has_depth=has_depth,
            has_ingredient_info=bool(meta_paths) or None,
            train_test_split_available=split_dir is not None,
            split_source="official_file" if split_dir is not None else None,
            notes=notes,
        )

    def _official_split_map(self) -> dict:
        """dish_id -> 'train'/'test', from the official rgb split files. Empty dict if absent."""
        split_dir = self._split_dir()
        if split_dir is None:
            return {}
        mapping = {}
        train_f = split_dir / "rgb_train_ids.txt"
        test_f = split_dir / "rgb_test_ids.txt"
        if train_f.is_file():
            for line in train_f.read_text().splitlines():
                line = line.strip()
                if line:
                    mapping[line] = "train"
        if test_f.is_file():
            for line in test_f.read_text().splitlines():
                line = line.strip()
                if line:
                    mapping[line] = "test"
        return mapping

    def build_manifest(self) -> list[ManifestSample]:
        meta_paths = self._metadata_paths()
        if not meta_paths:
            return []

        split_map = self._official_split_map()
        overhead_dir = self.root / "imagery" / "realsense_overhead"
        samples: list[ManifestSample] = []

        for p in meta_paths:
            with open(p, newline="") as f:
                for row in csv.reader(f):
                    if not row:
                        continue
                    dish_id = row[0]
                    try:
                        total_calories = float(row[1])
                        total_mass = float(row[2])
                        total_fat = float(row[3])
                        total_carb = float(row[4])
                        total_protein = float(row[5])
                    except (IndexError, ValueError):
                        # Malformed row -- skip rather than fabricate values.
                        continue

                    ingredients = []
                    rest = row[6:]
                    for i in range(0, len(rest) - 1, _INGR_STRIDE):
                        chunk = rest[i:i + _INGR_STRIDE]
                        if len(chunk) >= 2:
                            ingredients.append(chunk[1])  # ingr_name

                    rgb_path = overhead_dir / dish_id / "rgb.png"
                    depth_path = overhead_dir / dish_id / "depth_raw.png"

                    if not rgb_path.is_file():
                        # No image for this dish's metadata row -- don't fabricate an image_path.
                        continue

                    samples.append(ManifestSample(
                        image_path=str(rgb_path),
                        dataset=self.name,
                        instance_id=dish_id,
                        group_id=dish_id,
                        split=split_map.get(dish_id),
                        depth_path=str(depth_path) if depth_path.is_file() else None,
                        mass_g=total_mass,
                        calories=total_calories,
                        protein_g=total_protein,
                        carbohydrates_g=total_carb,
                        fat_g=total_fat,
                        ingredients=ingredients or None,
                    ))
        return samples
