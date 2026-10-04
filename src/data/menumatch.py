"""
MenuMatch adapter. EVALUATION-ONLY dataset per DATASET_PLAN.md -- never
used for training. Every sample this adapter emits gets split="eval_only".

Expected layout (typical distribution): a folder of restaurant meal photos
plus a table of restaurant-menu calorie labels per photo/dish. Exact
filenames vary by mirror, so this adapter searches for common candidates
and reports what it actually found rather than assuming a fixed layout.
"""
from __future__ import annotations

import csv

from src.data.base import BaseDatasetAdapter, count_images
from src.data.schema import DatasetStatus, ManifestSample

_IMAGE_DIR_CANDIDATES = ["images", "Images", "photos", "Photos"]
_CALORIE_HEADER_HINTS = ("calorie",)
_ID_HEADER_HINTS = ("id", "name", "image", "file")


class MenuMatchAdapter(BaseDatasetAdapter):
    name = "menumatch"
    display_name = "MenuMatch"

    def _images_dir(self):
        if self.root is None:
            return None
        for c in _IMAGE_DIR_CANDIDATES:
            p = self.root / c
            if p.is_dir():
                return p
        return None

    def _label_csvs(self):
        if self.root is None:
            return []
        return [p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]

    def _parse_calorie_table(self, path) -> dict[str, float]:
        mapping = {}
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return {}
                lower = {fn: fn.lower() for fn in reader.fieldnames}
                cal_col = next((fn for fn, lo in lower.items() if any(h in lo for h in _CALORIE_HEADER_HINTS)), None)
                id_col = next((fn for fn, lo in lower.items() if any(h in lo for h in _ID_HEADER_HINTS)), None)
                if not cal_col or not id_col:
                    return {}
                for row in reader:
                    key, val = row.get(id_col), row.get(cal_col)
                    if key is None or val is None:
                        continue
                    try:
                        mapping[key.strip()] = float(val)
                    except ValueError:
                        continue
        except (OSError, csv.Error):
            return {}
        return mapping

    def inspect(self) -> DatasetStatus:
        if self.root is None:
            return self._not_found_status()

        images_dir = self._images_dir()
        num_images = count_images(images_dir) if images_dir else None
        csvs = self._label_csvs()
        parsed = {}
        for c in csvs:
            parsed.update(self._parse_calorie_table(c))

        notes = []
        if images_dir is None:
            notes.append(f"No image directory found among {_IMAGE_DIR_CANDIDATES} under {self.root}.")
        if not csvs:
            notes.append("No .csv label file found in the dataset root.")
        elif not parsed:
            notes.append(f"Found {[c.name for c in csvs]} but could not auto-identify a calorie column.")

        return DatasetStatus(
            name=self.name,
            display_name=self.display_name,
            available=images_dir is not None,
            location=str(self.root),
            num_images=num_images,
            num_classes=None,
            annotation_format="csv" if csvs else None,
            has_nutrition_labels=(True if parsed else (None if csvs else False)),
            has_segmentation_labels=False,
            has_detection_labels=False,
            has_mass_labels=False,
            has_depth=False,
            has_ingredient_info=False,
            train_test_split_available=False,  # eval-only, by design -- see docstring
            split_source=None,
            notes=notes,
        )

    def build_manifest(self) -> list[ManifestSample]:
        images_dir = self._images_dir()
        if images_dir is None:
            return []
        parsed = {}
        for c in self._label_csvs():
            parsed.update(self._parse_calorie_table(c))

        samples: list[ManifestSample] = []
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            cal = parsed.get(img_path.stem) or parsed.get(img_path.name)
            samples.append(ManifestSample(
                image_path=str(img_path),
                dataset=self.name,
                instance_id=img_path.stem,
                split="eval_only",
                calories=cal,
            ))
        return samples
