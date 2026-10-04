"""
ECUSTFD adapter.

Expected layout (typical distribution):
  <root>/
    JPEGImages/ (or images/)   filenames like "<category><id>_<view>.jpg",
                                e.g. apple001_1.jpg / apple001_2.jpg for the
                                two calibration-object views of one physical
                                dish.
    a weight/density reference table (commonly an .xls/.xlsx or .csv) giving
    per-item mass, sometimes volume and density, used with the calibration
    object to derive mass from image.

GROUPING FOR LEAKAGE PREVENTION: multiple images are different camera views
of the SAME physical dish. group_id strips the trailing "_<digits>" view
suffix from the filename stem so all views of one dish share a group_id --
split assignment (done in scripts/create_manifests.py) must keep every
member of a group in the same split.

MASS PARSING: this adapter auto-parses a per-item mass table only when it
finds a .csv whose header contains a recognizable id/name column and a
weight/mass column -- the exact spreadsheet layout varies by distribution
mirror and hasn't been verified against a real copy of this dataset in this
environment. If the reference table is .xls/.xlsx, or a .csv with headers
this adapter doesn't recognize, it reports the file was found but not
parsed, rather than guessing a column mapping.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from src.data.base import BaseDatasetAdapter, count_images
from src.data.schema import DatasetStatus, ManifestSample

_IMAGE_DIR_CANDIDATES = ["JPEGImages", "images", "Images"]
_VIEW_SUFFIX_RE = re.compile(r"_\d+$")
_MASS_HEADER_HINTS = ("weight", "mass")
_ID_HEADER_HINTS = ("id", "name", "category", "image")


class ECUSTFDAdapter(BaseDatasetAdapter):
    name = "ecustfd"
    display_name = "ECUSTFD"

    def _images_dir(self):
        if self.root is None:
            return None
        for c in _IMAGE_DIR_CANDIDATES:
            p = self.root / c
            if p.is_dir():
                return p
        return None

    def _reference_tables(self) -> list[Path]:
        if self.root is None:
            return []
        return [p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in (".csv", ".xls", ".xlsx")]

    @staticmethod
    def _group_id(stem: str) -> str:
        return _VIEW_SUFFIX_RE.sub("", stem)

    def _parse_csv_mass_table(self, path: Path) -> dict[str, float]:
        """Best-effort: id/name column -> mass, only if headers are recognizable."""
        mapping: dict[str, float] = {}
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return {}
                lower_fields = {fn: fn.lower() for fn in reader.fieldnames}
                mass_col = next((fn for fn, lo in lower_fields.items() if any(h in lo for h in _MASS_HEADER_HINTS)), None)
                id_col = next((fn for fn, lo in lower_fields.items() if any(h in lo for h in _ID_HEADER_HINTS)), None)
                if not mass_col or not id_col:
                    return {}
                for row in reader:
                    key = row.get(id_col)
                    val = row.get(mass_col)
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
        tables = self._reference_tables()

        parsed_mass = {}
        for t in tables:
            if t.suffix.lower() == ".csv":
                parsed_mass.update(self._parse_csv_mass_table(t))

        notes = []
        if images_dir is None:
            notes.append(f"No image directory found among {_IMAGE_DIR_CANDIDATES} under {self.root}.")
        if not tables:
            notes.append("No reference mass/weight table (.csv/.xls/.xlsx) found in the dataset root.")
        elif not parsed_mass:
            notes.append(
                f"Found reference table(s) {[t.name for t in tables]} but could not auto-parse a "
                f"mass column from them -- see module docstring. has_mass_labels reported as "
                f"'file present, not parsed' (None) rather than True/False."
            )

        return DatasetStatus(
            name=self.name,
            display_name=self.display_name,
            available=images_dir is not None,
            location=str(self.root),
            num_images=num_images,
            num_classes=None,  # would require parsing category out of filenames/table; not assumed
            annotation_format="csv" if any(t.suffix.lower() == ".csv" for t in tables) else (
                "xls/xlsx" if tables else None
            ),
            has_nutrition_labels=False,
            has_segmentation_labels=False,
            has_detection_labels=images_dir is not None or None,
            has_mass_labels=(True if parsed_mass else (None if tables else False)),
            has_depth=False,
            has_ingredient_info=False,
            train_test_split_available=False,
            split_source=None,
            notes=notes,
        )

    def build_manifest(self) -> list[ManifestSample]:
        images_dir = self._images_dir()
        if images_dir is None:
            return []

        parsed_mass = {}
        for t in self._reference_tables():
            if t.suffix.lower() == ".csv":
                parsed_mass.update(self._parse_csv_mass_table(t))

        samples: list[ManifestSample] = []
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            group = self._group_id(img_path.stem)
            mass = parsed_mass.get(img_path.stem) or parsed_mass.get(group)
            samples.append(ManifestSample(
                image_path=str(img_path),
                dataset=self.name,
                instance_id=img_path.stem,
                group_id=group,
                mass_g=mass,
                split=None,
            ))
        return samples
