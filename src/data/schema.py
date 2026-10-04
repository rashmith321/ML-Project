"""
Standardized types shared by the whole data-engineering layer:

- `DatasetStatus`: the answer to every question the availability report
  needs, for one dataset, filled in only from what was actually found on
  disk (never from configs/datasets.yaml's declared expectations).
- `ManifestSample`: one row of a dataset manifest. Per the project rule
  ("missing values must remain missing, do not create fake values"),
  `as_dict()` omits any field that is None rather than writing it as
  null/0/"" -- a field's absence in the manifest IS the "missing" signal,
  and downstream code must treat a missing key as genuinely unknown, not
  coerce it to a default.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class DatasetStatus:
    """Availability report for exactly one dataset, measured from disk."""
    name: str
    display_name: str
    available: bool
    location: Optional[str] = None          # resolved absolute path, if found
    num_images: Optional[int] = None
    num_classes: Optional[int] = None
    annotation_format: Optional[str] = None  # e.g. "csv", "mat", "json", "polygon-xml", "none"
    has_nutrition_labels: Optional[bool] = None
    has_segmentation_labels: Optional[bool] = None
    has_detection_labels: Optional[bool] = None
    has_mass_labels: Optional[bool] = None
    has_depth: Optional[bool] = None
    has_ingredient_info: Optional[bool] = None
    train_test_split_available: Optional[bool] = None
    split_source: Optional[str] = None       # "official_file", "constructed", or None
    notes: list[str] = field(default_factory=list)  # anything an inspector wants to flag

    def as_dict(self) -> dict:
        """
        Unlike ManifestSample, None fields ARE included here: for a status
        report, "unknown / not measured" (None) is itself meaningful
        information the report must show, not omit.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class ManifestSample:
    """
    One standardized manifest row. Every field is Optional and is only set
    when the source dataset actually provides that value for that sample --
    never guessed, defaulted, or interpolated.
    """
    image_path: str
    dataset: str
    instance_id: Optional[str] = None
    food_class: Optional[str] = None
    split: Optional[str] = None
    group_id: Optional[str] = None        # leakage-prevention grouping key (dish id, recipe id, etc.)
    mask_path: Optional[str] = None
    bbox: Optional[tuple] = None          # (x, y, w, h)
    mass_g: Optional[float] = None
    depth_path: Optional[str] = None
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    fat_g: Optional[float] = None
    ingredients: Optional[list] = None

    def as_dict(self) -> dict:
        """Only fields that are not None are included -- absence == missing."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = v
        return out
