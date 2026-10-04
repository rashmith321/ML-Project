"""
UNIMIB2016 adapter.

UNIMIB2016 is distributed with a few different folder-naming conventions
depending on the download source, so this adapter tries several candidate
names for the image directory and the annotation directory rather than
assuming one exact layout.

Expected layout (most common distribution):
  <root>/
    images/ (or original/)          tray photos, one per meal
    annotations/ (or masks/, AnnotationsFiles/)   per-item annotations

IMPORTANT LIMITATION (documented rather than papered over): UNIMIB2016's
per-item annotation format (polygon coordinates + food-class label inside a
.mat/.xml/.txt file) is dataset-specific and this adapter has not been
run against real UNIMIB2016 files in this environment (no copy is present
here). It generically detects annotation files that share an image's
filename stem and reports whether they exist and what extension they are,
and sets `mask_path` only in the unambiguous case of an image-format mask
(.png/.bmp) with a matching stem. It deliberately does NOT parse polygon
coordinates or per-item class labels out of .mat/.xml/.txt files, since
doing so correctly requires inspecting an actual sample file first --
faking that parser against an assumed byte layout would violate the
"do not create fake values" rule. Extending this once real data is
available is a small, isolated follow-up (see inspect() notes it emits).
"""
from __future__ import annotations

from src.data.base import BaseDatasetAdapter, count_images
from src.data.schema import DatasetStatus, ManifestSample

_IMAGE_DIR_CANDIDATES = ["images", "original", "Images"]
_ANNOTATION_DIR_CANDIDATES = ["annotations", "Annotations", "masks", "Masks", "AnnotationsFiles"]
_MASK_IMAGE_EXTS = (".png", ".bmp")


class UNIMIB2016Adapter(BaseDatasetAdapter):
    name = "unimib2016"
    display_name = "UNIMIB2016"

    def _images_dir(self):
        if self.root is None:
            return None
        for c in _IMAGE_DIR_CANDIDATES:
            p = self.root / c
            if p.is_dir():
                return p
        return None

    def _annotations_dir(self):
        if self.root is None:
            return None
        for c in _ANNOTATION_DIR_CANDIDATES:
            p = self.root / c
            if p.is_dir():
                return p
        return None

    def inspect(self) -> DatasetStatus:
        if self.root is None:
            return self._not_found_status()

        images_dir = self._images_dir()
        ann_dir = self._annotations_dir()

        num_images = count_images(images_dir) if images_dir else None

        ann_ext = None
        has_seg = None
        if ann_dir is not None:
            ann_files = [p for p in ann_dir.iterdir() if p.is_file()]
            if ann_files:
                exts = {p.suffix.lower() for p in ann_files}
                ann_ext = ", ".join(sorted(exts))
                has_seg = True
            else:
                has_seg = False

        notes = []
        if images_dir is None:
            notes.append(
                f"No image directory found among {_IMAGE_DIR_CANDIDATES} under {self.root}."
            )
        if ann_dir is None:
            notes.append(
                f"No annotation directory found among {_ANNOTATION_DIR_CANDIDATES} under {self.root}."
            )
        elif ann_ext and not any(ext in _MASK_IMAGE_EXTS for ext in ann_ext.split(", ")):
            notes.append(
                f"Annotation files found with extension(s) [{ann_ext}] but this adapter only "
                f"auto-derives mask_path for image-format masks (.png/.bmp); parsing this format "
                f"into per-item polygons/classes is not yet implemented -- see module docstring."
            )

        return DatasetStatus(
            name=self.name,
            display_name=self.display_name,
            available=images_dir is not None,
            location=str(self.root),
            num_images=num_images,
            num_classes=None,  # not derivable without parsing per-item annotation content
            annotation_format=ann_ext,
            has_nutrition_labels=False,
            has_segmentation_labels=has_seg,
            has_detection_labels=has_seg,  # a bbox is derivable from a mask, once mask is parsed
            has_mass_labels=False,
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
        ann_dir = self._annotations_dir()

        samples: list[ManifestSample] = []
        for img_path in sorted(images_dir.rglob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            mask_path = None
            if ann_dir is not None:
                for ext in _MASK_IMAGE_EXTS:
                    candidate = ann_dir / f"{img_path.stem}{ext}"
                    if candidate.is_file():
                        mask_path = str(candidate)
                        break

            samples.append(ManifestSample(
                image_path=str(img_path),
                dataset=self.name,
                instance_id=img_path.stem,
                mask_path=mask_path,
                split=None,  # constructed at split-assignment time, not here
            ))
        return samples
