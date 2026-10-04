"""
Food-101 adapter.

Expected layout (official release):
  <root>/
    images/<class_name>/<id>.jpg
    meta/classes.txt
    meta/train.txt   (lines like "apple_pie/1005649", no extension)
    meta/test.txt
"""
from __future__ import annotations

from pathlib import Path

from src.data.base import BaseDatasetAdapter, count_images
from src.data.schema import DatasetStatus, ManifestSample


class Food101Adapter(BaseDatasetAdapter):
    name = "food101"
    display_name = "Food-101"

    def inspect(self) -> DatasetStatus:
        if self.root is None:
            return self._not_found_status()

        images_dir = self.root / "images"
        meta_dir = self.root / "meta"
        classes_file = meta_dir / "classes.txt"
        train_file = meta_dir / "train.txt"
        test_file = meta_dir / "test.txt"

        num_images = count_images(images_dir) if images_dir.is_dir() else None
        num_classes = None
        if classes_file.is_file():
            num_classes = len([l for l in classes_file.read_text().splitlines() if l.strip()])
        elif images_dir.is_dir():
            num_classes = len([p for p in images_dir.iterdir() if p.is_dir()])

        notes = []
        if not images_dir.is_dir():
            notes.append("images/ not found -- cannot report image counts.")
        if not (train_file.is_file() and test_file.is_file()):
            notes.append("meta/train.txt and/or meta/test.txt not found -- official split unavailable.")

        return DatasetStatus(
            name=self.name,
            display_name=self.display_name,
            available=images_dir.is_dir(),
            location=str(self.root),
            num_images=num_images,
            num_classes=num_classes,
            annotation_format="folder-per-class" if images_dir.is_dir() else None,
            has_nutrition_labels=False,
            has_segmentation_labels=False,
            has_detection_labels=False,
            has_mass_labels=False,
            has_depth=False,
            has_ingredient_info=False,
            train_test_split_available=train_file.is_file() and test_file.is_file(),
            split_source="official_file" if (train_file.is_file() and test_file.is_file()) else None,
            notes=notes,
        )

    def build_manifest(self) -> list[ManifestSample]:
        images_dir = self.root / "images" if self.root else None
        if images_dir is None or not images_dir.is_dir():
            return []

        meta_dir = self.root / "meta"
        samples: list[ManifestSample] = []

        def _rows_from_split(split_file: Path, split_name: str):
            if not split_file.is_file():
                return
            for line in split_file.read_text().splitlines():
                line = line.strip()
                if not line or "/" not in line:
                    continue
                food_class, image_id = line.split("/", 1)
                img_path = images_dir / food_class / f"{image_id}.jpg"
                if not img_path.is_file():
                    continue
                samples.append(ManifestSample(
                    image_path=str(img_path),
                    dataset=self.name,
                    instance_id=line.replace("/", "_"),
                    food_class=food_class,
                    split=split_name,
                ))

        train_file = meta_dir / "train.txt"
        test_file = meta_dir / "test.txt"
        if train_file.is_file() or test_file.is_file():
            _rows_from_split(train_file, "train")
            _rows_from_split(test_file, "test")
        else:
            # No official split files found -- enumerate images with split=None
            # (unknown) rather than inventing a split assignment here.
            for class_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                        continue
                    samples.append(ManifestSample(
                        image_path=str(img_path),
                        dataset=self.name,
                        instance_id=f"{class_dir.name}_{img_path.stem}",
                        food_class=class_dir.name,
                        split=None,
                    ))
        return samples
