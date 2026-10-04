"""
Vireo Food-172 adapter.

Expected layout (official release):
  <root>/
    ready_chinese_food/<class_id>/<image>.jpg     class_id is 1..172
    SplitAndIngreLabel/TR.txt                     relative image paths, training split
    SplitAndIngreLabel/VAL.txt                    validation split
    SplitAndIngreLabel/TE.txt                     test split
    SplitAndIngreLabel/FoodList.txt                line i = class name for class_id i
    SplitAndIngreLabel/IngreLabel.txt              "<relative_path> <353 binary flags>"
"""
from __future__ import annotations

from src.data.base import BaseDatasetAdapter, count_images
from src.data.schema import DatasetStatus, ManifestSample


class Vireo172Adapter(BaseDatasetAdapter):
    name = "vireo172"
    display_name = "Vireo Food-172"

    def _images_dir(self):
        if self.root is None:
            return None
        p = self.root / "ready_chinese_food"
        return p if p.is_dir() else None

    def _split_dir(self):
        if self.root is None:
            return None
        p = self.root / "SplitAndIngreLabel"
        return p if p.is_dir() else None

    def _food_list(self) -> dict[str, str]:
        """class_id (as it appears in the folder path, e.g. '1') -> class name."""
        split_dir = self._split_dir()
        if split_dir is None:
            return {}
        f = split_dir / "FoodList.txt"
        if not f.is_file():
            return {}
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        return {str(i + 1): name for i, name in enumerate(lines)}

    def _ingredients_index(self) -> dict[str, list]:
        """relative image path -> list of ingredient indices flagged 1 (names not resolved
        here since that requires the separate ingredient-name list, which some mirrors omit;
        stored as string indices rather than fabricated names)."""
        split_dir = self._split_dir()
        if split_dir is None:
            return {}
        f = split_dir / "IngreLabel.txt"
        if not f.is_file():
            return {}
        out = {}
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            path, flags = parts[0], parts[1:]
            active = [str(i) for i, v in enumerate(flags) if v == "1"]
            if active:
                out[path.lstrip("/")] = active
        return out

    def inspect(self) -> DatasetStatus:
        if self.root is None:
            return self._not_found_status()

        images_dir = self._images_dir()
        split_dir = self._split_dir()
        num_images = count_images(images_dir) if images_dir else None
        food_list = self._food_list()

        has_split = False
        if split_dir is not None:
            has_split = all((split_dir / f).is_file() for f in ("TR.txt", "TE.txt"))

        notes = []
        if images_dir is None:
            notes.append("ready_chinese_food/ not found -- cannot report image counts.")
        if split_dir is None:
            notes.append("SplitAndIngreLabel/ not found -- no split or ingredient-tag info available.")

        return DatasetStatus(
            name=self.name,
            display_name=self.display_name,
            available=images_dir is not None,
            location=str(self.root),
            num_images=num_images,
            num_classes=len(food_list) if food_list else (
                len([p for p in images_dir.iterdir() if p.is_dir()]) if images_dir else None
            ),
            annotation_format="folder-per-class" if images_dir is not None else None,
            has_nutrition_labels=False,
            has_segmentation_labels=False,
            has_detection_labels=False,
            has_mass_labels=False,
            has_depth=False,
            has_ingredient_info=(split_dir / "IngreLabel.txt").is_file() if split_dir else False,
            train_test_split_available=has_split,
            split_source="official_file" if has_split else None,
            notes=notes,
        )

    def build_manifest(self) -> list[ManifestSample]:
        images_dir = self._images_dir()
        if images_dir is None:
            return []

        split_dir = self._split_dir()
        food_list = self._food_list()
        ingredients_index = self._ingredients_index()

        split_by_relpath = {}
        if split_dir is not None:
            for fname, split_name in (("TR.txt", "train"), ("VAL.txt", "val"), ("TE.txt", "test")):
                f = split_dir / fname
                if f.is_file():
                    for line in f.read_text().splitlines():
                        rel = line.strip().lstrip("/")
                        if rel:
                            split_by_relpath[rel] = split_name

        samples: list[ManifestSample] = []
        for class_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
            class_name = food_list.get(class_dir.name, class_dir.name)
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                rel = f"{class_dir.name}/{img_path.name}"
                samples.append(ManifestSample(
                    image_path=str(img_path),
                    dataset=self.name,
                    instance_id=f"{class_dir.name}_{img_path.stem}",
                    food_class=class_name,
                    split=split_by_relpath.get(rel),
                    ingredients=ingredients_index.get(rel),
                ))
        return samples
