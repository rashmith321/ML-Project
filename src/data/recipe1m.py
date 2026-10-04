"""
Recipe1M+ adapter. AUXILIARY dataset -- ingredient text only, per
DATASET_PLAN.md (feeds Ingredient Understanding, not nutrient regression).

Expected layout (official release):
  <root>/
    layer1.json     recipes: [{id, ingredients: [{text}], partition: "train"|"val"|"test", ...}, ...]
    layer2.json     image index: [{id (recipe id), images: [{id: image_id}, ...]}, ...]
    images/<c0>/<c1>/<c2>/<c3>/<image_id>.jpg
           where c0..c3 are the first four characters of image_id (the
           dataset's documented nested-hash directory scheme).

Only recipes that have a `partition` field get a `split` value; only
images that are actually found on disk are included as samples -- a
recipe entry existing in layer1.json is not itself proof an image exists.
"""
from __future__ import annotations

import json

from src.data.base import BaseDatasetAdapter
from src.data.schema import DatasetStatus, ManifestSample


class Recipe1MAdapter(BaseDatasetAdapter):
    name = "recipe1m"
    display_name = "Recipe1M+"

    def _layer1(self):
        if self.root is None:
            return None
        p = self.root / "layer1.json"
        return p if p.is_file() else None

    def _layer2(self):
        if self.root is None:
            return None
        p = self.root / "layer2.json"
        return p if p.is_file() else None

    def _images_dir(self):
        if self.root is None:
            return None
        p = self.root / "images"
        return p if p.is_dir() else None

    @staticmethod
    def _image_path(images_dir, image_id: str):
        if len(image_id) < 4:
            return None
        c0, c1, c2, c3 = image_id[0], image_id[1], image_id[2], image_id[3]
        return images_dir / c0 / c1 / c2 / c3 / f"{image_id}.jpg"

    def inspect(self) -> DatasetStatus:
        if self.root is None:
            return self._not_found_status()

        layer1 = self._layer1()
        layer2 = self._layer2()
        images_dir = self._images_dir()

        num_recipes = None
        has_split = None
        if layer1 is not None:
            with open(layer1) as f:
                recipes = json.load(f)
            num_recipes = len(recipes)
            has_split = any("partition" in r for r in recipes[:100]) if recipes else False

        notes = []
        if layer1 is None:
            notes.append("layer1.json not found -- cannot report recipe/ingredient counts.")
        if layer2 is None:
            notes.append("layer2.json not found -- cannot map recipes to image files.")
        if images_dir is None:
            notes.append("images/ not found -- cannot verify any recipe actually has a downloaded photo.")

        return DatasetStatus(
            name=self.name,
            display_name=self.display_name,
            available=layer1 is not None,
            location=str(self.root),
            num_images=None,  # only meaningful after cross-referencing layer2 + disk, done in build_manifest
            num_classes=None,  # Recipe1M+ has no fixed class taxonomy
            annotation_format="json" if layer1 is not None else None,
            has_nutrition_labels=False,
            has_segmentation_labels=False,
            has_detection_labels=False,
            has_mass_labels=False,
            has_depth=False,
            has_ingredient_info=layer1 is not None or None,
            train_test_split_available=has_split,
            split_source="official_file" if has_split else None,
            notes=notes,
        )

    def build_manifest(self) -> list[ManifestSample]:
        layer1 = self._layer1()
        layer2 = self._layer2()
        images_dir = self._images_dir()
        if layer1 is None or layer2 is None or images_dir is None:
            return []

        with open(layer1) as f:
            recipes = json.load(f)
        with open(layer2) as f:
            image_index = json.load(f)

        split_by_id = {r["id"]: r.get("partition") for r in recipes if "id" in r}
        ingredients_by_id = {
            r["id"]: [ing.get("text") for ing in r.get("ingredients", []) if ing.get("text")]
            for r in recipes if "id" in r
        }

        samples: list[ManifestSample] = []
        for entry in image_index:
            recipe_id = entry.get("id")
            if recipe_id is None:
                continue
            for img in entry.get("images", []):
                image_id = img.get("id")
                if not image_id:
                    continue
                img_path = self._image_path(images_dir, image_id)
                if img_path is None or not img_path.is_file():
                    continue
                samples.append(ManifestSample(
                    image_path=str(img_path),
                    dataset=self.name,
                    instance_id=image_id,
                    group_id=recipe_id,
                    split=split_by_id.get(recipe_id),
                    ingredients=ingredients_by_id.get(recipe_id) or None,
                ))
        return samples
