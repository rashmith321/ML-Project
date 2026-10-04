"""
Unit and integration tests for Phase 5: Ingredient Understanding module.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingredients.dataset import MultiLabelIngredientDataset, create_ingredient_fixture
from src.ingredients.inference import IngredientPredictor, IngredientResult
from src.ingredients.ingredient_lookup import IngredientLookup
from src.ingredients.mapping import IngredientMapping, IngredientSource
from src.ingredients.model import MultiLabelIngredientModel
from src.ingredients.taxonomy import FOOD101_INGREDIENTS, get_canonical_ingredients
from src.utils.schema import FoodInstance


# ---------------------------------------------------------------------------
# 1. Source Provenance & Mapping Tests
# ---------------------------------------------------------------------------

def test_ingredient_sources_enum():
    assert IngredientSource.DATASET_GROUND_TRUTH.value == "dataset_ground_truth"
    assert IngredientSource.MODEL_PREDICTION.value == "model_prediction"
    assert IngredientSource.RECIPE_DERIVED.value == "recipe_derived"
    assert IngredientSource.REFERENCE_LOOKUP.value == "reference_lookup"
    assert IngredientSource.UNAVAILABLE.value == "unavailable"


def test_mapping_recipe_derived():
    mapping = IngredientMapping()
    cands, confs, source = mapping.lookup("pizza")
    assert source == IngredientSource.RECIPE_DERIVED
    assert "mozzarella cheese" in cands
    assert "tomato sauce" in cands
    assert confs["mozzarella cheese"] > 0.0


def test_mapping_reference_lookup():
    mapping = IngredientMapping()
    cands, confs, source = mapping.lookup("kung_pao_chicken")
    assert source == IngredientSource.REFERENCE_LOOKUP
    assert "chicken" in cands
    assert "peanuts" in cands


def test_mapping_dataset_ground_truth():
    gt = {"custom_bowl": ["brown_rice", "black_beans", "corn"]}
    mapping = IngredientMapping(ground_truth_index=gt)
    cands, confs, source = mapping.lookup("custom_bowl")
    assert source == IngredientSource.DATASET_GROUND_TRUTH
    assert cands == ["brown_rice", "black_beans", "corn"]
    assert confs["brown_rice"] == 1.0


def test_mapping_unknown_never_fabricates():
    mapping = IngredientMapping()
    cands, confs, source = mapping.lookup("martian_sand_cake")
    assert source == IngredientSource.UNAVAILABLE
    assert cands == []
    assert confs == {}

    cands_empty, _, src_empty = mapping.lookup("")
    assert src_empty == IngredientSource.UNAVAILABLE
    assert cands_empty == []


# ---------------------------------------------------------------------------
# 2. Multi-Label Model Architecture Tests
# ---------------------------------------------------------------------------

def test_multilabel_model_forward():
    model = MultiLabelIngredientModel(num_ingredients=10, backbone="efficientnet_b0", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 10)

    probs = model.predict_probs(x)
    assert probs.shape == (2, 10)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_multilabel_model_prediction_and_checkpoint(tmp_path):
    vocab = ["apple", "butter", "cheese", "dough", "egg"]
    model = MultiLabelIngredientModel(num_ingredients=5, backbone="efficientnet_b0", pretrained=False)
    ckpt_path = tmp_path / "model.pt"
    model.save_checkpoint(ckpt_path, ingredient_vocab=vocab)
    assert ckpt_path.is_file()

    loaded_model, loaded_vocab = MultiLabelIngredientModel.load_from_checkpoint(ckpt_path)
    assert loaded_vocab == vocab

    x = torch.randn(1, 3, 224, 224)
    preds = loaded_model.predict_ingredients(x, ingredient_vocab=loaded_vocab, threshold=0.0)
    assert len(preds) == 1
    # All 5 items present with threshold 0.0
    assert len(preds[0]) == 5
    assert all(item[0] in vocab for item in preds[0])


# ---------------------------------------------------------------------------
# 3. Dataset & Synthetic Fixture Tests
# ---------------------------------------------------------------------------

def test_ingredient_dataset_and_fixture(tmp_path):
    manifest_path, vocab_path = create_ingredient_fixture(tmp_path / "fix_ingr", samples_count=4)
    assert manifest_path.is_file()
    assert vocab_path.is_file()

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    ds = MultiLabelIngredientDataset(samples, ingredient_vocab=vocab, is_training=False)
    assert len(ds) == 4

    tensor, target = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert target.shape == (len(vocab),)
    # Check that at least one target is 1.0
    assert target.sum().item() > 0


# ---------------------------------------------------------------------------
# 4. Predictor & Instance Annotation Tests
# ---------------------------------------------------------------------------

def test_ingredient_predictor_fallback_and_annotation():
    predictor = IngredientPredictor()

    res = predictor.predict(food_name="caesar_salad")
    assert isinstance(res, IngredientResult)
    assert res.food_name == "caesar_salad"
    assert "romaine lettuce" in res.ingredient_candidates
    assert res.source == IngredientSource.RECIPE_DERIVED

    d = res.as_dict()
    assert d["source"] == "recipe_derived"
    assert len(d["ingredient_candidates"]) > 0

    # Test unknown food
    unknown_res = predictor.predict(food_name="unidentified_alien_snack")
    assert unknown_res.ingredient_candidates == []
    assert unknown_res.source == IngredientSource.UNAVAILABLE

    # Test instance annotation
    instances = [
        FoodInstance(instance_id="f1", bbox=(10, 10, 50, 50), class_name="pizza"),
        FoodInstance(instance_id="f2", bbox=(60, 60, 100, 100), class_name="unknown_dish"),
    ]
    img = Image.new("RGB", (120, 120), color=(200, 200, 200))
    annotated, results = predictor.annotate_instances(instances, image=img)

    assert len(annotated[0].ingredients) > 0
    assert "mozzarella cheese" in annotated[0].ingredients
    assert results[0].source == IngredientSource.RECIPE_DERIVED

    assert annotated[1].ingredients == []
    assert results[1].source == IngredientSource.UNAVAILABLE


# ---------------------------------------------------------------------------
# 5. Backwards Compatibility with IngredientLookup
# ---------------------------------------------------------------------------

def test_ingredient_lookup_backwards_compatible():
    lookup = IngredientLookup()
    lookup.load()
    assert len(lookup.lookup("pizza")) > 0
    assert lookup.lookup("non_existent_dish") == []
