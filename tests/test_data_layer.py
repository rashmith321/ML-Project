"""
Tests for the Phase 1 data-engineering layer: adapters correctly report
"not available" against an empty tree, correctly parse a real (synthetic
but format-accurate) layout, and the split/leakage utilities behave
correctly. Run with: pytest tests/
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ecustfd import ECUSTFDAdapter
from src.data.food101 import Food101Adapter
from src.data.nutrition5k import Nutrition5kAdapter
from src.data.schema import ManifestSample
from src.data.splits import assign_grouped_split, find_split_leakage

_COMMON_CFG_DEFAULTS = {"raw_subdir_aliases": []}


def _cfg(raw_subdir: str) -> dict:
    return {"raw_subdir": raw_subdir, **_COMMON_CFG_DEFAULTS}


# ---------------------------------------------------------------------------
# Absence must be reported honestly, never fabricated
# ---------------------------------------------------------------------------

def test_nutrition5k_reports_unavailable_when_absent(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    adapter = Nutrition5kAdapter(raw_dir=raw_dir, cfg=_cfg("nutrition5k"))
    status = adapter.inspect()
    assert status.available is False
    assert status.num_images is None
    assert status.location is None
    assert adapter.build_manifest() == []


def test_food101_reports_unavailable_when_absent(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    adapter = Food101Adapter(raw_dir=raw_dir, cfg=_cfg("food-101"))
    status = adapter.inspect()
    assert status.available is False
    assert adapter.build_manifest() == []


# ---------------------------------------------------------------------------
# ManifestSample.as_dict() omits missing fields rather than nulling them
# ---------------------------------------------------------------------------

def test_manifest_sample_omits_missing_fields():
    s = ManifestSample(image_path="/x.jpg", dataset="food101", food_class="sushi")
    d = s.as_dict()
    assert d == {"image_path": "/x.jpg", "dataset": "food101", "food_class": "sushi"}
    assert "mass_g" not in d
    assert "calories" not in d


# ---------------------------------------------------------------------------
# Nutrition5k adapter: real (synthetic, format-accurate) parse
# ---------------------------------------------------------------------------

def _make_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid-enough placeholder; adapters only check is_file(), not
    # pixel content, so a tiny stub file is sufficient and honest to use.
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def test_nutrition5k_parses_metadata_and_matches_official_split(tmp_path):
    raw_dir = tmp_path / "raw"
    root = raw_dir / "nutrition5k"
    (root / "metadata").mkdir(parents=True)
    with open(root / "metadata" / "dish_metadata_cafe1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dish_001", 450.5, 320.0, 12.5, 55.2, 20.1,
                    "ing1", "rice", 150.0, 180.0, 0.5, 40.0, 3.5])
    _make_png(root / "imagery" / "realsense_overhead" / "dish_001" / "rgb.png")
    (root / "dish_ids" / "splits").mkdir(parents=True)
    (root / "dish_ids" / "splits" / "rgb_train_ids.txt").write_text("dish_001\n")

    adapter = Nutrition5kAdapter(raw_dir=raw_dir, cfg=_cfg("nutrition5k"))
    status = adapter.inspect()
    assert status.available is True
    assert status.has_mass_labels is True
    assert status.train_test_split_available is True

    samples = adapter.build_manifest()
    assert len(samples) == 1
    assert samples[0].mass_g == 320.0
    assert samples[0].calories == 450.5
    assert samples[0].split == "train"
    assert samples[0].ingredients == ["rice"]


def test_nutrition5k_skips_metadata_row_with_no_image(tmp_path):
    raw_dir = tmp_path / "raw"
    root = raw_dir / "nutrition5k"
    (root / "metadata").mkdir(parents=True)
    with open(root / "metadata" / "dish_metadata_cafe1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dish_missing_image", 100.0, 50.0, 1.0, 5.0, 2.0])

    adapter = Nutrition5kAdapter(raw_dir=raw_dir, cfg=_cfg("nutrition5k"))
    samples = adapter.build_manifest()
    assert samples == []  # no image on disk -> no fabricated row


# ---------------------------------------------------------------------------
# ECUSTFD adapter: grouping by dish (view suffix stripped)
# ---------------------------------------------------------------------------

def test_ecustfd_groups_multiview_same_dish(tmp_path):
    raw_dir = tmp_path / "raw"
    root = raw_dir / "ecustfd"
    (root / "JPEGImages").mkdir(parents=True)
    _make_png(root / "JPEGImages" / "apple001_1.jpg")
    _make_png(root / "JPEGImages" / "apple001_2.jpg")
    with open(root / "weight.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "weight(g)"])
        w.writerow(["apple001_1", 152.3])
        w.writerow(["apple001_2", 152.3])

    adapter = ECUSTFDAdapter(raw_dir=raw_dir, cfg=_cfg("ecustfd"))
    samples = adapter.build_manifest()
    assert len(samples) == 2
    assert samples[0].group_id == samples[1].group_id == "apple001"
    assert all(s.mass_g == 152.3 for s in samples)


# ---------------------------------------------------------------------------
# Split assignment + leakage detection
# ---------------------------------------------------------------------------

def test_assign_grouped_split_keeps_group_together():
    samples = [
        ManifestSample(image_path="/a1.jpg", dataset="ecustfd", group_id="dishA"),
        ManifestSample(image_path="/a2.jpg", dataset="ecustfd", group_id="dishA"),
        ManifestSample(image_path="/b1.jpg", dataset="ecustfd", group_id="dishB"),
    ]
    assign_grouped_split(samples, seed=1)
    assert samples[0].split == samples[1].split
    assert all(s.split in ("train", "val", "test") for s in samples)


def test_assign_grouped_split_is_deterministic():
    def build():
        return [
            ManifestSample(image_path="/a.jpg", dataset="x", group_id="g1"),
            ManifestSample(image_path="/b.jpg", dataset="x", group_id="g2"),
            ManifestSample(image_path="/c.jpg", dataset="x", group_id="g3"),
        ]
    s1, s2 = build(), build()
    assign_grouped_split(s1, seed=7)
    assign_grouped_split(s2, seed=7)
    assert [s.split for s in s1] == [s.split for s in s2]


def test_assign_grouped_split_leaves_existing_split_untouched():
    samples = [ManifestSample(image_path="/a.jpg", dataset="x", group_id="g1", split="train")]
    assign_grouped_split(samples, seed=1)
    assert samples[0].split == "train"


def test_find_split_leakage_detects_group_in_two_splits():
    samples = [
        ManifestSample(image_path="/a1.jpg", dataset="x", group_id="dishA", split="train"),
        ManifestSample(image_path="/a2.jpg", dataset="x", group_id="dishA", split="test"),
    ]
    findings = find_split_leakage(samples)
    assert len(findings) == 1
    assert findings[0]["group_id"] == "dishA"


def test_find_split_leakage_clean_case_reports_nothing():
    samples = [
        ManifestSample(image_path="/a1.jpg", dataset="x", group_id="dishA", split="train"),
        ManifestSample(image_path="/a2.jpg", dataset="x", group_id="dishA", split="train"),
        ManifestSample(image_path="/b1.jpg", dataset="x", group_id="dishB", split="test"),
    ]
    assert find_split_leakage(samples) == []


def test_find_split_leakage_ignores_unassigned_split():
    samples = [
        ManifestSample(image_path="/a1.jpg", dataset="x", group_id="dishA", split=None),
        ManifestSample(image_path="/a2.jpg", dataset="x", group_id="dishA", split=None),
    ]
    assert find_split_leakage(samples) == []
