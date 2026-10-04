# Food Image Nutrition Estimation System

A final-year research project expanding food *calorie* estimation into full
multi-food, multi-nutrient estimation: detection → segmentation →
classification → ingredient understanding → portion/mass/volume estimation
→ multi-nutrient (calories/protein/carbs/fat) estimation → meal report.

## Attribution

This project takes the reference implementation at
[chetan-jarande/Food-calorie-estimations-Using-Deep-Learning-And-Computer-Vision](https://github.com/chetan-jarande/Food-calorie-estimations-Using-Deep-Learning-And-Computer-Vision)
(by Chetan Jarande, Mukta Bhagwat, Vishakha Patil, and Diya Ukirde,
CC BY-NC 4.0) as its starting reference and academic baseline. Per that
project's attribution requirement: **this is not that project** — the
reference repo is detection-only, single-target-nutrient (calories), and
built on a different, custom dataset. See `docs/PROJECT_AUDIT.md` for the
full audit of what was and wasn't reused. This project's own architecture,
dataset selection, and code are new, described in `docs/ARCHITECTURE.md`.

## Docs

- [`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md) — audit of the reference repo + status of prior Drive work.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline design and module responsibilities.
- [`docs/DATASET_PLAN.md`](docs/DATASET_PLAN.md) — which dataset does what, and why.
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — training sequence, evaluation, risks.

## Status

**Phase 1 (data engineering) is implemented and tested.** Scaffold stage
for the rest: directory structure, config system, shared schema, and the
nutrition-aggregation module are implemented and tested. Detection,
segmentation, classification, portion, and nutrient-regression stages are
interface-defined but not yet trained — see `docs/IMPLEMENTATION_PLAN.md`
for the build order and immediate next actions.

## Phase 1: data-engineering layer

`src/data/` has one adapter per approved dataset (Nutrition5k, Food-101,
UNIMIB2016, ECUSTFD, Recipe1M+, Vireo Food-172, MenuMatch). Each adapter
only reads its own dataset's real files on disk — nothing is assumed or
fabricated; a dataset not found at `<data_root>/raw/<name>/` is reported
`available: False`, not silently skipped or guessed at.

```bash
# 1. Point at wherever your data actually lives (see Setup below for
#    Windows/Colab env vars), then:
python scripts/inspect_datasets.py --env windows        # or --env colab
python scripts/create_manifests.py --env windows
python scripts/validate_datasets.py --env windows        # exits non-zero on any hard failure
```

- `configs/datasets.yaml` — declares where each dataset is expected under
  `data_root/raw/` and what label types it's documented to provide (used
  for report context only, never trusted over what's actually on disk).
- `scripts/inspect_datasets.py` → `reports/dataset_summary.{md,json}` —
  availability, image/class counts, annotation format, and which of
  {nutrition, segmentation, detection, mass, depth, ingredients, split}
  labels each dataset actually has, measured from disk.
- `scripts/create_manifests.py` → `data/metadata/<name>_manifest.jsonl` —
  one standardized row per sample (`src/data/schema.py:ManifestSample`);
  a field is present only if that dataset actually provides it for that
  sample. Assigns a grouped random split (seeded, reproducible) for the
  two datasets with no official split (UNIMIB2016, ECUSTFD), always
  keeping same-item samples (e.g. multiple camera views of one ECUSTFD
  dish) together — see `docs/DATA_SPLIT_STRATEGY.md`.
- `scripts/validate_datasets.py` → `reports/validation_report.md` — file
  existence, numeric sanity, duplicate-id, and cross-split leakage checks;
  non-zero exit on any hard failure.

`reports/example_run_with_fixture_data/` holds output from a synthetic
fixture (a minimal but format-accurate mini-copy of each dataset's real
layout) that was used to verify the adapters actually parse correctly and
that leakage detection actually fires — not just that they no-op on an
empty directory. It is a verification artifact, not real project data.
`tests/test_data_layer.py` covers the same cases as pytest.

## Setup

```bash
pip install -r requirements.txt

# Windows (PowerShell):
$env:FOOD_PROJECT_DATA_ROOT = "C:\path\to\data"
$env:FOOD_PROJECT_MODELS_ROOT = "C:\path\to\models"
$env:FOOD_PROJECT_OUTPUTS_ROOT = "C:\path\to\outputs"

# Colab:
import os
os.environ["FOOD_PROJECT_DATA_ROOT"] = "/content/drive/MyDrive/food_seg_project"
os.environ["FOOD_PROJECT_MODELS_ROOT"] = "/content/drive/MyDrive/food_seg_project/models"
os.environ["FOOD_PROJECT_OUTPUTS_ROOT"] = "/content/drive/MyDrive/food_seg_project/outputs"
```

```python
from src.utils.config import load_config
cfg = load_config(env="colab")   # or env="windows"
```

## Tests

```bash
pytest tests/
```

## Project structure

```
data/            raw / processed / metadata / splits (gitignored)
configs/         base.yaml (dataset roles + model choices) + env.*.yaml (paths, device)
notebooks/       exploratory / Colab notebooks
src/             the package: data, preprocessing, detection, segmentation,
                 classification, ingredients, portion, nutrition, models,
                 evaluation, utils
training/        training entry points (per-stage, to be added)
inference/       predict.py -- end-to-end pipeline runner
models/          saved weights (gitignored)
outputs/         run outputs (gitignored)
reports/         generated nutrition reports
app/             user-facing app (Streamlit/FastAPI, TBD)
tests/           pytest suite
scripts/         one-off utility scripts
docs/            the four planning docs above
```
