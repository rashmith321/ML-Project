#!/usr/bin/env python
"""
Inspect the ACTUAL local/available dataset locations for all seven approved
datasets and write reports/dataset_summary.{md,json}.

This script never assumes a dataset exists: it resolves <data_root>/raw/
via the env config, then asks each dataset's adapter to measure what is
really on disk. If <data_root> is unset, or a dataset's directory isn't
there, that dataset is reported unavailable -- this is the correct,
expected output when run somewhere the data hasn't been placed yet (e.g.
this is exactly what running it in a fresh sandbox with no data mounted
looks like); it is not a bug in the script.

Usage:
    python scripts/inspect_datasets.py --env windows
    python scripts/inspect_datasets.py --env colab
    python scripts/inspect_datasets.py --env local          # for local dev/testing only
    python scripts/inspect_datasets.py --env local --data-root /some/path   # override
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.registry import REGISTRY
from src.utils.config import load_config

EXCLUDED_NOTE = (
    "Explicitly excluded per project brief: foodx251 (unavailable, "
    "Kaggle-gated), khana (Indian food dataset; image set Kaggle-gated, "
    "only metadata obtained)."
)


def _load_datasets_cfg(config_dir: Path) -> dict:
    with open(config_dir / "datasets.yaml") as f:
        return yaml.safe_load(f)


def run_inspection(data_root: Path, config_dir: Path) -> list[dict]:
    datasets_cfg = _load_datasets_cfg(config_dir)
    raw_dir = data_root / "raw"

    results = []
    for name, adapter_cls in REGISTRY.items():
        cfg = datasets_cfg["datasets"][name]
        adapter = adapter_cls(raw_dir=raw_dir, cfg=cfg)
        status = adapter.inspect()
        results.append(status.as_dict())
    return results


def render_markdown(results: list[dict], data_root: Path) -> str:
    lines = []
    lines.append("# Dataset availability report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"data_root: `{data_root}`")
    lines.append("")
    n_available = sum(1 for r in results if r["available"])
    lines.append(f"**{n_available} / {len(results)} datasets available at this data_root.**")
    lines.append("")

    for r in results:
        lines.append(f"## {r['display_name']} (`{r['name']}`)")
        lines.append("")
        lines.append(f"- **Available:** {r['available']}")
        lines.append(f"- **Location:** {r['location'] or 'not found'}")
        lines.append(f"- **Images:** {r['num_images'] if r['num_images'] is not None else 'unknown'}")
        lines.append(f"- **Classes:** {r['num_classes'] if r['num_classes'] is not None else 'unknown / not applicable'}")
        lines.append(f"- **Annotation format:** {r['annotation_format'] or 'n/a'}")
        lines.append(f"- **Nutrition labels:** {r['has_nutrition_labels']}")
        lines.append(f"- **Segmentation labels:** {r['has_segmentation_labels']}")
        lines.append(f"- **Detection labels:** {r['has_detection_labels']}")
        lines.append(f"- **Mass labels:** {r['has_mass_labels']}")
        lines.append(f"- **Depth:** {r['has_depth']}")
        lines.append(f"- **Ingredient info:** {r['has_ingredient_info']}")
        lines.append(f"- **Train/test split available:** {r['train_test_split_available']} (source: {r['split_source'] or 'n/a'})")
        if r["notes"]:
            lines.append("- **Notes:**")
            for note in r["notes"]:
                lines.append(f"  - {note}")
        lines.append("")

    lines.append("## Excluded datasets")
    lines.append("")
    lines.append(EXCLUDED_NOTE)
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="windows | colab | local (configs/env.<env>.yaml)")
    parser.add_argument("--data-root", default=None, help="Override data_root instead of reading it from the env config")
    parser.add_argument("--config-dir", default=None, help="Override configs/ directory")
    parser.add_argument("--reports-dir", default=None, help="Override reports/ output directory")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = Path(args.config_dir) if args.config_dir else repo_root / "configs"
    reports_dir = Path(args.reports_dir) if args.reports_dir else repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        cfg = load_config(env=args.env, config_dir=config_dir)
        data_root = Path(cfg["data_root"])

    results = run_inspection(data_root, config_dir)

    json_path = reports_dir / "dataset_summary.json"
    md_path = reports_dir / "dataset_summary.md"

    with open(json_path, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "data_root": str(data_root),
                "datasets": results,
                "excluded": EXCLUDED_NOTE,
            },
            f,
            indent=2,
        )

    with open(md_path, "w") as f:
        f.write(render_markdown(results, data_root))

    n_available = sum(1 for r in results if r["available"])
    print(f"Inspected {len(results)} datasets under {data_root}/raw -- {n_available} available.")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
