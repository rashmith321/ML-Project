#!/usr/bin/env python
"""
Validate the manifests written by create_manifests.py:
  1. Schema: every row has the two always-required fields (image_path, dataset).
  2. File existence: image_path (and mask_path/depth_path if present) must
     exist on disk right now -- a manifest can go stale if files move.
  3. Numeric sanity: mass_g/calories/protein_g/carbohydrates_g/fat_g, when
     present, must be >= 0 (never fabricated, just range-checked).
  4. No duplicate instance_id within one dataset's manifest.
  5. Split leakage: no group_id (see src/data/splits.py) has members in
     more than one split.

Writes reports/validation_report.md. Exits non-zero if any hard failure
(existence/schema/duplicate/leakage) was found. Numeric range issues are
reported as warnings (soft) since they may indicate an upstream labeling
oddity worth a human look rather than a manifest bug.

Usage:
    python scripts/validate_datasets.py --env windows
    python scripts/validate_datasets.py --env local --data-root /some/path
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import ManifestSample
from src.data.splits import find_split_leakage
from src.utils.config import load_config

NUMERIC_FIELDS = ("mass_g", "calories", "protein_g", "carbohydrates_g", "fat_g")


def _load_manifest(path: Path) -> list[ManifestSample]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(ManifestSample(**d))
    return samples


def validate_dataset(name: str, path: Path) -> dict:
    findings = {"errors": [], "warnings": []}
    if not path.is_file():
        findings["errors"].append(f"Manifest file missing: {path}")
        return findings

    samples = _load_manifest(path)
    seen_ids = set()

    for i, s in enumerate(samples):
        if not s.image_path or not s.dataset:
            findings["errors"].append(f"Row {i}: missing required field(s) image_path/dataset.")
            continue

        if not Path(s.image_path).is_file():
            findings["errors"].append(f"Row {i} ({s.instance_id}): image_path does not exist: {s.image_path}")
        if s.mask_path and not Path(s.mask_path).is_file():
            findings["errors"].append(f"Row {i} ({s.instance_id}): mask_path does not exist: {s.mask_path}")
        if s.depth_path and not Path(s.depth_path).is_file():
            findings["errors"].append(f"Row {i} ({s.instance_id}): depth_path does not exist: {s.depth_path}")

        if s.instance_id is not None:
            if s.instance_id in seen_ids:
                findings["errors"].append(f"Duplicate instance_id within {name}: {s.instance_id}")
            seen_ids.add(s.instance_id)

        for field_name in NUMERIC_FIELDS:
            val = getattr(s, field_name)
            if val is not None and val < 0:
                findings["warnings"].append(
                    f"Row {i} ({s.instance_id}): {field_name}={val} is negative."
                )

    leakage = find_split_leakage(samples)
    for finding in leakage:
        findings["errors"].append(
            f"Split leakage in {name}: group_id={finding['group_id']!r} appears in splits {finding['splits']}"
        )

    findings["num_samples"] = len(samples)
    return findings


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

    metadata_dir = data_root / "metadata"
    index_path = metadata_dir / "manifest_index.json"

    lines = ["# Dataset validation report", ""]
    any_hard_failure = False

    if not index_path.is_file():
        lines.append(
            f"No manifest_index.json found at {index_path}. "
            f"Run scripts/create_manifests.py first."
        )
        report_path = reports_dir / "validation_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Wrote {report_path}")
        print("No manifests to validate.")
        sys.exit(0)

    with open(index_path) as f:
        index = json.load(f)

    for name, entry in index.items():
        lines.append(f"## {name}")
        if not entry.get("available"):
            lines.append("- Not available -- no manifest to validate.")
            lines.append("")
            continue

        findings = validate_dataset(name, Path(entry["manifest_path"]))
        lines.append(f"- Samples checked: {findings.get('num_samples', 0)}")
        lines.append(f"- Errors: {len(findings['errors'])}")
        lines.append(f"- Warnings: {len(findings['warnings'])}")
        if findings["errors"]:
            any_hard_failure = True
            lines.append("- **Error details:**")
            for e in findings["errors"][:50]:
                lines.append(f"  - {e}")
            if len(findings["errors"]) > 50:
                lines.append(f"  - ... and {len(findings['errors']) - 50} more")
        if findings["warnings"]:
            lines.append("- **Warning details:**")
            for w in findings["warnings"][:20]:
                lines.append(f"  - {w}")
            if len(findings["warnings"]) > 20:
                lines.append(f"  - ... and {len(findings['warnings']) - 20} more")
        lines.append("")

    report_path = reports_dir / "validation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {report_path}")
    if any_hard_failure:
        print("VALIDATION FAILED -- see error details above.")
        sys.exit(1)
    print("Validation passed (no hard failures).")


if __name__ == "__main__":
    main()
