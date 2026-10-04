#!/usr/bin/env python
"""
Build standardized manifests for every available dataset.

For each dataset in the registry: if it isn't found on disk, no manifest
file is written for it (an empty/fake manifest would be a fabricated
value). If it is found, its adapter builds real ManifestSample rows from
the files that actually exist, then:
  - datasets with NO official split (per configs/datasets.yaml
    `official_split: false`) get a grouped random split assigned
    (src/data/splits.assign_grouped_split) so that same-item samples
    (e.g. multiple ECUSTFD camera views of one dish) never straddle splits.
  - datasets WITH an official split keep whatever their adapter already
    set from the official source; any row the adapter couldn't match to
    the official split file is left split=None (unassigned), not guessed.

Writes one JSONL file per dataset to data/metadata/<name>_manifest.jsonl
(one standardized sample per line, only the fields that dataset actually
provides -- see src/data/schema.py:ManifestSample.as_dict). Also writes
data/metadata/manifest_index.json, a small catalog of which manifests
exist and their row counts -- NOT a merged training pool; each dataset's
manifest stays a separate file, per DATASET_PLAN.md's "no blind merging"
rule.

Usage:
    python scripts/create_manifests.py --env windows
    python scripts/create_manifests.py --env colab
    python scripts/create_manifests.py --env local --data-root /some/path
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.registry import REGISTRY
from src.data.splits import assign_grouped_split
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="windows | colab | local (configs/env.<env>.yaml)")
    parser.add_argument("--data-root", default=None, help="Override data_root instead of reading it from the env config")
    parser.add_argument("--config-dir", default=None, help="Override configs/ directory")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for grouped-random split assignment (datasets with no official split)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = Path(args.config_dir) if args.config_dir else repo_root / "configs"

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        cfg = load_config(env=args.env, config_dir=config_dir)
        data_root = Path(cfg["data_root"])

    with open(config_dir / "datasets.yaml") as f:
        datasets_cfg = yaml.safe_load(f)

    raw_dir = data_root / "raw"
    metadata_dir = data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    index = {}
    for name, adapter_cls in REGISTRY.items():
        cfg = datasets_cfg["datasets"][name]
        adapter = adapter_cls(raw_dir=raw_dir, cfg=cfg)

        if not adapter.is_present:
            print(f"[skip] {name}: not found under {raw_dir} -- no manifest written.")
            index[name] = {"available": False, "manifest_path": None, "num_samples": 0}
            continue

        samples = adapter.build_manifest()
        if not cfg.get("official_split", False):
            assign_grouped_split(samples, seed=args.split_seed)

        out_path = metadata_dir / f"{name}_manifest.jsonl"
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s.as_dict()) + "\n")

        n_unassigned = sum(1 for s in samples if s.split is None)
        print(f"[ok] {name}: {len(samples)} samples -> {out_path} ({n_unassigned} with split=unassigned)")
        index[name] = {
            "available": True,
            "manifest_path": str(out_path),
            "num_samples": len(samples),
            "num_split_unassigned": n_unassigned,
        }

    index_path = metadata_dir / "manifest_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
