"""
Split assignment + leakage checking.

Split strategy (documented, per project requirement -- see also
reports/dataset_summary.md "Split strategy" section, generated from this
module's docstring content, and configs/datasets.yaml's per-dataset
`split_notes`):

1. If a dataset ships an official split (Nutrition5k dish_ids/splits/,
   Food-101 meta/{train,test}.txt, Recipe1M+ layer1.json `partition`,
   Vireo-172 SplitAndIngreLabel/{TR,VAL,TE}.txt), the corresponding adapter
   sets `split` directly from that official source. This module is never
   used for those datasets' primary split assignment.
2. MenuMatch is evaluation-only (DATASET_PLAN.md) -- its adapter assigns
   split="eval_only" to every sample; nothing here applies to it.
3. UNIMIB2016 and ECUSTFD ship no official split. For these, this module
   performs a GROUPED random split: every sample's `group_id` (falling back
   to `instance_id` if a dataset has no natural grouping) is hashed with a
   fixed seed into train/val/test buckets, and every sample sharing a
   group_id goes to the same bucket. This is what prevents, e.g., two
   camera views of the same ECUSTFD dish -- or, in general, any
   same-underlying-item samples -- from ending up split across train and
   test.
4. Any sample whose split is still None after steps 1-3 (e.g. an official
   split file was expected but not found on disk) is left as split=None
   ("unassigned") rather than being forced into a bucket -- an unassigned
   sample must not be used for training or evaluation until this is
   resolved, and reports/dataset_summary.md surfaces the count.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from src.data.schema import ManifestSample

DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _bucket_for_group(group_id: str, seed: int, ratios: dict) -> str:
    """Deterministic hash -> bucket, so re-running produces the identical split."""
    h = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, ratio in ratios.items():
        cumulative += ratio
        if frac <= cumulative:
            return name
    return list(ratios.keys())[-1]


def assign_grouped_split(
    samples: Sequence[ManifestSample],
    seed: int = 42,
    ratios: dict | None = None,
) -> None:
    """
    In-place: for every sample with split is None, assign one based on a
    seeded hash of its group_id (or instance_id if group_id is absent).
    Samples that already have a split (e.g. from an official source) are
    left untouched.
    """
    ratios = ratios or DEFAULT_RATIOS
    for s in samples:
        if s.split is not None:
            continue
        key = s.group_id or s.instance_id or s.image_path
        s.split = _bucket_for_group(key, seed, ratios)


def find_split_leakage(samples: Sequence[ManifestSample]) -> list[dict]:
    """
    Returns a list of leakage findings: groups whose members appear in more
    than one non-None split. Each finding is
    {"group_id": ..., "splits": {"train": n, "test": m, ...}}.
    Samples with split=None are ignored (nothing to check yet).
    """
    by_group: dict[str, dict[str, int]] = {}
    for s in samples:
        if s.split is None:
            continue
        key = s.group_id or s.instance_id or s.image_path
        by_group.setdefault(key, {}).setdefault(s.split, 0)
        by_group[key][s.split] += 1

    findings = []
    for group_id, split_counts in by_group.items():
        if len(split_counts) > 1:
            findings.append({"group_id": group_id, "splits": split_counts})
    return findings
