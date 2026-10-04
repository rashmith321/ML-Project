"""
Base class for per-dataset adapters, plus small filesystem-probing helpers
shared by all of them. Every adapter is responsible only for ITS OWN
dataset's native layout -- no adapter reads another dataset's files, and
no adapter merges data across datasets (per registry.py's design note and
DATASET_PLAN.md's "no blind merging" rule).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

from src.data.schema import DatasetStatus, ManifestSample

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_dataset_root(raw_dir: Path, subdir: str, aliases: Iterable[str]) -> Optional[Path]:
    """
    Try `subdir` then each alias under raw_dir. Returns the first path that
    exists as a directory, or None if none exist. Never creates anything.
    """
    for candidate in (subdir, *aliases):
        p = raw_dir / candidate
        if p.is_dir():
            return p
    return None


def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


class BaseDatasetAdapter(ABC):
    """
    name           -- registry key, matches configs/datasets.yaml
    display_name   -- human-readable name
    """
    name: str
    display_name: str

    def __init__(self, raw_dir: Path, cfg: dict):
        """
        raw_dir: <data_root>/raw  (the adapter resolves its own subdir under this)
        cfg: this dataset's entry from configs/datasets.yaml (declarative only)
        """
        self.raw_dir = raw_dir
        self.cfg = cfg
        self.root = find_dataset_root(
            raw_dir, cfg["raw_subdir"], cfg.get("raw_subdir_aliases", [])
        )

    @property
    def is_present(self) -> bool:
        return self.root is not None

    @abstractmethod
    def inspect(self) -> DatasetStatus:
        """Measure real availability/counts/label-types from disk. Must not
        raise if self.root is None -- return an `available=False` status."""
        raise NotImplementedError

    @abstractmethod
    def build_manifest(self) -> list[ManifestSample]:
        """
        Return standardized manifest rows. MUST return an empty list (not
        raise, not fabricate rows) if self.root is None. Every row must only
        set fields this dataset genuinely provides for that sample.
        """
        raise NotImplementedError

    def _not_found_status(self) -> DatasetStatus:
        return DatasetStatus(
            name=self.name,
            display_name=self.display_name,
            available=False,
            notes=[
                f"Expected under one of: "
                f"{[self.cfg['raw_subdir'], *self.cfg.get('raw_subdir_aliases', [])]} "
                f"relative to {self.raw_dir}. None of these directories exist."
            ],
        )
