"""Factory helpers for dataset-specific stores."""

from __future__ import annotations

from pathlib import Path

from mmkgc.data.contracts import DatasetStore
from mmkgc.data.mkgw import MKGWDataStore


DATASET_STORE_TYPES = ("mkgw",)


def build_dataset_store(
    dataset_type: str,
    dataset_dir: str | Path,
    *,
    graph_split: str = "train",
) -> DatasetStore:
    """Build a dataset store for the requested dataset format."""
    if dataset_type == "mkgw":
        return MKGWDataStore(dataset_dir, graph_split=graph_split)

    choices = ", ".join(DATASET_STORE_TYPES)
    raise ValueError(f"dataset_type must be one of: {choices}")
