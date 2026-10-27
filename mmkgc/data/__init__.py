"""Data loading helpers and dataset adapter contracts."""

from mmkgc.data.contracts import (
    MODALITY_COMPLETION_STATUSES,
    DatasetStore,
    EntityRecord,
    build_modality_completion_policy,
    target_modalities,
)
from mmkgc.data.factory import DATASET_STORE_TYPES, build_dataset_store
from mmkgc.data.mkgw import GRAPH_SPLIT_FILES, MKGWDataStore

__all__ = [
    "DATASET_STORE_TYPES",
    "DatasetStore",
    "EntityRecord",
    "GRAPH_SPLIT_FILES",
    "MODALITY_COMPLETION_STATUSES",
    "MKGWDataStore",
    "build_modality_completion_policy",
    "build_dataset_store",
    "target_modalities",
]
