"""Dataset contracts shared by MMKGC data adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


MODALITY_COMPLETION_STATUSES = (
    "available",
    "masked_for_eval",
    "real_missing",
    "partial_missing",
)


@dataclass(frozen=True)
class EntityRecord:
    """Canonical entity payload consumed by the Stage 0-5 pipeline."""

    entity_id: int
    entity_key: str
    label: str
    description: str
    neighbors: List[Dict[str, Any]]
    existing_modalities: Dict[str, Optional[str]]
    ground_truth_modalities: Dict[str, Optional[str]]
    modality_completion_policy: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@runtime_checkable
class DatasetStore(Protocol):
    """Common interface implemented by dataset-specific stores."""

    def build_record(self, entity_id: int, missing_mode: str) -> EntityRecord:
        """Build one canonical entity record."""

    def select_records(
        self,
        missing_mode: str,
        limit: int | None = None,
    ) -> list[EntityRecord]:
        """Select records in the dataset's default order."""

    def select_records_by_keys(
        self,
        missing_mode: str,
        entity_keys: list[str],
    ) -> list[EntityRecord]:
        """Select records by public entity keys, preserving the input order."""


def target_modalities(missing_mode: str) -> list[str]:
    """Return modalities targeted by a completion run."""
    if missing_mode in {"text", "image"}:
        return [missing_mode]
    raise ValueError("missing_mode must be one of: text, image")


def build_modality_completion_policy(
    missing_mode: str,
    existing_modalities: Dict[str, Optional[str]],
    ground_truth_modalities: Dict[str, Optional[str]],
    *,
    requested_statuses: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Describe why each modality is kept, generated, or enhanced."""
    targets = set(target_modalities(missing_mode))
    requested_statuses = requested_statuses or {}
    policy: Dict[str, Dict[str, Any]] = {}
    for modality in ("text", "image"):
        requested = dict(requested_statuses.get(modality, {}))
        target = modality in targets
        has_existing = bool(existing_modalities.get(modality))
        has_ground_truth = bool(ground_truth_modalities.get(modality))
        requested_status = requested.get("status")
        if requested_status == "partial_missing":
            status = "partial_missing"
            action = "enhance" if target else "keep"
            can_evaluate = False
        elif target and has_ground_truth:
            status = "masked_for_eval"
            action = "generate"
            can_evaluate = True
        elif target:
            status = "real_missing"
            action = "generate"
            can_evaluate = False
        else:
            status = "available" if has_existing else "real_missing"
            action = "keep"
            can_evaluate = False
        row: Dict[str, Any] = {
            "status": status,
            "target": target,
            "action": action,
            "has_existing": has_existing,
            "has_ground_truth": has_ground_truth if status != "partial_missing" else False,
            "can_evaluate_against_ground_truth": can_evaluate,
        }
        if requested.get("reason"):
            row["reason"] = str(requested["reason"])
        policy[modality] = row
    return policy
