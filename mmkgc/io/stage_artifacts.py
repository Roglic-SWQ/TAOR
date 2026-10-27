"""Helpers for compact per-stage JSON artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


STAGE_ARTIFACT_SCHEMA_VERSION = "mmkgc_5stage_v1"
LEGACY_STAGE_ARTIFACT_SCHEMA_VERSION = "mmkgc_stage_artifact_v2"
LEGACY_SIX_STAGE_ARTIFACT_SCHEMA_VERSION = "mmkgc_6stage_v3"


def unwrap_stage_artifact(payload: Any) -> Any:
    """Return the inner data for wrapped stage artifacts; leave legacy payloads unchanged."""
    if not isinstance(payload, Mapping):
        return payload
    is_versioned_artifact = payload.get("schema_version") in {
        STAGE_ARTIFACT_SCHEMA_VERSION,
        LEGACY_STAGE_ARTIFACT_SCHEMA_VERSION,
        LEGACY_SIX_STAGE_ARTIFACT_SCHEMA_VERSION,
    }
    is_current_artifact = (
        isinstance(payload.get("stage"), str)
        and isinstance(payload.get("data"), Mapping)
        and "summary" in payload
    )
    if not is_versioned_artifact and not is_current_artifact:
        return payload
    data = payload.get("data")
    return data if data is not None else {}
