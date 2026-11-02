"""Normalize available perception outputs without treating media paths as evidence."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

MODALITIES = {"text", "image", "video", "audio"}
SOURCE_FIELDS = {
    "text": "text", "image_summary": "image", "video_summary": "video",
    "video_transcript": "video", "audio_transcript": "audio", "audio_summary": "audio",
}


def public_sources(record: Any, target: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Accept text, captions, transcripts and located observations from source adapters.

    A raw image/video/audio path is an asset, never a perceptual observation. A
    captioner/ASR/video reader must supply its observation before fact extraction.
    The target modality and observations derived from it stay outside the input.
    """
    if target not in MODALITIES:
        raise ValueError("unknown target modality")
    existing, context, catalog = record.existing_modalities, {}, []
    for field, modality in SOURCE_FIELDS.items():
        if modality == target:
            continue
        value = existing.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be source text, not a media object")
        context[field] = value
        if value and value.strip():
            item = {"evidence_id": "existing:" + field, "source": "existing_modality",
                    "modality": modality, "content": value}
            asset = existing.get(modality) if modality != "text" else None
            if isinstance(asset, str) and asset.strip():
                item["locator"] = {"asset": asset}
            catalog.append(item)
    for modality in ("image", "video", "audio"):
        if modality != target:
            context["has_" + modality] = bool(existing.get(modality))
    observations = existing.get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("observations must be a list of perception outputs")
    known = {item["evidence_id"] for item in catalog}
    public = []
    for observation in observations:
        if not isinstance(observation, dict) or observation.get("modality") not in MODALITIES:
            raise ValueError("each observation needs a known source modality")
        derived = observation.get("derived_from_modalities", [observation["modality"]])
        if not isinstance(derived, list) or any(value not in MODALITIES for value in derived):
            raise ValueError("invalid observation modality provenance")
        if observation["modality"] == target or target in derived:
            continue
        ref, content = observation.get("evidence_id"), observation.get("content")
        if not isinstance(ref, str) or not ref.startswith("existing:") or ref in known:
            raise ValueError("observation evidence IDs must be distinct existing: references")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("observation content must be nonempty perception text")
        if "locator" in observation and not isinstance(observation["locator"], dict):
            raise ValueError("observation locator must be an object")
        item = {"evidence_id": ref, "source": "existing_modality",
                "modality": observation["modality"], "content": content,
                "derived_from_modalities": list(derived)}
        if "locator" in observation:
            item["locator"] = deepcopy(observation["locator"])
        known.add(ref)
        catalog.append(item)
        public.append(deepcopy(item))
    if public:
        context["observations"] = public
    unprocessed = [modality for modality in ("image", "video", "audio")
                   if modality != target and existing.get(modality)
                   and not any(item["modality"] == modality for item in catalog)]
    if unprocessed:
        context["unprocessed_modalities"] = unprocessed
    return context, catalog
