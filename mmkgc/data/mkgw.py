"""MKG-W dataset loader."""

from __future__ import annotations

import json
import csv
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

from mmkgc.data.contracts import EntityRecord, build_modality_completion_policy
from mmkgc.utils import slugify

GRAPH_SPLIT_FILES = {
    "train": ("train.txt",),
    "train-valid": ("train.txt", "valid.txt"),
    "all": ("train.txt", "valid.txt", "test.txt"),
}

IMAGE_SUMMARY_KEYS = ("image_summary", "visual_summary", "image_description")
MODALITY_POLICY_KEYS = (
    "modality_completion_policy",
    "modality_status",
    "missing_modality_status",
)


def _load_json(path: str | Path, default):
    json_path = Path(path)
    if not json_path.exists():
        return default

    raw = json_path.read_text(encoding="utf-8")
    if not raw.strip():
        return default

    return json.loads(raw)


def _key(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", maxsplit=1)[-1]


def _label_variants(label: str, key: str) -> List[str]:
    variants = []
    for value in (
        label,
        label.replace(" ", "_"),
        label.replace("_", " "),
        key,
        key.replace(" ", "_"),
        key.replace("_", " "),
        slugify(label),
        slugify(key),
    ):
        if value and value not in variants:
            variants.append(value)
    return variants


class MKGWDataStore:
    """Load MKG-W mappings, triples, metadata, and modality records."""

    def __init__(self, dataset_dir: str | Path, graph_split: str = "train"):
        if graph_split not in GRAPH_SPLIT_FILES:
            choices = ", ".join(sorted(GRAPH_SPLIT_FILES))
            raise ValueError(f"graph_split must be one of: {choices}")

        self.dataset_dir = Path(dataset_dir)
        self.graph_split = graph_split
        self.graph_filenames = GRAPH_SPLIT_FILES[graph_split]
        self.entity_id_to_uri, self.entity_uri_to_id = self._load_mapping("entity2id.txt")
        self.relation_id_to_uri, self.relation_uri_to_id = self._load_mapping("relation2id.txt")
        self.entity_info = _load_json(self.dataset_dir / "entity_info.json", {})
        self.entity_texts = self._load_entity_texts()
        self.relation_info = _load_json(self.dataset_dir / "relation_info.json", {})
        self._image_indexes = [
            _load_json(self.dataset_dir / "image_alignment" / name, {})
            for name in ("qid2images.json", "entity2images.json", "entity_id2images.json")
        ]
        self.triples = []
        for filename in self.graph_filenames:
            self.triples.extend(self._load_triples(filename))
        self._adjacency = self._build_directed_adjacency()

    def _load_mapping(self, filename: str):
        id_to_uri = {}
        uri_to_id = {}
        path = self.dataset_dir / filename
        if not path.exists():
            return id_to_uri, uri_to_id

        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            uri, raw_id = parts[0], parts[1]
            item_id = int(raw_id)
            id_to_uri[item_id] = uri
            uri_to_id[uri] = item_id

        return id_to_uri, uri_to_id

    def _load_triples(self, filename: str):
        triples = []
        path = self.dataset_dir / filename
        if not path.exists():
            return triples

        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            head_uri, relation_uri, tail_uri = parts[:3]
            if head_uri not in self.entity_uri_to_id or tail_uri not in self.entity_uri_to_id:
                continue
            if relation_uri not in self.relation_uri_to_id:
                continue
            triples.append(
                (
                    self.entity_uri_to_id[head_uri],
                    self.relation_uri_to_id[relation_uri],
                    self.entity_uri_to_id[tail_uri],
                )
            )

        return triples

    def _build_directed_adjacency(self):
        adjacency = {entity_id: [] for entity_id in self.entity_id_to_uri}
        for head_id, relation_id, tail_id in self.triples:
            adjacency.setdefault(head_id, []).append(
                {"entity_id": tail_id, "relation_id": relation_id, "direction": "out"}
            )
            adjacency.setdefault(tail_id, []).append(
                {"entity_id": head_id, "relation_id": relation_id, "direction": "in"}
            )
        return adjacency

    def entity_key(self, entity_id: int) -> str:
        return _key(self.entity_id_to_uri[entity_id])

    def relation_label(self, relation_id: int) -> str:
        relation_key = _key(self.relation_id_to_uri[relation_id])
        return self.relation_info.get(relation_key, relation_key)

    def entity_metadata(self, entity_id: int) -> Dict[str, str]:
        entity_key = self.entity_key(entity_id)
        metadata = self.entity_info.get(entity_key, {})
        label = metadata.get("label") or entity_key
        description = self.entity_texts.get(entity_key) or metadata.get("description") or ""
        return {"label": label, "description": description}

    def _load_entity_texts(self) -> Dict[str, str]:
        path = self.dataset_dir / "entity_texts.tsv"
        if not path.exists():
            return {}
        texts: Dict[str, str] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("abstract_status") != "found":
                    continue
                entity_key = (row.get("entity_key") or "").strip()
                abstract = " ".join((row.get("abstract") or "").split())
                if entity_key and abstract:
                    texts[entity_key] = abstract
        return texts

    def image_path(self, entity_id: int) -> Optional[str]:
        paths = self.image_paths(entity_id)
        return paths[0] if paths else None

    def image_paths(self, entity_id: int) -> List[str]:
        """Load the aligned image list, falling back to entity image directories."""
        metadata = self.entity_metadata(entity_id)
        entity_key = self.entity_key(entity_id)
        paths: List[str] = []
        root = self.dataset_dir.resolve()

        def add(path):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("entity image path must stay within dataset_dir")
            value = str(resolved)
            if value not in paths:
                paths.append(value)

        for index in self._image_indexes:
            for key in (self.entity_id_to_uri[entity_id], entity_key, str(entity_id)):
                values = index.get(key, [])
                if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                    raise ValueError("image alignment entries must be lists of nonempty paths")
                for value in values:
                    # Declared missing/unreadable files fail explicitly when Stage 1 loads them.
                    add(root / value)
        if paths:
            return paths

        image_root = self.dataset_dir / "images"
        extensions = (".jpg", ".jpeg", ".png", ".webp", ".gif")
        directories = [self.dataset_dir / "images_by_id" / str(entity_id),
                       self.dataset_dir / "images_by_id" / entity_key,
                       *[image_root / variant for variant in _label_variants(metadata["label"], entity_key)]]
        for candidate_dir in directories:
            if not candidate_dir.is_dir():
                continue
            for name in ("image.jpg", "image.jpeg", "image.png", "image.webp", "image.gif"):
                candidate = candidate_dir / name
                if candidate.is_file():
                    add(candidate)
            for candidate in sorted(candidate_dir.iterdir()):
                if candidate.is_file() and candidate.suffix.lower() in extensions:
                    add(candidate)
        return paths

    def image_summary(self, entity_id: int) -> Optional[str]:
        """Return a precomputed semantic summary for the entity image, if available."""
        entity_key = self.entity_key(entity_id)
        metadata = self.entity_info.get(entity_key, {})
        for key in IMAGE_SUMMARY_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _neighbor_payload(self, neighbor: Dict[str, Any]) -> Dict[str, Any]:
        neighbor_id = neighbor["entity_id"]
        metadata = self.entity_metadata(neighbor_id)
        return {
            "entity_id": neighbor_id,
            "entity_key": self.entity_key(neighbor_id),
            "label": metadata["label"],
            "description": metadata["description"],
            "neighbor_description": metadata["description"],
            "relation_id": neighbor["relation_id"],
            "relation": self.relation_label(neighbor["relation_id"]),
            "direction": neighbor["direction"],
        }

    def _second_hop_neighbors(
        self,
        entity_id: int,
        *,
        source_entity_id: int,
        max_neighbors: int,
    ) -> List[Dict[str, Any]]:
        items = []
        if max_neighbors <= 0:
            return items
        # Round-robin relation/direction groups so one frequent relation cannot
        # consume the entire bounded second-hop context by file position.
        groups: Dict[tuple[int, str], list[Dict[str, Any]]] = {}
        for neighbor in self._adjacency.get(entity_id, []):
            if neighbor["entity_id"] != source_entity_id:
                groups.setdefault((neighbor["relation_id"], neighbor["direction"]), []).append(neighbor)
        buckets = [sorted(groups[key], key=lambda item: item["entity_id"]) for key in sorted(groups)]
        ordered = (bucket[index] for index in range(min(max_neighbors, max(map(len, buckets), default=0)))
                   for bucket in buckets if index < len(bucket))
        for neighbor in islice(ordered, max_neighbors):
            items.append(self._neighbor_payload(neighbor))
        return items

    def neighbors(
        self,
        entity_id: int,
        max_neighbors: int | None = None,
        max_second_hop_neighbors: int = 8,
    ) -> List[Dict[str, Any]]:
        """Keep every first-hop edge by default; only model views are compressed."""
        if max_neighbors is not None and (type(max_neighbors) is not int or max_neighbors < 0):
            raise ValueError("max_neighbors must be a nonnegative integer or None")
        items = []
        for neighbor in self._adjacency.get(entity_id, [])[:max_neighbors]:
            neighbor_id = neighbor["entity_id"]
            item = self._neighbor_payload(neighbor)
            item["graph_degree"] = len(self._adjacency.get(neighbor_id, []))
            item["neighbors"] = self._second_hop_neighbors(
                neighbor_id,
                source_entity_id=entity_id,
                max_neighbors=max_second_hop_neighbors,
            )
            items.append(item)
        return items

    def build_record(self, entity_id: int, missing_mode: str) -> EntityRecord:
        if missing_mode not in {"text", "image"}:
            raise ValueError("missing_mode must be one of: text, image")

        metadata = self.entity_metadata(entity_id)
        raw_metadata = self.entity_info.get(self.entity_key(entity_id), {})
        description = metadata["description"]
        images = self.image_paths(entity_id)
        image = images[0] if images else None
        image_summary = self.image_summary(entity_id) if image else None
        text = description.strip() or None
        existing_modalities = {"text": text, "image": image, "image_summary": image_summary}
        ground_truth_modalities = {"text": text, "image": image}
        requested_statuses = _requested_modality_statuses(raw_metadata)

        if missing_mode == "text":
            existing_modalities["text"] = None
        else:
            existing_modalities["image"] = None
            existing_modalities["image_summary"] = None
        for modality, requested in requested_statuses.items():
            if requested.get("status") != "partial_missing":
                continue
            if modality == "text":
                existing_modalities["text"] = text
                ground_truth_modalities["text"] = None
            elif modality == "image":
                existing_modalities["image"] = image
                existing_modalities["image_summary"] = image_summary
                ground_truth_modalities["image"] = None
        modality_completion_policy = build_modality_completion_policy(
            missing_mode,
            existing_modalities,
            ground_truth_modalities,
            requested_statuses=requested_statuses,
        )

        return EntityRecord(
            entity_id=entity_id,
            entity_key=self.entity_key(entity_id),
            label=metadata["label"],
            description=description,
            neighbors=self.neighbors(entity_id),
            existing_modalities=existing_modalities,
            ground_truth_modalities=ground_truth_modalities,
            modality_completion_policy=modality_completion_policy,
            image_paths=(images if existing_modalities.get("image") else []),
        )

    def select_records(self, missing_mode: str, limit: int | None = None) -> List[EntityRecord]:
        entity_ids = sorted(self.entity_id_to_uri)
        if limit is not None:
            entity_ids = entity_ids[:limit]
        return [self.build_record(entity_id, missing_mode) for entity_id in entity_ids]

    def select_records_by_keys(
        self,
        missing_mode: str,
        entity_keys: list[str],
    ) -> List[EntityRecord]:
        key_to_id = {
            self.entity_key(entity_id): entity_id
            for entity_id in sorted(self.entity_id_to_uri)
        }
        missing = [key for key in entity_keys if key not in key_to_id]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(f"unknown entity_key: {preview}{suffix}")
        return [self.build_record(key_to_id[key], missing_mode) for key in entity_keys]


def _requested_modality_statuses(metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    statuses: Dict[str, Dict[str, Any]] = {}
    for key in MODALITY_POLICY_KEYS:
        raw = metadata.get(key)
        if isinstance(raw, dict):
            for modality in ("text", "image"):
                value = raw.get(modality)
                parsed = _parse_modality_status(value)
                if parsed:
                    statuses[modality] = parsed
    partial = metadata.get("partial_missing_modalities")
    if isinstance(partial, list):
        for value in partial:
            modality = str(value).strip()
            if modality in {"text", "image"}:
                statuses[modality] = {"status": "partial_missing"}
    return statuses


def _parse_modality_status(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, str):
        status = value.strip()
        if status == "partial_missing":
            return {"status": status}
        return None
    if isinstance(value, dict):
        status = str(value.get("status") or "").strip()
        if status == "partial_missing":
            parsed = {"status": status}
            if value.get("reason"):
                parsed["reason"] = str(value["reason"])
            return parsed
    return None
