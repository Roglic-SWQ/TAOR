# MKG-W Both-Modality MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline, adapter-ready six-stage MMKGC pipeline that runs on MKG-W and produces text candidates, image prompt candidates, and text-image pair ranking for 100 entities.

**Architecture:** Create an independent Python package under `MMKGC/mmkgc` with focused modules for data loading, stage implementations, adapters, ranking, output writing, and CLI orchestration. The default path is deterministic and model-free, while adapter interfaces preserve clean upgrade points for later LMM, image generation, embedding, and judge integrations.

**Tech Stack:** Python 3.11 standard library, dataclasses, argparse, json, pathlib, pytest.

---

## File Structure

Create these files:

- `MMKGC/mmkgc/__init__.py`: package marker and version.
- `MMKGC/mmkgc/data/__init__.py`: data package exports.
- `MMKGC/mmkgc/data/mkgw.py`: MKG-W file loader and `EntityRecord` construction.
- `MMKGC/mmkgc/utils.py`: tokenization, scoring helpers, slug, JSON writing helpers.
- `MMKGC/mmkgc/adapters/__init__.py`: adapter exports.
- `MMKGC/mmkgc/adapters/base.py`: adapter protocols and deterministic implementations.
- `MMKGC/mmkgc/stages/__init__.py`: stage package exports.
- `MMKGC/mmkgc/stages/coarse.py`: Stage 0.
- `MMKGC/mmkgc/stages/modeling.py`: Stage 1.
- `MMKGC/mmkgc/stages/reasoning.py`: Stage 2.
- `MMKGC/mmkgc/stages/selection.py`: Stage 3.
- `MMKGC/mmkgc/stages/generation.py`: Stage 4.
- `MMKGC/mmkgc/ranking/__init__.py`: ranking package exports.
- `MMKGC/mmkgc/ranking/scoring.py`: Stage 5 scoring and pair ranking.
- `MMKGC/mmkgc/io/__init__.py`: output package exports.
- `MMKGC/mmkgc/io/writer.py`: run directory and artifact writer.
- `MMKGC/mmkgc/pipeline.py`: end-to-end orchestration.
- `MMKGC/mmkgc/run.py`: CLI.
- `MMKGC/configs/default.json`: default adapter and run config.
- `MMKGC/tests/test_data.py`: data loader tests.
- `MMKGC/tests/test_stages.py`: stage output tests.
- `MMKGC/tests/test_pipeline_outputs.py`: end-to-end output tests.
- `MMKGC/README.md`: usage and MVP scope.

Modify no files outside `MMKGC/`.

---

### Task 1: Package Skeleton and Utilities

**Files:**
- Create: `MMKGC/mmkgc/__init__.py`
- Create: `MMKGC/mmkgc/utils.py`
- Create: `MMKGC/tests/test_stages.py`

- [ ] **Step 1: Write utility tests**

Add to `MMKGC/tests/test_stages.py`:

```python
from mmkgc.utils import clamp01, slugify, token_cosine, token_overlap


def test_token_scores_are_deterministic():
    assert token_overlap("drama film from exampleland", ["drama film"]) == 1.0
    assert token_overlap("drama film", ["mountain"]) == 0.0
    assert token_cosine("drama film", "film drama") == 1.0
    assert token_cosine("drama film", "mountain river") == 0.0


def test_slugify_and_clamp():
    assert slugify("country of citizenship") == "country_of_citizenship"
    assert slugify("IMDb ID") == "imdb_id"
    assert clamp01(-1) == 0.0
    assert clamp01(2) == 1.0
```

- [ ] **Step 2: Run utility tests and verify failure**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_stages.py -v
```

Expected: FAIL because `mmkgc.utils` does not exist yet.

- [ ] **Step 3: Create package marker**

Create `MMKGC/mmkgc/__init__.py`:

```python
"""MMKGC: relation-guided missing modality completion for MKG-W."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Implement utilities**

Create `MMKGC/mmkgc/utils.py`:

```python
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def token_cosine(left: str, right: str) -> float:
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0
    common = set(left_counts) & set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return round(clamp01(dot / (left_norm * right_norm)), 6)


def token_overlap(text: str, required_terms: Iterable[str]) -> float:
    terms = [term for term in required_terms if term]
    if not terms:
        return 1.0
    text_tokens = set(tokenize(text))
    hits = 0
    for term in terms:
        term_tokens = set(tokenize(term))
        if term_tokens and term_tokens <= text_tokens:
            hits += 1
    return round(clamp01(hits / len(terms)), 6)


def slugify(value: str) -> str:
    tokens = tokenize(value)
    return "_".join(tokens) if tokens else "item"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 5: Run utility tests**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_stages.py -v
```

Expected: PASS for 2 tests.

- [ ] **Step 6: Commit or record checkpoint**

If the workspace is in a git repository, run:

```bash
git add MMKGC/mmkgc/__init__.py MMKGC/mmkgc/utils.py MMKGC/tests/test_stages.py
git commit -m "feat: add mmkgc package utilities"
```

If `git status` reports that this directory is not in a repository, record this checkpoint in the final handoff instead of committing.

---

### Task 2: MKG-W Data Loader

**Files:**
- Create: `MMKGC/mmkgc/data/__init__.py`
- Create: `MMKGC/mmkgc/data/mkgw.py`
- Create: `MMKGC/tests/test_data.py`

- [ ] **Step 1: Write data loader tests**

Create `MMKGC/tests/test_data.py`:

```python
import json
from pathlib import Path

from mmkgc.data import MKGWDataStore


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def make_fixture(tmp_path: Path) -> Path:
    write(
        tmp_path / "entity2id.txt",
        "\n".join(
            [
                "http://www.wikidata.org/entity/Q1 0",
                "http://www.wikidata.org/entity/Q2 1",
                "http://www.wikidata.org/entity/Q3 2",
            ]
        )
        + "\n",
    )
    write(
        tmp_path / "relation2id.txt",
        "\n".join(
            [
                "http://www.wikidata.org/entity/P31 0",
                "http://www.wikidata.org/entity/P17 1",
            ]
        )
        + "\n",
    )
    write(
        tmp_path / "train.txt",
        "\n".join(
            [
                "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P31 http://www.wikidata.org/entity/Q2 .",
                "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P17 http://www.wikidata.org/entity/Q3 .",
            ]
        )
        + "\n",
    )
    write(tmp_path / "valid.txt", "")
    write(tmp_path / "test.txt", "")
    (tmp_path / "entity_info.json").write_text(
        json.dumps(
            {
                "Q1": {"label": "Example Film", "description": "a fictional film"},
                "Q2": {"label": "film", "description": "type of creative work"},
                "Q3": {"label": "Exampleland", "description": "a country"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "relation_info.json").write_text(
        json.dumps({"P31": "instance of", "P17": "country"}),
        encoding="utf-8",
    )
    image_dir = tmp_path / "images" / "Example Film"
    image_dir.mkdir(parents=True)
    (image_dir / "image.jpg").write_bytes(b"fake")
    return tmp_path


def test_build_both_record_excludes_modalities_from_generation(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))
    record = store.build_record(0, missing_mode="both")

    assert record.entity_key == "Q1"
    assert record.label == "Example Film"
    assert record.existing_modalities == {"text": None, "image": None}
    assert record.ground_truth_modalities["text"] == "a fictional film"
    assert record.ground_truth_modalities["image"].endswith("image.jpg")
    assert [item["relation"] for item in record.neighbors] == ["instance of", "country"]
    assert record.neighbors[0]["direction"] == "out"


def test_select_records_respects_limit(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))
    records = store.select_records(missing_mode="both", limit=2)

    assert len(records) == 2
    assert records[0].entity_key == "Q1"
```

- [ ] **Step 2: Run data tests and verify failure**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_data.py -v
```

Expected: FAIL because `mmkgc.data` does not exist yet.

- [ ] **Step 3: Implement data package exports**

Create `MMKGC/mmkgc/data/__init__.py`:

```python
from .mkgw import EntityRecord, MKGWDataStore

__all__ = ["EntityRecord", "MKGWDataStore"]
```

- [ ] **Step 4: Implement MKG-W data loader**

Create `MMKGC/mmkgc/data/mkgw.py`:

```python
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _key(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _label_variants(label: str, key: str) -> List[str]:
    variants = [label, label.replace(" ", "_"), label.replace("_", " "), key]
    out = []
    seen = set()
    for value in variants:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


@dataclass(frozen=True)
class EntityRecord:
    entity_id: int
    entity_key: str
    label: str
    description: str
    neighbors: List[Dict[str, Any]]
    existing_modalities: Dict[str, Optional[str]]
    ground_truth_modalities: Dict[str, Optional[str]]


class MKGWDataStore:
    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.uri2id, self.id2uri = self._load_mapping("entity2id.txt")
        self.rel2id, self.id2rel = self._load_mapping("relation2id.txt")
        self.entity_info = _load_json(self.dataset_dir / "entity_info.json", {})
        self.relation_info = _load_json(self.dataset_dir / "relation_info.json", {})
        self.train_triples = self._load_triples("train.txt")
        self.valid_triples = self._load_triples("valid.txt")
        self.test_triples = self._load_triples("test.txt")
        self.directed_adj = self._build_directed_adjacency()

    def _load_mapping(self, filename: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        forward: Dict[str, int] = {}
        reverse: Dict[int, str] = {}
        path = self.dataset_dir / filename
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            value, idx = line.rsplit(" ", 1)
            forward[value] = int(idx)
            reverse[int(idx)] = value
        return forward, reverse

    def _load_triples(self, filename: str) -> List[Tuple[int, int, int]]:
        path = self.dataset_dir / filename
        if not path.exists():
            return []
        triples: List[Tuple[int, int, int]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().rstrip(" .")
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            h_uri, r_uri, t_uri = parts
            if h_uri in self.uri2id and r_uri in self.rel2id and t_uri in self.uri2id:
                triples.append((self.uri2id[h_uri], self.rel2id[r_uri], self.uri2id[t_uri]))
        return triples

    def _build_directed_adjacency(self) -> Dict[int, List[Tuple[int, int, str]]]:
        adj: Dict[int, List[Tuple[int, int, str]]] = defaultdict(list)
        for head, relation, tail in self.train_triples:
            adj[head].append((relation, tail, "out"))
            adj[tail].append((relation, head, "in"))
        return dict(adj)

    def entity_key(self, entity_id: int) -> str:
        return _key(self.id2uri[entity_id])

    def relation_label(self, relation_id: int) -> str:
        pid = _key(self.id2rel[relation_id])
        return self.relation_info.get(pid, pid)

    def entity_metadata(self, entity_id: int) -> Dict[str, str]:
        key = self.entity_key(entity_id)
        return self.entity_info.get(key, {"label": key, "description": ""})

    def image_path(self, entity_id: int) -> Optional[str]:
        key = self.entity_key(entity_id)
        info = self.entity_metadata(entity_id)
        label = info.get("label", key)
        for variant in _label_variants(label, key):
            directory = self.dataset_dir / "images" / variant
            if not directory.is_dir():
                continue
            images = sorted(
                item for item in directory.iterdir()
                if item.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if images:
                return str(images[0])
        return None

    def neighbors(self, entity_id: int, max_neighbors: int = 32) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for relation_id, neighbor_id, direction in self.directed_adj.get(entity_id, [])[:max_neighbors]:
            neighbor_key = self.entity_key(neighbor_id)
            neighbor_info = self.entity_metadata(neighbor_id)
            output.append(
                {
                    "relation_id": relation_id,
                    "relation": self.relation_label(relation_id),
                    "neighbor_id": neighbor_id,
                    "neighbor_key": neighbor_key,
                    "neighbor_label": neighbor_info.get("label", neighbor_key),
                    "neighbor_description": neighbor_info.get("description", ""),
                    "direction": direction,
                }
            )
        return output

    def build_record(self, entity_id: int, missing_mode: str) -> EntityRecord:
        if missing_mode not in {"text", "image", "both"}:
            raise ValueError("missing_mode must be one of: text, image, both")
        key = self.entity_key(entity_id)
        info = self.entity_metadata(entity_id)
        description = info.get("description", "")
        image = self.image_path(entity_id)
        if missing_mode == "both":
            existing = {"text": None, "image": None}
        elif missing_mode == "text":
            existing = {"text": None, "image": image}
        else:
            existing = {"text": description or None, "image": None}
        return EntityRecord(
            entity_id=entity_id,
            entity_key=key,
            label=info.get("label", key),
            description=description,
            neighbors=self.neighbors(entity_id),
            existing_modalities=existing,
            ground_truth_modalities={"text": description or None, "image": image},
        )

    def select_records(self, missing_mode: str, limit: int | None = None) -> List[EntityRecord]:
        selected: List[EntityRecord] = []
        for entity_id in sorted(self.id2uri):
            selected.append(self.build_record(entity_id, missing_mode))
            if limit is not None and len(selected) >= limit:
                break
        return selected
```

- [ ] **Step 5: Run data tests**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_data.py -v
```

Expected: PASS for 2 tests.

- [ ] **Step 6: Run all current tests**

Run from `MMKGC/`:

```bash
python -m pytest tests -v
```

Expected: PASS for 4 tests.

---

### Task 3: Stage 0-2 Implementations

**Files:**
- Create: `MMKGC/mmkgc/stages/__init__.py`
- Create: `MMKGC/mmkgc/stages/coarse.py`
- Create: `MMKGC/mmkgc/stages/modeling.py`
- Create: `MMKGC/mmkgc/stages/reasoning.py`
- Modify: `MMKGC/tests/test_stages.py`

- [ ] **Step 1: Add Stage 0-2 tests**

Append to `MMKGC/tests/test_stages.py`:

```python
from mmkgc.data.mkgw import EntityRecord
from mmkgc.stages import coarse_relation_filtering, graph_constrained_reasoning, knowledge_modeling


def example_record():
    return EntityRecord(
        entity_id=0,
        entity_key="Q1",
        label="Example Film",
        description="a fictional film",
        neighbors=[
            {
                "relation_id": 0,
                "relation": "instance of",
                "neighbor_id": 1,
                "neighbor_key": "Q2",
                "neighbor_label": "film",
                "neighbor_description": "type of creative work",
                "direction": "out",
            },
            {
                "relation_id": 1,
                "relation": "IMDb ID",
                "neighbor_id": 2,
                "neighbor_key": "Q3",
                "neighbor_label": "tt000000",
                "neighbor_description": "",
                "direction": "out",
            },
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": "a fictional film", "image": None},
    )


def test_stage_0_filters_low_value_relation():
    coarse = coarse_relation_filtering(example_record(), missing_mode="both")

    assert coarse["selected_relations"][0]["relation"] == "instance of"
    assert coarse["discarded_relations"][0]["relation"] == "IMDb ID"
    assert coarse["filtered_triples"] == ["(Example Film, instance of, film)"]


def test_stage_1_and_2_build_constraints():
    record = example_record()
    coarse = coarse_relation_filtering(record, missing_mode="both")
    knowledge = knowledge_modeling(record, missing_mode="both", coarse=coarse)
    reasoning = graph_constrained_reasoning(missing_mode="both", knowledge=knowledge)

    assert knowledge["entity_profile"]["entity_type"] == "film"
    assert knowledge["relation_facts"][0]["relation"] == "instance of"
    assert reasoning["modality_constraints"]["text"]["must_include"]
    assert reasoning["modality_constraints"]["image"]["must_include"]
    assert reasoning["forbidden_content"]
```

- [ ] **Step 2: Run Stage 0-2 tests and verify failure**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_stages.py -v
```

Expected: FAIL because `mmkgc.stages` does not exist yet.

- [ ] **Step 3: Implement stage exports**

Create `MMKGC/mmkgc/stages/__init__.py`:

```python
from .coarse import coarse_relation_filtering
from .modeling import knowledge_modeling
from .reasoning import graph_constrained_reasoning

__all__ = [
    "coarse_relation_filtering",
    "knowledge_modeling",
    "graph_constrained_reasoning",
]
```

- [ ] **Step 4: Implement Stage 0**

Create `MMKGC/mmkgc/stages/coarse.py`:

```python
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from mmkgc.data import EntityRecord
from mmkgc.utils import clamp01, token_cosine


LOW_VALUE_TERMS = {
    "id", "identifier", "url", "website", "wikimedia", "commons",
    "category", "catalog", "archive", "freebase", "imdb", "viaf",
}

IMAGE_RELATIONS = {
    "instance of", "occupation", "genre", "country", "country of origin",
    "located in", "location", "architectural style", "sport", "instrument",
    "material used", "depicts", "field of work",
}

TEXT_RELATIONS = {
    "instance of", "occupation", "author", "director", "cast member",
    "composer", "publisher", "member of", "educated at", "place of birth",
    "country of citizenship", "country of origin", "genre", "part of",
}


def _triple(label: str, item: Dict[str, Any]) -> str:
    if item["direction"] == "out":
        return f"({label}, {item['relation']}, {item['neighbor_label']})"
    return f"({item['neighbor_label']}, {item['relation']}, {label})"


def _is_low_value(relation: str) -> bool:
    relation_l = relation.lower()
    return any(term in relation_l for term in LOW_VALUE_TERMS)


def _modality_bonus(relation: str, missing_mode: str) -> float:
    relation_l = relation.lower()
    if missing_mode == "image":
        return 0.9 if relation_l in IMAGE_RELATIONS else 0.2
    if missing_mode == "text":
        return 0.9 if relation_l in TEXT_RELATIONS else 0.4
    bonus = 0.0
    if relation_l in IMAGE_RELATIONS:
        bonus += 0.45
    if relation_l in TEXT_RELATIONS:
        bonus += 0.45
    return bonus or 0.25


def coarse_relation_filtering(record: EntityRecord, missing_mode: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for neighbor in record.neighbors:
        grouped[neighbor["relation"]].append(neighbor)

    selected = []
    discarded = []
    modality_text = " ".join(
        value for value in record.existing_modalities.values() if isinstance(value, str)
    )

    for relation, items in grouped.items():
        if _is_low_value(relation):
            discarded.append({"relation": relation, "reason": "Low-value metadata relation."})
            continue
        evidence = [_triple(record.label, item) for item in items[:3]]
        semantic_text = " ".join(item.get("neighbor_label", "") for item in items)
        semantic_consistency = token_cosine(modality_text, semantic_text) if modality_text else 0.5
        raw_score = 1.0 + min(len(items), 3) * 0.45 + _modality_bonus(relation, missing_mode)
        if any(item.get("neighbor_description") for item in items):
            raw_score += 0.25
        raw_score += semantic_consistency * 0.35
        coarse_score = round(clamp01(raw_score / 3.5) * 5.0, 3)
        selected.append(
            {
                "relation": relation,
                "coarse_score": coarse_score,
                "modality_relevance": "high" if _modality_bonus(relation, missing_mode) >= 0.45 else "medium",
                "kgc_relevance": "high" if len(items) > 1 or relation.lower() == "instance of" else "medium",
                "semantic_consistency": "medium" if semantic_consistency >= 0.35 else "low",
                "reason": "Kept by high-recall rule-based coarse filtering.",
                "frequency": len(items),
                "directions": sorted({item["direction"] for item in items}),
                "evidence_triples": evidence,
            }
        )

    selected.sort(key=lambda item: (item["coarse_score"], item["frequency"]), reverse=True)
    kept = [item for item in selected if item["coarse_score"] >= 3.0]
    for item in selected:
        if item not in kept and len(kept) < 8:
            kept.append(item)
    kept = kept[:8]
    kept_relations = {item["relation"] for item in kept}
    for item in selected:
        if item["relation"] not in kept_relations:
            discarded.append({"relation": item["relation"], "reason": "Below Top-K coarse threshold."})
    return {
        "selected_relations": kept,
        "discarded_relations": discarded,
        "filtered_triples": [triple for item in kept for triple in item["evidence_triples"]],
    }
```

- [ ] **Step 5: Implement Stage 1**

Create `MMKGC/mmkgc/stages/modeling.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List

from mmkgc.data import EntityRecord


def _fact_type(relation: str) -> str:
    relation_l = relation.lower()
    if relation_l in {"instance of", "subclass of"}:
        return "type"
    if relation_l in {"occupation", "field of work"}:
        return "role"
    if "country" in relation_l or "origin" in relation_l:
        return "origin"
    if "located" in relation_l or "location" in relation_l:
        return "location"
    if relation_l in {"author", "director", "composer", "creator"}:
        return "creator"
    return "context"


def _infer_entity_type(record: EntityRecord, selected: List[Dict[str, Any]]) -> str:
    for relation in selected:
        if relation["relation"].lower() == "instance of" and relation["evidence_triples"]:
            triple = relation["evidence_triples"][0].rstrip(")")
            return triple.rsplit(", ", 1)[-1]
    words = record.description.split()
    return " ".join(words[:3]) if words else "entity"


def knowledge_modeling(record: EntityRecord, missing_mode: str, coarse: Dict[str, Any]) -> Dict[str, Any]:
    selected = coarse["selected_relations"]
    entity_type = _infer_entity_type(record, selected)
    relation_facts = []
    for relation in selected:
        relation_name = relation["relation"]
        fact = f"{record.label} has KG relation {relation_name} supported by {len(relation['evidence_triples'])} evidence triple(s)."
        relation_facts.append(
            {
                "relation": relation_name,
                "fact": fact,
                "fact_type": _fact_type(relation_name),
                "evidence_triples": relation["evidence_triples"],
                "support_level": "high" if relation["coarse_score"] >= 4 else "medium",
                "direction": "mixed" if len(relation.get("directions", [])) > 1 else relation.get("directions", ["out"])[0],
                "neighbor_entities": [
                    triple.rsplit(", ", 1)[-1].rstrip(")") for triple in relation["evidence_triples"]
                ],
                "modality_relevance": {
                    "image": "high" if relation["modality_relevance"] == "high" else "medium",
                    "text": "high",
                },
                "generation_use": "identity" if relation_name.lower() == "instance of" else "textual_fact",
                "risk": "Avoid unsupported specific visual or factual details.",
            }
        )

    missing_targets = ["text", "image"] if missing_mode == "both" else [missing_mode]
    clues = []
    for target in missing_targets:
        for fact in relation_facts[:4]:
            clues.append(
                {
                    "clue": fact["fact"],
                    "target_modality": target,
                    "clue_type": "relation_context",
                    "source_relations": [fact["relation"]],
                    "evidence_triples": fact["evidence_triples"],
                    "specificity": "medium",
                    "confidence": fact["support_level"],
                    "generation_role": "must_include" if fact["relation"].lower() == "instance of" else "should_include",
                    "risk": fact["risk"],
                }
            )

    return {
        "entity_profile": {
            "entity_type": entity_type,
            "secondary_types": [],
            "type_confidence": "high" if entity_type != "entity" else "low",
            "core_identity": record.label,
            "semantic_domain": [entity_type],
            "key_attributes": [
                {
                    "attribute": fact["neighbor_entities"][0] if fact["neighbor_entities"] else fact["relation"],
                    "attribute_type": "contextual",
                    "source": "KG",
                    "evidence": fact["evidence_triples"][0] if fact["evidence_triples"] else "",
                }
                for fact in relation_facts[:5]
            ],
            "type_evidence": relation_facts[:1],
            "conflicting_evidence": [],
            "uncertain_aspects": ["Exact visual appearance is unsupported by KG-only input."],
        },
        "modality_observations": {
            "available_modalities": record.existing_modalities,
            "ground_truth_available": {
                "text": record.ground_truth_modalities.get("text") is not None,
                "image": record.ground_truth_modalities.get("image") is not None,
            },
            "generation_input_policy": "both mode uses KG and label only" if missing_mode == "both" else "single missing modality may use existing modality",
        },
        "relation_facts": relation_facts,
        "missing_modality_clues": clues,
        "uncertainties": [
            {
                "claim": "Unsupported exact details",
                "reason": "The MVP uses local KG evidence and avoids inventing details absent from evidence triples.",
                "related_relations": [fact["relation"] for fact in relation_facts],
                "action": "avoid",
            }
        ],
    }
```

- [ ] **Step 6: Implement Stage 2**

Create `MMKGC/mmkgc/stages/reasoning.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List


def _constraint(text: str, kind: str, source: str, relation: str | None = None) -> Dict[str, Any]:
    return {
        "constraint": text,
        "constraint_type": kind,
        "source": source,
        "source_relations": [relation] if relation else [],
        "evidence_triples": [],
        "confidence": "high",
        "reason": "Derived by deterministic graph-constrained reasoning.",
    }


def graph_constrained_reasoning(missing_mode: str, knowledge: Dict[str, Any]) -> Dict[str, Any]:
    profile = knowledge["entity_profile"]
    relation_facts = knowledge["relation_facts"]
    supported = []
    for fact in relation_facts:
        supported.append(
            {
                "fact": fact["fact"],
                "fact_type": fact["fact_type"],
                "source": "relation_facts",
                "source_relations": [fact["relation"]],
                "evidence_triples": fact["evidence_triples"],
                "evidence_text": [],
                "usable_for": ["image", "text"],
                "confidence": fact["support_level"],
                "modality_usage": {
                    "image": "must_include" if fact["relation"].lower() == "instance of" else "should_include",
                    "text": "must_include" if fact["relation"].lower() == "instance of" else "should_include",
                },
                "reason": "Fact is directly supported by selected KG triples.",
            }
        )

    constraints = {"image": {"must_include": [], "should_include": [], "context_only": [], "avoid": []}, "text": {"must_include": [], "should_include": [], "context_only": [], "avoid": []}}
    constraints["text"]["must_include"].append(_constraint(f"Describe {profile['core_identity']} as {profile['entity_type']}.", "identity_definition", "entity_profile"))
    constraints["image"]["must_include"].append(_constraint(f"Represent {profile['core_identity']} as {profile['entity_type']}.", "entity_type", "entity_profile"))
    for fact in relation_facts[:4]:
        item = _constraint(fact["fact"], "relation_fact", "relation_facts", fact["relation"])
        constraints["text"]["should_include"].append(item)
        constraints["image"]["should_include"].append(item)
    constraints["image"]["avoid"].append(_constraint("Do not generate unsupported exact faces, logos, covers, dates, or locations.", "unsupported_specific_visual", "uncertainties"))
    constraints["text"]["avoid"].append(_constraint("Do not invent unsupported dates, awards, affiliations, creators, or relation directions.", "unsupported_specific_fact", "uncertainties"))

    forbidden = [
        {
            "content": "unsupported exact face, logo, cover, date, award, affiliation, creator, or location",
            "forbidden_type": "unsupported_specific_detail",
            "applies_to": "both",
            "source": "default_mmkg_guardrail",
            "trigger": "not supported by relation_facts",
            "reason": "Avoid hallucinating high-risk details.",
            "severity": "hard",
            "ranking_penalty": "high",
        }
    ]
    return {
        "reasoning_summary": "Converted selected KG facts into generation constraints and forbidden content.",
        "supported_facts": supported,
        "safe_inferences": [],
        "uncertain_facts": [
            {
                "claim": item["claim"],
                "uncertainty_type": "insufficient_evidence",
                "related_sources": ["knowledge_modeling"],
                "related_relations": item["related_relations"],
                "evidence_triples": [],
                "applies_to": "both",
                "recommended_action": item["action"],
                "risk": item["reason"],
                "reason": item["reason"],
            }
            for item in knowledge["uncertainties"]
        ],
        "modality_constraints": constraints,
        "forbidden_content": forbidden,
        "reasoning_confidence": "medium" if relation_facts else "low",
    }
```

- [ ] **Step 7: Run Stage 0-2 tests**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_stages.py -v
```

Expected: PASS for Stage 0-2 tests and utility tests.

---

### Task 4: Stage 3-4 Adapters, Selection, and Generation

**Files:**
- Create: `MMKGC/mmkgc/adapters/__init__.py`
- Create: `MMKGC/mmkgc/adapters/base.py`
- Create: `MMKGC/mmkgc/stages/selection.py`
- Create: `MMKGC/mmkgc/stages/generation.py`
- Modify: `MMKGC/mmkgc/stages/__init__.py`
- Modify: `MMKGC/tests/test_stages.py`

- [ ] **Step 1: Add Stage 3-4 tests**

Append to `MMKGC/tests/test_stages.py`:

```python
from mmkgc.adapters import RuleImagePromptGenerator, RuleTextGenerator
from mmkgc.stages import fine_grained_relation_selection, relation_alternation_generation


def test_stage_3_and_4_generate_candidates():
    record = example_record()
    coarse = coarse_relation_filtering(record, missing_mode="both")
    knowledge = knowledge_modeling(record, missing_mode="both", coarse=coarse)
    reasoning = graph_constrained_reasoning(missing_mode="both", knowledge=knowledge)
    selection = fine_grained_relation_selection(
        missing_mode="both",
        knowledge=knowledge,
        reasoning=reasoning,
        candidate_budget=5,
    )
    generation = relation_alternation_generation(
        record=record,
        missing_mode="both",
        knowledge=knowledge,
        reasoning=reasoning,
        selection=selection,
        text_generator=RuleTextGenerator(),
        image_prompt_generator=RuleImagePromptGenerator(),
        candidate_budget=5,
    )

    assert selection["fine_grained_relation_selection"]["primary_relations"]
    assert len(generation["text_candidates"]) == 5
    assert len(generation["image_candidates"]) == 5
    assert generation["image_candidates"][0]["positive_prompt"]
    assert generation["text_candidates"][0]["generated_content"]["type"] == "text"
```

- [ ] **Step 2: Run tests and verify failure**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_stages.py::test_stage_3_and_4_generate_candidates -v
```

Expected: FAIL because adapters and Stage 3-4 functions do not exist.

- [ ] **Step 3: Implement adapters**

Create `MMKGC/mmkgc/adapters/base.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List, Protocol


class TextGenerator(Protocol):
    def generate(self, record_label: str, entity_type: str, relation: str, angle: str, facts: List[str]) -> str:
        ...


class ImagePromptGenerator(Protocol):
    def generate(self, record_label: str, entity_type: str, relation: str, angle: str, facts: List[str], avoid: List[str]) -> Dict[str, str]:
        ...


class RuleTextGenerator:
    def generate(self, record_label: str, entity_type: str, relation: str, angle: str, facts: List[str]) -> str:
        fact_text = " ".join(facts[:2])
        if not fact_text:
            fact_text = f"{record_label} is represented as {entity_type} according to KG evidence."
        if angle == "concise-summary":
            return f"{record_label} is a {entity_type} described by KG-supported facts."
        return f"{record_label} is a {entity_type}. {fact_text}".strip()


class RuleImagePromptGenerator:
    def generate(self, record_label: str, entity_type: str, relation: str, angle: str, facts: List[str], avoid: List[str]) -> Dict[str, str]:
        fact_text = " ".join(facts[:2])
        positive = (
            f"An informative visual representation of {record_label} as {entity_type}, "
            f"guided by the {relation} relation, using a {angle} composition. {fact_text}"
        ).strip()
        negative = ", ".join(avoid) if avoid else "unsupported exact face, logo, cover, date, award, unrelated entity"
        return {
            "positive_prompt": positive,
            "negative_prompt": negative,
            "prompt_rationale": "Rule-based prompt assembled from KG-supported relation facts and constraints.",
        }
```

Create `MMKGC/mmkgc/adapters/__init__.py`:

```python
from .base import ImagePromptGenerator, RuleImagePromptGenerator, RuleTextGenerator, TextGenerator

__all__ = [
    "ImagePromptGenerator",
    "RuleImagePromptGenerator",
    "RuleTextGenerator",
    "TextGenerator",
]
```

- [ ] **Step 4: Implement Stage 3**

Create `MMKGC/mmkgc/stages/selection.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List


def _score_fact(fact: Dict[str, Any]) -> Dict[str, float]:
    relation = fact["relation"].lower()
    modality_fit = 5.0 if relation in {"instance of", "occupation", "genre", "country", "country of origin"} else 3.0
    constraint_alignment = 4.0
    evidence_strength = 4.0 if fact["support_level"] == "high" else 3.0
    generation_utility = 4.0 if fact["neighbor_entities"] else 2.0
    risk_penalty = 1.0
    fine_score = max(
        0.0,
        0.30 * modality_fit
        + 0.25 * constraint_alignment
        + 0.20 * evidence_strength
        + 0.20 * generation_utility
        - 0.25 * risk_penalty,
    )
    return {
        "modality_fit": modality_fit,
        "constraint_alignment": constraint_alignment,
        "evidence_strength": evidence_strength,
        "generation_utility": generation_utility,
        "risk_penalty": risk_penalty,
        "fine_score": round(fine_score, 3),
    }


def _relation_item(fact: Dict[str, Any], role: str, target_modality: str, candidate_quota: int, fallback: bool = False) -> Dict[str, Any]:
    scores = _score_fact(fact)
    return {
        "relation": fact["relation"],
        "relation_role": role,
        "fallback": fallback,
        "fine_score": scores.pop("fine_score"),
        "score_breakdown": scores,
        "target_modality": target_modality,
        "role_decision": {
            "hard_excluded": False,
            "matched_constraints": [],
            "matched_forbidden_content": [],
            "decision_rule": "rule-based fixed-weight scoring",
            "reason": f"Assigned {role} by deterministic MVP scoring.",
        },
        "generation_plan": {
            "generation_focus": f"Use {fact['relation']} as {role} evidence.",
            "supporting_relations": [],
            "candidate_quota": candidate_quota,
            "quota_reason": "Primary relations receive candidates until budget is filled.",
            "must_include_constraints": [],
            "should_include_constraints": [],
            "context_constraints": [],
            "avoid_constraints": [],
        },
        "evidence_triples": fact["evidence_triples"],
        "source_fact": fact,
    }


def fine_grained_relation_selection(missing_mode: str, knowledge: Dict[str, Any], reasoning: Dict[str, Any], candidate_budget: int = 5) -> Dict[str, Any]:
    del reasoning
    target_modality = "both" if missing_mode == "both" else missing_mode
    scored = []
    for fact in knowledge["relation_facts"]:
        scores = _score_fact(fact)
        scored.append((scores["fine_score"], fact))
    scored.sort(key=lambda item: item[0], reverse=True)
    primary_facts = [fact for score, fact in scored if score >= 3.5][:3]
    if not primary_facts and scored:
        primary_facts = [scored[0][1]]
    primary = [_relation_item(fact, "primary", target_modality, 2, fallback=False) for fact in primary_facts]
    assigned = sum(item["generation_plan"]["candidate_quota"] for item in primary)
    if assigned > candidate_budget:
        for item in primary:
            item["generation_plan"]["candidate_quota"] = 1
        assigned = len(primary)
    secondary_facts = [fact for _, fact in scored if fact not in primary_facts]
    secondary = [_relation_item(fact, "secondary", target_modality, 0) for fact in secondary_facts[:5]]
    if assigned < candidate_budget and secondary:
        secondary[0]["generation_plan"]["candidate_quota"] = candidate_budget - assigned
        assigned = candidate_budget
    return {
        "fine_grained_relation_selection": {
            "target_modality": target_modality,
            "selection_summary": "Selected primary and secondary relations with deterministic fixed-weight scoring.",
            "primary_relations": primary,
            "secondary_relations": secondary,
            "context_relations": [],
            "excluded_relations": [],
            "quota_summary": {"total_candidate_budget": candidate_budget, "assigned_candidates": assigned},
        }
    }
```

- [ ] **Step 5: Implement Stage 4**

Create `MMKGC/mmkgc/stages/generation.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List

from mmkgc.adapters import ImagePromptGenerator, TextGenerator
from mmkgc.data import EntityRecord
from mmkgc.utils import slugify


TEXT_ANGLES = ["identity-centered", "fact-centered", "relation-centered", "disambiguation-centered", "concise-summary"]
IMAGE_ANGLES = ["entity-type-centered", "role-centered", "scene-centered", "style-centered", "attribute-centered"]


def _facts_for_relation(knowledge: Dict[str, Any], relation: str) -> List[str]:
    if relation == "global_profile":
        return [fact["fact"] for fact in knowledge["relation_facts"][:3]]
    return [fact["fact"] for fact in knowledge["relation_facts"] if fact["relation"] == relation]


def _evidence_for_relation(knowledge: Dict[str, Any], relation: str) -> List[str]:
    triples: List[str] = []
    for fact in knowledge["relation_facts"]:
        if relation == "global_profile" or fact["relation"] == relation:
            triples.extend(fact["evidence_triples"])
    return triples[:5]


def _relation_plan(selection: Dict[str, Any]) -> List[Dict[str, Any]]:
    fg = selection["fine_grained_relation_selection"]
    plans = fg["primary_relations"] + [item for item in fg["secondary_relations"] if item["generation_plan"]["candidate_quota"] > 0]
    return plans or [{"relation": "global_profile", "generation_plan": {"candidate_quota": 5}, "evidence_triples": []}]


def _expand_relations(plans: List[Dict[str, Any]], budget: int) -> List[str]:
    relations: List[str] = []
    for plan in plans:
        quota = int(plan["generation_plan"].get("candidate_quota", 0))
        relations.extend([plan["relation"]] * quota)
    while len(relations) < budget:
        relations.append("global_profile")
    return relations[:budget]


def relation_alternation_generation(
    record: EntityRecord,
    missing_mode: str,
    knowledge: Dict[str, Any],
    reasoning: Dict[str, Any],
    selection: Dict[str, Any],
    text_generator: TextGenerator,
    image_prompt_generator: ImagePromptGenerator,
    candidate_budget: int = 5,
) -> Dict[str, Any]:
    profile = knowledge["entity_profile"]
    relations = _expand_relations(_relation_plan(selection), candidate_budget)
    avoid = [item["constraint"] for item in reasoning["modality_constraints"]["image"]["avoid"]]
    used_constraints = reasoning["modality_constraints"]
    text_candidates = []
    image_candidates = []

    if missing_mode in {"text", "both"}:
        for index in range(candidate_budget):
            relation = relations[index]
            angle = TEXT_ANGLES[index % len(TEXT_ANGLES)]
            facts = _facts_for_relation(knowledge, relation)
            text = text_generator.generate(record.label, profile["entity_type"], relation, angle, facts)
            candidate_id = f"text_{slugify(relation)}_{slugify(angle)}_{index}"
            text_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "target_modality": "text",
                    "primary_relation": relation,
                    "generation_angle": angle,
                    "supporting_relations": [],
                    "used_evidence_triples": _evidence_for_relation(knowledge, relation),
                    "used_constraints": used_constraints["text"],
                    "applied_forbidden_content": reasoning["forbidden_content"],
                    "prompt_or_instruction": f"Generate concise KG-grounded text for {record.label}.",
                    "positive_prompt": "",
                    "negative_prompt": "",
                    "generated_content": {"type": "text", "value": text},
                    "expected_relation_coverage": [] if relation == "global_profile" else [relation],
                    "metadata": {"candidate_quota_source": "fine_grained_relation_selection", "generation_model": "rule", "seed": None, "status": "generated"},
                }
            )

    if missing_mode in {"image", "both"}:
        for index in range(candidate_budget):
            relation = relations[index]
            angle = IMAGE_ANGLES[index % len(IMAGE_ANGLES)]
            facts = _facts_for_relation(knowledge, relation)
            prompt = image_prompt_generator.generate(record.label, profile["entity_type"], relation, angle, facts, avoid)
            candidate_id = f"image_{slugify(relation)}_{slugify(angle)}_{index}"
            image_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "target_modality": "image",
                    "primary_relation": relation,
                    "generation_angle": angle,
                    "supporting_relations": [],
                    "used_evidence_triples": _evidence_for_relation(knowledge, relation),
                    "used_constraints": used_constraints["image"],
                    "applied_forbidden_content": reasoning["forbidden_content"],
                    "prompt_or_instruction": "",
                    "positive_prompt": prompt["positive_prompt"],
                    "negative_prompt": prompt["negative_prompt"],
                    "generated_content": {"type": "image_prompt", "value": prompt["positive_prompt"]},
                    "expected_relation_coverage": [] if relation == "global_profile" else [relation],
                    "metadata": {"candidate_quota_source": "fine_grained_relation_selection", "generation_model": "rule_prompt", "seed": None, "status": "planned"},
                }
            )

    return {
        "relation_alternation_generation": {
            "target_modality": missing_mode,
            "generation_summary": "Generated deterministic text candidates and image prompt candidates.",
            "candidate_budget": candidate_budget,
            "candidate_specs": text_candidates + image_candidates,
            "generated_candidates": text_candidates + image_candidates,
            "failures": [],
        },
        "text_candidates": text_candidates,
        "image_candidates": image_candidates,
    }
```

- [ ] **Step 6: Update stage exports**

Modify `MMKGC/mmkgc/stages/__init__.py` to:

```python
from .coarse import coarse_relation_filtering
from .generation import relation_alternation_generation
from .modeling import knowledge_modeling
from .reasoning import graph_constrained_reasoning
from .selection import fine_grained_relation_selection

__all__ = [
    "coarse_relation_filtering",
    "fine_grained_relation_selection",
    "graph_constrained_reasoning",
    "knowledge_modeling",
    "relation_alternation_generation",
]
```

- [ ] **Step 7: Run Stage 3-4 tests**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_stages.py -v
```

Expected: PASS for utility and Stage 0-4 tests.

---

### Task 5: Stage 5 Ranking

**Files:**
- Create: `MMKGC/mmkgc/ranking/__init__.py`
- Create: `MMKGC/mmkgc/ranking/scoring.py`
- Modify: `MMKGC/tests/test_stages.py`

- [ ] **Step 1: Add ranking test**

Append to `MMKGC/tests/test_stages.py`:

```python
from mmkgc.ranking import kg_aware_candidate_ranking


def test_stage_5_pair_ranking_outputs_top_pair():
    record = example_record()
    coarse = coarse_relation_filtering(record, missing_mode="both")
    knowledge = knowledge_modeling(record, missing_mode="both", coarse=coarse)
    reasoning = graph_constrained_reasoning(missing_mode="both", knowledge=knowledge)
    selection = fine_grained_relation_selection("both", knowledge, reasoning, candidate_budget=5)
    generation = relation_alternation_generation(
        record,
        "both",
        knowledge,
        reasoning,
        selection,
        RuleTextGenerator(),
        RuleImagePromptGenerator(),
        candidate_budget=5,
    )
    ranking = kg_aware_candidate_ranking(record, "both", knowledge, reasoning, generation)

    assert len(ranking["text_ranking"]["ranked_candidates"]) == 5
    assert len(ranking["image_ranking"]["ranked_candidates"]) == 5
    assert len(ranking["pair_ranking"]["ranked_pairs"]) == 9
    assert len(ranking["pair_ranking"]["top_5_pairs"]) == 5
    assert ranking["pair_ranking"]["best_pair_id"] == ranking["pair_ranking"]["top_5_pairs"][0]["pair_id"]
```

- [ ] **Step 2: Run ranking test and verify failure**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_stages.py::test_stage_5_pair_ranking_outputs_top_pair -v
```

Expected: FAIL because `mmkgc.ranking` does not exist yet.

- [ ] **Step 3: Implement ranking exports**

Create `MMKGC/mmkgc/ranking/__init__.py`:

```python
from .scoring import kg_aware_candidate_ranking

__all__ = ["kg_aware_candidate_ranking"]
```

- [ ] **Step 4: Implement ranking**

Create `MMKGC/mmkgc/ranking/scoring.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List

from mmkgc.data import EntityRecord
from mmkgc.utils import clamp01, token_cosine, token_overlap


def _kg_reference(knowledge: Dict[str, Any], reasoning: Dict[str, Any], modality: str) -> str:
    profile = knowledge["entity_profile"]
    constraints = reasoning["modality_constraints"][modality]
    parts = [
        profile["core_identity"],
        profile["entity_type"],
        *[fact["fact"] for fact in knowledge["relation_facts"][:4]],
        *[item["constraint"] for item in constraints["must_include"]],
        *[item["constraint"] for item in constraints["should_include"][:3]],
    ]
    return " ".join(part for part in parts if part)


def _candidate_score(candidate: Dict[str, Any], reference: str, must_terms: List[str]) -> Dict[str, Any]:
    content = candidate["generated_content"]["value"]
    semantic = token_cosine(content, reference)
    relation_coverage = token_overlap(content, candidate.get("expected_relation_coverage", [])) if candidate.get("expected_relation_coverage") else 0.7
    constraint_satisfaction = token_overlap(content, must_terms)
    evidence_support = token_cosine(content, " ".join(candidate.get("used_evidence_triples", [])))
    kg_score = clamp01(0.30 * relation_coverage + 0.25 * constraint_satisfaction + 0.20 * 1.0 + 0.15 * constraint_satisfaction + 0.10 * evidence_support)
    single_quality = clamp01(0.40 * semantic + 0.35 * kg_score + 0.15 * relation_coverage + 0.10 * 1.0)
    final = clamp01(0.45 * single_quality + 0.55 * kg_score)
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate": candidate,
        "final_score": round(final, 6),
        "single_quality_score": round(single_quality, 6),
        "kg_consistency_score": round(kg_score, 6),
        "relation_coverage_score": round(relation_coverage, 6),
        "constraint_satisfaction": round(constraint_satisfaction, 6),
        "violations": [],
        "ranking_reason": "Deterministic MVP score using token similarity, relation coverage, and constraints.",
    }


def _rank_candidates(candidates: List[Dict[str, Any]], reference: str, must_terms: List[str]) -> Dict[str, Any]:
    ranked = [_candidate_score(candidate, reference, must_terms) for candidate in candidates]
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return {
        "best_candidate_id": ranked[0]["candidate_id"] if ranked else None,
        "ranked_candidates": ranked,
        "top_5_candidates": ranked[:5],
        "rejected_candidates": [],
    }


def _pair_score(text_item: Dict[str, Any], image_item: Dict[str, Any], knowledge: Dict[str, Any]) -> Dict[str, Any]:
    text_content = text_item["candidate"]["generated_content"]["value"]
    image_content = image_item["candidate"]["generated_content"]["value"]
    pair_consistency = token_cosine(text_content, image_content)
    shared_relation = 1.0 if text_item["candidate"]["primary_relation"] == image_item["candidate"]["primary_relation"] else 0.6
    entity_type = knowledge["entity_profile"]["entity_type"]
    entity_type_consistency = token_overlap(text_content + " " + image_content, [entity_type])
    kg_consistency = clamp01(0.45 * shared_relation + 0.35 * pair_consistency + 0.20 * entity_type_consistency)
    final = clamp01(0.18 * text_item["final_score"] + 0.18 * image_item["final_score"] + 0.20 * pair_consistency + 0.22 * kg_consistency + 0.22 * shared_relation)
    pair_id = f"{text_item['candidate_id']}__{image_item['candidate_id']}"
    return {
        "pair_id": pair_id,
        "text_candidate_id": text_item["candidate_id"],
        "image_candidate_id": image_item["candidate_id"],
        "text_candidate": text_item["candidate"],
        "image_candidate": image_item["candidate"],
        "final_score": round(final, 6),
        "text_score": text_item["final_score"],
        "image_score": image_item["final_score"],
        "missing_modality_consistency": round(pair_consistency, 6),
        "kg_consistency_score": round(kg_consistency, 6),
        "shared_relation_score": shared_relation,
        "violations": [],
        "ranking_reason": "Pair score combines individual candidate scores, relation agreement, and token-level text-prompt consistency.",
    }


def kg_aware_candidate_ranking(record: EntityRecord, missing_mode: str, knowledge: Dict[str, Any], reasoning: Dict[str, Any], generation: Dict[str, Any]) -> Dict[str, Any]:
    del record
    text_reference = _kg_reference(knowledge, reasoning, "text")
    image_reference = _kg_reference(knowledge, reasoning, "image")
    text_must = [item["constraint"] for item in reasoning["modality_constraints"]["text"]["must_include"]]
    image_must = [item["constraint"] for item in reasoning["modality_constraints"]["image"]["must_include"]]
    text_ranking = _rank_candidates(generation.get("text_candidates", []), text_reference, text_must)
    image_ranking = _rank_candidates(generation.get("image_candidates", []), image_reference, image_must)

    pair_ranking = {"ranked_pairs": [], "top_5_pairs": [], "best_pair_id": None, "best_pair": None}
    if missing_mode == "both":
        pairs = []
        for text_item in text_ranking["ranked_candidates"][:3]:
            for image_item in image_ranking["ranked_candidates"][:3]:
                pairs.append(_pair_score(text_item, image_item, knowledge))
        pairs.sort(key=lambda item: item["final_score"], reverse=True)
        pair_ranking = {
            "ranked_pairs": pairs,
            "top_5_pairs": pairs[:5],
            "best_pair_id": pairs[0]["pair_id"] if pairs else None,
            "best_pair": pairs[0] if pairs else None,
        }

    return {
        "kg_aware_candidate_consistency_ranking": {
            "target_modality": missing_mode,
            "candidate_num": 5,
            "kg_reference_text": text_reference,
        },
        "text_ranking": text_ranking,
        "image_ranking": image_ranking,
        "pair_ranking": pair_ranking,
    }
```

- [ ] **Step 5: Run ranking tests**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_stages.py -v
```

Expected: PASS for all stage tests.

---

### Task 6: Pipeline, Output Writer, and CLI

**Files:**
- Create: `MMKGC/mmkgc/io/__init__.py`
- Create: `MMKGC/mmkgc/io/writer.py`
- Create: `MMKGC/mmkgc/pipeline.py`
- Create: `MMKGC/mmkgc/run.py`
- Create: `MMKGC/configs/default.json`
- Create: `MMKGC/tests/test_pipeline_outputs.py`

- [ ] **Step 1: Write end-to-end output test**

Create `MMKGC/tests/test_pipeline_outputs.py`:

```python
import json
from pathlib import Path

from mmkgc.data import MKGWDataStore
from mmkgc.pipeline import MMKGCPipeline
from mmkgc.io import RunWriter
from tests.test_data import make_fixture


def test_pipeline_writes_stage_artifacts(tmp_path):
    dataset_dir = make_fixture(tmp_path / "dataset")
    output_root = tmp_path / "outputs"
    store = MKGWDataStore(dataset_dir)
    pipeline = MMKGCPipeline(candidate_budget=5)
    writer = RunWriter(output_root, run_name="test_run", config={"missing_mode": "both"})

    records = store.select_records("both", limit=1)
    result = pipeline.run(records[0], missing_mode="both")
    writer.write_entity(result)
    writer.write_summary()

    run_dir = output_root / "test_run"
    entity_dir = run_dir / "entities" / "Q1"
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "metrics_summary.json").exists()
    assert (entity_dir / "0_coarse_relation_filtering.json").exists()
    assert (entity_dir / "5_kg_aware_pair_ranking.json").exists()
    final = json.loads((entity_dir / "final.json").read_text(encoding="utf-8"))
    assert final["entity"]["entity_key"] == "Q1"
    assert final["counts"]["text_candidates"] == 5
    assert final["counts"]["image_candidates"] == 5
    assert final["counts"]["pair_candidates"] == 9
```

- [ ] **Step 2: Run output test and verify failure**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_pipeline_outputs.py -v
```

Expected: FAIL because pipeline and writer modules do not exist.

- [ ] **Step 3: Implement output exports and writer**

Create `MMKGC/mmkgc/io/__init__.py`:

```python
from .writer import RunWriter

__all__ = ["RunWriter"]
```

Create `MMKGC/mmkgc/io/writer.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from mmkgc.utils import write_json


class RunWriter:
    def __init__(self, output_root: str | Path, run_name: str, config: Dict[str, Any]):
        self.run_dir = Path(output_root) / run_name
        self.entities_dir = self.run_dir / "entities"
        self.results_path = self.run_dir / "results.jsonl"
        self.config = config
        self.processed = 0
        self.failed = 0
        self.text_counts = []
        self.image_counts = []
        self.pair_counts = []
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.entities_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.run_dir / "config.json", config)
        self.results_path.write_text("", encoding="utf-8")

    def write_entity(self, result: Dict[str, Any]) -> None:
        key = result["entity"]["entity_key"]
        entity_dir = self.entities_dir / key
        stage_files = [
            ("0_coarse_relation_filtering.json", result["coarse_relation_filtering"]),
            ("1_knowledge_modeling.json", result["knowledge_modeling"]),
            ("2_graph_constrained_reasoning.json", result["graph_constrained_reasoning"]),
            ("3_fine_grained_relation_selection.json", result["fine_grained_relation_selection"]),
            ("4_relation_alternation_generation.json", result["relation_alternation_generation"]),
            ("5_kg_aware_pair_ranking.json", result["kg_aware_candidate_consistency_ranking"]),
            ("final.json", result["final"]),
        ]
        for filename, payload in stage_files:
            write_json(entity_dir / filename, payload)
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result["final"], ensure_ascii=False) + "\n")
        self.processed += 1
        self.text_counts.append(result["final"]["counts"]["text_candidates"])
        self.image_counts.append(result["final"]["counts"]["image_candidates"])
        self.pair_counts.append(result["final"]["counts"]["pair_candidates"])

    def record_failure(self) -> None:
        self.failed += 1

    def write_summary(self) -> None:
        def avg(values):
            return round(sum(values) / len(values), 3) if values else 0
        write_json(
            self.run_dir / "metrics_summary.json",
            {
                "processed_entities": self.processed,
                "failed_entities": self.failed,
                "avg_text_candidates": avg(self.text_counts),
                "avg_image_candidates": avg(self.image_counts),
                "avg_pair_candidates": avg(self.pair_counts),
                "fallback_counts": {},
                "stage_failure_counts": {},
            },
        )
```

- [ ] **Step 4: Implement pipeline**

Create `MMKGC/mmkgc/pipeline.py`:

```python
from __future__ import annotations

from typing import Any, Dict

from mmkgc.adapters import RuleImagePromptGenerator, RuleTextGenerator
from mmkgc.data import EntityRecord
from mmkgc.ranking import kg_aware_candidate_ranking
from mmkgc.stages import (
    coarse_relation_filtering,
    fine_grained_relation_selection,
    graph_constrained_reasoning,
    knowledge_modeling,
    relation_alternation_generation,
)


class MMKGCPipeline:
    def __init__(self, candidate_budget: int = 5):
        if candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive")
        self.candidate_budget = candidate_budget
        self.text_generator = RuleTextGenerator()
        self.image_prompt_generator = RuleImagePromptGenerator()

    def run(self, record: EntityRecord, missing_mode: str) -> Dict[str, Any]:
        coarse = coarse_relation_filtering(record, missing_mode)
        knowledge = knowledge_modeling(record, missing_mode, coarse)
        reasoning = graph_constrained_reasoning(missing_mode, knowledge)
        selection = fine_grained_relation_selection(missing_mode, knowledge, reasoning, self.candidate_budget)
        generation = relation_alternation_generation(
            record,
            missing_mode,
            knowledge,
            reasoning,
            selection,
            self.text_generator,
            self.image_prompt_generator,
            self.candidate_budget,
        )
        ranking = kg_aware_candidate_ranking(record, missing_mode, knowledge, reasoning, generation)
        pair_candidates = len(ranking["pair_ranking"]["ranked_pairs"])
        final = {
            "entity": {"entity_id": record.entity_id, "entity_key": record.entity_key, "label": record.label},
            "missing_mode": missing_mode,
            "best_text_candidate_id": ranking["text_ranking"]["best_candidate_id"],
            "best_image_candidate_id": ranking["image_ranking"]["best_candidate_id"],
            "best_pair_id": ranking["pair_ranking"]["best_pair_id"],
            "best_pair": ranking["pair_ranking"]["best_pair"],
            "top_5_pairs": ranking["pair_ranking"]["top_5_pairs"],
            "counts": {
                "text_candidates": len(generation["text_candidates"]),
                "image_candidates": len(generation["image_candidates"]),
                "pair_candidates": pair_candidates,
            },
        }
        return {
            "entity": final["entity"],
            "coarse_relation_filtering": coarse,
            "knowledge_modeling": knowledge,
            "graph_constrained_reasoning": reasoning,
            "fine_grained_relation_selection": selection,
            "relation_alternation_generation": generation,
            "kg_aware_candidate_consistency_ranking": ranking,
            "final": final,
        }
```

- [ ] **Step 5: Implement CLI**

Create `MMKGC/mmkgc/run.py`:

```python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from mmkgc.data import MKGWDataStore
from mmkgc.io import RunWriter
from mmkgc.pipeline import MMKGCPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMKGC missing modality completion on MKG-W.")
    parser.add_argument("--dataset-dir", default="../KGC/datasets/MKG-W")
    parser.add_argument("--missing-mode", choices=["text", "image", "both"], default="both")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--candidate-budget", type=int, default=5)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"mkgw_{args.missing_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = vars(args) | {"run_name": run_name}
    store = MKGWDataStore(Path(args.dataset_dir))
    pipeline = MMKGCPipeline(candidate_budget=args.candidate_budget)
    writer = RunWriter(args.output_root, run_name, config)
    records = store.select_records(args.missing_mode, args.limit)
    for record in records:
        try:
            writer.write_entity(pipeline.run(record, args.missing_mode))
        except Exception:
            writer.record_failure()
            raise
    writer.write_summary()
    print(json.dumps({"run_dir": str(writer.run_dir), "processed_entities": writer.processed, "failed_entities": writer.failed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add default config**

Create `MMKGC/configs/default.json`:

```json
{
  "dataset": {
    "name": "MKG-W",
    "dataset_dir": "../KGC/datasets/MKG-W"
  },
  "run": {
    "missing_mode": "both",
    "limit": 100,
    "candidate_budget": 5
  },
  "adapters": {
    "text_generator": "rule",
    "image_prompt_generator": "rule",
    "image_generator": "noop",
    "text_embedder": "token",
    "image_text_scorer": "token",
    "kg_judge": "rule"
  }
}
```

- [ ] **Step 7: Run output tests**

Run from `MMKGC/`:

```bash
python -m pytest tests/test_pipeline_outputs.py -v
```

Expected: PASS for output writer test.

---

### Task 7: README and Full Verification

**Files:**
- Create: `MMKGC/README.md`

- [ ] **Step 1: Create README**

Create `MMKGC/README.md`:

```markdown
# MMKGC

Relation-guided missing modality completion for MKG-W.

This directory contains an independent MVP implementation of the six-stage plan in `../MMKG`.

## MVP Scope

- Offline deterministic pipeline.
- Supports `missing-mode text`, `image`, and `both`.
- Default acceptance mode is `both`.
- Generates 5 text candidates and 5 image prompt candidates.
- In `both` mode, ranks 9 text-image pairs from Top-3 text x Top-3 image candidates.
- Does not call LMMs, SDXL, CLIP, BLIP, Sentence-BERT, Gemini, or network services.

## Run

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m mmkgc.run --dataset-dir ../KGC/datasets/MKG-W --missing-mode both --limit 100
```

## Outputs

Each run writes:

```text
outputs/mkgw_both_YYYYMMDD_HHMMSS/
  config.json
  results.jsonl
  metrics_summary.json
  entities/Q.../
    0_coarse_relation_filtering.json
    1_knowledge_modeling.json
    2_graph_constrained_reasoning.json
    3_fine_grained_relation_selection.json
    4_relation_alternation_generation.json
    5_kg_aware_pair_ranking.json
    final.json
```

## Tests

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m pytest tests -v
```
```

- [ ] **Step 2: Run full test suite**

Run from `MMKGC/`:

```bash
python -m pytest tests -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run smoke command on fixture or real MKG-W**

Run from `MMKGC/`:

```bash
python -m mmkgc.run --dataset-dir ../KGC/datasets/MKG-W --missing-mode both --limit 3 --run-name smoke_test
```

Expected output includes:

```json
{
  "run_dir": "outputs/smoke_test",
  "processed_entities": 3,
  "failed_entities": 0
}
```

- [ ] **Step 4: Run acceptance command**

Run from `MMKGC/`:

```bash
python -m mmkgc.run --dataset-dir ../KGC/datasets/MKG-W --missing-mode both --limit 100
```

Expected:

- command exits with code 0
- `outputs/mkgw_both_<timestamp>/results.jsonl` has 100 lines
- `metrics_summary.json` reports `processed_entities = 100`
- a sampled entity directory contains all six stage JSON files and `final.json`

- [ ] **Step 5: Final checkpoint**

If the workspace is in a git repository, run:

```bash
git add MMKGC
git commit -m "feat: add mkgw both-modality mvp pipeline"
```

If the workspace is not in a git repository, report that no commit was created because git discovery fails outside a repository.

---

## Self-Review

Spec coverage:

- Independent `MMKGC` package: Tasks 1, 2, 6, 7.
- MKG-W data reading: Task 2.
- Six stage outputs: Tasks 3, 4, 5, 6.
- Text and image prompt candidates: Task 4.
- Pair ranking: Task 5.
- Mixed output layout: Task 6.
- CLI and 100-entity acceptance command: Tasks 6 and 7.
- Adapter-ready boundaries: Task 4.

Placeholder scan:

- No unresolved placeholders are present.
- No task depends on a function that is not introduced in an earlier task.

Type consistency:

- `EntityRecord` fields are defined in Task 2 and used consistently in Tasks 3-6.
- Stage function names exported in `mmkgc.stages` match test imports.
- Ranking output keys match `MMKGCPipeline` and writer expectations.
