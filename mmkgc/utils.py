"""Small deterministic helpers for the MMKGC MVP."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: object | None) -> list[str]:
    """Return lowercase alphanumeric tokens from text."""
    return TOKEN_RE.findall(str(text or "").lower())


def clamp01(value: float) -> float:
    """Clamp a numeric value into the inclusive [0.0, 1.0] range."""
    return float(min(1.0, max(0.0, value)))


def token_cosine(left: str, right: str) -> float:
    """Compute rounded cosine similarity between token-count vectors."""
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0

    shared = set(left_counts) & set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    return round(clamp01(dot / (left_norm * right_norm)), 6)


def token_overlap(text: str, required_terms: Iterable[str]) -> float:
    """Score how many required token groups are covered by text tokens."""
    term_token_sets = [set(tokens) for tokens in (tokenize(term) for term in required_terms) if tokens]
    if not term_token_sets:
        return 1.0

    text_tokens = set(tokenize(text))
    covered = 0
    for term_tokens in term_token_sets:
        if term_tokens <= text_tokens:
            covered += 1

    return round(covered / len(term_token_sets), 6)


def slugify(value: str) -> str:
    """Convert text into a lowercase underscore-separated slug."""
    tokens = tokenize(value)
    return "_".join(tokens) if tokens else "item"


def write_json(path: str | Path, data) -> None:
    """Write UTF-8 JSON, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
