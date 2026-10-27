#!/usr/bin/env python3
"""Rescore legacy image candidates with MMKGC's CLIP and BLIP/ITM models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Protocol


MMKGC_ROOT = Path(__file__).resolve().parents[1]
if str(MMKGC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMKGC_ROOT))

from mmkgc.evaluation.blip_itm import BlipITMConfig, BlipITMLocalScorer
from mmkgc.evaluation.image_quality import (
    ClipLocalConfig,
    ClipLocalImageQualityEvaluator,
)
from mmkgc.ranking.fusion import IMAGE_CANDIDATE_WEIGHTS


DEFAULT_LEGACY_ROOT = MMKGC_ROOT / "baselines/legacy_kb_mkgw_50/knowledge-generation"
DEFAULT_MANIFEST = MMKGC_ROOT / "baselines/legacy_kb_mkgw_50/entity_manifest.json"
DEFAULT_OUTPUT = MMKGC_ROOT / "outputs/legacy_kb_mkgw_50_mmkgc_clip_blip"
DEFAULT_CLIP_MODEL_NAME = "ViT-B/32"
DEFAULT_CLIP_MODEL_DIR = Path("CLIP")
DEFAULT_BLIP_MODEL_DIR = Path("blip-itm-large-coco")
SCORE_VERSION = "legacy_candidates_mmkgc_clip_blip_v1"
CANDIDATE_FIELDS = (
    "entity_index",
    "entity_key",
    "label",
    "category",
    "candidate_index",
    "rank",
    "image_path",
    "prompt",
    "clip_raw_0_100",
    "clip_normalized",
    "blip_itm_raw_0_100",
    "blip_itm_normalized",
    "clip_weight",
    "blip_itm_weight",
    "clip_blip_contribution",
    "clip_blip_score",
)
ENTITY_FIELDS = (
    "entity_index",
    "entity_key",
    "label",
    "category",
    "candidate_count",
    "clip_mean",
    "blip_itm_mean",
    "clip_blip_mean",
    "clip_blip_std",
    "clip_blip_max",
    "clip_blip_min",
    "best_candidate_index",
)


class ImageTextScorer(Protocol):
    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-model-name", default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--clip-model-dir", type=Path, default=DEFAULT_CLIP_MODEL_DIR)
    parser.add_argument("--blip-model-dir", type=Path, default=DEFAULT_BLIP_MODEL_DIR)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device shared by CLIP and BLIP, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--clip-weight",
        type=float,
        default=IMAGE_CANDIDATE_WEIGHTS["clip"],
        help="MMKGC image-fusion CLIP weight before Judge removal.",
    )
    parser.add_argument(
        "--blip-weight",
        type=float,
        default=IMAGE_CANDIDATE_WEIGHTS["blip_itm"],
        help="MMKGC image-fusion BLIP/ITM weight before Judge removal.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional entity limit for a smoke run.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-entity files with the same scoring signature.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and replace existing per-entity results.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to resolve --device auto") from exc
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def validate_fusion_weights(clip_weight: float, blip_weight: float) -> None:
    for name, value in (("clip_weight", clip_weight), ("blip_weight", blip_weight)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    if clip_weight + blip_weight <= 0:
        raise ValueError("clip_weight + blip_weight must be greater than zero")


def normalize_score(raw_score: float) -> float:
    value = float(raw_score)
    if not math.isfinite(value):
        raise ValueError(f"model score must be finite, got {raw_score!r}")
    return round(min(1.0, max(0.0, value / 100.0)), 8)


def fuse_clip_blip(
    clip_normalized: float,
    blip_normalized: float,
    *,
    clip_weight: float,
    blip_weight: float,
) -> tuple[float, float]:
    validate_fusion_weights(clip_weight, blip_weight)
    contribution = clip_weight * clip_normalized + blip_weight * blip_normalized
    normalized = contribution / (clip_weight + blip_weight)
    return round(contribution, 8), round(normalized, 8)


def load_entities(manifest_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    entities = manifest.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError(f"manifest has no entities: {manifest_path}")
    ordered = sorted(entities, key=lambda row: int(row["idx"]))
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero")
        ordered = ordered[:limit]
    return ordered


def load_candidate_pairs(
    entity: dict[str, Any],
    legacy_root: Path,
) -> list[dict[str, Any]]:
    entity_index = int(entity["idx"])
    entity_dir = legacy_root / str(entity_index)
    prompts_path = entity_dir / "prompts.json"
    if not prompts_path.is_file():
        raise FileNotFoundError(prompts_path)
    prompts = read_json(prompts_path)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"expected a non-empty prompt list: {prompts_path}")

    pairs: list[dict[str, Any]] = []
    for candidate_index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"empty prompt for entity {entity_index}, candidate {candidate_index}"
            )
        image_path = entity_dir / f"id_{candidate_index}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        pairs.append(
            {
                "candidate_index": candidate_index,
                "prompt": prompt.strip(),
                "image_path": image_path.resolve(),
            }
        )
    return pairs


def input_fingerprint(pairs: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        image_path = Path(pair["image_path"])
        stat = image_path.stat()
        digest.update(str(pair["candidate_index"]).encode("utf-8"))
        digest.update(str(pair["prompt"]).encode("utf-8"))
        digest.update(str(image_path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def scoring_signature(
    *,
    clip_model_name: str,
    clip_weight_path: Path,
    blip_model_dir: Path,
    device: str,
    clip_weight: float,
    blip_weight: float,
) -> dict[str, Any]:
    return {
        "score_version": SCORE_VERSION,
        "text_binding": "id_i.jpg paired with prompts.json[i]",
        "clip_model_name": clip_model_name,
        "clip_weight_path": str(clip_weight_path.resolve()),
        "blip_model_dir": str(blip_model_dir.resolve()),
        "blip_score": "softmax(itm_score)[positive_class]",
        "device": device,
        "clip_weight": clip_weight,
        "blip_itm_weight": blip_weight,
        "clip_blip_formula": (
            "(clip_weight * clip_normalized + blip_itm_weight * "
            "blip_itm_normalized) / (clip_weight + blip_itm_weight)"
        ),
    }


def score_entity(
    entity: dict[str, Any],
    pairs: list[dict[str, Any]],
    *,
    clip_scorer: ImageTextScorer,
    blip_scorer: ImageTextScorer,
    clip_weight: float,
    blip_weight: float,
    signature: dict[str, Any],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for pair in pairs:
        image_path = Path(pair["image_path"])
        prompt = str(pair["prompt"])
        clip_raw = _single_score(clip_scorer, image_path, prompt, "CLIP")
        blip_raw = _single_score(blip_scorer, image_path, prompt, "BLIP/ITM")
        clip_normalized = normalize_score(clip_raw)
        blip_normalized = normalize_score(blip_raw)
        contribution, combined = fuse_clip_blip(
            clip_normalized,
            blip_normalized,
            clip_weight=clip_weight,
            blip_weight=blip_weight,
        )
        candidates.append(
            {
                "candidate_index": int(pair["candidate_index"]),
                "image_path": str(image_path),
                "prompt": prompt,
                "clip_raw_0_100": round(clip_raw, 6),
                "clip_normalized": clip_normalized,
                "blip_itm_raw_0_100": round(blip_raw, 6),
                "blip_itm_normalized": blip_normalized,
                "clip_weight": clip_weight,
                "blip_itm_weight": blip_weight,
                "clip_blip_contribution": contribution,
                "clip_blip_score": combined,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda row: (-float(row["clip_blip_score"]), int(row["candidate_index"])),
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    candidates.sort(key=lambda row: int(row["candidate_index"]))

    summary = summarize_candidates(candidates)
    return {
        "entity": {
            "idx": int(entity["idx"]),
            "entity_key": str(entity["entity_key"]),
            "label": str(entity["label"]),
            "category": entity.get("category"),
        },
        "input_fingerprint": input_fingerprint(pairs),
        "scoring_signature": signature,
        "candidate_count": len(candidates),
        "summary": summary,
        "candidates": candidates,
    }


def _single_score(
    scorer: ImageTextScorer,
    image_path: Path,
    prompt: str,
    scorer_name: str,
) -> float:
    scores = scorer.score_image_texts(image_path, [prompt])
    if len(scores) != 1:
        raise ValueError(
            f"{scorer_name} returned {len(scores)} scores for one image-text pair"
        )
    score = float(scores[0])
    if not math.isfinite(score):
        raise ValueError(f"{scorer_name} returned a non-finite score: {score!r}")
    return score


def summarize_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("cannot summarize an empty candidate list")
    clip_scores = [float(row["clip_normalized"]) for row in candidates]
    blip_scores = [float(row["blip_itm_normalized"]) for row in candidates]
    combined_scores = [float(row["clip_blip_score"]) for row in candidates]
    best = max(
        candidates,
        key=lambda row: (float(row["clip_blip_score"]), -int(row["candidate_index"])),
    )
    return {
        "clip_mean": round(statistics.fmean(clip_scores), 8),
        "blip_itm_mean": round(statistics.fmean(blip_scores), 8),
        "clip_blip_mean": round(statistics.fmean(combined_scores), 8),
        "clip_blip_std": round(statistics.pstdev(combined_scores), 8),
        "clip_blip_max": round(max(combined_scores), 8),
        "clip_blip_min": round(min(combined_scores), 8),
        "best_candidate_index": int(best["candidate_index"]),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_resumable_result(
    path: Path,
    *,
    entity: dict[str, Any],
    pairs: list[dict[str, Any]],
    signature: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = read_json(path)
    expected_key = str(entity["entity_key"])
    if result.get("entity", {}).get("entity_key") != expected_key:
        raise ValueError(f"resume entity mismatch in {path}")
    if result.get("scoring_signature") != signature:
        raise ValueError(f"resume scoring signature mismatch in {path}; use --overwrite")
    if result.get("input_fingerprint") != input_fingerprint(pairs):
        raise ValueError(f"resume input fingerprint mismatch in {path}; use --overwrite")
    return result


def flatten_candidate_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        entity = result["entity"]
        for candidate in result["candidates"]:
            rows.append(
                {
                    "entity_index": entity["idx"],
                    "entity_key": entity["entity_key"],
                    "label": entity["label"],
                    "category": entity.get("category"),
                    **candidate,
                }
            )
    return rows


def flatten_entity_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        entity = result["entity"]
        rows.append(
            {
                "entity_index": entity["idx"],
                "entity_key": entity["entity_key"],
                "label": entity["label"],
                "category": entity.get("category"),
                "candidate_count": result["candidate_count"],
                **result["summary"],
            }
        )
    return rows


def build_run_summary(
    results: list[dict[str, Any]],
    *,
    signature: dict[str, Any],
    legacy_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    candidate_rows = flatten_candidate_rows(results)
    combined = [float(row["clip_blip_score"]) for row in candidate_rows]
    clip = [float(row["clip_normalized"]) for row in candidate_rows]
    blip = [float(row["blip_itm_normalized"]) for row in candidate_rows]
    return {
        "score_version": SCORE_VERSION,
        "legacy_root": str(legacy_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "scoring_signature": signature,
        "entity_count": len(results),
        "candidate_count": len(candidate_rows),
        "averages": {
            "clip_normalized": round(statistics.fmean(clip), 8),
            "blip_itm_normalized": round(statistics.fmean(blip), 8),
            "clip_blip_score": round(statistics.fmean(combined), 8),
        },
        "clip_blip_std": round(statistics.pstdev(combined), 8),
        "clip_blip_max": round(max(combined), 8),
        "clip_blip_min": round(min(combined), 8),
        "output_files": {
            "candidate_scores_csv": "candidate_scores.csv",
            "entity_summary_csv": "entity_summary.csv",
            "entity_results_dir": "entities",
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_fusion_weights(args.clip_weight, args.blip_weight)
    device = resolve_device(args.device)
    entities = load_entities(args.manifest, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entities_output_dir = args.output_dir / "entities"
    entities_output_dir.mkdir(parents=True, exist_ok=True)

    clip_scorer = ClipLocalImageQualityEvaluator(
        ClipLocalConfig(
            model_name=args.clip_model_name,
            model_dir=args.clip_model_dir,
            device=device,
            project_root=MMKGC_ROOT,
        )
    )
    blip_scorer = BlipITMLocalScorer(
        BlipITMConfig(
            model_dir=args.blip_model_dir,
            device=device,
            project_root=MMKGC_ROOT,
        )
    )
    signature = scoring_signature(
        clip_model_name=args.clip_model_name,
        clip_weight_path=clip_scorer.weight_path,
        blip_model_dir=blip_scorer.model_dir,
        device=device,
        clip_weight=args.clip_weight,
        blip_weight=args.blip_weight,
    )

    results: list[dict[str, Any]] = []
    for position, entity in enumerate(entities, start=1):
        pairs = load_candidate_pairs(entity, args.legacy_root)
        entity_key = str(entity["entity_key"])
        result_path = entities_output_dir / f"{entity_key}.json"
        result = None
        if args.resume:
            result = load_resumable_result(
                result_path,
                entity=entity,
                pairs=pairs,
                signature=signature,
            )
        elif result_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"result already exists: {result_path}; use --resume or --overwrite"
            )

        if result is None:
            result = score_entity(
                entity,
                pairs,
                clip_scorer=clip_scorer,
                blip_scorer=blip_scorer,
                clip_weight=args.clip_weight,
                blip_weight=args.blip_weight,
                signature=signature,
            )
            write_json(result_path, result)
            status = "scored"
        else:
            status = "resumed"
        results.append(result)
        print(
            f"[{position}/{len(entities)}] {entity_key} {status} "
            f"mean={result['summary']['clip_blip_mean']:.6f}",
            flush=True,
        )

    candidate_rows = flatten_candidate_rows(results)
    entity_rows = flatten_entity_rows(results)
    summary = build_run_summary(
        results,
        signature=signature,
        legacy_root=args.legacy_root,
        manifest_path=args.manifest,
    )
    write_csv(args.output_dir / "candidate_scores.csv", CANDIDATE_FIELDS, candidate_rows)
    write_csv(args.output_dir / "entity_summary.csv", ENTITY_FIELDS, entity_rows)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
