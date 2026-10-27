#!/usr/bin/env python3
"""Collect the original and best generated image for each comparison entity."""

import argparse
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MMKGC_RUN = ROOT / "MMKGC/outputs/mkgw_image_test_50_20261014_fixed2"
DEFAULT_LEGACY_RUN = ROOT / "MMKGC/baselines/legacy_kb_mkgw_50/knowledge-generation"
DEFAULT_LEGACY_RESCORE = ROOT / "MMKGC/outputs/legacy_kb_mkgw_50_mmkgc_clip_blip"
DEFAULT_MANIFEST = ROOT / "MMKGC/baselines/legacy_kb_mkgw_50/entity_manifest.json"
DEFAULT_OUTPUT = ROOT / "MMKGC/outputs/mkgw_image_comparison_50"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_legacy_score(score):
    """Convert legacy's percentage-scale semantic score to the common 0-1 scale."""
    if score is None:
        return None
    return round(max(0.0, min(1.0, float(score) / 100.0)), 6)


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return value or "entity"


def primary_original(entity_dir):
    manifest_path = entity_dir / "original_images" / "manifest.json"
    manifest = load_json(manifest_path)
    images = manifest.get("images", [])
    primary = next((item for item in images if item.get("is_primary_loader_image")), None)
    if primary is None and images:
        primary = images[0]
    if primary is None:
        raise FileNotFoundError(f"No original image listed in {manifest_path}")
    path = entity_dir / "original_images" / primary["file_name"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def mmkgc_best(entity_dir):
    ranking = load_json(entity_dir / "4_kg_aware_candidate_consistency_ranking.json")
    ranking_data = ranking["data"]
    image_ranking = ranking_data["image_ranking"]
    candidate_id = image_ranking["best_candidate_id"]
    candidate_by_id = {
        item["candidate_id"]: item
        for item in image_ranking.get("ranked_candidates", [])
    }
    candidate = candidate_by_id[candidate_id]
    components = candidate.get("scores", {}).get("components", {})
    image_weights = (
        ranking_data.get("ranking_profile", {})
        .get("candidate_score_weights", {})
        .get("image", {})
    )
    clip_score = float(components["clip"])
    blip_score = float(components["blip_itm"])
    judge_score = float(components["gpt4omini_judge"])
    clip_weight = float(image_weights["clip"])
    blip_weight = float(image_weights["blip_itm"])
    judge_weight = float(image_weights["judge"])
    clip_blip_contribution = clip_weight * clip_score + blip_weight * blip_score
    clip_blip_weight = clip_weight + blip_weight
    clip_blip_score = (
        clip_blip_contribution / clip_blip_weight
        if clip_blip_weight > 0
        else None
    )
    image_path = entity_dir / "generated_images" / f"{candidate_id}.png"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return image_path, candidate_id, {
        "final_score": float(candidate["final_score"]),
        "clip_score": clip_score,
        "blip_itm_score": blip_score,
        "judge_score": judge_score,
        "clip_weight": clip_weight,
        "blip_itm_weight": blip_weight,
        "judge_weight": judge_weight,
        "clip_blip_contribution": clip_blip_contribution,
        "clip_blip_score": clip_blip_score,
    }


def legacy_best(entity_dir, rescore_path):
    ranking = load_json(entity_dir / "ranking.json")
    rescore = load_json(rescore_path)
    candidate_index = int(rescore["summary"]["best_candidate_index"])
    candidate_by_index = {
        int(candidate["candidate_index"]): candidate
        for candidate in rescore["candidates"]
    }
    candidate = candidate_by_index[candidate_index]
    semantic_scores = ranking["semantic_score"]
    total_scores = ranking["total_score"]
    graph_scores = ranking.get("graph_score", [])
    semantic_score = float(semantic_scores[candidate_index])
    total_score = float(total_scores[candidate_index])
    graph_score = float(graph_scores[candidate_index]) if graph_scores else 0.0
    without_graph_score = total_score - graph_score
    image_path = entity_dir / f"id_{candidate_index}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return image_path, candidate_index, {
        "semantic_score": semantic_score,
        "total_score": total_score,
        "graph_score": graph_score,
        "without_graph_score": without_graph_score,
        "without_graph_normalized": normalize_legacy_score(without_graph_score),
        "original_semantic_best_candidate_index": int(ranking["best_candidate"]),
        "clip_raw_0_100": float(candidate["clip_raw_0_100"]),
        "clip_score": float(candidate["clip_normalized"]),
        "blip_itm_raw_0_100": float(candidate["blip_itm_raw_0_100"]),
        "blip_itm_score": float(candidate["blip_itm_normalized"]),
        "clip_weight": float(candidate["clip_weight"]),
        "blip_itm_weight": float(candidate["blip_itm_weight"]),
        "clip_blip_contribution": float(candidate["clip_blip_contribution"]),
        "clip_blip_score": float(candidate["clip_blip_score"]),
        "candidate_rank": int(candidate["rank"]),
        "candidate_summary": rescore["summary"],
        "scoring_signature": rescore["scoring_signature"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmkgc-run", type=Path, default=DEFAULT_MMKGC_RUN)
    parser.add_argument("--legacy-run", type=Path, default=DEFAULT_LEGACY_RUN)
    parser.add_argument("--legacy-rescore", type=Path, default=DEFAULT_LEGACY_RESCORE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    legacy_rescore_summary = load_json(args.legacy_rescore / "summary.json")
    entities = manifest["entities"]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    for entity in sorted(entities, key=lambda item: item["idx"]):
        idx = entity["idx"]
        qid = entity["entity_key"]
        label = entity["label"]
        mmkgc_dir = args.mmkgc_run / "entities" / qid
        legacy_dir = args.legacy_run / str(idx)
        folder = args.output / f"{idx:03d}_{safe_name(qid)}_{safe_name(label)}"
        folder.mkdir(parents=True, exist_ok=True)

        original_path = primary_original(mmkgc_dir)
        mmkgc_path, mmkgc_candidate, mmkgc_scores = mmkgc_best(mmkgc_dir)
        legacy_path, legacy_candidate, legacy_scores = legacy_best(
            legacy_dir,
            args.legacy_rescore / "entities" / f"{qid}.json",
        )

        original_target = folder / f"original{original_path.suffix.lower()}"
        mmkgc_target = folder / f"mmkgc_best{mmkgc_path.suffix.lower()}"
        legacy_target = folder / f"legacy_best{legacy_path.suffix.lower()}"
        shutil.copy2(original_path, original_target)
        shutil.copy2(mmkgc_path, mmkgc_target)
        shutil.copy2(legacy_path, legacy_target)

        records.append({
            "idx": idx,
            "entity_key": qid,
            "label": label,
            "category": entity.get("category"),
            "original": {
                "source": str(original_path),
                "copied": str(original_target),
            },
            "mmkgc": {
                "candidate_id": mmkgc_candidate,
                "score": mmkgc_scores["final_score"],
                "score_normalized": round(mmkgc_scores["final_score"], 6),
                "clip_score": mmkgc_scores["clip_score"],
                "blip_itm_score": mmkgc_scores["blip_itm_score"],
                "judge_score": mmkgc_scores["judge_score"],
                "score_weights": {
                    "clip": mmkgc_scores["clip_weight"],
                    "blip_itm": mmkgc_scores["blip_itm_weight"],
                    "judge": mmkgc_scores["judge_weight"],
                },
                "clip_blip_contribution": mmkgc_scores["clip_blip_contribution"],
                "clip_blip_score": mmkgc_scores["clip_blip_score"],
                "clip_blip_score_basis": (
                    "(clip_weight * clip + blip_itm_weight * blip_itm) / "
                    "(clip_weight + blip_itm_weight); judge excluded"
                ),
                "source": str(mmkgc_path),
                "copied": str(mmkgc_target),
            },
            "legacy_knowledge_bridger": {
                "candidate_index": legacy_candidate,
                "candidate_rank": legacy_scores["candidate_rank"],
                "score": legacy_scores["clip_blip_score"],
                "score_basis": (
                    "MMKGC CLIP+BLIP scorer: "
                    "(0.2 * clip + 0.3 * blip_itm) / 0.5"
                ),
                "clip_raw_0_100": legacy_scores["clip_raw_0_100"],
                "clip_score": legacy_scores["clip_score"],
                "blip_itm_raw_0_100": legacy_scores["blip_itm_raw_0_100"],
                "blip_itm_score": legacy_scores["blip_itm_score"],
                "score_weights": {
                    "clip": legacy_scores["clip_weight"],
                    "blip_itm": legacy_scores["blip_itm_weight"],
                },
                "clip_blip_contribution": legacy_scores["clip_blip_contribution"],
                "clip_blip_score": legacy_scores["clip_blip_score"],
                "clip_blip_score_basis": (
                    "(clip_weight * clip + blip_itm_weight * blip_itm) / "
                    "(clip_weight + blip_itm_weight); judge excluded"
                ),
                "candidate_summary": legacy_scores["candidate_summary"],
                "scoring_signature": legacy_scores["scoring_signature"],
                "original_semantic_best_candidate_index": legacy_scores[
                    "original_semantic_best_candidate_index"
                ],
                "semantic_score": legacy_scores["semantic_score"],
                "total_score": legacy_scores["total_score"],
                "graph_score": legacy_scores["graph_score"],
                "without_graph_score": legacy_scores["without_graph_score"],
                "without_graph_normalized": legacy_scores["without_graph_normalized"],
                "without_graph_score_basis": "(total_score - graph_score) / 100",
                "score_normalized": legacy_scores["clip_blip_score"],
                "source": str(legacy_path),
                "copied": str(legacy_target),
            },
        })

    manifest_path = args.output / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "entity_count": len(records),
            "selection_policy": {
                "original": "primary_loader_image from original_images/manifest.json",
                "mmkgc": (
                    "best_image_candidate_id from stage-4 ranking; also the highest "
                    "CLIP+BLIP candidate among candidates scored by both models"
                ),
                "legacy_knowledge_bridger": (
                    "highest clip_blip_score after rescoring all five candidates with "
                    "MMKGC ViT-B/32 and BLIP/ITM against prompts.json[i]"
                ),
            },
            "legacy_rescore_summary": legacy_rescore_summary,
            "entities": records,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"organized_entities={len(records)}")
    print(f"output={args.output}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
