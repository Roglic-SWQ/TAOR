"""Command-line entry point for offline MMKGC run evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from mmkgc.evaluation.blip_itm import (
    BlipITMConfig,
    BlipITMLocalScorer,
    evaluate_entity_blip_itm,
    validate_blip_model_dir,
)
from mmkgc.evaluation.evaluator import evaluate_entity
from mmkgc.evaluation.image_quality import (
    ClipLocalConfig,
    ClipLocalImageQualityEvaluator,
    evaluate_entity_image_quality,
    validate_clip_model_dir,
    validate_clip_weight_path,
)
from mmkgc.evaluation.image_quality_writer import write_clip_image_quality_outputs
from mmkgc.evaluation.loader import iter_entity_artifact_results, validate_run_dir
from mmkgc.evaluation.multi_image_quality_writer import (
    write_blip_itm_outputs,
    write_image_quality_model_comparison_summary,
    write_qwen_vl_judge_outputs,
)
from mmkgc.evaluation.gemini_embedding import (
    GeminiEmbeddingClient,
    evaluate_entity_gemini_embedding,
    load_gemini_embedding_config,
)
from mmkgc.evaluation.gemini_embedding_writer import write_gemini_embedding_outputs
from mmkgc.evaluation.qwen_vl_judge import (
    QwenVLJudgeClient,
    evaluate_entity_qwen_vl_judge,
    load_qwen_vl_judge_config,
)
from mmkgc.evaluation.text_quality import (
    BERTScoreLocalScorer,
    evaluate_entity_text_quality,
)
from mmkgc.evaluation.text_quality_writer import write_bertscore_text_quality_outputs
from mmkgc.evaluation.writer import write_evaluation_outputs


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate an existing MMKGC run directory and write offline reports."""
    args = _parse_args(argv)
    run_dir = validate_run_dir(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluation"
    clip_weight_path: Path | None = None
    clip_evaluator: ClipLocalImageQualityEvaluator | None = None
    blip_model_dir: Path | None = None
    blip_scorer: BlipITMLocalScorer | None = None
    qwen_vl_config = None
    qwen_vl_judge: QwenVLJudgeClient | None = None
    bertscore_scorer: BERTScoreLocalScorer | None = None
    gemini_embedding_config = None
    gemini_embedding_client: GeminiEmbeddingClient | None = None

    if args.image_quality_evaluator == "clip-local":
        clip_model_dir = validate_clip_model_dir(args.clip_model_dir)
        clip_weight_path = validate_clip_weight_path(
            clip_model_dir,
            args.clip_model_name,
        )
        clip_evaluator = ClipLocalImageQualityEvaluator(
            ClipLocalConfig(
                model_name=args.clip_model_name,
                model_dir=Path(args.clip_model_dir),
                device=args.clip_device,
            )
        )

    if args.blip_itm_evaluator == "blip-itm-local":
        blip_model_dir = validate_blip_model_dir(args.blip_model_dir)
        blip_scorer = BlipITMLocalScorer(
            BlipITMConfig(
                model_dir=Path(args.blip_model_dir),
                device=args.blip_device,
            )
        )

    if args.vlm_judge_evaluator == "qwen-vl-judge":
        qwen_vl_config = load_qwen_vl_judge_config(
            args.vlm_judge_model_config,
            timeout_seconds=args.vlm_judge_timeout_seconds,
        )
        qwen_vl_judge = QwenVLJudgeClient(qwen_vl_config)

    if args.text_quality_evaluator == "bertscore-local":
        bertscore_scorer = BERTScoreLocalScorer(
            model_type=args.bertscore_model_type,
            device=args.bertscore_device,
        )

    if args.gemini_embedding_evaluator == "gemini-api":
        gemini_embedding_config = load_gemini_embedding_config(
            args.gemini_embedding_config,
        )
        gemini_embedding_client = GeminiEmbeddingClient(gemini_embedding_config)

    load_results = list(iter_entity_artifact_results(run_dir))

    entity_evaluations: list[dict] = []
    clip_quality_results: list[dict] = []
    blip_itm_results: list[dict] = []
    qwen_vl_judge_results: list[dict] = []
    text_quality_results: list[dict] = []
    gemini_embedding_results: list[dict] = []
    failed_entity_keys: list[str] = []

    for load_result in load_results:
        if load_result.error is not None:
            failed_entity_keys.append(load_result.entity_key)
            print(
                f"Failed to load entity {load_result.entity_key}: {load_result.error}",
                file=sys.stderr,
            )
            continue

        artifacts = load_result.artifacts
        if artifacts is None:
            failed_entity_keys.append(load_result.entity_key)
            print(
                f"Failed to load entity {load_result.entity_key}: missing artifacts",
                file=sys.stderr,
            )
            continue

        try:
            entity_evaluations.append(evaluate_entity(artifacts))
        except Exception as exc:  # pragma: no cover - exercised via integration callers.
            failed_entity_keys.append(artifacts.entity_key)
            print(
                f"Failed to evaluate entity {artifacts.entity_key}: {exc}",
                file=sys.stderr,
            )

    if args.image_quality_evaluator == "clip-local":
        if clip_evaluator is None or clip_weight_path is None:
            raise RuntimeError("clip-local evaluator was not initialized")
        for load_result in load_results:
            if load_result.error is not None or load_result.artifacts is None:
                continue
            clip_quality_results.append(
                evaluate_entity_image_quality(
                    load_result.artifacts,
                    scorer=clip_evaluator,
                    clip_model_name=args.clip_model_name,
                    clip_weight_path=clip_weight_path,
                )
            )

    if args.blip_itm_evaluator == "blip-itm-local":
        if blip_scorer is None or blip_model_dir is None:
            raise RuntimeError("blip-itm-local evaluator was not initialized")
        for load_result in load_results:
            if load_result.error is not None or load_result.artifacts is None:
                continue
            blip_itm_results.append(
                evaluate_entity_blip_itm(
                    load_result.artifacts,
                    scorer=blip_scorer,
                    model_dir=blip_model_dir,
                )
            )

    if args.vlm_judge_evaluator == "qwen-vl-judge":
        if qwen_vl_judge is None or qwen_vl_config is None:
            raise RuntimeError("qwen-vl-judge evaluator was not initialized")
        for load_result in load_results:
            if load_result.error is not None or load_result.artifacts is None:
                continue
            qwen_vl_judge_results.append(
                evaluate_entity_qwen_vl_judge(
                    load_result.artifacts,
                    judge=qwen_vl_judge,
                    model_name=qwen_vl_config.model_name,
                    base_url=qwen_vl_config.base_url,
                )
            )

    if args.text_quality_evaluator == "bertscore-local":
        if bertscore_scorer is None:
            raise RuntimeError("bertscore-local evaluator was not initialized")
        for load_result in load_results:
            if load_result.error is not None or load_result.artifacts is None:
                continue
            text_quality_results.append(
                evaluate_entity_text_quality(
                    load_result.artifacts,
                    scorer=bertscore_scorer,
                    model_type=args.bertscore_model_type,
                )
            )

    if args.gemini_embedding_evaluator == "gemini-api":
        if gemini_embedding_client is None or gemini_embedding_config is None:
            raise RuntimeError("gemini-api evaluator was not initialized")
        for load_result in load_results:
            if load_result.error is not None or load_result.artifacts is None:
                continue
            gemini_embedding_results.append(
                evaluate_entity_gemini_embedding(
                    load_result.artifacts,
                    client=gemini_embedding_client,
                    model_name=gemini_embedding_config.model_name,
                )
            )

    write_evaluation_outputs(
        output_dir=output_dir,
        entity_evaluations=entity_evaluations,
        run_dir=run_dir,
        overwrite=args.overwrite,
        case_limit=args.case_limit,
        failed_entity_keys=failed_entity_keys,
    )
    clip_summary = None
    if args.image_quality_evaluator == "clip-local":
        clip_summary = write_clip_image_quality_outputs(
            output_dir=output_dir,
            run_dir=run_dir,
            entity_quality_results=clip_quality_results,
            overwrite=args.overwrite,
        )
    blip_summary = None
    if args.blip_itm_evaluator == "blip-itm-local":
        blip_summary = write_blip_itm_outputs(
            output_dir=output_dir,
            run_dir=run_dir,
            entity_itm_results=blip_itm_results,
            overwrite=args.overwrite,
        )
    qwen_vl_summary = None
    if args.vlm_judge_evaluator == "qwen-vl-judge":
        qwen_vl_summary = write_qwen_vl_judge_outputs(
            output_dir=output_dir,
            run_dir=run_dir,
            entity_judge_results=qwen_vl_judge_results,
            overwrite=args.overwrite,
        )
    bertscore_summary = None
    if args.text_quality_evaluator == "bertscore-local":
        bertscore_summary = write_bertscore_text_quality_outputs(
            output_dir=output_dir,
            run_dir=run_dir,
            entity_text_quality_results=text_quality_results,
            overwrite=args.overwrite,
        )
    gemini_summary = None
    if args.gemini_embedding_evaluator == "gemini-api":
        gemini_summary = write_gemini_embedding_outputs(
            output_dir=output_dir,
            run_dir=run_dir,
            entity_embedding_results=gemini_embedding_results,
            overwrite=args.overwrite,
        )
    comparison_summary = None
    if (
        clip_summary is not None
        or blip_summary is not None
        or qwen_vl_summary is not None
        or bertscore_summary is not None
        or gemini_summary is not None
    ):
        comparison_summary = write_image_quality_model_comparison_summary(
            output_dir=output_dir,
            run_dir=run_dir,
            clip_summary=clip_summary,
            blip_summary=blip_summary,
            qwen_vl_summary=qwen_vl_summary,
            bertscore_summary=bertscore_summary,
            gemini_summary=gemini_summary,
            overwrite=args.overwrite,
        )
    print(
        json.dumps(
            {
                "evaluation_dir": str(output_dir),
                "evaluated_entities": len(entity_evaluations),
                "failed_entities": len(failed_entity_keys),
                "failed_entity_keys": failed_entity_keys,
                "image_quality_evaluator": args.image_quality_evaluator,
                "clip_evaluated_entities": (
                    clip_summary["evaluated_entities"] if clip_summary else 0
                ),
                "clip_evaluated_images": (
                    clip_summary["evaluated_images"] if clip_summary else 0
                ),
                "blip_itm_evaluator": args.blip_itm_evaluator,
                "blip_evaluated_entities": (
                    blip_summary["evaluated_entities"] if blip_summary else 0
                ),
                "blip_evaluated_images": (
                    blip_summary["evaluated_images"] if blip_summary else 0
                ),
                "vlm_judge_evaluator": args.vlm_judge_evaluator,
                "vlm_judged_entities": (
                    qwen_vl_summary["evaluated_entities"] if qwen_vl_summary else 0
                ),
                "vlm_judged_images": (
                    qwen_vl_summary["evaluated_images"] if qwen_vl_summary else 0
                ),
                "text_quality_evaluator": args.text_quality_evaluator,
                "bertscore_evaluated_entities": (
                    bertscore_summary["evaluated_entities"] if bertscore_summary else 0
                ),
                "bertscore_evaluated_text_candidates": (
                    bertscore_summary["evaluated_text_candidates"]
                    if bertscore_summary
                    else 0
                ),
                "gemini_embedding_evaluator": args.gemini_embedding_evaluator,
                "gemini_evaluated_entities": (
                    gemini_summary["evaluated_entities"] if gemini_summary else 0
                ),
                "gemini_evaluated_text_candidates": (
                    gemini_summary["evaluated_text_candidates"]
                    if gemini_summary
                    else 0
                ),
                "gemini_evaluated_images": (
                    gemini_summary["evaluated_images"] if gemini_summary else 0
                ),
                "gemini_evaluated_pairs": (
                    gemini_summary["evaluated_pairs"] if gemini_summary else 0
                ),
                "image_quality_comparison_written": comparison_summary is not None,
                "image_quality_comparison_evaluated_images": (
                    comparison_summary["evaluated_images"] if comparison_summary else 0
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an existing MMKGC run offline.")
    parser.add_argument("--run-dir", required=True, help="Path to an existing MMKGC run directory.")
    parser.add_argument(
        "--output-dir",
        help="Directory for evaluation artifacts. Defaults to <run-dir>/evaluation.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=20,
        help="Number of best, worst, and fallback cases to export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite evaluation artifacts in an existing output directory.",
    )
    parser.add_argument(
        "--image-quality-evaluator",
        choices=("none", "clip-local"),
        default="none",
        help="Optional image quality evaluator to run after offline evaluation.",
    )
    parser.add_argument(
        "--clip-model-name",
        default="ViT-B/32",
        help="CLIP model name for --image-quality-evaluator clip-local.",
    )
    parser.add_argument(
        "--clip-model-dir",
        default="CLIP",
        help="Directory containing local CLIP weights.",
    )
    parser.add_argument(
        "--clip-device",
        default="cuda",
        help="Device used by the local CLIP evaluator.",
    )
    parser.add_argument(
        "--blip-itm-evaluator",
        choices=("none", "blip-itm-local"),
        default="none",
        help="Optional BLIP/ITM evaluator to run after offline evaluation.",
    )
    parser.add_argument(
        "--blip-model-dir",
        default="blip-itm-large-coco",
        help="Directory containing local BLIP/ITM weights.",
    )
    parser.add_argument(
        "--blip-device",
        default="cuda",
        help="Device used by the local BLIP/ITM evaluator.",
    )
    parser.add_argument(
        "--vlm-judge-evaluator",
        choices=("none", "qwen-vl-judge"),
        default="none",
        help="Optional VLM judge evaluator to run after offline evaluation.",
    )
    parser.add_argument(
        "--vlm-judge-model-config",
        default="configs/openai-gpt4o-mini-judge.yaml",
        help="YAML config for --vlm-judge-evaluator qwen-vl-judge.",
    )
    parser.add_argument(
        "--vlm-judge-timeout-seconds",
        type=float,
        default=120,
        help="HTTP timeout in seconds for --vlm-judge-evaluator qwen-vl-judge.",
    )
    parser.add_argument(
        "--text-quality-evaluator",
        choices=("none", "bertscore-local"),
        default="none",
        help="Optional text quality evaluator to run after offline evaluation.",
    )
    parser.add_argument(
        "--bertscore-model-type",
        default="roberta-large",
        help="BERTScore model type for --text-quality-evaluator bertscore-local.",
    )
    parser.add_argument(
        "--bertscore-device",
        default=None,
        help="Device used by the local BERTScore evaluator.",
    )
    parser.add_argument(
        "--gemini-embedding-evaluator",
        choices=("none", "gemini-api"),
        default="none",
        help="Optional Gemini embedding evaluator to run after offline evaluation.",
    )
    parser.add_argument(
        "--gemini-embedding-config",
        default="configs/gemini-embedding.yaml",
        help="YAML config for --gemini-embedding-evaluator gemini-api.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
