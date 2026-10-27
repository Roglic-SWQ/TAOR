"""Write offline evaluation summaries, CSVs, and case exports."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mmkgc.evaluation.cases import best_cases, fallback_cases, worst_cases
from mmkgc.utils import write_json


SUMMARY_FILENAME = "evaluation_summary.json"
SUMMARY_CSV_FILENAME = "summary.csv"
ENTITY_SCORES_CSV_FILENAME = "entity_scores.csv"
BEST_CASES_FILENAME = "best_cases.jsonl"
WORST_CASES_FILENAME = "worst_cases.jsonl"
FALLBACK_CASES_FILENAME = "fallback_cases.jsonl"
OWNED_OUTPUT_FILENAMES = (
    SUMMARY_FILENAME,
    SUMMARY_CSV_FILENAME,
    ENTITY_SCORES_CSV_FILENAME,
    BEST_CASES_FILENAME,
    WORST_CASES_FILENAME,
    FALLBACK_CASES_FILENAME,
)

SUMMARY_CSV_FIELDS = ["metric", "value"]
ENTITY_SCORE_FIELDS = [
    "entity_key",
    "label",
    "missing_mode",
    "text_best_offline_evaluation_score",
    "image_best_offline_evaluation_score",
    "pair_best_offline_evaluation_score",
    "overall_offline_evaluation_score",
    "calibrated_text_best_offline_evaluation_score",
    "calibrated_image_best_offline_evaluation_score",
    "calibrated_pair_best_offline_evaluation_score",
    "calibrated_overall_offline_evaluation_score",
    "calibration_tags",
    "text_candidate_count",
    "image_candidate_count",
    "pair_candidate_count",
    "best_text_candidate_id",
    "best_image_candidate_id",
    "best_pair_id",
    "fallback_candidate_count",
    "has_ground_truth_text",
    "case_tags",
]


class EvaluationWriter:
    """Writer for one evaluation output directory."""

    def __init__(self, output_dir: str | Path, overwrite: bool = False):
        self.output_dir = Path(output_dir)
        self.entities_dir = self.output_dir / "entities"

        if self.output_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Evaluation output directory already exists: {self.output_dir}. "
                "Use overwrite=True to replace evaluation artifacts."
            )

        if overwrite:
            self._clear_owned_artifacts()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entities_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        entity_evaluations: list[dict[str, Any]],
        run_dir: str | Path,
        case_limit: int = 20,
        failed_entity_keys: list[str] | None = None,
        failed_entities: int | None = None,
    ) -> dict[str, Any]:
        """Write all evaluation artifacts and return the run summary."""
        evaluations = [_json_ready(evaluation) for evaluation in entity_evaluations]
        failed_keys = _string_list(failed_entity_keys)
        case_exports = {
            "best": best_cases(evaluations, case_limit),
            "worst": worst_cases(evaluations, case_limit),
            "fallback": fallback_cases(evaluations, case_limit),
        }
        summary = _build_summary(
            run_dir=run_dir,
            output_dir=self.output_dir,
            entity_evaluations=evaluations,
            case_exports=case_exports,
            case_limit=case_limit,
            failed_entity_keys=failed_keys,
            failed_entities=failed_entities,
        )

        write_json(self.output_dir / SUMMARY_FILENAME, summary)
        self._write_summary_csv(summary)
        self._write_entity_scores_csv(evaluations)
        self._write_cases(BEST_CASES_FILENAME, case_exports["best"])
        self._write_cases(WORST_CASES_FILENAME, case_exports["worst"])
        self._write_cases(FALLBACK_CASES_FILENAME, case_exports["fallback"])
        self._write_entity_files(evaluations)

        return summary

    def _write_summary_csv(self, summary: dict[str, Any]) -> None:
        rows = [
            ("evaluated_entities", summary["evaluated_entities"]),
            ("failed_entities", summary["failed_entities"]),
            (
                "failed_entity_keys",
                json.dumps(summary["failed_entity_keys"], ensure_ascii=False),
            ),
            (
                "avg_text_best_offline_evaluation_score",
                summary["averages"]["text_best_offline_evaluation_score"],
            ),
            (
                "avg_image_best_offline_evaluation_score",
                summary["averages"]["image_best_offline_evaluation_score"],
            ),
            (
                "avg_pair_best_offline_evaluation_score",
                summary["averages"]["pair_best_offline_evaluation_score"],
            ),
            (
                "avg_overall_offline_evaluation_score",
                summary["averages"]["overall_offline_evaluation_score"],
            ),
            (
                "avg_calibrated_text_best_offline_evaluation_score",
                summary["averages"]["calibrated_text_best_offline_evaluation_score"],
            ),
            (
                "avg_calibrated_image_best_offline_evaluation_score",
                summary["averages"]["calibrated_image_best_offline_evaluation_score"],
            ),
            (
                "avg_calibrated_pair_best_offline_evaluation_score",
                summary["averages"]["calibrated_pair_best_offline_evaluation_score"],
            ),
            (
                "avg_calibrated_overall_offline_evaluation_score",
                summary["averages"]["calibrated_overall_offline_evaluation_score"],
            ),
            ("fallback_entities", summary["counts"]["fallback_entities"]),
            ("low_score_entities", summary["counts"]["low_score_entities"]),
            (
                "fallback_penalized_entities",
                summary["counts"]["fallback_penalized_entities"],
            ),
            (
                "generic_entity_type_entities",
                summary["counts"]["generic_entity_type_entities"],
            ),
            (
                "pair_single_side_penalized_entities",
                summary["counts"]["pair_single_side_penalized_entities"],
            ),
            (
                "template_prompt_penalized_entities",
                summary["counts"]["template_prompt_penalized_entities"],
            ),
        ]

        path = self.output_dir / SUMMARY_CSV_FILENAME
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUMMARY_CSV_FIELDS)
            writer.writerows(rows)

    def _write_entity_scores_csv(self, entity_evaluations: list[dict[str, Any]]) -> None:
        path = self.output_dir / ENTITY_SCORES_CSV_FILENAME
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ENTITY_SCORE_FIELDS)
            writer.writeheader()
            for evaluation in entity_evaluations:
                writer.writerow(_entity_score_row(evaluation))

    def _write_cases(self, filename: str, records: list[dict[str, Any]]) -> None:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _write_entity_files(self, entity_evaluations: list[dict[str, Any]]) -> None:
        for evaluation in entity_evaluations:
            entity_key = _entity_key(evaluation)
            write_json(
                self.entities_dir / _entity_dir_name(entity_key) / "evaluation.json",
                evaluation,
            )

    def _clear_owned_artifacts(self) -> None:
        if not self.output_dir.exists():
            return

        for filename in OWNED_OUTPUT_FILENAMES:
            path = self.output_dir / filename
            if path.is_symlink() or path.is_file():
                path.unlink()

        if self.entities_dir.is_symlink() or self.entities_dir.is_file():
            self.entities_dir.unlink()
        elif self.entities_dir.exists():
            shutil.rmtree(self.entities_dir)


def write_evaluation_outputs(
    output_dir: str | Path,
    entity_evaluations: list[dict[str, Any]],
    run_dir: str | Path,
    overwrite: bool = False,
    case_limit: int = 20,
    failed_entity_keys: list[str] | None = None,
    failed_entities: int | None = None,
) -> dict[str, Any]:
    """Write offline evaluation artifacts to an output directory."""
    _validate_output_dir(output_dir, run_dir)
    writer = EvaluationWriter(output_dir, overwrite=overwrite)
    return writer.write(
        entity_evaluations,
        run_dir,
        case_limit=case_limit,
        failed_entity_keys=failed_entity_keys,
        failed_entities=failed_entities,
    )


def _validate_output_dir(output_dir: str | Path, run_dir: str | Path) -> None:
    resolved_output_dir = Path(output_dir).resolve()
    resolved_run_dir = Path(run_dir).resolve()
    if resolved_output_dir == resolved_run_dir:
        raise ValueError(
            "Evaluation output_dir must not be the run_dir. "
            "Use the default <run-dir>/evaluation directory or choose a separate "
            "output directory."
        )


def _build_summary(
    *,
    run_dir: str | Path,
    output_dir: Path,
    entity_evaluations: list[dict[str, Any]],
    case_exports: dict[str, list[dict[str, Any]]],
    case_limit: int,
    failed_entity_keys: list[str],
    failed_entities: int | None,
) -> dict[str, Any]:
    fallback_entities = [
        evaluation
        for evaluation in entity_evaluations
        if "fallback" in _string_list(evaluation.get("case_tags"))
    ]
    low_score_entities = [
        evaluation
        for evaluation in entity_evaluations
        if "low_score" in _string_list(evaluation.get("case_tags"))
    ]
    failed_entity_count = (
        len(failed_entity_keys)
        if failed_entity_keys or failed_entities is None
        else max(int(failed_entities), 0)
    )

    return {
        "run_dir": str(Path(run_dir)),
        "output_dir": str(output_dir),
        "evaluated_entities": len(entity_evaluations),
        "failed_entities": failed_entity_count,
        "failed_entity_keys": failed_entity_keys,
        "averages": {
            "text_best_offline_evaluation_score": _average_best(
                entity_evaluations,
                "text_evaluation",
            ),
            "image_best_offline_evaluation_score": _average_best(
                entity_evaluations,
                "image_evaluation",
            ),
            "pair_best_offline_evaluation_score": _average_best(
                entity_evaluations,
                "pair_evaluation",
            ),
            "overall_offline_evaluation_score": _average(
                _score(evaluation.get("overall_offline_evaluation_score"))
                for evaluation in entity_evaluations
            ),
            "calibrated_text_best_offline_evaluation_score": _average(
                _score_with_fallback(
                    evaluation.get("calibrated_text_best_offline_evaluation_score"),
                    _dict_value(evaluation.get("text_evaluation")).get(
                        "best_offline_evaluation_score"
                    ),
                )
                for evaluation in entity_evaluations
            ),
            "calibrated_image_best_offline_evaluation_score": _average(
                _score_with_fallback(
                    evaluation.get("calibrated_image_best_offline_evaluation_score"),
                    _dict_value(evaluation.get("image_evaluation")).get(
                        "best_offline_evaluation_score"
                    ),
                )
                for evaluation in entity_evaluations
            ),
            "calibrated_pair_best_offline_evaluation_score": _average(
                _score_with_fallback(
                    evaluation.get("calibrated_pair_best_offline_evaluation_score"),
                    _dict_value(evaluation.get("pair_evaluation")).get(
                        "best_offline_evaluation_score"
                    ),
                )
                for evaluation in entity_evaluations
            ),
            "calibrated_overall_offline_evaluation_score": _average(
                _score_with_fallback(
                    evaluation.get("calibrated_overall_offline_evaluation_score"),
                    evaluation.get("overall_offline_evaluation_score"),
                )
                for evaluation in entity_evaluations
            ),
        },
        "counts": {
            "fallback_entities": len(fallback_entities),
            "low_score_entities": len(low_score_entities),
            "fallback_penalized_entities": _calibration_tag_count(
                entity_evaluations,
                "fallback_penalized",
            ),
            "generic_entity_type_entities": _calibration_tag_count(
                entity_evaluations,
                "generic_entity_type",
            ),
            "pair_single_side_penalized_entities": _calibration_tag_count(
                entity_evaluations,
                "pair_single_side_penalized",
            ),
            "template_prompt_penalized_entities": _calibration_tag_count(
                entity_evaluations,
                "template_prompt_penalized",
            ),
        },
        "fallback_counts": {
            "entities": len(fallback_entities),
            "candidates": sum(
                int(
                    _dict_value(evaluation.get("fallback_summary")).get(
                        "fallback_candidate_count",
                        0,
                    )
                    or 0
                )
                for evaluation in entity_evaluations
            ),
        },
        "case_exports": {
            "case_limit": case_limit,
            "best_cases": len(case_exports["best"]),
            "worst_cases": len(case_exports["worst"]),
            "fallback_cases": len(case_exports["fallback"]),
        },
        "output_files": {
            "evaluation_summary_json": SUMMARY_FILENAME,
            "summary_csv": SUMMARY_CSV_FILENAME,
            "entity_scores_csv": ENTITY_SCORES_CSV_FILENAME,
            "best_cases_jsonl": BEST_CASES_FILENAME,
            "worst_cases_jsonl": WORST_CASES_FILENAME,
            "fallback_cases_jsonl": FALLBACK_CASES_FILENAME,
            "entity_evaluations_dir": "entities/",
        },
    }


def _entity_score_row(evaluation: dict[str, Any]) -> dict[str, Any]:
    entity = _dict_value(evaluation.get("entity"))
    text_evaluation = _dict_value(evaluation.get("text_evaluation"))
    image_evaluation = _dict_value(evaluation.get("image_evaluation"))
    pair_evaluation = _dict_value(evaluation.get("pair_evaluation"))
    fallback_summary = _dict_value(evaluation.get("fallback_summary"))
    calibration_summary = _dict_value(evaluation.get("calibration_summary"))

    return {
        "entity_key": _entity_key(evaluation),
        "label": entity.get("label", ""),
        "missing_mode": entity.get("missing_mode", ""),
        "text_best_offline_evaluation_score": _score(
            text_evaluation.get("best_offline_evaluation_score")
        ),
        "image_best_offline_evaluation_score": _score(
            image_evaluation.get("best_offline_evaluation_score")
        ),
        "pair_best_offline_evaluation_score": _score(
            pair_evaluation.get("best_offline_evaluation_score")
        ),
        "overall_offline_evaluation_score": _score(
            evaluation.get("overall_offline_evaluation_score")
        ),
        "calibrated_text_best_offline_evaluation_score": _score_with_fallback(
            evaluation.get("calibrated_text_best_offline_evaluation_score"),
            text_evaluation.get("best_offline_evaluation_score"),
        ),
        "calibrated_image_best_offline_evaluation_score": _score_with_fallback(
            evaluation.get("calibrated_image_best_offline_evaluation_score"),
            image_evaluation.get("best_offline_evaluation_score"),
        ),
        "calibrated_pair_best_offline_evaluation_score": _score_with_fallback(
            evaluation.get("calibrated_pair_best_offline_evaluation_score"),
            pair_evaluation.get("best_offline_evaluation_score"),
        ),
        "calibrated_overall_offline_evaluation_score": _score_with_fallback(
            evaluation.get("calibrated_overall_offline_evaluation_score"),
            evaluation.get("overall_offline_evaluation_score"),
        ),
        "calibration_tags": "|".join(
            _string_list(calibration_summary.get("calibration_tags"))
        ),
        "text_candidate_count": int(text_evaluation.get("candidate_count", 0) or 0),
        "image_candidate_count": int(image_evaluation.get("candidate_count", 0) or 0),
        "pair_candidate_count": int(pair_evaluation.get("candidate_count", 0) or 0),
        "best_text_candidate_id": text_evaluation.get("best_candidate_id"),
        "best_image_candidate_id": image_evaluation.get("best_candidate_id"),
        "best_pair_id": pair_evaluation.get("best_pair_id"),
        "fallback_candidate_count": int(
            fallback_summary.get("fallback_candidate_count", 0) or 0
        ),
        "has_ground_truth_text": _has_ground_truth_text(text_evaluation),
        "case_tags": "|".join(_string_list(evaluation.get("case_tags"))),
    }


def _has_ground_truth_text(text_evaluation: dict[str, Any]) -> bool:
    candidates = text_evaluation.get("ranked_candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    first_candidate = candidates[0]
    return (
        isinstance(first_candidate, dict)
        and first_candidate.get("ground_truth_available") is True
    )


def _average_best(entity_evaluations: list[dict[str, Any]], key: str) -> float:
    return _average(
        _score(_dict_value(evaluation.get(key)).get("best_offline_evaluation_score"))
        for evaluation in entity_evaluations
    )


def _calibration_tag_count(entity_evaluations: list[dict[str, Any]], tag: str) -> int:
    return sum(
        1
        for evaluation in entity_evaluations
        if tag in _string_list(
            _dict_value(evaluation.get("calibration_summary")).get("calibration_tags")
        )
    )


def _average(values: Any) -> float:
    numeric_values = list(values)
    if not numeric_values:
        return 0.0
    return round(sum(numeric_values) / len(numeric_values), 6)


def _entity_key(evaluation: dict[str, Any]) -> str:
    entity = _dict_value(evaluation.get("entity"))
    return str(entity.get("entity_key") or entity.get("label") or "entity")


def _entity_dir_name(entity_key: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in entity_key
    )
    slug = safe.strip("._") or "entity"
    digest = hashlib.sha256(entity_key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}"


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _score(value: Any) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _score_with_fallback(value: Any, fallback: Any) -> float:
    return _score(fallback if value is None else value)


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
