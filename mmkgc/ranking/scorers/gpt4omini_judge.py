"""GPT-4o mini judge prompt builders and strict response parser for Stage 4 ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
import math
import re
import time
from typing import Any
from urllib.error import HTTPError

from mmkgc.ranking.context import RankingContext
from mmkgc.ranking.contracts import GateResult, ScoreResult
from mmkgc.ranking.image_input import ImageInput
from mmkgc.utils import clamp01
from mmkgc.observation import bind_observation, observation_response_schema, validate_verification_targets


ALLOWED_HARD_REJECTION_CODES = frozenset(
    {
        "forbidden_content",
        "identity_mismatch",
        "unsupported_claim",
        "missing_required_content",
        "wrong_modality",
        "unsafe_content",
        "malformed_candidate",
        "pair_mismatch",
    }
)
REQUIRED_RESPONSE_FIELDS = frozenset(
    {
        "score",
        "confidence",
        "hard_reject",
        "hard_rejection_codes",
        "soft_violations",
        "evidence",
        "reason",
    }
)
MAX_REPAIR_RESPONSE_CHARS = 4000
OBSERVATION_CONTRACT_VERSION = "candidate_observation_contract_v2"


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def observation_judgment_schema(targets: Any, modality: str) -> dict[str, Any]:
    """Constrain output structure and legal status/mode pairs, retaining local validation."""
    targets = validate_verification_targets(targets)
    text = {"type": "string"}
    number = {"type": "number", "minimum": 0, "maximum": 1}
    strings = {"type": "array", "items": text}
    branches = []
    for target in targets:
        pairs = [("unverifiable", ["unverifiable"])]
        if target["verification_mode"] != "unverifiable":
            pairs += [("direct", ["supported", "contradicted", "missing", "uncertain"]),
                      ("appearance_only", ["unverifiable", "uncertain"])]
        for mode, statuses in pairs:
            branches.append(_object_schema({
                "check_id": {"type": "string", "enum": [target["check_id"]]},
                "status": {"type": "string", "enum": statuses},
                "verification_mode": {"type": "string", "enum": [mode]},
                "observed": text,
                "evidence": {"type": "array", "items": {
                    "type": "string", "pattern": "^" + ("pixel:" if modality == "image" else "text:") + r"\s*\S"}},
                "confidence": number, "reason": text,
            }))
    return _object_schema({
        "score": {"type": "number", "minimum": 0, "maximum": 100},
        "confidence": number, "hard_reject": {"type": "boolean"},
        "hard_rejection_codes": {"type": "array", "items": {
            "type": "string", "enum": sorted(ALLOWED_HARD_REJECTION_CODES)}},
        "soft_violations": strings, "evidence": strings, "reason": text,
        "observation": _object_schema({
            "fact_checks": {"type": "array", "minItems": len(targets), "maxItems": len(targets),
                            "items": {"anyOf": branches}},
            "unsupported_claims": {"type": "array", "items": _object_schema({
                "claim": text, "evidence": {"type": "array", "minItems": 1,
                    "items": {"type": "string", "pattern": "^" + ("pixel:" if modality == "image" else "text:") + r"\s*\S"}},
                "confidence": number, "reason": text})},
        }),
    })


def observation_output_template(targets: Any) -> dict[str, Any]:
    return {"score": 0, "confidence": 0, "hard_reject": False, "hard_rejection_codes": [],
            "soft_violations": [], "evidence": [], "reason": "",
            "observation": {"fact_checks": [
                {"check_id": t["check_id"], "status": "", "verification_mode": "", "observed": "",
                 "evidence": [], "confidence": 0, "reason": ""}
                for t in validate_verification_targets(targets)], "unsupported_claims": []}}


def observation_contract_instructions(targets: Any, modality: str) -> str:
    targets = validate_verification_targets(targets)
    rules = [{"check_id": t["check_id"], "required_mode": t["verification_mode"],
              "legal_status_mode_pairs": (["unverifiable/unverifiable"] if t["verification_mode"] == "unverifiable" else
                  ["supported/direct", "contradicted/direct", "missing/direct", "uncertain/direct",
                   "uncertain/appearance_only", "unverifiable/appearance_only", "unverifiable/unverifiable"])}
             for t in targets]
    return (
        "\nMANDATORY OBSERVATION OUTPUT CONTRACT " + OBSERVATION_CONTRACT_VERSION + ":\n"
        "Return ALL eight top-level fields in the template, including observation. "
        "Complete every listed check_id exactly once. Empty strings are placeholders to fill. "
        "Do not copy 'assess' from an input target into verification_mode. "
        "For invisible facts choose status unverifiable and verification_mode unverifiable. "
        "For appearance_only choose only uncertain or unverifiable. Never use supported, missing or "
        "contradicted for an invisible fact or mere appearance. Evidence entries must each start with "
        + ("pixel:" if modality == "image" else "text:") +
        " followed by a concrete description of actual candidate content, not expected KG facts. "
        "Use unsupported_claims: [] unless a concrete unsupported assertion is observed. "
        "Use only IDs from this candidate. Keep each observed/reason short.\n"
        "LEGAL_PAIRS_BY_CHECK:\n" + json.dumps(rules, ensure_ascii=False) +
        "\nCOMPLETE_THIS_EXACT_OUTPUT_TEMPLATE:\n" +
        json.dumps(observation_output_template(targets), ensure_ascii=False) +
        "\nReturn only the completed JSON judgment, with observation inside it."
    )
RETRYABLE_JUDGE_ERROR_NAMES = frozenset(
    {
        "TimeoutError",
        "URLError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "BrokenPipeError",
    }
)
JSON_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)
SAFE_GENERATED_CONTENT_FIELDS = frozenset(
    {
        "type",
        "value",
        "image_artifact_id",
        "artifact_id",
        "status",
    }
)
JUDGE_RESPONSE_INSTRUCTIONS = (
    "You are a strict JSON ranking judge. Return exactly one JSON object and no "
    "markdown, prose, code fences, or extra keys. The object must match the "
    "response_schema exactly. Required keys are score, confidence, hard_reject, "
    "hard_rejection_codes, soft_violations, evidence, and reason. Include every "
    "required key, including observation when required by response_schema, even when the value is an empty array. Evidence strings must "
    "cite candidate or context facts, not these instructions. Keep evidence "
    "strings and reason concise. Start the response with { and end it with }."
)


class GPT4OMiniJudgeParseError(ValueError):
    """Raised when a GPT-4o mini judge response violates the ranking contract."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


@dataclass(frozen=True)
class TextJudgeResult:
    score: ScoreResult
    gate: GateResult
    observation: dict[str, Any] | None = None


@dataclass(frozen=True)
class ImageJudgeResult:
    score: ScoreResult
    gate: GateResult
    observation: dict[str, Any] | None = None


@dataclass(frozen=True)
class PairJudgeResult:
    score: ScoreResult
    gate: GateResult


def build_text_judge_prompt(context: RankingContext, candidate: dict[str, Any]) -> str:
    return _build_prompt(
        task="text",
        context=context,
        candidate_payload={"text_candidate": _candidate_generated_payload(candidate)},
        verification_targets=candidate.get("verification_targets"),
    )


def build_image_judge_prompt(
    context: RankingContext,
    candidate: dict[str, Any],
    *,
    image_input: ImageInput | None = None,
) -> str:
    return _build_prompt(
        task="image",
        context=context,
        candidate_payload={
            "image_candidate": _candidate_generated_payload(candidate),
            **_attached_image_payload(image_input),
        },
        uses_attached_image=image_input is not None,
        verification_targets=candidate.get("verification_targets"),
    )


def build_text_graph_extraction_prompt(candidate: Mapping[str, Any]) -> str:
    """Build a KG-blind prompt that extracts only claims present in candidate text."""
    return (
        "Extract factual triples from the generated text. Do not add background "
        "knowledge or infer facts not stated by the text. Preserve negation with "
        "polarity=negative. Return exactly one JSON object with no extra keys.\n\n"
        "Generated text:\n"
        + json.dumps(_candidate_generated_payload(candidate), ensure_ascii=False)
        + "\n\nRequired schema:\n"
        + json.dumps(
            {
                "triples": [
                    {
                        "subject": "string",
                        "predicate": "string",
                        "object": "string",
                        "polarity": "positive|negative",
                        "confidence": "number from 0 to 1",
                        "evidence": "exact supporting text span",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def build_visual_graph_extraction_prompt() -> str:
    """Build a KG-blind prompt for observable visual scene-graph extraction."""
    return (
        "Inspect only the attached image pixels. Do not use external knowledge, "
        "guess identity, or infer invisible biographical or administrative facts. "
        "Return visible objects/scenes as nodes, visible attributes, observable "
        "relations as edges, and readable text as OCR. For a non-blank image, "
        "return at least one broad visible scene or object node, such as "
        "landscape, vegetation, building, road, water, person, vehicle, or sky. "
        "Return empty arrays only when the image is blank, unreadable, or no visual "
        "content can be observed. Return exactly one JSON object with no extra "
        "keys.\n\nRequired schema:\n"
        + json.dumps(
            {
                "nodes": [
                    {
                        "id": "unique string",
                        "label": "visible object or scene",
                        "attributes": ["visible attribute"],
                        "bbox": ["x1", "y1", "x2", "y2"],
                        "confidence": "number from 0 to 1",
                        "evidence": "short pixel observation",
                    }
                ],
                "edges": [
                    {
                        "subject": "node id or visible subject",
                        "predicate": "observable relation",
                        "object": "node id or visible object",
                        "confidence": "number from 0 to 1",
                        "evidence": "short pixel observation",
                    }
                ],
                "ocr": [
                    {
                        "text": "readable visible text",
                        "confidence": "number from 0 to 1",
                        "evidence": "location or visual observation",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )


def _build_visual_graph_reobserve_prompt() -> str:
    return (
        "Re-inspect the attached image pixels and return a concise visual scene graph. "
        "The previous response was structurally valid but empty. Do not use external "
        "knowledge or guess identity, country, biography, or administrative facts. "
        "For any non-blank image, include at least one broad visible node such as "
        "landscape, vegetation, building, road, water, person, vehicle, or sky. "
        "Use empty edges and OCR when none are visibly supported. Return only this "
        "JSON shape: {\"nodes\": [], \"edges\": [], \"ocr\": []}."
    )


def _visual_graph_is_empty(graph: Mapping[str, Any]) -> bool:
    return not any(
        isinstance(graph.get(field), (list, tuple)) and graph.get(field)
        for field in ("nodes", "edges", "ocr")
    )



def _build_visual_graph_contract_repair_prompt(
    response_text: str,
    error: GPT4OMiniJudgeParseError,
) -> str:
    previous_response = _truncate_repair_response(_repair_source_text(response_text))
    return (
        "Repair the previous visual graph JSON so it satisfies the exact Stage 4 "
        "visual graph contract. Return exactly one JSON object with only the keys "
        "nodes, edges, and ocr. Do not re-observe or re-evaluate the image. Preserve "
        "the previous observations whenever possible. Every node attributes field "
        "must be an array of strings; use [] when no visible attributes are present. "
        "Every bbox must be [] or an array of four values. Every confidence must be "
        "a number from 0 to 1. Edges and OCR entries must be objects matching the "
        "schema. Return only the repaired JSON, with no markdown or explanation.\n\n"
        "Required schema:\n"
        + json.dumps(
            {
                "nodes": [
                    {
                        "id": "unique string",
                        "label": "visible object or scene",
                        "attributes": ["visible attribute"],
                        "bbox": ["x1", "y1", "x2", "y2"],
                        "confidence": "number from 0 to 1",
                        "evidence": "short pixel observation",
                    }
                ],
                "edges": [
                    {
                        "subject": "node id or visible subject",
                        "predicate": "observable relation",
                        "object": "node id or visible object",
                        "confidence": "number from 0 to 1",
                        "evidence": "short pixel observation",
                    }
                ],
                "ocr": [
                    {
                        "text": "readable visible text",
                        "confidence": "number from 0 to 1",
                        "evidence": "location or visual observation",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\nParser error: "
        + str(error)
        + "\nPrevious response:\n"
        + previous_response
        + "\n\nReturn only the repaired visual graph JSON:"
    )


def parse_text_graph_response(response_text: str) -> dict[str, Any]:
    payload = _load_graph_payload(response_text)
    if set(payload) != {"triples"}:
        raise GPT4OMiniJudgeParseError("text graph response must contain only triples")
    triples = _mapping_list("triples", payload["triples"])
    normalized = []
    for item in triples:
        subject = _required_text("subject", item.get("subject"))
        predicate = _required_text("predicate", item.get("predicate"))
        object_value = _required_text("object", item.get("object"))
        polarity = _required_text("polarity", item.get("polarity"))
        if polarity not in {"positive", "negative"}:
            raise GPT4OMiniJudgeParseError("polarity must be positive or negative")
        normalized.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_value,
                "polarity": polarity,
                "confidence": _unit_number("confidence", item.get("confidence")),
                "evidence": _required_text("evidence", item.get("evidence")),
            }
        )
    return {"triples": normalized}


def parse_visual_graph_response(response_text: str) -> dict[str, Any]:
    payload = _load_graph_payload(response_text)
    if set(payload) != {"nodes", "edges", "ocr"}:
        raise GPT4OMiniJudgeParseError(
            "visual graph response must contain only nodes, edges, and ocr"
        )
    nodes = []
    for item in _mapping_list("nodes", payload["nodes"]):
        bbox = item.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) not in {0, 4}:
            raise GPT4OMiniJudgeParseError("bbox must be empty or contain four values")
        attributes = item.get("attributes", [])
        if not isinstance(attributes, list) or any(
            not isinstance(value, str) for value in attributes
        ):
            raise GPT4OMiniJudgeParseError("attributes must be an array of strings")
        nodes.append(
            {
                "id": _required_text("id", item.get("id")),
                "label": _required_text("label", item.get("label")),
                "attributes": [value.strip() for value in attributes if value.strip()],
                "bbox": list(bbox),
                "confidence": _unit_number("confidence", item.get("confidence")),
                "evidence": _required_text("evidence", item.get("evidence")),
            }
        )
    edges = []
    for item in _mapping_list("edges", payload["edges"]):
        edges.append(
            {
                "subject": _required_text("subject", item.get("subject")),
                "predicate": _required_text("predicate", item.get("predicate")),
                "object": _required_text("object", item.get("object")),
                "confidence": _unit_number("confidence", item.get("confidence")),
                "evidence": _required_text("evidence", item.get("evidence")),
            }
        )
    ocr = []
    for item in _mapping_list("ocr", payload["ocr"]):
        ocr.append(
            {
                "text": _required_text("text", item.get("text")),
                "confidence": _unit_number("confidence", item.get("confidence")),
                "evidence": _required_text("evidence", item.get("evidence")),
            }
        )
    return {"nodes": nodes, "edges": edges, "ocr": ocr}


def build_pair_judge_prompt(
    context: RankingContext,
    text_candidate: dict[str, Any],
    image_candidate: dict[str, Any],
    *,
    image_input: ImageInput | None = None,
) -> str:
    return _build_prompt(
        task="pair",
        context=context,
        candidate_payload={
            "text_candidate": _candidate_generated_payload(text_candidate),
            "image_candidate": _candidate_generated_payload(image_candidate),
            **_attached_image_payload(image_input),
        },
        uses_attached_image=image_input is not None,
    )


def parse_text_judge_response(response_text: str, *, model_name: str = "gpt4omini-ranking-judge") -> TextJudgeResult:
    score, gate = _parse_judge_response(response_text, model_name=model_name)
    return TextJudgeResult(score=score, gate=gate, observation=_load_judge_payload(response_text).get("observation"))


def parse_image_judge_response(response_text: str, *, model_name: str = "gpt4omini-ranking-judge") -> ImageJudgeResult:
    score, gate = _parse_judge_response(response_text, model_name=model_name)
    return ImageJudgeResult(score=score, gate=gate, observation=_load_judge_payload(response_text).get("observation"))


def parse_pair_judge_response(response_text: str, *, model_name: str = "gpt4omini-ranking-judge") -> PairJudgeResult:
    score, gate = _parse_judge_response(response_text, model_name=model_name)
    return PairJudgeResult(score=score, gate=gate)


class GPT4OMiniRankingJudge:
    """Thin one-call-per-candidate adapter around a GPT-4o-mini-compatible text client."""

    def __init__(
        self,
        client: Any,
        *,
        model_name: str = "gpt4omini-ranking-judge",
        max_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be a non-negative integer")
        if max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if (
            isinstance(retry_backoff_seconds, bool)
            or not isinstance(retry_backoff_seconds, (int, float))
            or not math.isfinite(float(retry_backoff_seconds))
            or float(retry_backoff_seconds) < 0.0
        ):
            raise ValueError("retry_backoff_seconds must be a non-negative finite number")
        self.client = client
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_backoff_seconds = float(retry_backoff_seconds)

    def judge_text(
        self,
        context: RankingContext,
        candidate: dict[str, Any],
    ) -> TextJudgeResult:
        prompt = build_text_judge_prompt(context, candidate)
        return self._judge_with_contract_repair(
            prompt,
            parse_text_judge_response,
            kind="text",
            candidate_id=str(candidate.get("candidate_id", "")),
            verification_targets=candidate.get("verification_targets"),
        )

    def extract_text_graph(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        response = self._complete_with_retries(
            build_text_graph_extraction_prompt(candidate),
            image_input=None,
            kind="text_graph_extraction",
            candidate_id=str(candidate.get("candidate_id", "")),
        )
        return parse_text_graph_response(response)

    def extract_visual_graph(
        self,
        candidate: Mapping[str, Any],
        *,
        image_input: ImageInput,
    ) -> dict[str, Any]:
        response = self._complete_with_retries(
            build_visual_graph_extraction_prompt(),
            image_input=image_input,
            kind="visual_graph_extraction",
            candidate_id=str(candidate.get("candidate_id", "")),
        )
        try:
            graph = parse_visual_graph_response(response)
        except GPT4OMiniJudgeParseError as exc:
            repaired_response = self._complete_with_retries(
                _build_visual_graph_contract_repair_prompt(response, exc),
                image_input=None,
                kind="visual_graph_extraction_contract_repair",
                candidate_id=str(candidate.get("candidate_id", "")),
            )
            try:
                graph = parse_visual_graph_response(repaired_response)
            except GPT4OMiniJudgeParseError as repair_exc:
                details = {
                    "contract_repair_retries": 1,
                    "contract_repair_error_type": type(exc).__name__,
                    "contract_repair_error": str(exc),
                    "contract_repair_failed": True,
                    "contract_repair_failure_error_type": type(repair_exc).__name__,
                    "contract_repair_failure_error": str(repair_exc),
                }
                raise GPT4OMiniJudgeParseError(
                    "visual graph contract repair failed",
                    details=details,
                ) from repair_exc
            return graph

        if not _visual_graph_is_empty(graph):
            return graph

        reobserved_response = self._complete_with_retries(
            _build_visual_graph_reobserve_prompt(),
            image_input=image_input,
            kind="visual_graph_extraction_empty_retry",
            candidate_id=str(candidate.get("candidate_id", "")),
        )
        try:
            reobserved_graph = parse_visual_graph_response(reobserved_response)
        except GPT4OMiniJudgeParseError as exc:
            raise GPT4OMiniJudgeParseError(
                "visual graph empty retry returned invalid JSON",
                details={
                    "empty_graph_retry": True,
                    "empty_graph_retry_error_type": type(exc).__name__,
                    "empty_graph_retry_error": str(exc),
                },
            ) from exc
        if _visual_graph_is_empty(reobserved_graph):
            raise GPT4OMiniJudgeParseError(
                "visual graph remained empty after re-observation",
                details={"empty_graph_retry": True, "empty_graph_retry_failed": True},
            )
        return reobserved_graph

    def judge_image(
        self,
        context: RankingContext,
        candidate: dict[str, Any],
        *,
        image_input: ImageInput | None = None,
    ) -> ImageJudgeResult:
        prompt = build_image_judge_prompt(context, candidate, image_input=image_input)
        return self._judge_with_contract_repair(
            prompt,
            parse_image_judge_response,
            image_input=image_input,
            kind="image",
            candidate_id=str(candidate.get("candidate_id", "")),
            verification_targets=candidate.get("verification_targets"),
        )

    def judge_pair(
        self,
        context: RankingContext,
        text_candidate: dict[str, Any],
        image_candidate: dict[str, Any],
        *,
        image_input: ImageInput | None = None,
    ) -> PairJudgeResult:
        prompt = build_pair_judge_prompt(
            context,
            text_candidate,
            image_candidate,
            image_input=image_input,
        )
        return self._judge_with_contract_repair(
            prompt,
            parse_pair_judge_response,
            image_input=image_input,
            kind="pair",
            candidate_id=(
                f"{text_candidate.get('candidate_id', '')}__"
                f"{image_candidate.get('candidate_id', '')}"
            ),
        )

    def _judge_with_contract_repair(
        self,
        prompt: str,
        parser: Any,
        *,
        image_input: ImageInput | None = None,
        kind: str,
        candidate_id: str,
        verification_targets: Any = None,
    ) -> Any:
        if verification_targets:
            return self._judge_observation_with_repairs(
                prompt, parser, image_input=image_input, kind=kind,
                candidate_id=candidate_id, targets=verification_targets,
            )

        response = self._complete_with_retries(
            prompt,
            image_input=image_input,
            kind=kind,
            candidate_id=candidate_id,
        )
        try:
            return parser(response, model_name=self.model_name)
        except GPT4OMiniJudgeParseError as exc:
            salvaged = _try_salvage_partial_response(
                response,
                parser,
                model_name=self.model_name,
                error=exc,
            )
            if salvaged is not None:
                return salvaged
            repaired_response = self._complete_with_retries(
                (prompt + "\n\nThe previous response violated the contract: " + str(exc)
                 + "\nPrevious response:\n" + _truncate_repair_response(response)
                 + "\nReturn a corrected judgment, checking every supplied check_id exactly once."
                 if verification_targets else _build_contract_repair_prompt(response, exc)),
                image_input=image_input if verification_targets else None,
                kind=f"{kind}_contract_repair",
                candidate_id=candidate_id,
            )
            repair_details = _contract_repair_details(response, exc)
            try:
                result = parser(repaired_response, model_name=self.model_name)
            except GPT4OMiniJudgeParseError as repair_exc:
                salvaged = _try_salvage_partial_response(
                    repaired_response,
                    parser,
                    model_name=self.model_name,
                    error=repair_exc,
                    previous_details=repair_details,
                )
                if salvaged is not None:
                    return salvaged
                repair_details.update(
                    {
                        "contract_repair_failed": True,
                        "contract_repair_failure_error_type": type(repair_exc).__name__,
                        "contract_repair_failure_error": str(repair_exc),
                    }
                )
                raise GPT4OMiniJudgeParseError(
                    "contract repair failed",
                    details=repair_details,
                ) from repair_exc
            details = dict(result.score.details)
            details.update(repair_details)
            score = ScoreResult(
                raw_score=result.score.raw_score,
                normalized_score=result.score.normalized_score,
                status=result.score.status,
                model_name=result.score.model_name,
                model_version=result.score.model_version,
                latency_ms=result.score.latency_ms,
                details=details,
            )
            return replace(result, score=score)

    def _judge_observation_with_repairs(self, prompt, parser, *, image_input, kind, candidate_id, targets):
        schema = observation_judgment_schema(targets, kind)
        attempts = []
        details = {"observation_contract_version": OBSERVATION_CONTRACT_VERSION}
        request_prompt = prompt
        seen = set()
        for attempt in range(3):
            response = self._complete_with_retries(
                request_prompt, image_input=image_input, kind=kind if not attempt else f"{kind}_contract_repair",
                candidate_id=candidate_id, response_schema=schema,
            )
            try:
                result = parser(response, model_name=self.model_name)
                observation = bind_observation(result.observation, targets, kind, candidate_id)
            except (GPT4OMiniJudgeParseError, ValueError, TypeError) as error:
                attempts.append({"attempt": attempt, "error_type": type(error).__name__,
                                 "error": str(error), "response_text": response})
                if not attempt:
                    details.update(_contract_repair_details(response, error))
                details.update(contract_repair_retries=attempt, contract_repair_failed=True,
                               contract_repair_failure_error_type=type(error).__name__,
                               contract_repair_failure_error=str(error))
                fingerprint = response.strip()
                if fingerprint in seen or attempt == 2:
                    details["contract_repair_stopped_reason"] = (
                        "repeated_invalid_response" if fingerprint in seen else "repair_limit_reached")
                    details["validation_attempts"] = attempts
                    raise GPT4OMiniJudgeParseError("observation contract repair failed", details=details) from error
                seen.add(fingerprint)
                request_prompt = (
                    prompt + "\n\nOBSERVATION_VALIDATION_ERROR: " + str(error) +
                    "\nThe previous judgment did not satisfy the output contract. Return a COMPLETE "
                    "replacement. Retain valid judgments; correct every illegal status/mode pair "
                    "and evidence field, and fill every required observation check. "
                    "If the FULL claim is invisible from the image, use unverifiable/unverifiable. "
                    "Do not use missing/direct or supported/appearance_only for such a claim.\n"
                    "PREVIOUS_JUDGMENT:\n" + _truncate_repair_response(response) +
                    observation_contract_instructions(targets, kind)
                )
            else:
                attempts.append({"attempt": attempt, "response_text": response, "errors": []})
                details.update(contract_repair_retries=attempt, validation_attempts=attempts)
                for key in ("contract_repair_failed", "contract_repair_failure_error_type", "contract_repair_failure_error"):
                    details.pop(key, None)
                return replace(result, observation=observation,
                               score=replace(result.score, details={**result.score.details, **details}))
        raise RuntimeError("Observation repair loop exhausted")

    def _complete_with_retries(
        self,
        prompt: str,
        *,
        image_input: ImageInput | None,
        kind: str,
        candidate_id: str,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                return self._complete(prompt, image_input=image_input, response_schema=response_schema)
            except Exception as exc:
                retryable = _is_retryable_judge_error(exc)
                will_retry = retryable and attempt < attempts
                _attach_judge_diagnostic(
                    exc,
                    kind=kind,
                    candidate_id=candidate_id,
                    prompt=prompt,
                    image_input=image_input,
                    started=started,
                    attempt=attempt,
                    max_attempts=attempts,
                    retryable=retryable,
                    will_retry=will_retry,
                )
                if not will_retry:
                    raise
                delay = self.retry_backoff_seconds
                if isinstance(exc, HTTPError) and exc.headers is not None:
                    try:
                        retry_after = float(exc.headers.get("Retry-After", "0"))
                    except (TypeError, ValueError):
                        retry_after = 0.0
                    if math.isfinite(retry_after):
                        delay = max(delay, min(max(retry_after, 0.0), 60.0))
                if delay:
                    time.sleep(delay)
        raise RuntimeError("unreachable judge retry state")

    def _complete(self, prompt: str, *, image_input: ImageInput | None = None,
                  response_schema: dict[str, Any] | None = None) -> str:
        if image_input is not None:
            if response_schema is not None and hasattr(self.client, "complete_multimodal_structured"):
                return str(self.client.complete_multimodal_structured(prompt, image_input.as_data_url(), response_schema))
            if hasattr(self.client, "complete_multimodal"):
                return str(self.client.complete_multimodal(prompt, image_input.as_data_url()))
            raise TypeError(
                "GPT4OMiniRankingJudge multimodal image/pair judge requires "
                "client.complete_multimodal(prompt, image_data_url)"
            )
        if response_schema is not None and hasattr(self.client, "complete_structured"):
            return str(self.client.complete_structured(prompt, response_schema))
        if hasattr(self.client, "complete"):
            return str(self.client.complete(prompt))
        if callable(self.client):
            return str(self.client(prompt))
        raise TypeError("GPT4OMiniRankingJudge client must be callable or expose complete(prompt)")


def _build_contract_repair_prompt(response_text: str, error: GPT4OMiniJudgeParseError) -> str:
    previous_response = _truncate_repair_response(_repair_source_text(response_text))
    return (
        JUDGE_RESPONSE_INSTRUCTIONS
        + "\n\nRepair the previous judgment JSON so it satisfies the Stage 4 "
        "response contract. Do not re-evaluate the candidate. Preserve existing "
        "score, confidence, evidence, and reason when possible. Use only these "
        "hard_rejection_codes values: "
        + json.dumps(sorted(ALLOWED_HARD_REJECTION_CODES), ensure_ascii=False)
        + ". If an invalid code cannot be mapped exactly to one allowed value, "
        "move it to soft_violations and set hard_reject to false unless a listed "
        "hard rejection code clearly applies. Use at most one short evidence "
        "string and keep reason under 20 words. Do not preserve long malformed "
        "text; synthesize concise compliant fields when needed. Return exactly "
        "this object shape with no extra keys: "
        + _judge_response_template()
        + "\nParser error: "
        + str(error)
        + "\nPrevious response or extracted judgment fragment:\n"
        + previous_response
        + "\n\nReturn only the repaired judgment JSON:"
    )


def _contract_repair_details(
    response_text: str,
    error: GPT4OMiniJudgeParseError,
) -> dict[str, Any]:
    return {
        "contract_repair_retries": 1,
        "contract_repair_error_type": type(error).__name__,
        "contract_repair_error": str(error),
        "contract_repair_previous_response_truncated": len(response_text)
        > MAX_REPAIR_RESPONSE_CHARS,
    }


def _try_salvage_partial_response(
    response_text: str,
    parser: Any,
    *,
    model_name: str,
    error: GPT4OMiniJudgeParseError,
    previous_details: Mapping[str, Any] | None = None,
) -> Any | None:
    payload = _partial_judge_payload(response_text)
    if payload is None:
        return None
    try:
        result = parser(json.dumps(payload, ensure_ascii=False), model_name=model_name)
    except GPT4OMiniJudgeParseError:
        return None
    details = dict(result.score.details)
    details.update(dict(previous_details or {}))
    details.update(
        {
            "contract_local_salvage": True,
            "contract_local_salvage_error_type": type(error).__name__,
            "contract_local_salvage_error": str(error),
        }
    )
    score = ScoreResult(
        raw_score=result.score.raw_score,
        normalized_score=result.score.normalized_score,
        status=result.score.status,
        model_name=result.score.model_name,
        model_version=result.score.model_version,
        latency_ms=result.score.latency_ms,
        details=details,
    )
    return replace(result, score=score)


def _build_prompt(
    *,
    task: str,
    context: RankingContext,
    candidate_payload: dict[str, Any],
    uses_attached_image: bool = False,
    verification_targets: Any = None,
) -> str:
    payload = {
        "task": f"stage4_{task}_ranking_judge",
        "evaluation_guidance": _task_evaluation_guidance(
            task,
            uses_attached_image=uses_attached_image,
        ),
        "context": context.to_prompt_payload(),
        "candidate": candidate_payload,
        "response_schema": {
            "score": "number from 0 to 100",
            "confidence": "number from 0 to 1",
            "hard_reject": "boolean",
            "hard_rejection_codes": sorted(ALLOWED_HARD_REJECTION_CODES),
            "soft_violations": "array of strings",
            "evidence": "array of strings",
            "reason": "string",
        },
    }
    if verification_targets:
        payload["verification_targets"] = verification_targets
        payload["response_schema"]["observation"] = observation_response_schema()
        payload["observation_guidance"] = (
            "Return observation in addition to ALL normal judgment fields. Check exactly the supplied "
            "candidate-local targets, each check_id once; never add targets or borrow another candidate's "
            "requirements. The expected claims are requirements, not observations. Assess subject, "
            "relation, object, direction and negation. supported means expressed with actual content "
            "evidence; contradicted requires an explicit conflict, not mere absence; missing requires "
            "an expressible, directly verifiable target and enough content visibility to establish absence. "
            "Use uncertain when evidence is ambiguous, cropped, occluded or too weak. For images first "
            "decide whether the FULL fact is verifiable from pixels. Authorship, credits, dates, birthplace, "
            "administrative relations and named identity usually cannot be verified from appearance. "
            "Return unverifiable for invisible facts, never missing; appearance_only may describe matching "
            "visible cues but cannot support, contradict or establish absence of the full KG fact. "
            "Do not invent portraits, logos, readable text or settings to visualize non-visual relations. "
            "A fixed unverifiable target must stay unverifiable. Evidence must describe actual pixels "
            "with pixel: or quote actual text with text:. Missing text may cite absence explicitly. "
            "unsupported_claims lists concrete assertions unsupported by ANY supplied evidence, not "
            "ordinary rendering choices; absence from KG alone does not mean false. Give confidence "
            "per check. A should_include omission can lower quality without requiring regeneration."
        )
    return (
        JUDGE_RESPONSE_INSTRUCTIONS
        + "\n\nInput JSON:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\n\nDo not repeat the input JSON. Return only the judgment object with "
        "the required response_schema fields.\nJudgment JSON:"
        + (observation_contract_instructions(verification_targets, task) if verification_targets else "")
    )


def _task_evaluation_guidance(task: str, *, uses_attached_image: bool = False) -> str:
    if task == "image":
        image_source = (
            "Use the attached PNG pixels as the primary visual evidence. "
            "Use image prompt metadata only as auxiliary provenance. "
            if uses_attached_image
            else (
                "The image candidate is represented by generation prompt metadata, "
                "not by pixels. "
            )
        )
        return (
            image_source
            + _visual_evidence_source_guidance(
                uses_attached_image=uses_attached_image,
            )
            + "Judge whether the visual candidate is grounded in "
            "the context and safe for image generation. Do not reject solely "
            "because the candidate type is image. Forbidden or avoid-list items "
            "are checks to apply, not evidence that the candidate contains those "
            "items; hard-reject only when the candidate content actually requests "
            "or contains them. must_include and should_include are positive "
            "grounding targets, not unsafe-content categories. The words "
            "entity, identity, film, and grounded details are allowed "
            "descriptors in an image-generation prompt. Use low score and "
            "soft_violations for ordinary grounding weakness; reserve hard_reject "
            "for explicit forbidden content, malformed candidates, wrong modality, "
            "or clear identity mismatch."
        )
    if task == "pair":
        image_source = (
            "Use the attached PNG pixels as the primary visual evidence for the image candidate. "
            "Use image prompt metadata only as auxiliary provenance. "
            if uses_attached_image
            else (
                "The image candidate is represented by generation prompt metadata, "
                "not by pixels. "
            )
        )
        return (
            image_source
            + _visual_evidence_source_guidance(
                uses_attached_image=uses_attached_image,
            )
            + "Judge whether the text candidate and visual image content are "
            "mutually consistent and grounded in the context. Do "
            "not reject solely because the candidate type is image. Forbidden or "
            "avoid-list items are checks to apply, not evidence that the candidate "
            "contains those items; hard-reject only when the candidate content "
            "actually requests or contains them. must_include and should_include "
            "are positive grounding targets, not unsafe-content categories. The "
            "words entity, identity, film, and grounded details are allowed "
            "descriptors in an image-generation prompt. Use low score and "
            "soft_violations for ordinary grounding weakness; reserve hard_reject "
            "for explicit forbidden content, malformed candidates, wrong modality, "
            "or clear identity mismatch."
        )
    return (
        "Judge whether the text candidate is grounded in the context and safe "
        "for the requested missing modality."
    )


def _visual_evidence_source_guidance(*, uses_attached_image: bool) -> str:
    if uses_attached_image:
        pixel_requirement = (
            "Scores above 60 require at least one pixel: evidence item that "
            "names a concrete visible object, scene, text, style, composition, "
            "or attribute, plus one entity-specific context requirement that "
            "the pixels satisfy. "
        )
    else:
        pixel_requirement = (
            "Without attached pixels, score must be <= 50 and confidence must "
            "be <= 0.50 because only prompt-level generation plausibility can "
            "be judged. "
        )
    return (
        "Evidence strings must start with pixel:, metadata:, or context:. "
        "KG/context facts are requirements, not pixel evidence; metadata is "
        "provenance, not proof that pixels contain a fact. "
        + pixel_requirement
        + "Generic entity-type matches alone must score <= 50 and should use "
        "soft_violations such as generic_entity_type_only. "
    )


def _candidate_generated_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    generated = candidate.get("generated_content")
    if not isinstance(generated, Mapping):
        return {"candidate_id": _string_or_empty(candidate.get("candidate_id"))}
    payload: dict[str, Any] = {
        "candidate_id": _string_or_empty(candidate.get("candidate_id")),
    }
    for key in sorted(SAFE_GENERATED_CONTENT_FIELDS):
        value = generated.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
    return payload


def _attached_image_payload(image_input: ImageInput | None) -> dict[str, Any]:
    if image_input is None:
        return {}
    return {
        "attached_image": {
            "candidate_id": image_input.candidate_id,
            "media_type": image_input.media_type,
            "transport": "openai_image_url_data_url",
        }
    }


def _attach_judge_diagnostic(
    exc: Exception,
    *,
    kind: str,
    candidate_id: str,
    prompt: str,
    image_input: ImageInput | None,
    started: float,
    attempt: int | None = None,
    max_attempts: int | None = None,
    retryable: bool | None = None,
    will_retry: bool | None = None,
) -> None:
    latency_ms = max(0, int(round((time.monotonic() - started) * 1000)))
    diagnostic: dict[str, Any] = {
        "kind": kind,
        "candidate_id": candidate_id,
        "prompt_chars": len(prompt),
        "has_image": image_input is not None,
        "latency_ms": latency_ms,
    }
    if attempt is not None:
        diagnostic["attempt"] = attempt
    if max_attempts is not None:
        diagnostic["max_attempts"] = max_attempts
    if retryable is not None:
        diagnostic["retryable"] = retryable
    if will_retry is not None:
        diagnostic["will_retry"] = will_retry
    if image_input is not None:
        diagnostic["image_candidate_id"] = image_input.candidate_id
        diagnostic["image_media_type"] = image_input.media_type
    try:
        setattr(exc, "gpt4omini_judge_diagnostic", diagnostic)
    except Exception:
        return


def _is_retryable_judge_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    return type(exc).__name__ in RETRYABLE_JUDGE_ERROR_NAMES


def _parse_judge_response(response_text: str, *, model_name: str) -> tuple[ScoreResult, GateResult]:
    payload = _load_judge_payload(response_text)
    if not isinstance(payload, Mapping):
        raise GPT4OMiniJudgeParseError("judge response must be a JSON object")

    missing = sorted(REQUIRED_RESPONSE_FIELDS - payload.keys())
    if missing:
        raise GPT4OMiniJudgeParseError(f"missing required field: {missing[0]}")
    extra = sorted(payload.keys() - REQUIRED_RESPONSE_FIELDS - {"observation"})
    if extra:
        raise GPT4OMiniJudgeParseError(f"unexpected field: {extra[0]}")
    if "observation" in payload and not isinstance(payload["observation"], Mapping):
        raise GPT4OMiniJudgeParseError("observation must be an object")

    score = _finite_number("score", payload["score"])
    if not 0.0 <= score <= 100.0:
        raise GPT4OMiniJudgeParseError("score must be between 0 and 100")
    confidence = _finite_number("confidence", payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise GPT4OMiniJudgeParseError("confidence must be between 0 and 1")
    hard_reject = payload["hard_reject"]
    if not isinstance(hard_reject, bool):
        raise GPT4OMiniJudgeParseError("hard_reject must be a boolean")
    hard_codes = _string_tuple("hard_rejection_codes", payload["hard_rejection_codes"])
    for code in hard_codes:
        if code not in ALLOWED_HARD_REJECTION_CODES:
            raise GPT4OMiniJudgeParseError(f"unknown hard rejection code: {code}")
    if hard_reject and not hard_codes:
        raise GPT4OMiniJudgeParseError("hard_reject requires at least one hard_rejection_code")
    if not hard_reject and hard_codes:
        raise GPT4OMiniJudgeParseError("hard_rejection_codes require hard_reject to be true")
    soft_violations = _string_tuple("soft_violations", payload["soft_violations"])
    evidence = _string_tuple("evidence", payload["evidence"])
    reason = payload["reason"]
    if not isinstance(reason, str):
        raise GPT4OMiniJudgeParseError("reason must be a string")

    score_result = ScoreResult(
        raw_score=score,
        normalized_score=clamp01(score / 100.0),
        status="scored",
        model_name=model_name,
    )
    gate = GateResult(
        "reject" if hard_reject else "pass",
        model_name,
        hard_violations=hard_codes,
        soft_violations=soft_violations,
        judge_confidence=confidence,
        evidence=evidence,
        reason=reason,
    )
    return score_result, gate


def _load_judge_payload(response_text: str) -> Any:
    text = str(response_text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as direct_exc:
        payloads: list[Any] = []
        for candidate_text in _iter_json_candidate_texts(text):
            try:
                payload = json.loads(candidate_text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping) and REQUIRED_RESPONSE_FIELDS <= payload.keys():
                return payload
            payloads.append(payload)
        if payloads:
            return payloads[-1]
        raise GPT4OMiniJudgeParseError("malformed JSON judge response") from direct_exc


def _load_graph_payload(response_text: str) -> Mapping[str, Any]:
    text = str(response_text).strip()
    candidates = (text, *_iter_json_candidate_texts(text))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return payload
    raise GPT4OMiniJudgeParseError("malformed JSON graph extraction response")


def _mapping_list(field_name: str, value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise GPT4OMiniJudgeParseError(f"{field_name} must be an array of objects")
    return list(value)


def _required_text(field_name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GPT4OMiniJudgeParseError(f"{field_name} must be a non-empty string")
    return value.strip()


def _unit_number(field_name: str, value: Any) -> float:
    number = _finite_number(field_name, value)
    if not 0.0 <= number <= 1.0:
        raise GPT4OMiniJudgeParseError(f"{field_name} must be between 0 and 1")
    return number


def _iter_json_candidate_texts(response_text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        stripped = candidate.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            candidates.append(stripped)

    for match in JSON_CODE_FENCE_RE.finditer(response_text):
        add(match.group(1))
    for candidate in _iter_balanced_json_objects(response_text):
        add(candidate)
    return tuple(candidates)


def _iter_balanced_json_objects(text: str) -> tuple[str, ...]:
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return tuple(objects)


def _repair_source_text(response_text: str) -> str:
    text = str(response_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for candidate_text in _iter_json_candidate_texts(text):
            if _looks_like_judge_fragment(candidate_text):
                return candidate_text
        return text
    if isinstance(payload, Mapping):
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return text


def _looks_like_judge_fragment(candidate_text: str) -> bool:
    return any(f'"{field}"' in candidate_text for field in REQUIRED_RESPONSE_FIELDS)


def _judge_response_template() -> str:
    return json.dumps(
        {
            "score": 0,
            "confidence": 0,
            "hard_reject": False,
            "hard_rejection_codes": [],
            "soft_violations": [],
            "evidence": [],
            "reason": "",
        },
        ensure_ascii=False,
        sort_keys=True,
    )


_MISSING = object()


def _partial_judge_payload(response_text: str) -> dict[str, Any] | None:
    score = _first_valid_number_field(response_text, "score", minimum=0.0, maximum=100.0)
    confidence = _first_valid_number_field(
        response_text,
        "confidence",
        minimum=0.0,
        maximum=1.0,
    )
    hard_reject = _first_value_of_type(response_text, "hard_reject", bool)
    if score is _MISSING or confidence is _MISSING or hard_reject is _MISSING:
        return None

    hard_codes = _first_string_list_field(response_text, "hard_rejection_codes")
    valid_hard_codes = [
        code for code in hard_codes if code in ALLOWED_HARD_REJECTION_CODES
    ]
    if hard_reject and not valid_hard_codes:
        return None
    if not hard_reject:
        valid_hard_codes = []

    reason = _first_value_of_type(response_text, "reason", str)
    if reason is _MISSING or not reason.strip() or len(reason) > 200:
        reason = "Partial malformed judge response salvaged."

    return {
        "score": score,
        "confidence": confidence,
        "hard_reject": hard_reject,
        "hard_rejection_codes": valid_hard_codes,
        "soft_violations": _first_string_list_field(response_text, "soft_violations"),
        "evidence": _short_string_list(_first_string_list_field(response_text, "evidence")),
        "reason": reason,
    }


def _first_valid_number_field(
    text: str,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float | object:
    for value in _field_values(text, field_name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        normalized = float(value)
        if math.isfinite(normalized) and minimum <= normalized <= maximum:
            return normalized
    return _MISSING


def _first_value_of_type(text: str, field_name: str, expected_type: type) -> Any:
    for value in _field_values(text, field_name):
        if isinstance(value, expected_type):
            return value
    return _MISSING


def _first_string_list_field(text: str, field_name: str) -> list[str]:
    for value in _field_values(text, field_name):
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
    return []


def _field_values(text: str, field_name: str) -> tuple[Any, ...]:
    values: list[Any] = []
    decoder = json.JSONDecoder()
    marker = json.dumps(field_name)
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            return tuple(values)
        colon = text.find(":", index + len(marker))
        if colon < 0:
            return tuple(values)
        value_start = colon + 1
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        try:
            value, _ = decoder.raw_decode(text[value_start:])
        except json.JSONDecodeError:
            start = value_start + 1
            continue
        values.append(value)
        start = value_start + 1


def _short_string_list(values: list[str]) -> list[str]:
    short_values = [value.strip() for value in values if value.strip() and len(value) <= 240]
    return short_values[:1]


def _truncate_repair_response(response_text: str) -> str:
    text = str(response_text)
    if len(text) <= MAX_REPAIR_RESPONSE_CHARS:
        return text
    return (
        text[:MAX_REPAIR_RESPONSE_CHARS]
        + "\n...[truncated previous response for contract repair]..."
    )


def _finite_number(field_name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GPT4OMiniJudgeParseError(f"{field_name} must be finite")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise GPT4OMiniJudgeParseError(f"{field_name} must be finite")
    return normalized


def _string_tuple(field_name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise GPT4OMiniJudgeParseError(f"{field_name} must be an array of strings")
    if any(not isinstance(item, str) for item in value):
        raise GPT4OMiniJudgeParseError(f"{field_name} must contain only strings")
    return tuple(value)


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""
