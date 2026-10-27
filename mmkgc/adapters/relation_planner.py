"""OpenAI-compatible GPT-4o mini Stage 2 planner using the existing config format."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib import request

from mmkgc.adapters.qwen import QwenConfig
from mmkgc.stages.relation_planning import (
    SLOT_SCHEMA, RelationPlanValidationError, apply_candidate_slot_plans,
    candidate_slot_response_schema, selection_prompt,
)

PROMPT_VERSION = "gpt4omini_stage2_candidate_slots_v3"


class GPT4oMiniRelationPlanner:
    def __init__(
        self, config: QwenConfig, timeout_seconds: float = 60.0, *, max_contract_repairs: int = 2
    ) -> None:
        if type(max_contract_repairs) is not int or max_contract_repairs < 0:
            raise ValueError("max_contract_repairs must be a nonnegative integer")
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.max_contract_repairs = max_contract_repairs
        self.calls: list[dict[str, Any]] = []

    def _complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": 0.0, "max_tokens": max(4096, self.config.max_token),
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "relation_plans", "strict": True, "schema": schema,
            }},
        }
        req = request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        choice = result["choices"][0]
        if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
            raise ValueError("Stage 2 response was refused or incomplete")
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise ValueError("Stage 2 response content must be text")
        return content

    def plan(self, record: Any, missing_mode: str, knowledge: dict[str, Any], candidate_budget: int) -> dict[str, Any]:
        system, prompt = selection_prompt(record.label, missing_mode, knowledge, candidate_budget)
        call = {
            "stage": "stage2_selection", "provider": "gpt4omini-global-relation-planner",
            "model_name": self.config.model_name, "prompt": prompt,
            "system_prompt": system, "ground_truth_used": False,
            "prompt_version": PROMPT_VERSION,
            "response_schema_version": SLOT_SCHEMA,
        }
        started = time.perf_counter()
        try:
            schema = candidate_slot_response_schema(knowledge, candidate_budget)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            invalid_responses: set[str] = set()
            attempts: list[dict[str, Any]] = []
            call["validation_attempts"] = attempts
            for attempt in range(self.max_contract_repairs + 1):
                if attempt:
                    call["contract_repair_retries"] = attempt
                response = self._complete(messages, schema)
                call["response_text"] = response
                try:
                    parsed = json.loads(response)
                except json.JSONDecodeError as exc:
                    errors = [{"code": "invalid_json", "path": "", "message": exc.msg,
                               "line": exc.lineno, "column": exc.colno}]
                    parsed = None
                    fingerprint = response.strip()
                else:
                    errors = []
                    fingerprint = json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                try:
                    if errors:
                        raise RelationPlanValidationError(errors)
                    result = apply_candidate_slot_plans(knowledge, parsed, missing_mode, candidate_budget)
                except RelationPlanValidationError as exc:
                    attempts.append({"attempt": attempt, "response_text": response, "errors": exc.errors})
                    call.update(contract_repair_error=str(exc), contract_repair_error_type=type(exc).__name__,
                                validation_errors=exc.errors)
                    call.setdefault("contract_repair_original_response", response)
                    if fingerprint in invalid_responses:
                        call["contract_repair_stopped_reason"] = "repeated_invalid_response"
                        raise
                    if attempt == self.max_contract_repairs:
                        call["contract_repair_stopped_reason"] = "repair_limit_reached"
                        raise
                    invalid_responses.add(fingerprint)
                    repair_prompt = (
                        "Correct your previous Stage 2 plan using the ORIGINAL input and ALL "
                        "validator errors below. Return a complete replacement JSON, not a patch. "
                        "Preserve the supplied facts and evidence; previous assistant responses "
                        "are not evidence. Each path is a JSON Pointer with zero-based array indices; "
                        "an empty path refers to the entire response. "
                        "Cite only the current fact's allowed evidence; use supporting_fact_ids to "
                        "link other facts. Keep every fact_id exactly once, give every role a "
                        "nonblank reason, and recheck support roles and candidate slot count. "
                        "Return {plan: {status, relation_plans, candidate_anchors}} with no "
                        "candidate_quota fields. Keeping a fact does not require a slot. "
                        "For ready, choose exactly candidate_budget anchor IDs, allowing repeats; "
                        "primary facts need a slot and secondary facts can keep zero slots. "
                        "For insufficient_evidence, use only context/excluded roles and no slots. "
                        "Do not repeat the invalid response.\nVALIDATION_ERRORS_JSON:\n"
                        + json.dumps({"candidate_budget": candidate_budget, "validation_errors": exc.errors},
                                     ensure_ascii=False, separators=(",", ":"))
                    )
                    call["contract_repair_prompt"] = repair_prompt
                    messages = [*messages, {"role": "assistant", "content": response},
                                {"role": "user", "content": repair_prompt}]
                else:
                    attempts.append({"attempt": attempt, "response_text": response, "errors": []})
                    call.update(status="applied", validation_errors=[])
                    return result
        except Exception as exc:
            call.update(status="failed", error_type=type(exc).__name__, error_message=str(exc))
            setattr(exc, "model_call", call)
            raise
        finally:
            call["latency_ms"] = int((time.perf_counter() - started) * 1000)
            if self.config.api_key:
                sanitized = json.dumps(call, ensure_ascii=False).replace(self.config.api_key, "[REDACTED]")
                call.clear()
                call.update(json.loads(sanitized))
            self.calls.append(call)
