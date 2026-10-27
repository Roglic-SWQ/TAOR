from __future__ import annotations

import inspect
from urllib.error import URLError

import pytest

from mmkgc.adapters.qwen import QwenConfig, QwenLocalTextGenerator
from mmkgc.data import EntityRecord
from mmkgc.pipeline import MMKGCPipeline
from mmkgc.ranking.engine import ModelFusionRankingEngine
from mmkgc.run import build_parser
from mmkgc.stages.semantic import StrictJsonStagePlanner


REMOVED_MODE_FLAGS = {
    "--text-generator",
    "--image-prompt-generator",
    "--image-generator",
    "--stage02-planner",
    "--stage3-planner",
    "--stage03-planner",
    "--stage3-scorer",
    "--stage3-llm-assessor",
    "--stage4-ranker",
}


def _record() -> EntityRecord:
    return EntityRecord(
        entity_id=1,
        entity_key="Q1",
        label="Example",
        description="",
        neighbors=[],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": None, "image": None},
    )


def test_cli_does_not_expose_alternate_chain_selectors() -> None:
    option_strings = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }

    assert REMOVED_MODE_FLAGS.isdisjoint(option_strings)


def test_pipeline_rejects_incomplete_fixed_mainline() -> None:
    class RankingEngine:
        def rank(self, context):  # pragma: no cover - validation runs first.
            raise AssertionError(context)

    pipeline = MMKGCPipeline(ranking_engine=RankingEngine())

    with pytest.raises(RuntimeError, match="fixed mainline is missing required components"):
        pipeline.run(_record(), "text")


def test_model_fusion_engine_has_no_legacy_ranker_injection() -> None:
    parameters = inspect.signature(ModelFusionRankingEngine).parameters

    assert "legacy_ranker" not in parameters


def test_qwen_generation_failure_is_not_replaced_by_rule_output(monkeypatch) -> None:
    generator = QwenLocalTextGenerator(
        QwenConfig(
            model_name="qwen-test",
            base_url="http://localhost:12345/v1",
            api_key="test-key",
        )
    )

    def fail_request(*_args, **_kwargs):
        raise URLError("forced failure")

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", fail_request)

    with pytest.raises(URLError, match="forced failure"):
        generator.generate("Example", "film", "director", "identity", [])

    call = generator.consume_last_call("Q1", "text_001")
    assert call is not None
    assert call["status"] == "failed"
    assert call["fallback_used"] is False


def test_semantic_contract_failure_does_not_return_deterministic_baseline() -> None:
    planner = StrictJsonStagePlanner(lambda _prompt: "not-json")

    with pytest.raises(Exception, match="semantic stage response must be JSON"):
        planner.refine_coarse(
            _record(),
            "text",
            {
                "selected_relations": ["director"],
                "high_recall_relation_candidates": [],
            },
        )

    assert planner.calls[-1]["status"] == "failed"
