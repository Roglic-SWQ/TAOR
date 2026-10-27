from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading

import pytest

from mmkgc.adapters.base import GeneratedImageArtifact
from mmkgc.adapters.qwen import QwenConfig, QwenLocalImagePromptGenerator, QwenLocalTextGenerator, load_qwen_config
from mmkgc.data import EntityRecord
from mmkgc.stages.generation import relation_alternation_generation, safer_relation_alternation_generation
from mmkgc.stages.modeling import apply_knowledge_response, build_knowledge_input
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.relation_planning import apply_relation_plans


def fixture(mode, budget=5):
    record = EntityRecord(
        entity_id=1, entity_key="Q1", label="Example Museum", description="SECRET_TARGET",
        neighbors=[], existing_modalities={"text": "A museum in Example City." if mode == "image" else None, "image": None},
        ground_truth_modalities={"image": "SECRET_IMAGE", "text": "SECRET_TARGET"},
    )
    inputs = build_knowledge_input(record, mode, {"selected_relations": [
        {"relation": "located in", "direction": "out", "neighbor_entities": ["Example City"],
         "evidence_triples": ["(Example Museum, located in, Example City)"]},
    ]})
    knowledge = apply_knowledge_response(inputs, {
        "entity_profile": {"primary_type": "museum", "core_identity": "Example Museum", "secondary_types": [],
                           "semantic_domains": [], "evidence_refs": []},
        "text_visual_profile": {}, "key_attributes": [], "uncertain_aspects": [],
        "relation_facts": [{"fact_id": fact["fact_id"], "claim": "The museum is in Example City."} for fact in inputs["fact_candidates"]],
    })
    fact = knowledge["relation_facts"][0]
    selection, reasoning = compile_plan_context(knowledge, apply_relation_plans(knowledge, {
        "status": "ready", "relation_plans": [{
            "fact_id": fact["fact_id"], "role": "primary", "candidate_quota": budget,
            "evidence_refs": list(fact["evidence_refs"]), "supporting_fact_ids": [], "reason": "Supplied evidence reviewed.",
        }],
    }, mode, budget), mode)
    return record, knowledge, selection, reasoning


class Response:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": self.text}}]}).encode()


@pytest.mark.parametrize("mode", ["text", "image"])
@pytest.mark.parametrize("workers", [1, 2, 5])
@pytest.mark.parametrize("safe_retry", [False, True])
def test_bounded_stage3_requests_keep_candidates_logs_and_gpu_on_caller(monkeypatch, mode, workers, safe_retry):
    record, knowledge, selection, reasoning = fixture(mode)
    original = deepcopy((knowledge, selection, reasoning))
    config = QwenConfig("gpt-4o-mini", "http://localhost/v1", "test-key", stage3_max_concurrency=workers)
    text, image = QwenLocalTextGenerator(config), QwenLocalImagePromptGenerator(config)
    lock, barrier = threading.Lock(), threading.Barrier(workers)
    active, peak, admitted, finished = 0, 0, 0, []
    main_thread = threading.get_ident()
    gpu_calls = []

    def request(req, **kwargs):
        nonlocal active, peak, admitted
        body = json.loads(req.data)
        payload = json.loads(body["messages"][1]["content"].split("INPUT_JSON:\n", 1)[1])
        assert "SECRET_TARGET" not in json.dumps(payload) and "SECRET_IMAGE" not in json.dumps(payload)
        angle = payload["task"]["generation_angle"]
        with lock:
            admitted += 1
            first_wave = admitted <= workers
            active += 1
            peak = max(peak, active)
        if first_wave:
            barrier.wait(timeout=5)
        with lock:
            active -= 1
            finished.append(angle)
        content = "Text " + angle if mode == "text" else json.dumps({"positive_prompt": "Image " + angle})
        return Response(content)

    class ImageGenerator:
        def generate(self, entity_key, candidate_id, positive_prompt, negative_prompt):
            assert threading.get_ident() == main_thread
            assert len(finished) == 5
            gpu_calls.append(candidate_id)
            return GeneratedImageArtifact(
                entity_key=entity_key, candidate_id=candidate_id,
                relative_image_path=f"generated_images/{candidate_id}.png",
                relative_metadata_path=f"generated_images/{candidate_id}.json", image_bytes=b"PNG",
                metadata={"generator": "test-sdxl", "width": 1024, "height": 1024, "seed": 100 + len(gpu_calls)},
                model_call={"modality": "image", "candidate_id": candidate_id, "positive_prompt": positive_prompt},
            )

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", request)
    args = (record, mode, knowledge, reasoning, selection, text, image, ImageGenerator())
    if safe_retry:
        result = safer_relation_alternation_generation(*args, candidate_budget=5, failed_ranking={},
            stage3_feedback={"repair_actions": ["remove unsupported details"], "preserve": ["entity identity"]})
    else:
        result = relation_alternation_generation(*args, candidate_budget=5)
    candidates = result["relation_alternation_generation"][f"{mode}_candidates"]
    ids = [f"{mode}_{'safe_' if safe_retry else ''}{index:03d}" for index in range(1, 6)]
    assert [candidate["candidate_id"] for candidate in candidates] == ids
    assert peak == workers
    calls = [call for call in result["model_calls"] if call.get("modality") != "image"]
    assert [call["candidate_id"] for call in calls] == ids
    for candidate, call in zip(candidates, calls, strict=True):
        angle = candidate["generation_angle"]
        assert call["entity_key"] == "Q1"
        assert angle in call["response_text"]
        if mode == "image":
            assert candidate["positive_prompt"] == "Image " + angle
        else:
            assert candidate["generated_content"]["value"] == "Text " + angle
            assert candidate["metadata"]["candidate_text"] == "Text " + angle
    assert gpu_calls == (ids if mode == "image" else [])
    assert (knowledge, selection, reasoning) == original
    assert text.consume_last_call("Q1", "unused") is None
    assert image.consume_last_call("Q1", "unused") is None


@pytest.mark.parametrize("cls", [QwenLocalTextGenerator, QwenLocalImagePromptGenerator])
def test_out_of_order_completion_does_not_swap_thread_audit_records(cls):
    generator = cls(QwenConfig("gpt-4o-mini", "http://localhost/v1", "key"))
    first_recorded, second_recorded, first_consumed = threading.Event(), threading.Event(), threading.Event()

    def first():
        generator._record_call({"response_text": "first", "metadata": {"value": "first"}})
        first_recorded.set()
        assert second_recorded.wait(5)
        result = generator.consume_last_call("Q1", "image_001")
        metadata = generator.last_candidate_metadata()
        first_consumed.set()
        return result, metadata

    def second():
        assert first_recorded.wait(5)
        generator._record_call({"response_text": "second", "metadata": {"value": "second"}})
        second_recorded.set()
        assert first_consumed.wait(5)
        return generator.consume_last_call("Q1", "image_002"), generator.last_candidate_metadata()

    with ThreadPoolExecutor(max_workers=2) as executor:
        a, b = executor.submit(first), executor.submit(second)
        first_result, second_result = a.result(), b.result()
    assert first_result[0]["response_text"] == first_result[1]["value"] == "first"
    assert second_result[0]["response_text"] == second_result[1]["value"] == "second"


@pytest.mark.parametrize("mode", ["text", "image"])
def test_failure_preserves_candidate_audit_and_does_not_start_gpu(monkeypatch, mode):
    record, knowledge, selection, reasoning = fixture(mode)
    config = QwenConfig("gpt-4o-mini", "http://localhost/v1", "test-key", stage3_max_concurrency=5)
    barrier = threading.Barrier(5)
    gpu_calls = []

    def request(req, **kwargs):
        payload = json.loads(json.loads(req.data)["messages"][1]["content"].split("INPUT_JSON:\n", 1)[1])
        angle = payload["task"]["generation_angle"]
        barrier.wait(timeout=5)
        if angle == "identity anchor":
            raise TimeoutError("test timeout")
        return Response("Text " + angle if mode == "text" else json.dumps({"positive_prompt": "Image " + angle}))

    class ImageGenerator:
        def generate(self, *args):
            gpu_calls.append(args)

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", request)
    with pytest.raises(TimeoutError) as caught:
        relation_alternation_generation(record, mode, knowledge, reasoning, selection,
            QwenLocalTextGenerator(config), QwenLocalImagePromptGenerator(config), ImageGenerator())
    assert caught.value.candidate_id == f"{mode}_001"
    assert caught.value.model_call["candidate_id"] == f"{mode}_001"
    assert caught.value.model_call["stage"] == "stage3_generation"
    assert caught.value.model_call["status"] == "failed"
    assert not gpu_calls


def test_workers_are_capped_by_candidate_count_and_legacy_generator_stays_serial():
    record, knowledge, selection, reasoning = fixture("text", budget=1)
    calls = []

    class Generator:
        max_concurrency = 10

        def generate(self, *args):
            calls.append(threading.get_ident())
            return "The museum is in Example City."

    result = relation_alternation_generation(record, "text", knowledge, reasoning, selection, Generator(), None, candidate_budget=1)
    assert calls == [threading.get_ident()]
    assert len(result["relation_alternation_generation"]["text_candidates"]) == 1


@pytest.mark.parametrize("value", [0, -1, True, 1.2, "5"])
def test_invalid_config_concurrency_fails_early(tmp_path, value):
    with pytest.raises(ValueError, match="positive integer"):
        QwenConfig("gpt-4o-mini", "http://localhost/v1", "key", stage3_max_concurrency=value)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"lmm": {"model_name": "gpt-4o-mini", "base_url": "http://localhost/v1"},
                                "inference": {"stage3_max_concurrency": value}}))
    with pytest.raises(ValueError, match="positive integer"):
        load_qwen_config(path)


def test_legacy_config_defaults_to_serial_and_mainline_enables_five(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text('lmm: {model_name: test, base_url: "http://localhost/v1"}\n')
    assert load_qwen_config(path).stage3_max_concurrency == 1
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost/v1")
    assert load_qwen_config("configs/gpt4omini-mainline.yaml").stage3_max_concurrency == 5
