"""Two image-bearing requests, description provenance and shared failure bounds."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from mmkgc.adapters.image_knowledge import IMAGE_REF, DESCRIPTION_REF, DESCRIPTION_PROMPT, IMAGE_MODELING_VERSION
from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner
from mmkgc.data import EntityRecord
from mmkgc.io.writer import _stage_artifact
from mmkgc.pipeline import _semantic_model_calls
from mmkgc.ranking.image_input import ImageInputError
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT
from mmkgc.stages.triples import TRIPLE_SCHEMA


def fixture(tmp_path):
    image = Image.new("RGB", (160, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 20, 70, 80), fill="blue")
    draw.rectangle((95, 20, 145, 80), fill="red")
    path = tmp_path / "source.png"
    image.save(path)
    target = EntityRecord(1, "TEST_SCENE", "Scene", "PRIVATE_DESCRIPTION", [],
        {"text": "PRIVATE_TARGET_TEXT", "image": "source.png", "image_summary": "OUTDATED_SUMMARY"},
        {"text": "PRIVATE_GROUND_TRUTH"})
    return target, path


def extraction(refs=None):
    return {"entity_profile": {"primary_type": "drawing", "core_identity": "Scene", "evidence_refs": [IMAGE_REF]},
        "extracted_triples": [{"subject": "Scene", "relation": "contains", "object": "a blue circle",
                               "visual_support": {"basis": "direct", "scope": "target", "confidence": 0.95, "uncertainty": ""},
                               "evidence_refs": refs if refs is not None else [DESCRIPTION_REF, IMAGE_REF]}],
        "uncertain_aspects": []}


def transport(monkeypatch, replies, on_request=None, timeouts=None):
    requests = []
    class Response:
        def __init__(self, reply): self.reply = reply
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            reply, finish = self.reply if isinstance(self.reply, tuple) else (self.reply, "stop")
            return json.dumps({"choices": [{"finish_reason": finish, "message": {"content": reply}}]}).encode()
    def send(request, **kwargs):
        requests.append(json.loads(request.data))
        if timeouts is not None: timeouts.append(kwargs["timeout"])
        if on_request: on_request(len(requests))
        return Response(replies[len(requests) - 1])
    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", send)
    return requests


def planner(tmp_path, budget=180):
    return QwenLocalSemanticStagePlanner(QwenConfig("vision-test", "http://localhost/v1", "TEST_SECRET"),
                                        dataset_root=tmp_path, knowledge_stage_timeout_seconds=budget)


def text_and_image(request):
    content = request["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]
    return content[0]["text"], content[1]["image_url"]["url"]


def test_describe_then_extract_both_receive_the_original_image_and_save_both_sources(tmp_path, monkeypatch):
    target, path = fixture(tmp_path)
    original = deepcopy(target.existing_modalities)
    requests = transport(monkeypatch, ["A blue circle and red rectangle on a white background.", json.dumps(extraction())])
    client = planner(tmp_path)
    result = client.build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))
    assert len(requests) == 2 and target.existing_modalities == original
    assert "guided_json" not in requests[0]
    first, first_image = text_and_image(requests[0])
    second, second_image = text_and_image(requests[1])
    assert first == DESCRIPTION_PROMPT
    assert "natural-language" in requests[0]["messages"][0]["content"]
    assert first_image == second_image
    assert base64.b64decode(first_image.split(",", 1)[1]) == path.read_bytes()
    assert "A blue circle and red rectangle" in second and "Inspect the original image again" in second
    assert "PRIVATE_" not in first + second and "OUTDATED_SUMMARY" not in second
    assert result["schema_version"] == TRIPLE_SCHEMA
    assert result["source_inputs"]["image_summary"] == "OUTDATED_SUMMARY"
    fact = result["relation_facts"][0]
    assert set(fact["evidence_refs"]) == {IMAGE_REF, DESCRIPTION_REF}
    catalog = {item["evidence_id"]: item for item in result["evidence_catalog"]}
    assert catalog[IMAGE_REF]["representation"] == "original_image" and catalog[DESCRIPTION_REF]["processing"] == "model_generated_description"
    assert result["image_modeling"]["original_image"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["image_modeling"]["extraction_uses_original_image"] is True
    calls = _semantic_model_calls(client, start_index=0)
    assert [call["stage"] for call in calls] == ["stage1_image_description", "stage1_knowledge"]
    assert calls[0]["model_call_ref"] == calls[1]["image_description_model_call_ref"]
    assert "base64," not in json.dumps(calls) and "TEST_SECRET" not in json.dumps(calls)
    artifact = _stage_artifact("knowledge_modeling", result)["data"]
    assert artifact["image_modeling"]["schema_version"] == IMAGE_MODELING_VERSION
    assert artifact["source_inputs"] == result["source_inputs"]
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_description_errors_can_be_corrected_and_missing_description_details_added_from_the_image(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    corrected = extraction([IMAGE_REF])
    requests = transport(monkeypatch, ["A green triangle.", json.dumps(corrected)])
    result = planner(tmp_path).build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))
    prompt, _ = text_and_image(requests[1])
    assert "correct its errors" in prompt and "recover useful visible details" in prompt
    assert result["relation_facts"][0]["object"] == "a blue circle"
    assert result["relation_facts"][0]["evidence_refs"] == [IMAGE_REF]


def test_qwen_visual_extraction_requires_source_metadata_in_the_first_decode(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    requests = transport(monkeypatch, ["A blue circle.", json.dumps(extraction())])
    client = QwenLocalSemanticStagePlanner(QwenConfig("Qwen/Qwen2-VL-7B-Instruct", "http://localhost/v1", "EMPTY"),
                                         dataset_root=tmp_path)
    client.build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))
    assert "guided_json" not in requests[0]
    schema = requests[1]["guided_json"]["properties"]["extracted_triples"]["items"]
    assert "visual_support" in schema["required"]
    assert requests[1]["guided_decoding_backend"] == "lm-format-enforcer"


def test_contract_repairs_keep_image_attached_and_require_original_image_citations(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    requests = transport(monkeypatch, ["A blue circle.", json.dumps(extraction([DESCRIPTION_REF])), json.dumps(extraction())])
    client = planner(tmp_path)
    client.build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))
    assert len(requests) == 3
    parts = [text_and_image(request) for request in requests]
    assert len({image for _, image in parts}) == 1
    assert "missing_original_image_ref" in parts[2][0]
    assert client.calls[-1]["contract_repair_retries"] == 1


@pytest.mark.parametrize("reply", ["", ("A partial description", "length")])
def test_failed_or_truncated_description_stops_before_extraction(tmp_path, monkeypatch, reply):
    target, _ = fixture(tmp_path)
    requests = transport(monkeypatch, [reply])
    client = planner(tmp_path)
    with pytest.raises(ValueError):
        client.build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))
    assert len(requests) == 1 and client.calls[0]["status"] == "failed"
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


@pytest.mark.parametrize('reply,code', [('', 'empty_response'),
    (('{"entity_profile":', 'length'), 'output_token_limit')])
def test_incomplete_image_extraction_is_not_retried_as_a_json_format_error(tmp_path, monkeypatch, reply, code):
    from mmkgc.stages.semantic import Stage1CompletionError
    target, _ = fixture(tmp_path)
    requests = transport(monkeypatch, ['A blue circle.', reply])
    client = planner(tmp_path)
    with pytest.raises(Stage1CompletionError) as error:
        client.build_knowledge(target, 'text', coarse_relation_filtering(target, 'text', catalogue_only=True))
    assert error.value.code == code
    assert len(requests) == 2
    assert client.calls[-1]['completion_failure'] == code
    assert client.calls[-1]['contract_repair_retries'] == 0


def test_description_and_extraction_share_one_total_deadline(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    now = {"value": 0.0}
    monkeypatch.setattr("mmkgc.adapters.qwen.time.perf_counter", lambda: now["value"])
    def slow(_): now["value"] = 11.0
    requests = transport(monkeypatch, ["A blue circle."], slow)
    with pytest.raises(TimeoutError, match="total time budget"):
        planner(tmp_path, budget=10).build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))
    assert len(requests) == 1 and KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_extraction_receives_only_time_remaining_after_description(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    now, timeouts = {"value": 0.0}, []
    monkeypatch.setattr("mmkgc.adapters.qwen.time.perf_counter", lambda: now["value"])
    def advance(index): now["value"] = 6.0 if index == 1 else 8.0
    transport(monkeypatch, ["A blue circle.", json.dumps(extraction())], advance, timeouts)
    planner(tmp_path, budget=10).build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))
    assert timeouts == [10.0, 4.0]


def test_two_steps_use_frozen_bytes_even_if_the_source_file_changes(tmp_path, monkeypatch):
    target, path = fixture(tmp_path)
    original = path.read_bytes()
    def mutate(index):
        if index == 1: Image.new("RGB", (10, 10), "green").save(path)
    requests = transport(monkeypatch, ["A blue circle.", json.dumps(extraction())], mutate)
    planner(tmp_path).build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))
    for request in requests:
        _, url = text_and_image(request)
        assert base64.b64decode(url.split(",", 1)[1]) == original


def test_missing_target_image_is_never_opened_or_attached(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    target.existing_modalities["image"] = "DOES_NOT_EXIST.png"
    target.existing_modalities["image_summary"] = "PRIVATE_TARGET_IMAGE_SUMMARY"
    response = {"entity_profile": {"primary_type": "entity", "core_identity": "Scene", "evidence_refs": []},
                "extracted_triples": [], "uncertain_aspects": []}
    requests = transport(monkeypatch, [json.dumps(response)])
    result = planner(tmp_path).build_knowledge(target, "image", coarse_relation_filtering(target, "image", catalogue_only=True))
    assert len(requests) == 1 and isinstance(requests[0]["messages"][-1]["content"], str)
    assert "PRIVATE_TARGET_IMAGE_SUMMARY" not in json.dumps(requests)
    assert "image_modeling" not in result


@pytest.mark.parametrize("asset", ["missing.png", "../escape.png"])
def test_invalid_available_image_fails_instead_of_silently_skipping_vision(tmp_path, monkeypatch, asset):
    target, _ = fixture(tmp_path)
    target.existing_modalities["image"] = asset
    requests = transport(monkeypatch, [])
    with pytest.raises(ImageInputError):
        planner(tmp_path).build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))
    assert requests == []
