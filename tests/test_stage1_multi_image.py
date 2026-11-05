"""All original images survive perception, repairs, batching and persisted provenance."""
import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import wave

from PIL import Image
import pytest

from mmkgc.adapters.image_knowledge import MULTI_IMAGE_MODELING_VERSION
from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner, load_qwen_config
from mmkgc.data import EntityRecord, MKGWDataStore, existing_image_paths
from mmkgc.io.resume import _validate_saved_media_sources
from mmkgc.io.writer import _stage_artifact
from mmkgc.pipeline import _semantic_model_calls
from mmkgc.ranking.image_input import ImageInputError
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT


def original_ref(index):
    return f"existing:image:{index:03d}:original"


def description_ref(index):
    return f"existing:image:{index:03d}:description"


def scene(tmp_path, count=3):
    paths = []
    for index in range(1, count + 1):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", (20, 16), (index * 25, index * 15, index * 5)).save(path)
        paths.append(path)
    target = EntityRecord(1, "Q1", "Scene", "PRIVATE_TARGET_DESCRIPTION", [
        {"entity_key": "Q2", "label": "Town", "relation": "located in", "direction": "out"}],
        {"text": "PRIVATE_TARGET_TEXT", "image": paths[0].name, "image_summary": "OLD_SUMMARY"},
        {"text": "PRIVATE_TRUTH"}, image_paths=[path.name for path in paths])
    return target, paths


def response(indices, *, descriptions=False, gaps=None):
    return {"entity_profile": {"primary_type": "building", "core_identity": "Scene",
                              "evidence_refs": [original_ref(indices[0])]},
        "extracted_triples": [{"subject": "Scene", "relation": "wall material", "object": "red brick",
            "visual_support": direct(),
            "evidence_refs": [original_ref(index)] + ([description_ref(index)] if descriptions else [])}
            for index in indices], "uncertain_aspects": gaps or []}


def direct():
    return {"basis": "direct", "scope": "target", "confidence": 0.95, "uncertainty": ""}


def transport(monkeypatch, replies, on_request=None):
    requests, timeouts = [], []
    class Reply:
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            content = json.dumps(self.value) if isinstance(self.value, dict) else self.value
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()
    def send(request, timeout):
        requests.append(json.loads(request.data))
        timeouts.append(timeout)
        if on_request:
            on_request(len(requests))
        return Reply(replies[len(requests) - 1])
    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", send)
    return requests, timeouts


def planner(root, *, batch_size=4, budget=180):
    return QwenLocalSemanticStagePlanner(QwenConfig("vision-test", "http://localhost/v1", "SECRET_KEY",
        stage1_image_batch_size=batch_size), dataset_root=root, knowledge_stage_timeout_seconds=budget)


def run(client, target, mode="text"):
    return client.build_knowledge(target, mode, coarse_relation_filtering(target, mode, catalogue_only=True))


def image_bytes(request):
    return [base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            for part in request["messages"][-1]["content"] if part["type"] == "image_url"]


def test_three_images_are_described_separately_then_jointly_extracted_and_deduplicated(tmp_path, monkeypatch):
    target, paths = scene(tmp_path)
    original = [path.read_bytes() for path in paths]
    extraction = response([1, 2, 3], descriptions=True)
    extraction["extracted_triples"].append({"subject": "Scene", "relation": "entrance shape", "object": "arch",
                                           "visual_support": direct(),
                                           "evidence_refs": [original_ref(3)]})
    requests, _ = transport(monkeypatch, ["View one.", "View two.", "View three.", extraction])
    client = planner(tmp_path)
    result = run(client, target)
    assert [image_bytes(request) for request in requests[:3]] == [[payload] for payload in original]
    assert image_bytes(requests[3]) == original
    assert "PRIVATE_" not in json.dumps(requests) and "OLD_SUMMARY" not in json.dumps(requests)
    assert result["source_inputs"]["image_summary"] == "OLD_SUMMARY"
    facts = result["relation_facts"]
    common = next(fact for fact in facts if fact["relation"] == "wall material")
    assert len(facts) == 3  # One KG fact and two distinct visual facts.
    assert set(common["evidence_refs"]) == {ref for index in (1, 2, 3)
                                            for ref in (original_ref(index), description_ref(index))}
    modeling = result["image_modeling"]
    assert modeling["schema_version"] == MULTI_IMAGE_MODELING_VERSION
    assert [entry["description"] for entry in modeling["images"]] == ["View one.", "View two.", "View three."]
    assert [entry["original_image"]["sha256"] for entry in modeling["images"]] == [
        hashlib.sha256(payload).hexdigest() for payload in original]
    artifact = json.loads(json.dumps(_stage_artifact("knowledge_modeling", result)["data"]))
    assert artifact["image_modeling"] == modeling
    _validate_saved_media_sources(artifact, target)
    calls = _semantic_model_calls(client, start_index=0)
    assert [call["stage"] for call in calls] == ["stage1_image_description"] * 3 + ["stage1_knowledge"]
    assert len(calls[-1]["image_description_model_call_refs"]) == 3
    assert {entry["model_call_ref"] for entry in calls[:3]} == set(calls[-1]["image_description_model_call_refs"].values())
    assert "base64," not in json.dumps(calls) and "SECRET_KEY" not in json.dumps(calls)
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_description_cannot_be_grounded_in_a_different_original_and_repairs_keep_all_images(tmp_path, monkeypatch):
    target, paths = scene(tmp_path, 2)
    bad = response([2])
    bad["extracted_triples"][0]["evidence_refs"].append(description_ref(1))
    requests, _ = transport(monkeypatch, ["First.", "Second.", bad, response([1, 2], descriptions=True)])
    client = planner(tmp_path)
    result = run(client, target)
    assert len(requests) == 4
    assert "missing_original_image_ref" in json.dumps(requests[-1])
    assert image_bytes(requests[-1]) == image_bytes(requests[-2]) == [path.read_bytes() for path in paths]
    assert client.calls[-1]["contract_repair_retries"] == 1
    assert len(result["image_modeling"]["images"]) == 2


def test_batches_see_only_their_originals_then_merge_facts_and_all_sources(tmp_path, monkeypatch):
    target, paths = scene(tmp_path, 5)
    gap = {"aspect": "roof color", "reason": "Conflicting lighting; color remains uncertain."}
    requests, _ = transport(monkeypatch, [*[f"View {index}." for index in range(1, 6)],
        response([1, 2], gaps=[gap]), response([3, 4], gaps=[gap]), response([5])])
    client = planner(tmp_path, batch_size=2)
    result = run(client, target)
    assert len(requests) == 8
    batches = ([1, 2], [3, 4], [5])
    for request, indices in zip(requests[5:], batches):
        assert image_bytes(request) == [paths[index - 1].read_bytes() for index in indices]
        prompt = request["messages"][-1]["content"][0]["text"]
        for index in range(1, 6):
            assert (original_ref(index) in prompt) == (index in indices)
    visual = next(fact for fact in result["relation_facts"] if fact["relation"] == "wall material")
    assert set(visual["evidence_refs"]) == {original_ref(index) for index in range(1, 6)}
    assert len(result["relation_facts"]) == 2
    assert result["uncertainties"] == [gap]
    assert len(result["image_modeling"]["images"]) == 5
    assert len(result["image_modeling"]["extraction_batches"]) == 3
    assert len(result["existing_modalities"]["image_summaries"]) == 5
    assert len({fact["fact_id"] for fact in result["relation_facts"]}) == 2


def test_batch_repair_cannot_cite_an_original_from_another_batch(tmp_path, monkeypatch):
    target, paths = scene(tmp_path, 3)
    wrong = response([1])
    wrong["extracted_triples"][0]["evidence_refs"] = [original_ref(3)]
    requests, _ = transport(monkeypatch, ["First.", "Second.", "Third.", wrong,
                                          response([1, 2]), response([3])])
    client = planner(tmp_path, batch_size=2)
    result = run(client, target)
    assert "unknown_evidence_ref" in json.dumps(requests[4])
    assert image_bytes(requests[3]) == image_bytes(requests[4]) == [path.read_bytes() for path in paths[:2]]
    assert len(result["image_modeling"]["images"]) == 3
    assert [call["contract_repair_retries"] for call in client.calls if call["stage"] == "stage1_knowledge"] == [1, 0]


def test_multiple_images_and_original_audio_share_one_extraction(tmp_path, monkeypatch):
    target, _ = scene(tmp_path, 2)
    with wave.open(str(tmp_path / "source.wav"), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000)
    target.existing_modalities["audio"] = "source.wav"
    extraction = response([1, 2])
    extraction["extracted_triples"].append({"subject": "Scene", "relation": "audio contains sound",
        "visual_support": None,
        "object": "silence", "evidence_refs": ["existing:audio:original", "existing:audio_summary"]})
    requests, _ = transport(monkeypatch, ["First image.", "Second image.", "Silent audio.", extraction])
    result = run(planner(tmp_path), target)
    assert len(requests) == 4 and len(image_bytes(requests[-1])) == 2
    assert any(part["type"] == "input_audio" for part in requests[-1]["messages"][-1]["content"])
    assert "audio" in result["media_modeling"] and len(result["image_modeling"]["images"]) == 2
    assert any(fact["source_modalities"] == ["audio"] for fact in result["relation_facts"])


def test_view_specific_details_and_reported_conflicts_are_not_erased(tmp_path, monkeypatch):
    target, _ = scene(tmp_path, 2)
    gap = {"aspect": "front door color", "reason": "The originals disagree; exact color is uncertain."}
    extraction = response([1, 2], gaps=[gap])
    extraction["extracted_triples"] += [
        {"subject": "Scene", "relation": "left facade material", "object": "stone", "evidence_refs": [original_ref(1)], "visual_support": direct()},
        {"subject": "Scene", "relation": "right facade material", "object": "brick", "evidence_refs": [original_ref(2)], "visual_support": direct()}]
    transport(monkeypatch, ["First view.", "Second view.", extraction])
    result = run(planner(tmp_path), target)
    assert result["uncertainties"] == [gap]
    assert {fact["relation"] for fact in result["relation_facts"]} >= {"left facade material", "right facade material"}


def test_all_originals_are_frozen_before_the_first_description(tmp_path, monkeypatch):
    target, paths = scene(tmp_path, 2)
    original = [path.read_bytes() for path in paths]
    def mutate(index):
        if index == 1:
            Image.new("RGB", (20, 16), "black").save(paths[1])
    requests, _ = transport(monkeypatch, ["First.", "Second.", response([1, 2])], mutate)
    run(planner(tmp_path), target)
    assert image_bytes(requests[1]) == [original[1]]
    assert image_bytes(requests[2]) == original


def test_missing_second_original_fails_before_any_model_call(tmp_path, monkeypatch):
    target, paths = scene(tmp_path, 2)
    paths[1].unlink()
    requests, _ = transport(monkeypatch, [])
    with pytest.raises(ImageInputError):
        run(planner(tmp_path), target)
    assert requests == []


def test_descriptions_and_batches_share_one_total_deadline(tmp_path, monkeypatch):
    target, _ = scene(tmp_path, 2)
    now = {"value": 0.0}
    monkeypatch.setattr("mmkgc.adapters.qwen.time.perf_counter", lambda: now["value"])
    def advance(index): now["value"] = 6.0 * index
    requests, timeouts = transport(monkeypatch, ["First.", "Second."], advance)
    with pytest.raises(TimeoutError, match="total time budget"):
        run(planner(tmp_path, budget=10), target)
    assert len(requests) == 2 and timeouts == [10.0, 4.0]
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_target_image_lists_are_never_loaded_or_leaked(tmp_path, monkeypatch):
    target, _ = scene(tmp_path, 2)
    target = replace(target, existing_modalities={"text": "Public source text", "image": "MISSING_TARGET.png"},
                     image_paths=["MISSING_TARGET.png", "SECOND_PRIVATE_TARGET.png"])
    reply = {"entity_profile": {"primary_type": "entity", "core_identity": "Scene", "evidence_refs": []},
             "extracted_triples": [], "uncertain_aspects": []}
    requests, _ = transport(monkeypatch, [reply])
    result = run(planner(tmp_path), target, "image")
    assert len(requests) == 1 and "PRIVATE_TARGET" not in json.dumps(requests)
    assert "MISSING_TARGET" not in json.dumps(requests) and "image_modeling" not in result


def test_image_list_without_a_legacy_primary_is_supported_and_duplicate_paths_are_removed(tmp_path, monkeypatch):
    target, _ = scene(tmp_path, 2)
    target = replace(target, existing_modalities={"image": None}, image_paths=target.image_paths * 2)
    requests, _ = transport(monkeypatch, ["First.", "Second.", response([1, 2])])
    result = run(planner(tmp_path), target)
    assert len(existing_image_paths(target)) == 2 and len(requests) == 3
    assert result["source_inputs"]["has_image"] is True


def test_saved_multi_image_sources_detect_changed_bytes_and_changed_image_lists(tmp_path, monkeypatch):
    target, paths = scene(tmp_path, 2)
    transport(monkeypatch, ["First.", "Second.", response([1, 2])])
    result = run(planner(tmp_path), target)
    with pytest.raises(ValueError, match="assets differ"):
        _validate_saved_media_sources(result, replace(target, image_paths=target.image_paths[:1]))
    Image.new("RGB", (20, 16), "black").save(paths[1])
    with pytest.raises(ValueError, match="bytes differ"):
        _validate_saved_media_sources(result, target)


@pytest.mark.parametrize("index_name,key", [("qid2images.json", "Q1"),
    ("entity2images.json", "http://www.wikidata.org/entity/Q1"), ("entity_id2images.json", "0")])
def test_dataset_loads_all_aligned_originals_and_masks_the_whole_list(tmp_path, index_name, key):
    _, paths = scene(tmp_path)
    (tmp_path / "entity2id.txt").write_text("http://www.wikidata.org/entity/Q1 0\n")
    (tmp_path / "relation2id.txt").write_text("")
    alignment = tmp_path / "image_alignment"
    alignment.mkdir()
    (alignment / index_name).write_text(json.dumps({key: [path.name for path in paths] + [paths[0].name]}))
    store = MKGWDataStore(tmp_path)
    target = store.build_record(0, "text")
    assert target.image_paths == [str(path) for path in paths]
    assert target.existing_modalities["image"] == str(paths[0])
    masked = store.build_record(0, "image")
    assert masked.image_paths == [] and masked.existing_modalities["image"] is None


@pytest.mark.parametrize("directory", ["images_by_id/0", "images/Q1"])
def test_dataset_directory_discovery_preserves_primary_and_collects_remaining_images(tmp_path, directory):
    (tmp_path / "entity2id.txt").write_text("Q1 0\n")
    root = tmp_path / directory
    root.mkdir(parents=True)
    for name in ["b.png", "image.jpg", "a.png"]:
        (root / name).write_bytes(b"placeholder; decoding occurs at Stage 1")
    paths = MKGWDataStore(tmp_path).image_paths(0)
    assert [Path(path).name for path in paths] == ["image.jpg", "a.png", "b.png"]


@pytest.mark.parametrize("value", [0, -1, True, "2"])
def test_image_batch_size_rejects_invalid_config_values(value):
    with pytest.raises(ValueError, match="stage1_image_batch_size"):
        QwenConfig("vision-test", "http://localhost/v1", "key", stage1_image_batch_size=value)


def test_image_batch_size_loads_from_yaml(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("lmm:\n  model_name: vision-test\n  base_url: http://localhost/v1\ninference:\n  stage1_image_batch_size: 2\n")
    assert load_qwen_config(path).stage1_image_batch_size == 2


def test_qwen_launcher_sets_the_server_multi_image_limit(tmp_path):
    conda_root = tmp_path / "conda"
    shell_root = conda_root / "etc/profile.d"
    shell_root.mkdir(parents=True)
    (shell_root / "conda.sh").write_text("conda() { return 0; }\n")
    bin_root = tmp_path / "bin"
    bin_root.mkdir()
    conda = bin_root / "conda"
    conda.write_text(f"#!/bin/sh\nprintf '%s\\n' '{conda_root}'\n")
    conda.chmod(0o755)
    python = tmp_path / "fake-python"
    python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    python.chmod(0o755)
    environment = {**os.environ, "PATH": str(bin_root) + os.pathsep + os.environ["PATH"],
        "QWEN_PYTHON": str(python), "QWEN_MODEL_DIR": str(tmp_path), "QWEN_MAX_IMAGES_PER_PROMPT": "4"}
    script = Path(__file__).resolve().parents[1] / "scripts/run_qwen7b.sh"
    process = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True, check=True)
    assert "--limit-mm-per-prompt\nimage=4\n" in process.stdout
