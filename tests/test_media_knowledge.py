"""Original video frames/audio survive description, extraction, repairs and artifacts."""
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import wave

from PIL import Image
import pytest

from mmkgc.adapters.media_knowledge import MediaInputError, SourceMedia, load_source_media
from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner, load_qwen_config
from mmkgc.data import EntityRecord
from mmkgc.io.writer import _stage_artifact
from mmkgc.io.resume import _validate_saved_media_sources
from mmkgc.pipeline import _semantic_model_calls
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT


@pytest.fixture(scope="module")
def media_files(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required for real media decoding checks")
    root = tmp_path_factory.mktemp("av-fixtures")
    with wave.open(str(root / "source.wav"), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(b"\0\0" * 32000)
    Image.new("RGB", (24, 24), "green").save(root / "source.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=red:s=64x48:d=1:r=4", "-f", "lavfi", "-i", "color=c=blue:s=64x48:d=1:r=4",
        "-i", str(root / "source.wav"), "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(root / "source.mp4")], check=True, capture_output=True, timeout=20)
    return root


def record(root, modalities=("video", "audio")):
    files = {"image": "source.png", "video": "source.mp4", "audio": "source.wav"}
    existing = {modality: files[modality] for modality in modalities}
    existing.update({modality + "_summary": "OUTDATED_" + modality for modality in modalities})
    existing["text"] = "PRIVATE_TARGET_TEXT"
    return EntityRecord(1, "AV_CASE", "Clip", "PRIVATE_DESCRIPTION", [], existing, {})


def extraction(modality="audio", refs=None):
    return {"entity_profile": {"primary_type": "clip", "core_identity": "Clip",
                              "evidence_refs": [f"existing:{modality}:original"]},
        "extracted_triples": [{"subject": "Clip", "relation": "contains", "object": "sound at 0-2 seconds",
            "evidence_refs": refs if refs is not None else [f"existing:{modality}:original", f"existing:{modality}_summary"]}],
        "uncertain_aspects": []}


def transport(monkeypatch, replies, on_request=None):
    requests, timeouts = [], []
    def send(req, timeout=None):
        body = json.loads(req.data)
        requests.append(body)
        timeouts.append(timeout)
        if on_request:
            on_request(len(requests))
        reply = replies[len(requests) - 1]
        text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": text}, "finish_reason": finish}]}).encode())
    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", send)
    return requests, timeouts


def planner(root, budget=180, audio_input_type="input_audio"):
    return QwenLocalSemanticStagePlanner(QwenConfig("omni-test", "http://localhost/v1", "TEST_SECRET",
        audio_input_type=audio_input_type, stage1_image_policy="visual_observation_policy_v1"), dataset_root=root, knowledge_stage_timeout_seconds=budget)


def run(client, target, mode="text"):
    return client.build_knowledge(target, mode, coarse_relation_filtering(target, mode, catalogue_only=True))


def payloads(body):
    parts = body["messages"][-1]["content"]
    return [base64.b64decode(part[key]["data"] if key == "input_audio" else part[key]["url"].split(",", 1)[1], validate=True)
            for part in parts for key in ("image_url", "input_audio", "audio_url") if part["type"] == key]


@pytest.mark.parametrize("fmt,mime", [("wav", "audio/wav"), ("mp3", "audio/mpeg")])
@pytest.mark.parametrize("wire_type", ["input_audio", "audio_url"])
def test_audio_wire_payload_decodes_strictly_to_original_bytes(fmt, mime, wire_type):
    original = b"original frozen audio bytes"
    source = SourceMedia("audio", {"media_type": mime, "format": fmt}, original)
    part, = source.request_parts(wire_type)
    assert part["type"] == wire_type
    if wire_type == "input_audio":
        encoded = part["input_audio"]["data"]
        assert part["input_audio"]["format"] == fmt
    else:
        url = part["audio_url"]["url"]
        assert url.startswith(f"data:{mime};base64,")
        encoded = url.split(",", 1)[1]
    assert base64.b64decode(encoded, validate=True) == original


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_two_passes_receive_identical_media_and_save_provenance(media_files, monkeypatch, modality):
    target = record(media_files, (modality,))
    original = deepcopy(target.existing_modalities)
    requests, _ = transport(monkeypatch, ["[0-2 seconds] Source observation.", json.dumps(extraction(modality))])
    client = planner(media_files)
    result = run(client, target)
    assert len(requests) == 2 and payloads(requests[0]) == payloads(requests[1])
    assert "response_format" not in requests[0]
    assert requests[1]["response_format"] == {"type": "json_object"}
    assert target.existing_modalities == original
    first, second = (request["messages"][-1]["content"][0]["text"] for request in requests)
    assert "natural-language" in requests[0]["messages"][0]["content"]
    assert "OUTDATED_" not in second and "PRIVATE_" not in first + second
    assert "Rewatch or relisten" in second and "not independent confirmation" in second
    assert result["source_inputs"][modality + "_summary"] == "OUTDATED_" + modality
    assert result["media_modeling"][modality]["extraction_uses_original_media"]
    assert result["relation_facts"][0]["source_modalities"] == [modality]
    catalog = {item["evidence_id"]: item for item in result["evidence_catalog"]}
    assert catalog[f"existing:{modality}:original"]["locator"]["end_seconds"] == 2.0
    calls = _semantic_model_calls(client, start_index=0)
    assert [call["stage"] for call in calls] == [f"stage1_{modality}_description", "stage1_knowledge"]
    assert calls[0]["model_call_ref"] == calls[1]["media_description_model_call_refs"][modality]
    assert "base64," not in json.dumps(calls) and "TEST_SECRET" not in json.dumps(calls)
    artifact = _stage_artifact("knowledge_modeling", result)["data"]
    assert artifact["media_modeling"] == result["media_modeling"]
    assert artifact["source_inputs"] == result["source_inputs"]
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_video_includes_later_scene_and_full_audio_with_actual_timestamps(media_files):
    source = load_source_media(record(media_files), "video", media_files)
    assert [time for time, _ in source.frames] == [0.0, 1.0]
    colors = [Image.open(io.BytesIO(data)).getpixel((20, 20)) for _, data in source.frames]
    assert colors[0][0] > 200 and colors[0][2] < 20
    assert colors[1][2] > 200 and colors[1][0] < 20
    assert source.audio_payload.startswith(b"RIFF")
    assert source.metadata["audio_sha256"] == hashlib.sha256(source.audio_payload).hexdigest()
    assert "sampled" in source.metadata["input_coverage"] and source.metadata["limitations"]


def test_video_frame_times_keep_the_offset_from_the_audio_timeline(media_files, tmp_path):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-itsoffset", "0.5", "-i",
        str(media_files / "source.mp4"), "-i", str(media_files / "source.wav"), "-map", "0:v:0",
        "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", str(tmp_path / "source.mp4")],
        check=True, capture_output=True, timeout=20)
    source = load_source_media(record(tmp_path, ("video",)), "video", tmp_path)
    assert [timestamp for timestamp, _ in source.frames] == [0.5, 1.5]


def test_silent_video_has_frames_without_inventing_audio(media_files, tmp_path, monkeypatch):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(media_files / "source.mp4"),
        "-an", "-c:v", "copy", str(tmp_path / "source.mp4")], check=True, capture_output=True, timeout=20)
    requests, _ = transport(monkeypatch, ["Red then blue; no audio stream.", json.dumps(extraction("video"))])
    result = run(planner(tmp_path), record(tmp_path, ("video",)))
    assert not result["media_modeling"]["video"]["original_media"]["has_audio_stream"]
    assert all(part["type"] != "input_audio" for request in requests for part in request["messages"][-1]["content"])


def test_image_video_audio_have_separate_descriptions_then_one_joint_extraction(media_files, monkeypatch):
    target = record(media_files, ("image", "video", "audio"))
    audio_extraction = extraction()
    audio_extraction["extracted_triples"][0]["visual_support"] = None
    requests, _ = transport(monkeypatch, ["An image.", "Video timeline.", "Audio timeline.", json.dumps(audio_extraction)])
    client = planner(media_files)
    result = run(client, target)
    assert [call["stage"] for call in client.calls] == ["stage1_image_description", "stage1_video_description",
                                                        "stage1_audio_description", "stage1_knowledge"]
    assert payloads(requests[3]) == sum([payloads(request) for request in requests[:3]], [])
    assert result["image_modeling"] and set(result["media_modeling"]) == {"video", "audio"}
    assert result["source_inputs"]["audio_summary"] == "OUTDATED_audio"
    assert "OUTDATED_" not in requests[-1]["messages"][-1]["content"][0]["text"]


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_extraction_repairs_keep_originals_attached_and_require_original_refs(media_files, monkeypatch, modality):
    requests, _ = transport(monkeypatch, ["Source description.",
        json.dumps(extraction(modality, [f"existing:{modality}_summary"])), json.dumps(extraction(modality))])
    client = planner(media_files)
    run(client, record(media_files, (modality,)))
    assert len(requests) == 3
    assert payloads(requests[0]) == payloads(requests[1]) == payloads(requests[2])
    assert f"missing_original_{modality}_ref" in requests[2]["messages"][-1]["content"][0]["text"]
    assert client.calls[-1]["contract_repair_retries"] == 1


@pytest.mark.parametrize("modality", ["video", "audio"])
@pytest.mark.parametrize("reply", ["", ("Partial source description", "length")])
def test_empty_or_truncated_description_stops(media_files, monkeypatch, modality, reply):
    requests, _ = transport(monkeypatch, [reply])
    client = planner(media_files)
    with pytest.raises(ValueError):
        run(client, record(media_files, (modality,)))
    assert len(requests) == 1 and client.calls[0]["status"] == "failed"
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_source_file_changes_after_description_cannot_change_extraction(media_files, tmp_path, monkeypatch, modality):
    target = record(media_files, (modality,))
    source = tmp_path / target.existing_modalities[modality]
    shutil.copyfile(media_files / source.name, source)
    requests, _ = transport(monkeypatch, ["Source observation.", json.dumps(extraction(modality))],
                            lambda index: source.write_bytes(b"CHANGED") if index == 1 else None)
    run(planner(tmp_path), target)
    assert payloads(requests[0]) == payloads(requests[1])


def test_all_descriptions_and_extraction_share_deadline(media_files, monkeypatch):
    now = {"value": 0.0}
    monkeypatch.setattr("mmkgc.adapters.qwen.time.perf_counter", lambda: now["value"])
    def advance(index): now["value"] = index * 3.0
    _, timeouts = transport(monkeypatch, ["Video observation.", "Audio observation.", json.dumps(extraction())], advance)
    run(planner(media_files, budget=20), record(media_files))
    assert timeouts == [20.0, 17.0, 14.0]


def test_audio_url_wire_dialect_is_explicitly_configurable(media_files, monkeypatch, tmp_path):
    requests, _ = transport(monkeypatch, ["Source observation.", json.dumps(extraction())])
    run(planner(media_files, audio_input_type="audio_url"), record(media_files, ("audio",)))
    assert all(request["messages"][-1]["content"][1]["type"] == "audio_url" for request in requests)
    config = tmp_path / "config.yaml"
    config.write_text("lmm:\n  model_name: test\n  base_url: http://localhost/v1\n  audio_input_type: audio_url\n")
    assert load_qwen_config(config).audio_input_type == "audio_url"
    with pytest.raises(ValueError, match="audio_input_type"):
        QwenConfig("test", "http://localhost", "test", audio_input_type="unsupported")


@pytest.mark.parametrize("asset", ["missing.wav", "../outside.wav", "https://example.com/a.wav", "source.png"])
def test_invalid_audio_asset_fails_before_any_model_call(media_files, monkeypatch, asset):
    target = record(media_files, ("audio",))
    target.existing_modalities["audio"] = asset
    requests, _ = transport(monkeypatch, [])
    with pytest.raises(MediaInputError):
        run(planner(media_files), target)
    assert not requests


def test_oversized_media_is_rejected_without_clipping(media_files, monkeypatch):
    monkeypatch.setattr("mmkgc.adapters.media_knowledge.MAX_MEDIA_BYTES", 16)
    requests, _ = transport(monkeypatch, [])
    with pytest.raises(MediaInputError, match="no source was truncated"):
        run(planner(media_files), record(media_files, ("audio",)))
    assert not requests


def test_long_media_is_rejected_without_using_its_opening_only(media_files, monkeypatch):
    monkeypatch.setattr("mmkgc.adapters.media_knowledge.MAX_MEDIA_DURATION_SECONDS", 1.0)
    requests, _ = transport(monkeypatch, [])
    with pytest.raises(MediaInputError, match="no source was truncated"):
        run(planner(media_files), record(media_files, ("audio",)))
    assert not requests


def test_missing_probe_or_bad_stream_does_not_silently_use_old_summary(media_files, monkeypatch):
    monkeypatch.setattr("mmkgc.adapters.media_knowledge.shutil.which", lambda _: None)
    with pytest.raises(MediaInputError, match="ffprobe"):
        run(planner(media_files), record(media_files, ("audio",)))


def test_target_image_is_masked_while_audio_is_read(media_files, monkeypatch):
    target = record(media_files, ("image", "audio"))
    target.existing_modalities["image"] = "PRIVATE_MISSING.png"
    requests, _ = transport(monkeypatch, ["Audio observation.", json.dumps(extraction())])
    result = run(planner(media_files), target, "image")
    assert "image_modeling" not in result
    assert "PRIVATE_MISSING" not in json.dumps(requests) and "OUTDATED_image" not in json.dumps(requests)
    assert len(requests) == 2


def test_vision_only_qwen_rejects_audio_before_api_instead_of_ignoring_it(media_files, monkeypatch):
    requests, _ = transport(monkeypatch, [])
    client = QwenLocalSemanticStagePlanner(QwenConfig("Qwen/Qwen2-VL-7B-Instruct", "http://localhost", "test"),
                                          dataset_root=media_files)
    with pytest.raises(ValueError, match="cannot listen"):
        run(client, record(media_files, ("audio",)))
    assert not requests


def test_resume_rejects_changed_media_assets_and_bytes(media_files, tmp_path):
    shutil.copyfile(media_files / "source.wav", tmp_path / "source.wav")
    target = record(tmp_path, ("audio",))
    media = load_source_media(target, "audio", tmp_path)
    knowledge = {"media_modeling": {"audio": {"original_media": media.metadata}}}
    _validate_saved_media_sources(knowledge, target)
    target.existing_modalities["audio"] = "other.wav"
    with pytest.raises(ValueError, match="asset differs"):
        _validate_saved_media_sources(knowledge, target)
    target.existing_modalities["audio"] = "source.wav"
    (tmp_path / "source.wav").write_bytes(b"changed audio")
    with pytest.raises(ValueError, match="bytes differ"):
        _validate_saved_media_sources(knowledge, target)
