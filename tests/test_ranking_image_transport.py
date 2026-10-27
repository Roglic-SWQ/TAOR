from __future__ import annotations

import base64
from io import BytesIO
import json

from PIL import Image
import pytest

from mmkgc.adapters.qwen import QwenConfig
from mmkgc.ranking.factory import _OpenAICompatibleCompletionClient


def image_url(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.parametrize("kind", ["large_rgb", "small_rgb", "large_rgba"])
def test_judge_upload_preserves_dimensions_and_only_compacts_large_rgb_png(monkeypatch, kind):
    source = Image.effect_noise((1024, 1024), 60).convert("RGBA" if kind == "large_rgba" else "RGB")
    if kind == "small_rgb":
        source = Image.new("RGB", (64, 64), (32, 64, 96))
    original = image_url(source)
    captured = []
    client = _OpenAICompatibleCompletionClient(
        QwenConfig("gpt-4o-mini", "http://localhost/v1", "test-key"), timeout_seconds=1,
    )

    def respond(request):
        captured.append(json.loads(request.data))
        return {"choices": [{"message": {"content": "test-result"}}]}

    monkeypatch.setattr(client, "_open_json_with_timeout", respond)
    assert client.complete_multimodal("Inspect this image.", original) == "test-result"
    sent = captured[0]["messages"][1]["content"][1]["image_url"]["url"]
    if kind == "large_rgb":
        assert len(base64.b64decode(original.split(",", 1)[1])) > 512 * 1024
        assert sent.startswith("data:image/jpeg;base64,")
        assert len(sent) < len(original)
    else:
        assert sent == original
    with Image.open(BytesIO(base64.b64decode(sent.split(",", 1)[1]))) as decoded:
        assert decoded.size == source.size
        assert decoded.mode == source.mode
    assert captured[0]["messages"][1]["content"][0]["text"] == "Inspect this image."


def test_judge_preserves_existing_jpeg_or_remote_url(monkeypatch):
    client = _OpenAICompatibleCompletionClient(
        QwenConfig("gpt-4o-mini", "http://localhost/v1", "test-key"), timeout_seconds=1,
    )
    captured = []
    monkeypatch.setattr(client, "_chat_completion", lambda messages: captured.append(messages) or "ok")
    for url in ["https://example.test/image.png", "data:image/jpeg;base64,dGVzdA=="]:
        client.complete_multimodal("Inspect.", url)
        assert captured[-1][1]["content"][1]["image_url"]["url"] == url
