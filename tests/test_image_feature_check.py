"""A source-only pixel recheck can quarantine a fluent but unsupported extraction."""
import json

import pytest

from test_image_knowledge import fixture, transport, text_and_image
from test_entity_visual_features import (assessment, feature, response, planner, run, pixel_reply)
from mmkgc.adapters.image_feature_check import parse_checks, proposed_features, apply_checks
from mmkgc.adapters.image_knowledge import IMAGE_REF
from mmkgc.stages.candidate_planning import planning_facts, fact_selection_prompt
from mmkgc.stages.semantic import Stage1CompletionError, KNOWLEDGE_REQUEST_CONTEXT


def test_uncertain_pixel_property_is_retained_but_not_selected(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    requests = transport(monkeypatch, [json.dumps(assessment()), "CAPTION_UNSUPPORTED_GUESS",
        json.dumps(response([feature()])), pixel_reply(status="uncertain")])
    result = run(planner(tmp_path), target)
    assert planning_facts(result) == []
    fact = result["relation_facts"][0]
    assert fact["visual_observations"][0]["basis"] == "direct"  # original extraction not rewritten
    assert fact["visual_source_checks"][0]["status"] == "uncertain"
    assert fact["generation_eligible"] is False and fact["source_weight"] == 0
    assert any("Pixel recheck uncertain" in gap["reason"] for gap in result["uncertainties"])
    prompt, pixels = text_and_image(requests[-1])
    assert "CAPTION_UNSUPPORTED_GUESS" not in prompt and '"Scene"' not in prompt
    assert pixels == text_and_image(requests[0])[1]


def test_selector_receives_pixel_support_without_unfiltered_caption(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    transport(monkeypatch, [json.dumps(assessment()), "CAPTION_UNSUPPORTED_GUESS",
        json.dumps(response([feature()])), pixel_reply()])
    result = run(planner(tmp_path), target)
    _, raw = fact_selection_prompt("Scene", "text", result, 8)
    assert "CAPTION_UNSUPPORTED_GUESS" not in raw
    assert "Visible shapes agree" not in raw  # no graph-informed gate reasoning
    assert json.loads(raw)["triples"][0]["visual_source_checks"][0]["status"] == "supported"
    assert "CAPTION_UNSUPPORTED_GUESS" in json.dumps(result)  # provenance intact


@pytest.mark.parametrize("reply,code", [("", "empty_response"), (("{", "length"), "output_token_limit")])
def test_incomplete_pixel_check_is_not_repaired_as_json(tmp_path, monkeypatch, reply, code):
    target, _ = fixture(tmp_path)
    requests = transport(monkeypatch, [json.dumps(assessment()), "A blue circle.",
        json.dumps(response([feature()])), reply])
    client = planner(tmp_path)
    with pytest.raises(Stage1CompletionError) as error:
        run(client, target)
    assert error.value.code == code and len(requests) == 4
    assert client.calls[-1]["contract_repair_retries"] == 0
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_missing_pixel_check_repaired_once_with_same_original(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    requests = transport(monkeypatch, [json.dumps(assessment()), "A blue circle.",
        json.dumps(response([feature()])), '{"checks":[]}', pixel_reply()])
    client = planner(tmp_path)
    result = run(client, target)
    assert len(requests) == 5 and len(planning_facts(result)) == 1
    assert client.calls[-1]["contract_repair_retries"] == 1
    assert text_and_image(requests[-1])[1] == text_and_image(requests[-2])[1]


@pytest.mark.parametrize("change", ["unknown_id", "wrong_source", "duplicate", "empty_observation"])
def test_pixel_check_cannot_borrow_sources_or_skip_contract(change):
    value = json.loads(pixel_reply())
    check = value['checks'][0]
    if change == "unknown_id": check['observation_id'] = 'not-in-proposals'
    if change == "wrong_source": check['evidence_refs'] = ['existing:image:002:original']
    if change == "duplicate": value['checks'].append(dict(check))
    if change == "empty_observation": check['observed'] = ' '
    with pytest.raises(ValueError):
        parse_checks(json.dumps(value), [{'observation_id':'mf:0001:visual:0', 'evidence_refs':[IMAGE_REF]}], {IMAGE_REF})


def test_only_reobserved_originals_receive_positive_support():
    originals = ["existing:image:001:original", "existing:image:002:original"]
    captions = [ref.replace(":original", ":description") for ref in originals]
    support = feature()["visual_support"]
    knowledge = {"stage1_v3": {"visual_fact_policy": {}}, "uncertainties": [],
        "evidence_catalog": [
            *[{"evidence_id": ref, "modality": "image", "representation": "original_image"} for ref in originals],
            *[{"evidence_id": ref, "modality": "image", "original_image_ref": original}
              for ref, original in zip(captions, originals)]],
        "relation_facts": [{"fact_id": "mf:0001", "relation": "has", "object": "curved body",
            "evidence_refs": originals + captions,
            "visual_observations": [{**support, "evidence_refs": originals + captions}]}]}
    checks = json.loads(pixel_reply(ref=originals[0]))['checks']
    result = apply_checks(knowledge, proposed_features(knowledge), checks)
    fact = result["relation_facts"][0]
    assert fact["evidence_refs"] == sorted([originals[0], captions[0]])
    assert fact["visual_observations"][0]["evidence_refs"] == originals + captions
    assert fact["generation_eligible"] is True
