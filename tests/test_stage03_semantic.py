import json

import pytest

from mmkgc.data import EntityRecord
from mmkgc.pipeline import MMKGCPipeline
from mmkgc.stages import (
    coarse_relation_filtering,
    fine_grained_relation_selection,
    constraint_compilation,
    knowledge_modeling,
)
from mmkgc.stages.semantic import (
    SemanticStageParseError,
    StrictJsonStagePlanner,
    _apply_stage1_response,
    build_stage_prompt,
    parse_strict_json_object,
)


def semantic_record() -> EntityRecord:
    relations = [
        ("instance of", "film", "a motion picture work"),
        ("genre", "Drama", "dramatic work"),
        ("director", "Jane Example", "film director"),
    ]
    return EntityRecord(
        entity_id=1,
        entity_key="semantic_probe",
        label="Semantic Probe",
        description="ZZZ_SECRET_DESCRIPTION must not enter semantic prompts.",
        neighbors=[
            {
                "entity_id": index + 2,
                "entity_key": label.lower().replace(" ", "_"),
                "label": label,
                "description": description,
                "relation_id": index + 10,
                "relation": relation,
                "direction": "out",
            }
            for index, (relation, label, description) in enumerate(relations)
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={
            "text": "ZZZ_SECRET_DESCRIPTION must not enter semantic prompts.",
            "image": "secret.jpg",
        },
    )


def prompt_input_payload(prompt: str) -> dict:
    return json.loads(prompt.split("INPUT_JSON:\n", 1)[1])


def prompt_response_shape(prompt: str) -> dict:
    return json.loads(prompt.split("Response JSON shape:\n", 1)[1].split("\nINPUT_JSON:", 1)[0])


def test_semantic_stage_parser_rejects_prose_or_markdown() -> None:
    with pytest.raises(SemanticStageParseError):
        parse_strict_json_object("```json\n{}\n```")

    with pytest.raises(SemanticStageParseError):
        parse_strict_json_object("Here is the JSON: {}")


def test_stage0_prompt_bounds_verbose_modalities_and_neighbors() -> None:
    record = semantic_record()
    record.existing_modalities["text"] = "target " * 2000
    record.neighbors[:] = [
        {
            "entity_key": f"neighbor_{index}",
            "label": f"Neighbor {index}",
            "relation": "related to",
            "direction": "out",
            "description": "neighbor description " * 1000,
        }
        for index in range(30)
    ]

    prompt = build_stage_prompt(
        stage="stage0_coarse",
        record=record,
        missing_mode="image",
        deterministic={"relation_facts": []},
        prior={},
    )
    payload = prompt_input_payload(prompt)
    existing_text = payload["record"]["existing_modalities"]["text"]
    descriptions = [item["description"] for item in payload["record"]["neighbors"]]

    assert len(existing_text) <= 1600
    assert all(description is not None and len(description) <= 320 for description in descriptions)
    assert "target target target" in existing_text
    assert all("neighbor description" in description for description in descriptions)
    assert all("truncated for semantic prompt" in description for description in descriptions)


def test_stage1_rejects_unknown_fact_ids_without_rule_credibility_fallback() -> None:
    from mmkgc.stages.modeling import build_knowledge_input

    record = semantic_record()
    inputs = build_knowledge_input(record, "image", coarse_relation_filtering(record, "image"))
    response = {
        "entity_profile": {
            "primary_type": "film", "core_identity": "a film",
            "secondary_types": [], "semantic_domains": ["cinema"], "evidence_refs": [],
        },
        "text_visual_profile": {},
        "relation_facts": [{"fact_id": "invented", "claim": "An invented fact"}],
    }
    with pytest.raises(ValueError, match="unknown or duplicate fact_id"):
        _apply_stage1_response(inputs, response, missing_mode="image")
