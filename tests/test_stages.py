import json
import math
import sys
from types import SimpleNamespace

from mmkgc.data import EntityRecord
from mmkgc.stages import (
    coarse_relation_filtering,
    fine_grained_relation_selection,
    constraint_compilation,
    knowledge_modeling,
    relation_alternation_generation,
    safer_relation_alternation_generation,
)
from mmkgc.stages.reasoning import build_knowledge_from_v3_artifacts
from mmkgc.stages.generation import _generation_angle_specs
from mmkgc.stages.selection import (
    STAGE2_EMBEDDING_SCORE_METHOD,
    _embedding_modality_fit_anchor_groups,
    _score_fact,
)
from mmkgc.adapters import RuleImagePromptGenerator, RuleTextGenerator
from mmkgc.utils import clamp01, slugify, token_cosine, token_overlap, tokenize, write_json


class RecordingImagePromptGenerator:
    def __init__(self):
        self._pending_call = None

    def generate(self, record_label, entity_type, relation, angle, facts, avoid):
        self._pending_call = {
            "provider": "fake-image-prompt-planner",
            "stage": "relation_alternation_generation_gpt_prompt",
            "modality": "image_prompt",
            "record_label": record_label,
            "entity_type": entity_type,
            "relation": relation,
            "angle": angle,
            "facts": list(facts),
            "avoid": list(avoid),
            "status": "generated",
        }
        return {
            "prompt_or_instruction": "fake prompt planner instruction",
            "positive_prompt": "rich fake image prompt",
            "negative_prompt": ", ".join(avoid),
        }

    def consume_last_call(self, entity_key, candidate_id):
        if self._pending_call is None:
            return None
        call = dict(self._pending_call)
        self._pending_call = None
        call["entity_key"] = entity_key
        call["candidate_id"] = candidate_id
        return call


class FixedStage3EmbeddingScorer:
    def __init__(self):
        self.calls = []

    def similarity(self, left, right):
        self.calls.append((left, right))
        if "unsupported exact face" in right and "specific location" in right:
            raise AssertionError("risk anchor should not be used for Stage3 embedding scoring")
        if "visible object" in right:
            return 0.9
        if "person name" in right:
            return 0.2
        if "date, award" in right:
            return 0.1
        if "specific named entity" in right:
            return 0.8
        if "specific visual or descriptive content" in right:
            return 0.85
        if "background context" in right:
            return 0.2
        if "administrative metadata" in right:
            return 0.1
        if "cast member: Example Actor" in right:
            return 0.9
        return 0.0


def test_sentence_bert_stage2_embedding_scorer_caches_normalized_rescaled_embeddings(monkeypatch):
    from mmkgc.adapters.stage2_embedding import build_stage2_embedding_scorer

    created_models = []

    class FakeSentenceTransformer:
        def __init__(self, model_dir):
            self.model_dir = model_dir
            self.encode_calls = []
            created_models.append(self)

        def encode(self, text, **_kwargs):
            self.encode_calls.append(text)
            return {
                "same": [3.0, 4.0],
                "opposite": [-3.0, -4.0],
            }[text]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    scorer = build_stage2_embedding_scorer(
        "embedding-component",
        "Sentence-BERT/fake-model",
    )

    assert scorer.similarity("same", "same") == 1.0
    assert scorer.similarity("same", "opposite") == 0.0
    assert created_models[0].model_dir == "Sentence-BERT/fake-model"
    assert created_models[0].encode_calls == ["same", "opposite"]


def example_record():
    return EntityRecord(
        entity_id=1,
        entity_key="example_film",
        label="Example Film",
        description="Example Film is a drama film.",
        neighbors=[
            {
                "entity_id": 2,
                "entity_key": "film",
                "label": "film",
                "description": "a motion picture work",
                "relation_id": 10,
                "relation": "instance of",
                "direction": "out",
            },
            {
                "entity_id": 3,
                "entity_key": "tt000000",
                "label": "tt000000",
                "description": "",
                "relation_id": 11,
                "relation": "IMDb ID",
                "direction": "out",
            },
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": "Example Film is a drama film.", "image": "image.jpg"},
    )


def bidirectional_record():
    return EntityRecord(
        entity_id=1,
        entity_key="example_film",
        label="Example Film",
        description="Example Film is a drama film.",
        neighbors=[
            {
                "entity_id": 2,
                "entity_key": "film",
                "label": "film",
                "description": "a motion picture work",
                "relation_id": 10,
                "relation": "influenced by",
                "direction": "out",
            },
            {
                "entity_id": 3,
                "entity_key": "later_work",
                "label": "Later Work",
                "description": "a later film",
                "relation_id": 10,
                "relation": "influenced by",
                "direction": "in",
            },
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": None, "image": None},
    )


def crowded_relation_record():
    neighbors = []
    for index in range(5):
        neighbors.append(
            {
                "entity_id": index + 2,
                "entity_key": f"genre_{index}",
                "label": f"Genre {index}",
                "description": "genre clue",
                "relation_id": 20,
                "relation": "genre",
                "direction": "out",
            }
        )
    return EntityRecord(
        entity_id=1,
        entity_key="example_film",
        label="Example Film",
        description="",
        neighbors=neighbors,
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": None, "image": None},
    )


def empty_record():
    return EntityRecord(
        entity_id=99,
        entity_key="empty",
        label="Empty Entity",
        description="",
        neighbors=[],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": None, "image": None},
    )


def inbound_instance_record():
    return EntityRecord(
        entity_id=1,
        entity_key="target_work",
        label="Target Work",
        description="mystery artifact with sparse metadata",
        neighbors=[
            {
                "entity_id": 2,
                "entity_key": "child_entity",
                "label": "Child Entity",
                "description": "an entity typed by the target",
                "relation_id": 10,
                "relation": "instance of",
                "direction": "in",
            },
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": None, "image": None},
    )


def image_summary_record():
    return EntityRecord(
        entity_id=8,
        entity_key="image_summary_probe",
        label="Image Summary Probe",
        description="Hidden text target should not be used.",
        neighbors=[],
        existing_modalities={
            "text": None,
            "image": "relative_existing_image.jpg",
            "image_summary": "A red sculpture displayed in a museum gallery.",
        },
        ground_truth_modalities={
            "text": "Hidden text target should not be used.",
            "image": "relative_existing_image.jpg",
        },
    )


def multi_type_description_record():
    return EntityRecord(
        entity_id=9,
        entity_key="multi_type_probe",
        label="Multi Type Probe",
        description="",
        neighbors=[
            {
                "entity_id": 2,
                "entity_key": "film",
                "label": "film",
                "description": "a motion picture work",
                "relation_id": 10,
                "relation": "instance of",
                "direction": "out",
            },
            {
                "entity_id": 3,
                "entity_key": "creative_work",
                "label": "creative work",
                "description": "a created cultural work",
                "relation_id": 10,
                "relation": "instance of",
                "direction": "out",
            },
            {
                "entity_id": 4,
                "entity_key": "drama",
                "label": "drama",
                "description": "genre clue",
                "relation_id": 20,
                "relation": "genre",
                "direction": "out",
            },
        ],
        existing_modalities={
            "text": "A theatrical drama film released for festival audiences.",
            "image": None,
        },
        ground_truth_modalities={"text": None, "image": None},
    )


def secret_description_record(missing_mode="both"):
    existing_modalities = {"text": None, "image": None}
    if missing_mode == "text":
        existing_modalities["image"] = "existing.jpg"
    return EntityRecord(
        entity_id=7,
        entity_key="leak_probe",
        label="Leak Probe",
        description="ZZZ_SECRET_MODAL_TEXT should stay in ground truth only.",
        neighbors=[],
        existing_modalities=existing_modalities,
        ground_truth_modalities={
            "text": "ZZZ_SECRET_MODAL_TEXT should stay in ground truth only.",
            "image": "secret.jpg",
        },
    )


def many_relation_record():
    relations = [
        ("instance of", "film", "a motion picture work"),
        ("genre", "Drama", "dramatic work"),
        ("country of origin", "Exampleland", "fictional country"),
        ("director", "Jane Example", "film director"),
        ("part of", "Example Series", "film series"),
    ]
    return EntityRecord(
        entity_id=1,
        entity_key="many_relation_film",
        label="Many Relation Film",
        description="Many Relation Film is a drama film.",
        neighbors=[
            {
                "entity_id": index + 2,
                "entity_key": label.lower().replace(" ", "_"),
                "label": label,
                "description": description,
                "relation_id": index + 20,
                "relation": relation,
                "direction": "out",
            }
            for index, (relation, label, description) in enumerate(relations)
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": None, "image": None},
    )


def test_token_scores_are_deterministic():
    assert token_overlap("drama film from exampleland", ["drama film"]) == 1.0
    assert token_overlap("drama film", ["mountain"]) == 0.0
    assert token_cosine("drama film", "film drama") == 1.0
    assert token_cosine("drama film", "mountain river") == 0.0


def test_slugify_and_clamp():
    assert slugify("country of citizenship") == "country_of_citizenship"
    assert slugify("IMDb ID") == "imdb_id"
    assert clamp01(-1) == 0.0
    assert clamp01(2) == 1.0


def test_tokenize_handles_empty_text():
    assert tokenize(None) == []
    assert tokenize("") == []
    assert tokenize("!!!") == []


def test_token_overlap_ignores_empty_required_terms():
    assert token_overlap("anything", []) == 1.0
    assert token_overlap("drama film", ["", "!!!", "drama film"]) == 1.0


def test_repeated_token_cosine_is_clamped():
    score = token_cosine("film film drama", "film")

    assert 0.0 <= score <= 1.0


def test_write_json_creates_parent_dirs_and_writes_utf8(tmp_path):
    output_path = tmp_path / "nested" / "data.json"
    payload = {"title": "电影", "scores": [1, 2]}

    write_json(output_path, payload)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert "电影" in output_path.read_text(encoding="utf-8")












def test_stage_1_v3_extracts_trustworthy_music_profile_from_existing_text():
    record = EntityRecord(
        entity_id=9,
        entity_key="Q15935",
        label="Kanye West",
        description="",
        neighbors=[
            {
                "entity_id": 2,
                "entity_key": "song_1",
                "label": "Example Song",
                "description": "song",
                "relation_id": 3,
                "relation": "performer",
                "direction": "in",
            },
            {
                "entity_id": 3,
                "entity_key": "album_1",
                "label": "Example Album",
                "description": "album",
                "relation_id": 5,
                "relation": "producer",
                "direction": "in",
            },
        ],
        existing_modalities={
            "text": (
                "Ye (YAY; born Kanye Omari West) is an American rapper, songwriter, "
                "and record producer. His music is associated with hip-hop and stage "
                "performance contexts."
            ),
            "image": None,
            "image_summary": None,
        },
        ground_truth_modalities={"text": None, "image": None},
    )
    coarse = coarse_relation_filtering(record, "image")

    knowledge = knowledge_modeling(record, "image", coarse)

    stage1 = knowledge["stage1_v3"]
    profile = stage1["entity_profile"]
    assert profile["primary_type"] == "music artist"
    assert profile["secondary_types"] == ["rapper", "songwriter", "record producer"]
    assert profile["semantic_domains"] == ["music", "entertainment"]
    assert "ye yay born" not in json.dumps(profile).lower()
    assert profile["confidence"] >= 0.8
    attributes = {item["value"]: item["type"] for item in stage1["key_attributes"]}
    assert attributes["rapper"] == "identity_role"
    assert attributes["record producer"] == "identity_role"
    assert attributes["music performance context"] == "visual_scene"
    attribute_ids = {item["value"]: item["attribute_id"] for item in stage1["key_attributes"]}
    assert attribute_ids["rapper"] == "attr:identity_role:0001"
    assert attribute_ids["record producer"] == "attr:identity_role:0003"
    assert attribute_ids["music performance context"] == "attr:visual_scene:0001"
    assert attribute_ids["music performance context"] != "attr:music_performance_context"
    assert "YAY" not in stage1["text_visual_profile"]["context_cues"]
    assert "avoid" not in stage1["text_visual_profile"]
    assert stage1["modality_state"]["available"] == {
        "text": True,
        "image": False,
        "image_summary": False,
    }
    assert stage1["modality_state"]["targets"] == ["image"]
    payload = json.dumps(stage1)
    assert record.existing_modalities["text"] not in payload


def test_stage_1_extracts_text_visual_profile_for_image_generation():
    record = EntityRecord(
        entity_id=12,
        entity_key="actor_text_probe",
        label="Actor Text Probe",
        description="",
        neighbors=[],
        existing_modalities={
            "text": (
                "Actor Text Probe is an American actor known for film and television "
                "roles in Point Break and Shaft."
            ),
            "image": None,
        },
        ground_truth_modalities={"text": None, "image": None},
    )
    coarse = coarse_relation_filtering(record, "image")

    knowledge = knowledge_modeling(record, "image", coarse)

    profile = knowledge["text_visual_profile"]
    assert profile["source"] == "existing_text"
    assert profile["entity_type_hint"] == "actor"
    assert "cinematic portrait" in profile["visual_scene_cues"]
    assert "film and television context" in profile["context_cues"]
    assert "Point Break" in profile["context_cues"]
    assert profile["source_text_char_len"] == len(record.existing_modalities["text"])
    assert record.existing_modalities["text"] not in json.dumps(profile)
    assert "avoid" not in profile










def test_stage_2_compiles_credibility_policy_into_constraints():
    relation_facts = [
        {
            "fact_id": "rf:direct",
            "relation": "depicts",
            "fact": "Example Film depicts a river",
            "evidence_triples": ["(Example Film, depicts, river)"],
            "credibility_score": 0.9,
            "modality_relevance": ["image"],
            "risk": "low",
        },
        {
            "fact_id": "rf:supporting",
            "relation": "genre",
            "fact": "Example Film has genre drama",
            "evidence_triples": ["(Example Film, genre, drama)"],
            "credibility_score": 0.7,
            "modality_relevance": ["image"],
            "risk": "low",
        },
        {
            "fact_id": "rf:context",
            "relation": "country",
            "fact": "Example Film has country Exampleland",
            "evidence_triples": ["(Example Film, country, Exampleland)"],
            "credibility_score": 0.5,
            "modality_relevance": ["image"],
            "risk": "low",
        },
        {
            "fact_id": "rf:excluded",
            "relation": "logo",
            "fact": "Example Film has an exact logo",
            "evidence_triples": ["(Example Film, logo, Exact Logo)"],
            "credibility_score": 0.95,
            "modality_relevance": ["image"],
            "risk": "high",
        },
    ]
    knowledge = {
        "entity_profile": {
            "type": "film",
            "entity_type": "film",
            "confidence": 0.9,
            "core_identity": "a film",
            "type_evidence": ["(Example Film, instance of, film)"],
        },
        "relation_facts": relation_facts,
        "uncertainties": [],
    }

    reasoning = constraint_compilation("image", knowledge)
    assessments = {
        item["fact_id"]: item
        for item in reasoning["fact_credibility_assessments"]
    }
    assert assessments["rf:direct"]["allowed_use"] == "direct_generation"
    assert assessments["rf:supporting"]["allowed_use"] == "supporting_only"
    assert assessments["rf:context"]["allowed_use"] == "context_only"
    assert assessments["rf:excluded"]["allowed_use"] == "excluded"

    image_constraints = reasoning["modality_constraints"]["image"]
    assert any("depicts" in item["constraint"] for item in image_constraints["must_include"])
    assert any("genre" in item["constraint"] for item in image_constraints["should_include"])
    assert any("country" in item["constraint"] for item in image_constraints["context_only"])
    assert all(
        "logo" not in item["constraint"]
        for bucket in image_constraints.values()
        for item in bucket
    )

    supported_ids = {item.get("fact_id") for item in reasoning["supported_facts"]}
    assert "rf:direct" in supported_ids
    assert "rf:supporting" in supported_ids
    assert "rf:context" not in supported_ids
    assert "rf:excluded" not in supported_ids


def test_stage_3_caps_supporting_only_and_excludes_untrusted_facts():
    knowledge = {
        "entity_profile": {"type": "film", "entity_type": "film", "confidence": 0.9},
        "relation_facts": [
            {
                "fact_id": "rf:direct",
                "relation": "depicts",
                "fact": "Example Film depicts a river",
                "evidence_triples": ["(Example Film, depicts, river)"],
                "credibility_score": 0.9,
                "modality_relevance": ["image"],
                "risk": "low",
            },
            {
                "fact_id": "rf:supporting",
                "relation": "genre",
                "fact": "Example Film has genre drama",
                "evidence_triples": ["(Example Film, genre, drama)"],
                "credibility_score": 0.7,
                "modality_relevance": ["image"],
                "risk": "low",
            },
            {
                "fact_id": "rf:excluded",
                "relation": "logo",
                "fact": "Example Film has exact logo",
                "evidence_triples": ["(Example Film, logo, Exact Logo)"],
                "credibility_score": 0.95,
                "modality_relevance": ["image"],
                "risk": "high",
            },
        ],
        "uncertainties": [],
    }
    reasoning = constraint_compilation("image", knowledge)

    selection = fine_grained_relation_selection(
        "image",
        knowledge,
        reasoning,
        candidate_budget=2,
    )["fine_grained_relation_selection"]

    assert [item["relation"] for item in selection["primary_relations"]] == ["depicts"]
    assert [item["relation"] for item in selection["secondary_relations"]] == ["genre"]
    assert [item["relation"] for item in selection["excluded_relations"]] == ["logo"]






def test_stage_2_prefers_stage1_v3_clean_entity_profile():
    knowledge = {
        "entity_profile": {
            "type": "ye yay born",
            "entity_type": "ye yay born",
            "confidence": 0.6,
            "core_identity": "a ye yay born related to cinema",
        },
        "stage1_v3": {
            "entity_profile": {
                "primary_type": "music artist",
                "secondary_types": ["rapper", "songwriter", "record producer"],
                "semantic_domains": ["music", "entertainment"],
                "core_identity": "a music artist, rapper, songwriter, and record producer",
                "confidence": 0.85,
                "type_confidence": "high",
                "evidence_refs": ["text:description:identity_roles"],
            },
            "text_visual_profile": {
                "visual_scene_cues": ["music performance context"],
                "context_cues": ["music context"],
                "avoid": ["unsupported exact likeness"],
            },
            "uncertain_aspects": [
                {
                    "aspect": "exact_visual_appearance",
                    "reason": "image modality is missing",
                }
            ],
        },
        "relation_facts": [
            {
                "fact_id": "rf:0001",
                "relation": "performer",
                "source_relations": ["performer"],
                "subject": "Example Song",
                "predicate": "performer",
                "object": "Kanye West",
                "fact": "Example Song has performer to Kanye West",
                "fact_type": "graph_relation",
                "evidence_triples": ["(Example Song, performer, Kanye West)"],
                "support_level": "strong",
                "direction": "in",
                "neighbor_entities": ["Example Song"],
                "modality_relevance": ["text", "image"],
            }
        ],
        "text_visual_profile": {},
        "uncertainties": [],
    }

    reasoning = constraint_compilation("image", knowledge)

    payload = json.dumps(reasoning).lower()
    assert "ye yay born" not in payload
    assert reasoning["supported_facts"][0]["fact"] == (
        "a music artist, rapper, songwriter, and record producer has entity type: music artist"
    )
    assert reasoning["modality_constraints"]["image"]["must_include"][0]["constraint"] == (
        "entity type: music artist"
    )
    assert reasoning["safe_inferences"][0]["based_on_facts"] == [
        "entity type: music artist"
    ]


def test_stage_2_v3_removes_redundant_fact_and_uncertainty_fields():
    knowledge = {
        "entity_profile": {
            "type": "music artist",
            "entity_type": "music artist",
            "confidence": 0.85,
            "core_identity": "a music artist",
            "type_evidence": ["text:description:identity_roles"],
        },
        "relation_facts": [
            {
                "fact_id": "rf:0001",
                "relation": "performer",
                "source_relations": ["performer"],
                "fact": "Example Song has performer to Kanye West",
                "fact_type": "graph_relation",
                "evidence_triples": ["(Example Song, performer, Kanye West)"],
                "support_level": "strong",
                "direction": "in",
                "neighbor_entities": ["Example Song"],
                "modality_relevance": ["text", "image"],
                "stage0_fact_ids": ["fact:0001"],
            }
        ],
        "uncertainties": [
            "Exact visual appearance is unsupported unless present in image ground truth or relation evidence.",
            "unsupported exact likeness",
            "unsupported exact face",
            "unsupported exact logos",
            "unsupported exact likeness",
        ],
    }

    reasoning = constraint_compilation("image", knowledge)

    supported = reasoning["supported_facts"][0]
    assert "evidence" not in supported
    assert "evidence_text" not in supported
    assert "modality_usage" not in supported
    assert supported["evidence_refs"] == ["text:description:identity_roles"]

    relation_supported = reasoning["supported_facts"][1]
    assert relation_supported["source_fact_refs"] == ["rf:0001"]
    assert relation_supported["evidence_refs"] == ["fact:0001"]
    assert "modality_usage" not in relation_supported

    uncertainties = reasoning["uncertain_facts"]
    assert len(uncertainties) == 2
    assert uncertainties[0]["uncertainty_type"] == "unsupported_specific_visual_detail"
    assert uncertainties[0]["blocked_items"] == [
        "exact_face",
        "exact_likeness",
        "logo",
    ]
    assert "uncertainty" not in uncertainties[0]
    assert "risk" not in uncertainties[0]


def test_stage_2_can_be_built_from_v3_stage_artifacts():
    stage0 = {
        "relations": [
            {
                "relation_id": "rel:performer:in",
                "relation": "performer",
                "direction": "in",
                "score": 6.0,
                "neighbor_entities": ["Example Song"],
                "fact_ids": ["fact:0001"],
                "decision": "selected",
            }
        ],
        "facts": [
            {
                "fact_id": "fact:0001",
                "subject": "Example Song",
                "predicate": "performer",
                "object": "Kanye West",
                "relation_id": "rel:performer:in",
                "hop": 1,
                "triple": "(Example Song, performer, Kanye West)",
            }
        ],
    }
    stage1 = {
        "entity_profile": {
            "primary_type": "music artist",
            "secondary_types": ["rapper"],
            "semantic_domains": ["music", "entertainment"],
            "core_identity": "a music artist and rapper",
            "confidence": 0.9,
            "type_confidence": "high",
            "evidence_refs": ["text:description:identity_roles"],
        },
        "text_visual_profile": {
            "visual_scene_cues": ["music performance context"],
            "context_cues": ["music context"],
            "avoid": ["unsupported exact likeness"],
        },
        "relation_fact_refs": [
            {
                "fact_id": "rf:0001",
                "relation": "performer",
                "support_level": "strong",
                "modality_relevance": ["text", "image"],
            }
        ],
        "uncertain_aspects": [
            {"aspect": "exact_visual_appearance", "reason": "image modality is missing"}
        ],
    }

    knowledge = build_knowledge_from_v3_artifacts(stage0, stage1)
    reasoning = constraint_compilation("image", knowledge)

    assert knowledge["relation_facts"][0]["object"] == "Kanye West"
    assert knowledge["relation_facts"][0]["stage0_fact_ids"] == ["fact:0001"]
    assert reasoning["modality_constraints"]["image"]["must_include"][0]["constraint"] == (
        "entity type: music artist"
    )
    assert "text visual cue: music performance context" in [
        item["constraint"]
        for item in reasoning["modality_constraints"]["image"]["should_include"]
    ]


def test_stage_1_carries_stage0_fact_refs_into_stage2_supported_facts():
    record = EntityRecord(
        entity_id=99,
        entity_key="actor_fact_ref_probe",
        label="Actor Fact Ref Probe",
        description="Actor Fact Ref Probe is an American actor.",
        neighbors=[
            {
                "entity_id": 100,
                "entity_key": "example_movie",
                "label": "Example Movie",
                "description": "a film",
                "relation_id": 1,
                "relation": "cast member",
                "direction": "in",
            },
            {
                "entity_id": 101,
                "entity_key": "example_movie_2",
                "label": "Example Movie 2",
                "description": "a television film",
                "relation_id": 1,
                "relation": "cast member",
                "direction": "in",
            },
            {
                "entity_id": 102,
                "entity_key": "example_movie_3",
                "label": "Example Movie 3",
                "description": "a drama film",
                "relation_id": 1,
                "relation": "cast member",
                "direction": "in",
            },
        ],
        existing_modalities={
            "text": "Actor Fact Ref Probe is an American actor known for film roles.",
            "image": None,
        },
        ground_truth_modalities={"text": None, "image": None},
    )
    coarse = coarse_relation_filtering(record, "image")
    knowledge = knowledge_modeling(record, "image", coarse)
    reasoning = constraint_compilation("image", knowledge)

    relation_fact = next(
        item
        for item in reasoning["supported_facts"]
        if item.get("fact_type") == "graph_relation"
    )

    assert relation_fact["evidence_refs"] == ["fact:0001", "fact:0002", "fact:0003"]


def test_stage_2_and_4_use_text_visual_profile_for_image_prompts():
    record = EntityRecord(
        entity_id=13,
        entity_key="actor_prompt_probe",
        label="Actor Prompt Probe",
        description="",
        neighbors=[],
        existing_modalities={
            "text": (
                "Actor Prompt Probe is an American actor known for film and television "
                "roles in Point Break and Shaft."
            ),
            "image": None,
        },
        ground_truth_modalities={"text": None, "image": None},
    )
    coarse = coarse_relation_filtering(record, "image")
    knowledge = knowledge_modeling(record, "image", coarse)

    reasoning = constraint_compilation("image", knowledge)

    image_constraints = reasoning["modality_constraints"]["image"]
    should_constraints = [
        item["constraint"] for item in image_constraints["should_include"]
    ]
    context_constraints = [
        item["constraint"] for item in image_constraints["context_only"]
    ]
    assert "text visual cue: cinematic portrait" in should_constraints
    assert "text context cue: Point Break" in context_constraints

    selection = fine_grained_relation_selection("image", knowledge, reasoning, candidate_budget=1)
    output = relation_alternation_generation(
        record,
        "image",
        knowledge,
        reasoning,
        selection,
        RuleTextGenerator(),
        RuleImagePromptGenerator(),
        candidate_budget=1,
    )

    candidate = output["relation_alternation_generation"]["image_candidates"][0]
    assert "cinematic" in candidate["positive_prompt"]
    assert "text visual cue: cinematic portrait" in candidate["generation_brief"]["should_include"]
    assert "Point Break" in candidate["prompt_or_instruction"]


def test_stage_4_rule_image_prompt_includes_supported_entity_names():
    record = EntityRecord(
        entity_id=13,
        entity_key="Q60493",
        label="Lee Tergesen",
        description="",
        neighbors=[
            {
                "entity_id": 14,
                "entity_key": "movie",
                "label": "The Collection",
                "description": "a film",
                "relation_id": 1,
                "relation": "cast member",
                "direction": "in",
            },
            {
                "entity_id": 15,
                "entity_key": "movie_2",
                "label": "Point Break",
                "description": "a film",
                "relation_id": 1,
                "relation": "cast member",
                "direction": "in",
            },
            {
                "entity_id": 16,
                "entity_key": "movie_3",
                "label": "Shaft",
                "description": "a film",
                "relation_id": 1,
                "relation": "cast member",
                "direction": "in",
            },
        ],
        existing_modalities={
            "text": "Lee Tergesen is an American actor known for film and television roles.",
            "image": None,
        },
        ground_truth_modalities={"text": None, "image": None},
    )
    coarse = coarse_relation_filtering(record, "image")
    knowledge = knowledge_modeling(record, "image", coarse)
    reasoning = constraint_compilation("image", knowledge)
    selection = fine_grained_relation_selection("image", knowledge, reasoning, candidate_budget=1)

    output = relation_alternation_generation(
        record,
        "image",
        knowledge,
        reasoning,
        selection,
        RuleTextGenerator(),
        RuleImagePromptGenerator(),
        candidate_budget=1,
    )

    prompt = output["relation_alternation_generation"]["image_candidates"][0]["positive_prompt"]

    assert "Lee Tergesen" in prompt
    assert "The Collection" in prompt
    assert "generation intent:" in prompt
    assert "diversity axis:" in prompt
    assert "must include:" in prompt
    assert "actor" in prompt.lower()
    assert "cast member context" in prompt


def test_stage_4_collects_image_prompt_planner_model_calls():
    record = example_record()
    coarse = coarse_relation_filtering(record, "image")
    knowledge = knowledge_modeling(record, "image", coarse)
    reasoning = constraint_compilation("image", knowledge)
    selection = fine_grained_relation_selection("image", knowledge, reasoning, candidate_budget=1)
    prompt_generator = RecordingImagePromptGenerator()

    output = relation_alternation_generation(
        record,
        "image",
        knowledge,
        reasoning,
        selection,
        RuleTextGenerator(),
        prompt_generator,
        candidate_budget=1,
    )

    assert len(output["model_calls"]) == 1
    call = output["model_calls"][0]
    assert call["provider"] == "fake-image-prompt-planner"
    assert call["stage"] == "relation_alternation_generation_gpt_prompt"
    assert call["modality"] == "image_prompt"
    assert call["record_label"] == "Example Film"
    assert call["entity_type"] == "film"
    assert call["angle"] == "identity anchor"
    assert call["status"] == "generated"
    assert call["entity_key"] == "example_film"
    assert call["candidate_id"] == "image_001"
    assert "unsupported exact faces" in call["avoid"]


def test_stage_2_assigns_stable_constraint_ids():
    knowledge = {
        "entity_profile": {"entity_type": "person", "type": "person"},
        "relation_facts": [
            {
                "fact_id": "rf:0001",
                "relation": "occupation",
                "fact": "Example Person has occupation: lawyer",
                "evidence_triples": ["(Example Person, occupation, lawyer)"],
                "support_level": "strong",
                "source_relations": ["occupation"],
                "modality_relevance": ["text"],
            }
        ],
    }

    reasoning = constraint_compilation("text", knowledge)

    text_constraints = reasoning["modality_constraints"]["text"]
    assert text_constraints["must_include"][0]["constraint_id"] == (
        "constraint:text:must_include:0001"
    )










def test_stage_1_text_mode_uses_existing_image_summary_not_image_path():
    record = image_summary_record()
    coarse = coarse_relation_filtering(record, "text")
    knowledge = knowledge_modeling(record, "text", coarse)
    payload = json.dumps(knowledge).lower()

    assert [item["attribute"] for item in knowledge["entity_profile"]["key_attributes"][:4]] == [
        "red",
        "sculpture",
        "displayed",
        "museum",
    ]
    assert knowledge["modality_observations"]["image_summary"] == (
        "A red sculpture displayed in a museum gallery."
    )
    assert knowledge["modality_observations"]["image_summary_available"] is True
    assert "relative_existing_image.jpg" in payload
    assert "relative_existing_image jpg" not in [
        item["attribute"] for item in knowledge["entity_profile"]["key_attributes"]
    ]


def test_stage_1_modality_observations_include_completion_policy():
    record = EntityRecord(
        entity_id=1,
        entity_key="partial_text",
        label="Partial Text",
        description="weak stub",
        neighbors=[],
        existing_modalities={"text": "weak stub", "image": None},
        ground_truth_modalities={"text": None, "image": None},
        modality_completion_policy={
            "text": {
                "status": "partial_missing",
                "target": True,
                "action": "enhance",
                "has_existing": True,
                "has_ground_truth": False,
                "can_evaluate_against_ground_truth": False,
            }
        },
    )
    coarse = coarse_relation_filtering(record, "text")

    knowledge = knowledge_modeling(record, "text", coarse)

    assert knowledge["modality_observations"]["completion_policy"]["text"]["status"] == (
        "partial_missing"
    )
    assert knowledge["modality_observations"]["completion_policy"]["text"]["action"] == "enhance"








def test_stage_3_links_relation_plans_to_stage_2_supported_facts():
    knowledge = {
        "entity_profile": {"entity_type": "person", "type": "person"},
        "relation_facts": [
            {
                "fact_id": "rf:0007",
                "relation": "occupation",
                "fact": "Example Person has occupation: lawyer",
                "evidence_triples": ["(Example Person, occupation, lawyer)"],
                "support_level": "strong",
                "direction": "out",
                "neighbor_entities": ["lawyer"],
                "source_relations": ["occupation"],
                "modality_relevance": ["text"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            }
        ],
    }
    reasoning = constraint_compilation("text", knowledge)

    selection = fine_grained_relation_selection("text", knowledge, reasoning, candidate_budget=1)
    selected = selection["fine_grained_relation_selection"]

    relation = selected["primary_relations"][0]
    assert relation["source_fact_refs"] == ["rf:0007"]
    assert relation["supported_fact_refs"] == ["rf:0007"]


def test_stage_3_rule_weighted_v2_scores_relation_specific_constraints_and_risk():
    fact = {
        "fact_id": "rf:0001",
        "relation": "cast member",
        "fact": "Example film has cast member: Example Actor",
        "evidence_triples": [
            "(Example Film, cast member, Example Actor)",
            "(Another Film, cast member, Example Actor)",
        ],
        "support_level": "strong",
        "neighbor_entities": ["Example Actor"],
        "modality_relevance": ["image"],
        "generation_use": "use_as_grounded_constraint",
        "risk": "medium",
    }
    plan = {
        "must_include": [
            {
                "constraint": "cast member: Example Actor",
                "source_relations": ["cast member"],
            }
        ],
        "should_include": [],
        "context_only": [],
        "avoid": [
            {
                "constraint": "unsupported exact faces",
                "source_relations": [],
            }
        ],
    }

    fine_score, components = _score_fact(fact, ["image"], plan)

    assert components == {
        "modality_fit": 0.6,
        "constraint_alignment": 1.0,
        "evidence_strength": 0.95,
        "generation_utility": 0.78,
        "risk_penalty": 0.35,
    }
    assert fine_score == 0.6885


def test_stage_3_rule_weighted_v2_prefers_direct_visual_relations_for_image():
    common_plan = {
        "must_include": [
            {
                "constraint": "grounded image cue",
                "source_relations": [],
            }
        ],
        "should_include": [],
        "context_only": [],
        "avoid": [],
    }
    common_fact = {
        "evidence_triples": ["t1", "t2", "t3"],
        "support_level": "strong",
        "neighbor_entities": ["visual cue"],
        "modality_relevance": ["image"],
        "generation_use": "use_as_grounded_constraint",
        "risk": "low",
    }

    depicts_score, depicts_components = _score_fact(
        {**common_fact, "relation": "depicts"},
        ["image"],
        common_plan,
    )
    credit_score, credit_components = _score_fact(
        {**common_fact, "relation": "composer"},
        ["image"],
        common_plan,
    )

    assert depicts_components["modality_fit"] == 1.0
    assert credit_components["modality_fit"] == 0.6
    assert credit_components["risk_penalty"] == 0.1
    assert depicts_score > credit_score


def test_stage_3_embedding_component_formula_scores_only_semantic_components():
    knowledge = {
        "entity_profile": {"entity_type": "film", "type": "film"},
        "relation_facts": [
            {
                "fact_id": "rf:0001",
                "relation": "cast member",
                "fact": "Example Film has cast member: Example Actor",
                "evidence_triples": [
                    "(Example Film, cast member, Example Actor)",
                    "(Another Film, cast member, Example Actor)",
                ],
                "support_level": "strong",
                "neighbor_entities": ["Example Actor"],
                "modality_relevance": ["image"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "medium",
            }
        ],
    }
    reasoning = {
        "supported_facts": [],
        "modality_constraints": {
            "image": {
                "must_include": [
                    {
                        "constraint": "cast member: Example Actor",
                        "source_relations": ["cast member"],
                    }
                ],
                "should_include": [],
                "context_only": [],
                "avoid": [
                    {
                        "constraint": "unsupported exact faces",
                        "source_relations": [],
                    }
                ],
            }
        },
    }
    scorer = FixedStage3EmbeddingScorer()

    selection = fine_grained_relation_selection(
        "image",
        knowledge,
        reasoning,
        candidate_budget=1,
        embedding_scorer=scorer,
    )

    relation = selection["fine_grained_relation_selection"]["primary_relations"][0]
    assert relation["score_method"] == STAGE2_EMBEDDING_SCORE_METHOD
    assert relation["score_breakdown"] == {
        "modality_fit": 0.435,
        "constraint_alignment": 0.9,
        "evidence_strength": 0.95,
        "generation_utility": 0.4875,
        "risk_penalty": 0.35,
    }
    assert relation["fine_score"] == 0.5555
    assert scorer.calls
    assert all("unsupported exact face" not in right for _, right in scorer.calls)
    serialized = json.dumps(selection)
    assert "embedding_prompt" not in serialized
    assert "raw_vector" not in serialized
    assert "anchor_similarity" not in serialized


def test_stage_3_embedding_modality_fit_uses_entity_specific_positive_anchors():
    class EntityAwareSimilarity:
        def similarity(self, _left, right):
            if "visible representation of a film" in right:
                return 0.9
            if "visible representation of a building" in right:
                return 0.1
            return 0.5

    fact = {
        "relation": "depicts",
        "fact": "Example Film depicts a river scene",
        "neighbor_entities": ["river scene"],
    }
    plan = {"must_include": [], "should_include": [], "context_only": []}
    scorer = EntityAwareSimilarity()

    film_score, _ = _score_fact(
        fact,
        ["image"],
        plan,
        embedding_scorer=scorer,
        entity_context={"primary_type": "film"},
    )
    building_score, _ = _score_fact(
        fact,
        ["image"],
        plan,
        embedding_scorer=scorer,
        entity_context={"primary_type": "building"},
    )

    assert film_score > building_score
    anchor_groups = _embedding_modality_fit_anchor_groups(
        fact,
        ["image"],
        {"primary_type": "film"},
    )
    modality_anchors = [anchor for anchors in anchor_groups.values() for anchor in anchors]
    assert all("creator information" not in anchor for anchor in modality_anchors)
    assert all("administrative metadata" not in anchor for anchor in modality_anchors)


def test_stage_3_clamps_negative_fine_score_to_zero():
    fine_score, components = _score_fact(
        {
            "relation": "cast member",
            "fact": "unsupported exact face logo date",
            "generation_use": "forbidden",
            "risk": "high",
        },
        [],
        {"must_include": [], "should_include": [], "context_only": [], "avoid": []},
    )

    assert components["risk_penalty"] == 0.8
    assert fine_score == 0.0


def test_stage_3_risk_penalty_ignores_free_text_risk_keywords():
    plan = {"must_include": [], "should_include": [], "context_only": [], "avoid": []}
    common_fact = {
        "relation": "depicts",
        "evidence_triples": ["(Example Film, depicts, river scene)"],
        "support_level": "strong",
        "neighbor_entities": ["river scene"],
        "modality_relevance": ["image"],
        "generation_use": "use_as_grounded_constraint",
        "risk": "low",
    }

    _, keyword_components = _score_fact(
        {
            **common_fact,
            "fact": "Example Film won an award at a location with an exact face",
        },
        ["image"],
        plan,
    )
    _, neutral_components = _score_fact(
        {**common_fact, "fact": "Example Film depicts a river scene"},
        ["image"],
        plan,
    )

    assert keyword_components["risk_penalty"] == neutral_components["risk_penalty"] == 0.0


def test_stage_3_full_score_contract_is_point_ninety_five():
    class AlwaysOneSimilarity:
        def similarity(self, left, right):
            return 1.0

    fine_score, components = _score_fact(
        {
            "fact_id": "rf:0001",
            "relation": "depicts",
            "fact": "Example Film depicts a detailed river scene",
            "evidence_triples": ["t1", "t2", "t3"],
            "support_level": "strong",
            "neighbor_entities": ["river scene"],
            "modality_relevance": ["image"],
            "generation_use": "use_as_grounded_constraint",
            "risk": "low",
        },
        ["image"],
        {
            "must_include": [{"constraint": "detailed river scene", "source_relations": ["depicts"]}],
            "should_include": [],
            "context_only": [],
            "avoid": [],
        },
        embedding_scorer=AlwaysOneSimilarity(),
    )

    assert components == {
        "modality_fit": 1.0,
        "constraint_alignment": 1.0,
        "evidence_strength": 1.0,
        "generation_utility": 1.0,
        "risk_penalty": 0.0,
    }
    assert fine_score == 0.95
    assert fine_score <= 1.0


def test_stage_3_embedding_scorer_failures_fall_back_per_relation():
    class MissingSimilarity:
        pass

    class RaisingSimilarity:
        def similarity(self, left, right):
            raise RuntimeError("embedding backend unavailable")

    class NonFiniteSimilarity:
        def __init__(self, value):
            self.value = value

        def similarity(self, left, right):
            return self.value

    knowledge = {
        "entity_profile": {"entity_type": "film", "type": "film"},
        "relation_facts": [
            {
                "relation": "depicts",
                "fact": "Example Film depicts a river scene",
                "evidence_triples": ["(Example Film, depicts, river scene)"],
                "support_level": "strong",
                "neighbor_entities": ["river scene"],
                "modality_relevance": ["image"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            }
        ],
    }
    reasoning = {
        "supported_facts": [],
        "modality_constraints": {
            "image": {"must_include": [], "should_include": [], "context_only": [], "avoid": []}
        },
    }

    for scorer in (
        MissingSimilarity(),
        RaisingSimilarity(),
        NonFiniteSimilarity(math.nan),
        NonFiniteSimilarity(math.inf),
        NonFiniteSimilarity(-math.inf),
    ):
        selection = fine_grained_relation_selection(
            "image",
            knowledge,
            reasoning,
            candidate_budget=1,
            embedding_scorer=scorer,
        )

        relation = selection["fine_grained_relation_selection"]["primary_relations"][0]
        assert relation["score_method"] == STAGE3_SCORE_METHOD
        assert relation["fine_score"] == 0.722


def test_stage_3_embedding_scorer_changes_relation_ranking_and_default_uses_rules():
    class PreferenceScorer:
        def similarity(self, left, right):
            return 0.95 if "preferred relation" in left else 0.05

    knowledge = {
        "entity_profile": {"entity_type": "film", "type": "film"},
        "relation_facts": [
            {
                "relation": "depicts",
                "fact": "Example Film depicts a generic scene",
                "evidence_triples": ["(Example Film, depicts, generic scene)"],
                "support_level": "strong",
                "neighbor_entities": ["generic scene"],
                "modality_relevance": ["image"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            },
            {
                "relation": "composer",
                "fact": "Example Film has preferred relation",
                "evidence_triples": ["(Example Film, composer, Preferred Person)"],
                "support_level": "strong",
                "neighbor_entities": ["Preferred Person"],
                "modality_relevance": ["image"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            },
        ],
    }
    reasoning = {
        "supported_facts": [],
        "modality_constraints": {
            "image": {"must_include": [], "should_include": [], "context_only": [], "avoid": []}
        },
    }

    default_selection = fine_grained_relation_selection("image", knowledge, reasoning, candidate_budget=1)
    embedding_selection = fine_grained_relation_selection(
        "image",
        knowledge,
        reasoning,
        candidate_budget=1,
        embedding_scorer=PreferenceScorer(),
    )

    default_primary = default_selection["fine_grained_relation_selection"]["primary_relations"]
    embedding_primary = embedding_selection["fine_grained_relation_selection"]["primary_relations"]
    assert default_primary[0]["relation"] == "depicts"
    assert all(item["score_method"] == STAGE3_SCORE_METHOD for item in default_primary)
    assert embedding_primary[0]["relation"] == "composer"
    assert all(item["score_method"] == STAGE2_EMBEDDING_SCORE_METHOD for item in embedding_primary)


def test_stage_3_embedding_failure_falls_back_only_for_its_relation():
    class PartiallyFailingSimilarity:
        def similarity(self, left, right):
            if "broken relation" in left:
                raise RuntimeError("embedding backend unavailable for this relation")
            return 0.95 if "preferred relation" in left else 0.05

    knowledge = {
        "entity_profile": {"entity_type": "film", "type": "film"},
        "relation_facts": [
            {
                "relation": "composer",
                "fact": "Example Film has broken relation",
                "evidence_triples": ["(Example Film, composer, Broken Person)"],
                "support_level": "strong",
                "neighbor_entities": ["Broken Person"],
                "modality_relevance": ["image"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            },
            {
                "relation": "depicts",
                "fact": "Example Film depicts a preferred relation",
                "evidence_triples": ["(Example Film, depicts, preferred scene)"],
                "support_level": "strong",
                "neighbor_entities": ["preferred scene"],
                "modality_relevance": ["image"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            },
        ],
    }
    reasoning = {
        "supported_facts": [],
        "modality_constraints": {
            "image": {"must_include": [], "should_include": [], "context_only": [], "avoid": []}
        },
    }

    selection = fine_grained_relation_selection(
        "image",
        knowledge,
        reasoning,
        candidate_budget=1,
        embedding_scorer=PartiallyFailingSimilarity(),
    )

    primary = selection["fine_grained_relation_selection"]["primary_relations"]
    by_relation = {item["relation"]: item for item in primary}
    assert primary[0]["relation"] == "depicts"
    assert by_relation["depicts"]["score_method"] == STAGE2_EMBEDDING_SCORE_METHOD
    assert by_relation["composer"]["score_method"] == STAGE3_SCORE_METHOD




def test_stage_3_blocks_unsafe_relations_before_generation_role_assignment():
    knowledge = {
        "entity_profile": {"entity_type": "person", "type": "person"},
        "relation_facts": [
            {
                "relation": "occupation",
                "fact": "Example Person has occupation: architect",
                "evidence_triples": ["(Example Person, occupation, architect)"],
                "support_level": "strong",
                "direction": "out",
                "neighbor_entities": ["architect"],
                "source_relations": ["occupation"],
                "modality_relevance": ["text"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            },
            {
                "relation": "award received",
                "fact": "Example Person has award received: unsupported prize",
                "evidence_triples": ["(Example Person, award received, unsupported prize)"],
                "support_level": "strong",
                "direction": "out",
                "neighbor_entities": ["unsupported prize"],
                "source_relations": ["award received"],
                "modality_relevance": ["text"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            },
            {
                "relation": "date of birth",
                "fact": "Example Person has date of birth: 1901",
                "evidence_triples": ["(Example Person, date of birth, 1901)"],
                "support_level": "strong",
                "direction": "out",
                "neighbor_entities": ["1901"],
                "source_relations": ["date of birth"],
                "modality_relevance": ["text"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "high",
            },
            {
                "relation": "country of citizenship",
                "fact": "Example Person has country of citizenship: Exampleland",
                "evidence_triples": ["(Example Person, country of citizenship, Exampleland)"],
                "support_level": "strong",
                "direction": "out",
                "neighbor_entities": ["Exampleland"],
                "source_relations": ["country of citizenship"],
                "modality_relevance": ["text"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            },
            {
                "relation": "claim without evidence",
                "fact": "Example Person has claim without evidence: unsourced detail",
                "evidence_triples": [],
                "support_level": "strong",
                "direction": "out",
                "neighbor_entities": ["unsourced detail"],
                "source_relations": ["claim without evidence"],
                "modality_relevance": ["text"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
            },
        ],
    }
    reasoning = {
        "modality_constraints": {
            "text": {
                "must_include": [
                    {
                        "constraint": "occupation: architect",
                        "source_relations": ["occupation"],
                    }
                ],
                "should_include": [],
                "context_only": [
                    {
                        "constraint": "country of citizenship: Exampleland",
                        "source_relations": ["country of citizenship"],
                    }
                ],
                "avoid": [
                    {
                        "constraint": "unsupported award details",
                        "source_relations": ["award received"],
                        "severity": "hard",
                        "gate_action": "hard_forbidden",
                    }
                ],
            },
            "image": {"must_include": [], "should_include": [], "context_only": [], "avoid": []},
        }
    }

    selection = fine_grained_relation_selection(
        "text",
        knowledge,
        reasoning,
        candidate_budget=3,
    )
    selected = selection["fine_grained_relation_selection"]
    by_relation = {
        item["relation"]: item
        for bucket in (
            "primary_relations",
            "secondary_relations",
            "context_relations",
            "excluded_relations",
        )
        for item in selected[bucket]
    }

    assert by_relation["occupation"]["relation_role"] == "primary"
    assert by_relation["award received"]["relation_role"] == "excluded"
    assert by_relation["date of birth"]["relation_role"] == "excluded"
    assert by_relation["claim without evidence"]["relation_role"] == "excluded"
    assert by_relation["country of citizenship"]["relation_role"] == "context"
    generated_roles = selected["primary_relations"] + selected["secondary_relations"]
    assert {item["relation"] for item in generated_roles} == {"occupation"}
    assert selected["quota_summary"]["relation_quotas"] == {"occupation": 3}




def test_stage_4_generation_angles_change_with_entity_and_modality():
    relation_item = {
        "relation": "director",
        "source_fact": {
            "neighbor_entities": ["Jane Example"],
        },
    }
    film_knowledge = {
        "stage1_v3": {
            "entity_profile": {
                "primary_type": "film",
                "semantic_domains": ["cinema"],
                "core_identity": "a film work",
            },
            "text_visual_profile": {
                "visual_scene_cues": ["cinematic scene"],
                "context_cues": ["production context"],
            },
        }
    }
    building_knowledge = {
        "stage1_v3": {
            "entity_profile": {
                "primary_type": "building",
                "semantic_domains": ["architecture"],
                "core_identity": "an architectural structure",
            },
            "text_visual_profile": {
                "visual_scene_cues": ["exterior view"],
                "context_cues": ["urban context"],
            },
        }
    }

    film_image = _generation_angle_specs("image", relation_item, film_knowledge)
    building_image = _generation_angle_specs("image", relation_item, building_knowledge)
    film_text = _generation_angle_specs("text", relation_item, film_knowledge)

    assert [item["angle"] for item in film_image[:2]] == [
        "identity anchor",
        "evidence-grounded description",
    ]
    assert film_image[2]["angle"] == "relation focus: director with Jane Example"
    assert film_image[3]["angle"] == "film attribute and context"
    assert building_image[3]["angle"] == "building attribute and context"
    assert "cinematic scene" in film_image[4]["diversity_axis"]
    assert "exterior view" in building_image[4]["diversity_axis"]
    assert film_image[4]["angle"] == "image semantic bridge for film"
    assert film_text[4]["angle"] == "text semantic bridge for film"
