import json
from pathlib import Path

from mmkgc.data import DatasetStore, MKGWDataStore, build_dataset_store


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_fixture(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    write(
        tmp_path / "entity2id.txt",
        "\n".join(
            [
                "http://www.wikidata.org/entity/Q1 0",
                "http://www.wikidata.org/entity/Q2 1",
                "http://www.wikidata.org/entity/Q3 2",
            ]
        )
        + "\n",
    )
    write(
        tmp_path / "relation2id.txt",
        "\n".join(
            [
                "http://www.wikidata.org/entity/P31 0",
                "http://www.wikidata.org/entity/P17 1",
            ]
        )
        + "\n",
    )
    write(
        tmp_path / "train.txt",
        "\n".join(
            [
                "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P31 http://www.wikidata.org/entity/Q2 .",
                "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P17 http://www.wikidata.org/entity/Q3 .",
            ]
        )
        + "\n",
    )
    write(tmp_path / "valid.txt", "")
    write(tmp_path / "test.txt", "")
    (tmp_path / "entity_info.json").write_text(
        json.dumps(
                {
                    "Q1": {
                        "label": "Example Film",
                        "description": "a fictional film",
                        "image_summary": "A poster-like image with cinematic composition.",
                    },
                    "Q2": {"label": "film", "description": "type of creative work"},
                    "Q3": {"label": "Exampleland", "description": "a country"},
                }
        ),
        encoding="utf-8",
    )
    (tmp_path / "relation_info.json").write_text(
        json.dumps({"P31": "instance of", "P17": "country"}),
        encoding="utf-8",
    )
    image_dir = tmp_path / "images" / "Example Film"
    image_dir.mkdir(parents=True)
    (image_dir / "image.jpg").write_bytes(b"fake")
    return tmp_path


def make_split_fixture(tmp_path: Path) -> Path:
    root = make_fixture(tmp_path)
    write(
        root / "valid.txt",
        "http://www.wikidata.org/entity/Q2 http://www.wikidata.org/entity/P17 http://www.wikidata.org/entity/Q3 .\n",
    )
    write(
        root / "test.txt",
        "http://www.wikidata.org/entity/Q3 http://www.wikidata.org/entity/P31 http://www.wikidata.org/entity/Q2 .\n",
    )
    return root


def test_build_both_record_excludes_modalities_from_generation(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))
    record = store.build_record(0, missing_mode="both")

    assert record.entity_key == "Q1"
    assert record.label == "Example Film"
    assert record.existing_modalities == {"text": None, "image": None, "image_summary": None}
    assert record.modality_completion_policy["text"]["status"] == "masked_for_eval"
    assert record.modality_completion_policy["text"]["action"] == "generate"
    assert record.modality_completion_policy["text"]["can_evaluate_against_ground_truth"] is True
    assert record.modality_completion_policy["image"]["status"] == "masked_for_eval"
    assert record.modality_completion_policy["image"]["action"] == "generate"
    assert record.modality_completion_policy["image"]["can_evaluate_against_ground_truth"] is True


def test_ground_truth_text_and_image_are_present_when_available(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))
    record = store.build_record(0, missing_mode="both")

    assert record.ground_truth_modalities["text"] == "a fictional film"
    assert record.ground_truth_modalities["image"].endswith("image.jpg")


def test_entity_texts_file_overrides_short_description_for_text_modality(tmp_path):
    root = make_fixture(tmp_path)
    write(
        root / "entity_texts.tsv",
        "\n".join(
            [
                "entity_id\tentity_key\twikidata_uri\tdbpedia_uri\tabstract_status\tabstract",
                (
                    "0\tQ1\thttp://www.wikidata.org/entity/Q1\t"
                    "http://dbpedia.org/resource/Example_Film\tfound\t"
                    "Example Film is a longer Wikipedia-derived abstract."
                ),
                (
                    "1\tQ2\thttp://www.wikidata.org/entity/Q2\t"
                    "http://dbpedia.org/resource/Film\tmissing\t"
                ),
            ]
        )
        + "\n",
    )

    store = MKGWDataStore(root)
    q1_record = store.build_record(0, missing_mode="image")
    q2_record = store.build_record(1, missing_mode="image")

    assert q1_record.description == "Example Film is a longer Wikipedia-derived abstract."
    assert q1_record.existing_modalities["text"] == "Example Film is a longer Wikipedia-derived abstract."
    assert q1_record.ground_truth_modalities["text"] == "Example Film is a longer Wikipedia-derived abstract."
    assert q2_record.description == "type of creative work"


def test_text_mode_keeps_existing_image_when_available(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))
    record = store.build_record(0, missing_mode="text")

    assert record.existing_modalities["text"] is None
    assert record.existing_modalities["image"].endswith("image.jpg")
    assert record.existing_modalities["image_summary"] == (
        "A poster-like image with cinematic composition."
    )


def test_image_mode_keeps_existing_text_when_available(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))
    record = store.build_record(0, missing_mode="image")

    assert record.existing_modalities["text"] == "a fictional film"
    assert record.existing_modalities["image"] is None
    assert record.existing_modalities["image_summary"] is None


def test_image_lookup_uses_underscore_label_variant(tmp_path):
    root = make_fixture(tmp_path)
    original = root / "images" / "Example Film" / "image.jpg"
    underscore_dir = root / "images" / "Example_Film"
    underscore_dir.mkdir()
    (underscore_dir / "image.jpg").write_bytes(original.read_bytes())
    original.unlink()
    original.parent.rmdir()

    store = MKGWDataStore(root)
    record = store.build_record(0, missing_mode="both")

    assert Path(record.ground_truth_modalities["image"]).parts[-2:] == ("Example_Film", "image.jpg")


def test_empty_description_is_not_a_text_modality(tmp_path):
    root = make_fixture(tmp_path)
    (root / "entity_info.json").write_text(
        json.dumps({"Q1": {"label": "Example Film", "description": ""}}),
        encoding="utf-8",
    )

    store = MKGWDataStore(root)
    text_record = store.build_record(0, missing_mode="image")

    assert text_record.description == ""
    assert text_record.existing_modalities["text"] is None
    assert text_record.ground_truth_modalities["text"] is None


def test_real_missing_text_has_quality_only_policy(tmp_path):
    root = make_fixture(tmp_path)
    (root / "entity_info.json").write_text(
        json.dumps({"Q1": {"label": "Example Film", "description": ""}}),
        encoding="utf-8",
    )

    store = MKGWDataStore(root)
    record = store.build_record(0, missing_mode="text")

    policy = record.modality_completion_policy["text"]
    assert policy["status"] == "real_missing"
    assert policy["action"] == "generate"
    assert policy["has_ground_truth"] is False
    assert policy["can_evaluate_against_ground_truth"] is False


def test_whitespace_description_is_not_a_text_modality(tmp_path):
    root = make_fixture(tmp_path)
    (root / "entity_info.json").write_text(
        json.dumps({"Q1": {"label": "Example Film", "description": "  \n\t  "}}),
        encoding="utf-8",
    )

    store = MKGWDataStore(root)
    text_record = store.build_record(0, missing_mode="image")

    assert text_record.description == "  \n\t  "
    assert text_record.existing_modalities["text"] is None
    assert text_record.ground_truth_modalities["text"] is None


def test_partial_missing_text_keeps_weak_text_for_enhancement(tmp_path):
    root = make_fixture(tmp_path)
    (root / "entity_info.json").write_text(
        json.dumps(
            {
                "Q1": {
                    "label": "Example Film",
                    "description": "weak stub",
                    "modality_completion_policy": {
                        "text": {
                            "status": "partial_missing",
                            "reason": "description is a weak stub",
                        }
                    },
                },
                "Q2": {"label": "film", "description": "type of creative work"},
                "Q3": {"label": "Exampleland", "description": "a country"},
            }
        ),
        encoding="utf-8",
    )

    store = MKGWDataStore(root)
    record = store.build_record(0, missing_mode="text")

    policy = record.modality_completion_policy["text"]
    assert record.existing_modalities["text"] == "weak stub"
    assert record.ground_truth_modalities["text"] is None
    assert policy["status"] == "partial_missing"
    assert policy["action"] == "enhance"
    assert policy["has_ground_truth"] is False
    assert policy["can_evaluate_against_ground_truth"] is False
    assert policy["reason"] == "description is a weak stub"


def test_missing_metadata_files_use_defaults(tmp_path):
    root = make_fixture(tmp_path)
    (root / "entity_info.json").unlink()
    (root / "relation_info.json").unlink()

    store = MKGWDataStore(root)
    record = store.build_record(0, missing_mode="both")

    assert record.label == "Q1"
    assert record.description == ""
    assert record.ground_truth_modalities["text"] is None
    assert record.neighbors[0]["relation"] == "P31"


def test_blank_metadata_files_use_defaults(tmp_path):
    root = make_fixture(tmp_path)
    (root / "entity_info.json").write_text("", encoding="utf-8")
    (root / "relation_info.json").write_text("\n  \n", encoding="utf-8")

    store = MKGWDataStore(root)
    record = store.build_record(0, missing_mode="both")

    assert record.label == "Q1"
    assert record.description == ""
    assert record.ground_truth_modalities["text"] is None
    assert record.neighbors[0]["relation"] == "P31"


def test_neighbors_preserve_relation_labels_and_direction(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))
    outgoing_record = store.build_record(0, missing_mode="both")
    incoming_record = store.build_record(1, missing_mode="both")

    assert [item["relation"] for item in outgoing_record.neighbors] == ["instance of", "country"]
    assert [item["direction"] for item in outgoing_record.neighbors] == ["out", "out"]
    assert [item["neighbor_description"] for item in outgoing_record.neighbors] == [
        "type of creative work",
        "a country",
    ]
    assert incoming_record.neighbors[0]["relation"] == "instance of"
    assert incoming_record.neighbors[0]["direction"] == "in"
    assert incoming_record.neighbors[0]["neighbor_description"] == "a fictional film"


def test_neighbors_include_bounded_second_hop_facts(tmp_path):
    root = make_fixture(tmp_path)
    write(
        root / "train.txt",
        "\n".join(
            [
                "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P31 http://www.wikidata.org/entity/Q2 .",
                "http://www.wikidata.org/entity/Q2 http://www.wikidata.org/entity/P17 http://www.wikidata.org/entity/Q3 .",
            ]
        )
        + "\n",
    )
    store = MKGWDataStore(root)

    record = store.build_record(0, missing_mode="both")

    first_hop = record.neighbors[0]
    assert first_hop["relation"] == "instance of"
    assert first_hop["neighbors"] == [
        {
            "entity_id": 2,
            "entity_key": "Q3",
            "label": "Exampleland",
            "description": "a country",
            "neighbor_description": "a country",
            "relation_id": 1,
            "relation": "country",
            "direction": "out",
        }
    ]


def test_default_graph_split_uses_train_edges_only(tmp_path):
    store = MKGWDataStore(make_split_fixture(tmp_path))

    assert store.graph_split == "train"
    assert store.graph_filenames == ("train.txt",)
    assert len(store.triples) == 2

    q2_record = store.build_record(1, missing_mode="both")
    q3_record = store.build_record(2, missing_mode="both")

    assert [(item["relation"], item["direction"]) for item in q2_record.neighbors] == [
        ("instance of", "in")
    ]
    assert [(item["relation"], item["direction"]) for item in q3_record.neighbors] == [
        ("country", "in")
    ]


def test_graph_split_can_include_validation_or_all_edges(tmp_path):
    root = make_split_fixture(tmp_path)

    train_valid_store = MKGWDataStore(root, graph_split="train-valid")
    all_store = MKGWDataStore(root, graph_split="all")

    assert train_valid_store.graph_filenames == ("train.txt", "valid.txt")
    assert all_store.graph_filenames == ("train.txt", "valid.txt", "test.txt")
    assert len(train_valid_store.triples) == 3
    assert len(all_store.triples) == 4

    q2_train_valid = train_valid_store.build_record(1, missing_mode="both")
    q3_all = all_store.build_record(2, missing_mode="both")

    assert ("country", "out") in [
        (item["relation"], item["direction"]) for item in q2_train_valid.neighbors
    ]
    assert ("instance of", "out") in [
        (item["relation"], item["direction"]) for item in q3_all.neighbors
    ]


def test_graph_split_rejects_unknown_value(tmp_path):
    try:
        MKGWDataStore(make_fixture(tmp_path), graph_split="test")
    except ValueError as exc:
        assert "graph_split must be one of" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported graph_split")


def test_select_records_respects_limit(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))
    records = store.select_records(missing_mode="both", limit=2)

    assert len(records) == 2
    assert records[0].entity_key == "Q1"


def test_mkgw_store_satisfies_dataset_store_contract(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))

    assert isinstance(store, DatasetStore)


def test_build_dataset_store_returns_mkgw_store(tmp_path):
    store = build_dataset_store("mkgw", make_fixture(tmp_path), graph_split="train")

    assert isinstance(store, MKGWDataStore)
    assert store.graph_split == "train"


def test_build_dataset_store_rejects_unknown_dataset_type(tmp_path):
    try:
        build_dataset_store("unknown", make_fixture(tmp_path), graph_split="train")
    except ValueError as exc:
        assert "dataset_type must be one of" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported dataset type")


def test_select_records_by_keys_preserves_manifest_order(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))

    records = store.select_records_by_keys("both", ["Q3", "Q1"])

    assert [record.entity_key for record in records] == ["Q3", "Q1"]


def test_select_records_by_keys_rejects_unknown_entity_key(tmp_path):
    store = MKGWDataStore(make_fixture(tmp_path))

    try:
        store.select_records_by_keys("both", ["Q1", "Q404"])
    except ValueError as exc:
        assert "unknown entity_key" in str(exc)
        assert "Q404" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown entity key")
