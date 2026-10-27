from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_mkgw_dbpedia_abstracts.py"
spec = importlib.util.spec_from_file_location("export_mkgw_dbpedia_abstracts", SCRIPT_PATH)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
export_abstract_rows = module.export_abstract_rows
dbpedia_uri_to_wikipedia_title = module.dbpedia_uri_to_wikipedia_title
normalize_abstract = module.normalize_abstract


def test_exports_abstracts_aligned_to_entity2id_order(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "MKG-W"
    dataset_dir.mkdir()
    (dataset_dir / "entity2id.txt").write_text(
        "\n".join(
            [
                "http://www.wikidata.org/entity/Q2 0",
                "http://www.wikidata.org/entity/Q1 1",
                "http://www.wikidata.org/entity/Q3 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "ent_links").write_text(
        "\n".join(
            [
                "http://dbpedia.org/resource/Entity_One\thttp://www.wikidata.org/entity/Q1",
                "http://dbpedia.org/resource/Entity_Three\thttp://www.wikidata.org/entity/Q3",
                "http://dbpedia.org/resource/Entity_Two\thttp://www.wikidata.org/entity/Q2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "abstracts.tsv"
    summary = export_abstract_rows(
        dataset_dir=dataset_dir,
        output_path=output_path,
        abstracts_by_dbpedia_uri={
            "http://dbpedia.org/resource/Entity_One": "Entity One is a sample page.",
            "http://dbpedia.org/resource/Entity_Two": "Entity Two is another page.",
        },
    )

    assert summary == {
        "ok": True,
        "entities": 3,
        "with_abstract": 2,
        "missing_abstract": 1,
        "output_path": str(output_path),
    }

    rows = list(csv.DictReader(output_path.open(encoding="utf-8"), delimiter="\t"))
    assert rows == [
        {
            "entity_id": "0",
            "entity_key": "Q2",
            "wikidata_uri": "http://www.wikidata.org/entity/Q2",
            "dbpedia_uri": "http://dbpedia.org/resource/Entity_Two",
            "abstract_status": "found",
            "abstract": "Entity Two is another page.",
        },
        {
            "entity_id": "1",
            "entity_key": "Q1",
            "wikidata_uri": "http://www.wikidata.org/entity/Q1",
            "dbpedia_uri": "http://dbpedia.org/resource/Entity_One",
            "abstract_status": "found",
            "abstract": "Entity One is a sample page.",
        },
        {
            "entity_id": "2",
            "entity_key": "Q3",
            "wikidata_uri": "http://www.wikidata.org/entity/Q3",
            "dbpedia_uri": "http://dbpedia.org/resource/Entity_Three",
            "abstract_status": "missing",
            "abstract": "",
        },
    ]


def test_converts_dbpedia_resource_uri_to_wikipedia_title() -> None:
    assert (
        dbpedia_uri_to_wikipedia_title("http://dbpedia.org/resource/Milano_trema:_la_polizia_vuole_giustizia")
        == "Milano_trema:_la_polizia_vuole_giustizia"
    )
    assert dbpedia_uri_to_wikipedia_title("http://dbpedia.org/resource/Chi%C8%99in%C4%83u") == "Chișinău"


def test_normalizes_abstract_to_single_line_text() -> None:
    assert normalize_abstract("First line.\n\nSecond\tline.  Third line.") == "First line. Second line. Third line."
