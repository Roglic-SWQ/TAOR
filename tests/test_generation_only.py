"""Generation-only runs preserve real candidates and never initialize or call judges."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mmkgc.io import RunWriter
from mmkgc.io.resume import load_saved_image_run
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
from mmkgc.plan_schema import FACT_SELECTION_SCHEMA
from mmkgc.run import build_parser
from test_shared_fact_selection import selected, target, Views, Images, reply


def prepared():
    knowledge, selection, reasoning = selected()
    record = replace(target(), entity_key="Q1", entity_id=1)
    return PreparedEntity(record, "image", {"stage0_v3": {"relations": [], "facts": [], "paths": [], "decisions": {}}},
        knowledge, selection, reasoning, [{"stage": "stage2_selection", "status": "applied",
        "response_text": json.dumps(reply()), "response_schema_version": FACT_SELECTION_SCHEMA,
        "max_selected_facts": 8}])


def pipeline(monkeypatch, *, images=None, duplicate=False):
    def forbid(*args, **kwargs):
        raise AssertionError("Stage 4 must not initialize or run")
    monkeypatch.setattr("mmkgc.ranking.factory.build_model_fusion_engine", forbid)
    return MMKGCPipeline(candidate_budget=3, image_prompt_generator=Views(duplicate),
        image_generator=images or Images(), skip_stage4=True)


def test_generation_only_skips_judge_initialization_and_preserves_candidates_for_later_scoring(tmp_path, monkeypatch):
    item = prepared()
    engine = pipeline(monkeypatch)
    result = engine.finish(item)
    assert engine.ranking_engine is None
    assert len(result["generated_image_artifacts"]) == 3
    assert result["ranking_model_calls"] == []
    assert result["final"]["counts"]["image"] == 3 and result["final"]["stage4_deferred"]
    assert result["final"]["best_image_candidate_id"] is None
    assert "stage3_regeneration" not in result["relation_alternation_generation"]["relation_alternation_generation"]
    writer = RunWriter(tmp_path, "generation", {})
    writer.write_entity(result)
    root = writer.entities_dir / "Q1"
    stage4 = json.loads((root / "4_kg_aware_candidate_consistency_ranking.json").read_text())["data"]
    assert stage4["ranking_status"] == "not_evaluated" and stage4["evaluation_skipped"]
    assert "image_ranking" not in stage4
    assert not (root / "ranking_model_calls.jsonl").exists()
    restored, generated = load_saved_image_run(root, item.record, 3)
    assert restored.record == item.record
    assert len(generated["generated_image_artifacts"]) == 3
    assert [image.image_bytes for image in generated["generated_image_artifacts"]] == [
        image.image_bytes for image in result["generated_image_artifacts"]]


def test_batch_checkpoints_all_generated_images_and_resume_checks_their_hashes(tmp_path, monkeypatch):
    pytest.importorskip("fcntl", reason="This batch checkpoint test requires POSIX file locking")
    from test_missing_image_completion import batch, make_job

    job = make_job(tmp_path, 1)
    job.config.update(skip_stage4=True, entity_pipeline=False)
    item = prepared()
    result = pipeline(monkeypatch).finish(item)
    writer = RunWriter(job.root / "attempts", "attempt_0001", {})
    entry = batch.begin_entity(job, job.manifest[0], 1)
    entry["record"] = item.record
    batch.persist_entity_result(job, writer, entry, result)
    batch.finish_entity(job, writer, entry)
    row = job.rows["Q1"]
    assert row["status"] == "generated" and row["generated_image_count"] == 3
    assert not (job.root / "completed_images").exists()
    assert batch.resumable_generated(row, job.root)
    assert list(batch.pending_entities(job, False)) == []
    progress = batch.load(job.root / "progress.json")
    assert progress["generated_entities"] == 1 and progress["generated_image_count"] == 3
    assert progress["attempted_entities"] == 1 and progress["stage4_deferred"]
    assert batch.load(job.root / "failures.json") == []
    Path(row["generated_images"][1]["image_path"]).write_bytes(b"changed")
    assert not batch.resumable_generated(row, job.root)
    assert len(list(batch.pending_entities(job, False))) == 1


def test_generation_only_fills_seed_variants_without_scoring(tmp_path, monkeypatch):
    result = pipeline(monkeypatch, duplicate=True).finish(prepared())
    assert len(result["generated_image_artifacts"]) == 3
    assert result["final"]["counts"]["image"] == 3
    summary=result['relation_alternation_generation']['relation_alternation_generation']['image_prompt_search']
    assert summary['unique_prompt_count']==1 and summary['seed_variant_count']==2
    assert summary['generated_image_count']==3


def test_generation_only_does_not_count_failed_image_metadata_as_a_generated_image(monkeypatch):
    class FailedImages:
        def generate(self, *args):
            artifact = Images().generate(*args)
            return replace(artifact, image_bytes=b"", metadata={"generator": "fake", "error_type": "Failure"})
    result = pipeline(monkeypatch, images=FailedImages()).finish(prepared())
    assert result["final"]["counts"]["image"] == 0
    assert result["final"]["generation_status"] == "generation_empty"


def test_generation_only_is_an_explicit_cli_flag():
    assert build_parser().parse_args(["--skip-stage4"]).skip_stage4
    assert not build_parser().parse_args([]).skip_stage4


@pytest.mark.parametrize("value", [None, "yes", 1])
def test_invalid_generation_only_switch_is_rejected(value):
    with pytest.raises(ValueError, match="skip_stage4"):
        MMKGCPipeline(skip_stage4=value)
