# MKG-W Both-Modality MVP Design

## Goal

Build an independent `MMKGC` pipeline that implements the six-stage Relation-guided Knowledge Bridger design on the MKG-W dataset for missing modality completion.

The first implementation is a verifiable MVP:

- It supports `missing_mode=text`, `missing_mode=image`, and `missing_mode=both`.
- The default acceptance mode is `both`.
- It processes 100 MKG-W entities by default when `--limit 100` is provided.
- It does not require external model services or network access.
- It generates text candidates, image prompt candidates, and text-image pair rankings with complete provenance.
- It leaves adapter boundaries for later LMM, SDXL, CLIP, BLIP, Sentence-BERT, Gemini embedding, and LLM/VLM judge integrations.

## Source Material

The implementation follows these stage plans:

- `MMKG/0 Coarse_Relation_Filtering.md`
- `MMKG/1 knowledge_modeling_stage_design.md`
- `MMKG/2 graph_constrained_reasoning_stage_design.md`
- `MMKG/3 modality_aware_fine_grained_relation_selection_stage_design.md`
- `MMKG/4 relation_alternation_generation_stage_design.md`
- `MMKG/5 kg_aware_candidate_consistency_ranking_stage_design.md`

The implementation uses the existing MKG-W dataset as a read-only source:

- `KGC/datasets/MKG-W/entity2id.txt`
- `KGC/datasets/MKG-W/relation2id.txt`
- `KGC/datasets/MKG-W/train.txt`
- `KGC/datasets/MKG-W/valid.txt`
- `KGC/datasets/MKG-W/test.txt`
- `KGC/datasets/MKG-W/entity_info.json`
- `KGC/datasets/MKG-W/relation_info.json`
- `KGC/datasets/MKG-W/images/`

## Project Boundary

`MMKGC/` is an independent Python package. It must not modify `KGC/` code or dataset files.

Recommended package layout:

```text
MMKGC/
  mmkgc/
    adapters/
    data/
    io/
    ranking/
    stages/
    run.py
  configs/
  docs/
  outputs/
  tests/
```

The package may reuse the MKG-W file format, but it should own its own data loader so the MVP can evolve independently from the existing `KGC` implementation.

## Task Modes

The command line interface supports:

```text
missing_mode=text
missing_mode=image
missing_mode=both
```

For `missing_mode=text`:

- Input context may include KG, entity label, and existing image path.
- The pipeline generates text candidates and ranks them.

For `missing_mode=image`:

- Input context may include KG, entity label, and existing text/description.
- The pipeline generates image prompt candidates and ranks them.

For `missing_mode=both`:

- Generation input uses only KG structure, entity key, entity label, relation labels, and neighbor labels/descriptions.
- Description and real image path are stored only under `ground_truth_modalities` for inspection and later evaluation.
- The pipeline generates both text candidates and image prompt candidates, then ranks text-image pairs.

The default acceptance path is `missing_mode=both`.

## Core Data Model

Each selected entity becomes an `EntityRecord`:

```json
{
  "entity_id": 0,
  "entity_key": "Q...",
  "label": "...",
  "description": "...",
  "neighbors": [],
  "existing_modalities": {
    "text": null,
    "image": null
  },
  "ground_truth_modalities": {
    "text": "...",
    "image": "..."
  }
}
```

For `both`, `existing_modalities.text` and `existing_modalities.image` are null by design.

Neighbor records preserve direction:

```json
{
  "relation_id": 0,
  "relation": "instance of",
  "neighbor_id": 1,
  "neighbor_key": "Q...",
  "neighbor_label": "film",
  "neighbor_description": "type of creative work",
  "direction": "out"
}
```

## Six-Stage Pipeline

### Stage 0: Coarse Relation Filtering

Input:

- target entity metadata
- available modality summary if present
- missing mode
- one-hop directed KG neighbors

Responsibilities:

- group triples by relation and direction
- remove low-value relations such as IDs, URLs, websites, category pages, and metadata links
- keep at most 3 evidence triples per relation group
- score relation groups with a deterministic high-recall heuristic
- keep Top-8 relations and all relations with `coarse_score >= 3`

Output:

```json
{
  "selected_relations": [],
  "discarded_relations": [],
  "filtered_triples": []
}
```

### Stage 1: Knowledge Modeling

Input:

- selected relations
- filtered triples
- entity metadata
- available modality context if present
- missing mode

Responsibilities:

- build `entity_profile`
- build `modality_observations`
- convert selected relation groups into traceable `relation_facts`
- generate missing-modality clues for text and image
- record unsupported details in `uncertainties`

Output follows the fields from `MMKG/1`:

```json
{
  "entity_profile": {},
  "modality_observations": {},
  "relation_facts": [],
  "missing_modality_clues": [],
  "uncertainties": []
}
```

### Stage 2: Graph-Constrained Reasoning

Responsibilities:

- stratify knowledge into `supported_facts`, `safe_inferences`, and `uncertain_facts`
- construct modality constraints for both `text` and `image`
- create hard and soft `forbidden_content`

Output:

```json
{
  "reasoning_summary": "",
  "supported_facts": [],
  "safe_inferences": [],
  "uncertain_facts": [],
  "modality_constraints": {
    "image": {
      "must_include": [],
      "should_include": [],
      "context_only": [],
      "avoid": []
    },
    "text": {
      "must_include": [],
      "should_include": [],
      "context_only": [],
      "avoid": []
    }
  },
  "forbidden_content": [],
  "reasoning_confidence": "high|medium|low"
}
```

### Stage 3: Fine-Grained Relation Selection

Responsibilities:

- score each candidate relation with the fixed formula from `MMKG/3`
- assign roles: `primary`, `secondary`, `context`, `excluded`
- assign candidate quota
- keep at most 3 primary relations

Default candidate budget:

```text
text candidates = 5
image prompt candidates = 5
```

### Stage 4: Relation Alternation Generation

Responsibilities:

- generate 5 text candidates
- generate 5 image prompt candidates
- preserve provenance for every candidate
- use `global_profile` fallback if relation quotas do not fill the budget

The MVP does not generate image files. Image candidates contain:

- `positive_prompt`
- `negative_prompt`
- `generated_content.type = "image_prompt"`
- `generated_content.value = positive_prompt`

Text candidates contain:

- `prompt_or_instruction`
- `generated_content.type = "text"`
- `generated_content.value = generated text`

### Stage 5: KG-Aware Candidate Consistency Ranking

For single-modality modes:

- rank generated candidates with local deterministic scoring
- output Top-1 and Top-5 breakdown

For `both`:

- rank 5 text candidates
- rank 5 image prompt candidates
- construct Top-3 text x Top-3 image pairs
- rank 9 pairs
- output Top-1 pair and Top-5 pair breakdown

Scoring uses local fallback metrics:

- token cosine for text semantic similarity
- token overlap for relation coverage
- rule-based constraint satisfaction
- rule-based forbidden violation penalty
- pair consistency using text candidate vs image positive prompt token similarity

## Adapter Interfaces

The MVP includes deterministic adapters:

```text
TextGenerator: rule/template text generation
ImagePromptGenerator: rule/template positive and negative prompts
ImageGenerator: no-op
TextEmbedder: token vectors
ImageTextScorer: token overlap between prompt text and reference text
KGJudge: rule-based hard gate and penalties
```

Later adapters can replace these without changing stage orchestration:

```text
TextGenerator -> Qwen/OpenAI-compatible LMM
ImagePromptGenerator -> LMM prompt refiner
ImageGenerator -> SDXL
TextEmbedder -> Sentence-BERT or Gemini Embedding
ImageTextScorer -> CLIP, BLIP, or Gemini multimodal embedding
KGJudge -> LLM/VLM judge
```

## Output Layout

Each run creates:

```text
MMKGC/outputs/
  mkgw_both_YYYYMMDD_HHMMSS/
    config.json
    results.jsonl
    metrics_summary.json
    entities/
      Qxxx/
        0_coarse_relation_filtering.json
        1_knowledge_modeling.json
        2_graph_constrained_reasoning.json
        3_fine_grained_relation_selection.json
        4_relation_alternation_generation.json
        5_kg_aware_pair_ranking.json
        final.json
```

`results.jsonl` contains one compact final result per entity.

Each entity directory contains full stage artifacts for traceability.

## Acceptance Criteria

The MVP is accepted when:

1. `python -m mmkgc.run --dataset-dir ../KGC/datasets/MKG-W --missing-mode both --limit 100` runs from `MMKGC/`.
2. The run does not require network access or external model services.
3. `results.jsonl` contains 100 lines when 100 entities are successfully processed.
4. Each processed entity has six stage JSON files and `final.json`.
5. Each `both` entity has 5 text candidates.
6. Each `both` entity has 5 image prompt candidates.
7. Each `both` entity has 9 text-image pair candidates.
8. Each `both` entity has Top-1 pair and Top-5 pair ranking output.
9. `metrics_summary.json` reports processed count, failed count, average candidate counts, fallback counts, and stage failure counts.
10. Pytest covers data loading, stage schemas, candidate counts, pair ranking, and output file structure.

## Out of Scope for MVP

The MVP does not:

- call Qwen or any LMM
- call SDXL or generate actual image files
- compute CLIP, BLIP, Sentence-BERT, Gemini, or BERTScore metrics
- perform full dataset quality evaluation
- modify `KGC/`

## Self-Review

- Placeholder scan: no unresolved placeholders are present.
- Scope check: the MVP focuses on one independent, testable system: offline six-stage completion over MKG-W.
- Ambiguity check: `both` mode input excludes description and real images from generation context; they remain ground truth metadata only.
