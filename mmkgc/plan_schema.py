"""Candidate-plan versions shared by planning, generation, ranking and persistence."""

CANDIDATE_PLAN_SCHEMA = "candidate_fact_plan_v2"
LEGACY_CANDIDATE_PLAN_SCHEMA = "candidate_fact_plan_v1"
CANDIDATE_PLAN_SCHEMAS = frozenset({CANDIDATE_PLAN_SCHEMA, LEGACY_CANDIDATE_PLAN_SCHEMA})
FACT_SELECTION_SCHEMA = "shared_fact_selection_v1"
FACT_GROUNDED_SCHEMAS = CANDIDATE_PLAN_SCHEMAS | {FACT_SELECTION_SCHEMA}
