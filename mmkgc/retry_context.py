"""Per-entity retry diagnostics, kept separate from source evidence."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib

CURRENT_RETRY = ContextVar('taor_retry_diagnostics', default=None)


@contextmanager
def retry_scope(value):
    token = CURRENT_RETRY.set(deepcopy(value) if value else None)
    try:
        yield
    finally:
        CURRENT_RETRY.reset(token)


def current_retry():
    return deepcopy(CURRENT_RETRY.get())


RETRY_GUIDANCE = (
    'This retries a previously unresolved entity. Prior Judge comments and generated '
    'scenes are diagnostics, not new evidence. Keep the intended entity sense and source '
    'qualifiers. Choose one useful core semantic feature rather than covering every fact. '
    'Use the requested rendering strategy when helpful; an illustrative scene does not prove '
    'a named identity or a universal property. Address the actual reported mismatch; avoid '
    'repeating the failed scene or substituting another meaning of the label. Put the central '
    'subject/action early in a short SDXL prompt; keep incidental surroundings simple. '
    'If no candidate was generated, reconsider whether the supplied definition supports '
    'a meaningful illustration even without a unique physical appearance. '
    'A usage example already present in the source may provide a concrete illustration; '
    'keep it an example rather than inventing a new physical mechanism or universal property. '
    'No readable text is needed. The original evidence remains the factual boundary. '
)


class RetrySeededGenerator:
    """Keep one SDXL weight set while varying seeds across entities and rounds."""
    def __init__(self, delegate, round_number, seed_namespace=0):
        self.delegate, self.round_number, self.seed_namespace = delegate, round_number, seed_namespace

    def generate(self, entity_key, candidate_id, positive_prompt, negative_prompt):
        offset = (self.seed_namespace * 1000 + self.round_number) * 1_000_003 + int(hashlib.sha256(entity_key.encode()).hexdigest()[:8], 16)
        artifact = self.delegate.generate(entity_key, candidate_id, positive_prompt, negative_prompt,
                                          seed_offset=offset)
        for metadata in [artifact.metadata, artifact.model_call]:
            metadata.update(retry_round=self.round_number, retry_seed_offset=offset,
                            retry_seed_namespace=self.seed_namespace)
        return artifact
