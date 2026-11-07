"""Read-only conversion of historical coverage buckets at the import boundary."""
from copy import deepcopy


def evidence_context(value):
    if isinstance(value, list):
        return [evidence_context(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: evidence_context(item) for key, item in value.items()
              if key not in {'must_include', 'should_include', 'requirement'}}
    if 'must_include' in value or 'should_include' in value:
        checks = list(result.get('checks', []))
        for bucket in ('must_include', 'should_include'):
            for item in value.get(bucket, []):
                item = evidence_context(item)
                if item not in checks:
                    checks.append(item)
        result['checks'] = checks
    return result


def convert_saved_checks(candidate, expected):
    """Validate old scope before attaching source metadata from the saved facts."""
    from mmkgc.observation import validate_verification_targets, validate_repair_targets
    from mmkgc.observation import SEMANTIC_ISSUE_ID
    # Early v3 controller output used this reserved issue ID before adding the
    # explicit kind. Fact/delete IDs occupy separate controller namespaces.
    for target in candidate.get('repair_targets', []):
        if target.get('issue_id') == SEMANTIC_ISSUE_ID:
            target.setdefault('issue_kind', 'semantic')
    if candidate.get('repair_targets') is not None:
        candidate['repair_targets'] = validate_repair_targets(candidate['repair_targets'])
        known_refs = {ref for target in expected for ref in target['evidence_refs']}
        if any(not set(item['evidence_refs']) <= known_refs for item in candidate['repair_targets']):
            raise ValueError('Saved repair issues reference foreign source evidence')
    current = candidate.get('verification_targets', [])
    legacy = any('requirement' in item for item in current)
    if not legacy:
        if validate_verification_targets(current) != expected:
            raise ValueError('Saved candidate checks differ from its original fact plan')
        return
    if any(item.get('requirement') not in {'must_include', 'should_include'} for item in current):
        raise ValueError('Malformed historical coverage requirement')
    converted = validate_verification_targets(evidence_context(deepcopy(current)))
    fields = ('check_id', 'fact_id', 'claim', 'subject', 'relation', 'object', 'direction',
              'verification_mode', 'evidence_refs')
    if ([{key: item[key] for key in fields} for item in converted]
            != [{key: item[key] for key in fields} for item in expected]):
        raise ValueError('Saved candidate checks differ from its original fact plan')
    candidate['verification_targets'] = deepcopy(expected)
    for key in ('generation_context', 'generation_brief', 'used_constraints'):
        if key in candidate:
            candidate[key] = evidence_context(candidate[key])
    candidate.setdefault('metadata', {})['evidence_protocol_migrated'] = True


def recover_saved_repair_targets(candidate, ranking):
    """Early Stage3 writers omitted issues that were retained in Stage4."""
    if not candidate.get('source_candidate_id') or candidate.get('repair_targets') is not None:
        return
    section = ranking.get(candidate['target_modality'] + '_ranking', {})
    matches = [item for bucket in ('ranked_candidates', 'rejected_candidates')
               for item in section.get(bucket, []) if item.get('candidate_id') == candidate['candidate_id']
               and item.get('repair_targets') is not None]
    if not matches:
        raise ValueError('Saved repaired candidate has no fixed-issue record')
    record = matches[0]
    if (record.get('source_candidate_id') != candidate['source_candidate_id'] or
            any(item['repair_targets'] != record['repair_targets'] for item in matches)):
        raise ValueError('Saved repair issue parent or scope differs')
    known_refs = {ref for target in candidate['verification_targets'] for ref in target['evidence_refs']}
    if any(not set(item['evidence_refs']) <= known_refs for item in record['repair_targets']):
        raise ValueError('Saved repair issues reference foreign source evidence')
    candidate['repair_targets'] = deepcopy(record['repair_targets'])
