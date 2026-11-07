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
    from mmkgc.observation import validate_verification_targets
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
