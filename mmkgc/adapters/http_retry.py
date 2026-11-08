"""HTTP failures eligible for the callers' existing bounded transport retries."""
from io import BytesIO
import json
from urllib.parse import urlsplit


def retryable_http_error(exc):
    if exc.code in {408, 429, 500, 502, 503, 504}:
        return True
    if exc.code != 400 or urlsplit(exc.url).hostname in {'localhost', '127.0.0.1', '::1'}:
        return False
    # The remote gateway intermittently rejects an unchanged, valid request
    # with this normalized upstream error. Ordinary validation 400s stay final.
    raw = getattr(exc, '_mmkgc_error_body', None)
    if raw is None:
        try:
            raw = exc.read()
        except (OSError, ValueError):
            return False
        exc._mmkgc_error_body = raw
        exc.fp = BytesIO(raw)  # Preserve the original body for diagnostics.
        exc.read = exc.fp.read
    try:
        payload = json.loads(raw)
        detail = payload.get('error', {}) if isinstance(payload, dict) else {}
        return (isinstance(detail, dict)
                and detail.get('code') == 'invalid_request_format'
                and detail.get('type') == 'invalid_request_error'
                and detail.get('message') ==
                'The request format is invalid or unsupported by the upstream provider.')
    except (ValueError, TypeError):
        return False
