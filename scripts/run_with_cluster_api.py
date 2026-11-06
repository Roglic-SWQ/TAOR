"""Load private API/proxy settings for one process without sourcing shell startup files."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from urllib.parse import urlsplit

from dotenv import load_dotenv
from scripts.run_wn18rr_api_preflight import resolve_configuration, pace_api_requests, ROOT


def load_proxy_settings(path):
    values = json.loads(Path(path).read_text())
    allowed = {'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'}
    if not isinstance(values, dict) or not values or not set(values) <= allowed:
        raise ValueError('Proxy settings must contain only literal proxy environment entries')
    for value in values.values():
        if not isinstance(value, str):
            raise ValueError('Proxy setting must be a literal URL')
        parsed = urlsplit(value)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError('Proxy setting must be an HTTP or HTTPS URL')
    os.environ.update(values)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--proxy-settings', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--module')
    parser.add_argument('module_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    missing = [str(path) for path in (args.env_file, args.proxy_settings) if not path.is_file()]
    if missing:
        print(json.dumps({'status':'missing_configuration_files','missing_files':missing}))
        raise SystemExit(2)
    load_dotenv(args.env_file, override=False)
    load_proxy_settings(args.proxy_settings)
    config, status = resolve_configuration(ROOT/'configs/independent-visual-verifier.yaml')
    if config is None:
        print(json.dumps(status)); raise SystemExit(2)
    if args.check_only:
        print(json.dumps(status)); return
    if not args.module:
        parser.error('--module is required unless --check-only is set')
    restore = pace_api_requests(status['api_host'], os.environ.get('OPENAI_MIN_REQUEST_INTERVAL_MS','0'))
    previous = sys.argv
    module_args = args.module_args[1:] if args.module_args[:1] == ['--'] else args.module_args
    sys.argv = [args.module, *module_args]
    try:
        runpy.run_module(args.module, run_name='__main__')
    finally:
        restore()
        sys.argv = previous


if __name__ == '__main__':
    main()
