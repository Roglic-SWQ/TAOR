"""Export real English encyclopedia introductions for MKG-Y, with resumable provenance."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import re
import threading
import time
from urllib import error, parse, request

from export_mkgw_dbpedia_abstracts import FIELDS, normalize_abstract, resolve_title_alias

API = 'https://en.wikipedia.org/w/api.php'
USER_AGENT = 'Knowledge-Bridger-MKGY-text-export/1.0 (academic dataset enrichment)'
URI_SAFE = "()_-'!~,:@"
RATE_LOCK = threading.Lock()
NEXT_REQUEST_AT = 0.0


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def title_key(value):
    return parse.unquote(value).replace(' ', '_')


def looks_like_disambiguation(text):
    return bool(re.search(r'\b(?:may|can)\s+(?:also\s+)?refer to\b', text[:180], re.I))


def entities(dataset):
    rows = []
    for number, line in enumerate((dataset / 'entity2id.txt').read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            raise ValueError(f'Invalid entity mapping at line {number}')
        key, ident = parts
        rows.append({'entity_id': int(ident), 'entity_key': key})
    if len({row['entity_id'] for row in rows}) != len(rows) or len({row['entity_key'] for row in rows}) != len(rows):
        raise ValueError('Duplicate entity ID or entity key')
    return sorted(rows, key=lambda row: row['entity_id'])


def entity_links(dataset, keys):
    path = dataset / 'ent_links.tsv'
    if not path.exists():
        path = dataset / 'ent_links'
    links = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for number, row in enumerate(csv.reader(stream, delimiter='\t'), 1):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f'Invalid entity link at line {number}')
            uri, key = row
            if key in links:
                raise ValueError('Duplicate entity link: ' + key)
            if not uri.startswith(('http://dbpedia.org/resource/', 'https://dbpedia.org/resource/')):
                raise ValueError('Expected a DBpedia resource URI: ' + uri)
            links[key] = uri
    if set(links) != keys or len(set(links.values())) != len(links):
        raise ValueError('Entity links must match entity2id exactly and be one-to-one')
    return links, path


def reuse_texts(path, keys):
    result = {}
    if not path or not path.is_file():
        return result
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            uri = row.get('dbpedia_uri', '')
            prefix = 'http://dbpedia.org/resource/'
            if not uri.startswith(prefix):
                continue
            key = title_key(uri[len(prefix):])
            text = normalize_abstract(row.get('abstract') or '')
            if key not in keys or row.get('abstract_status') != 'found' or not text or looks_like_disambiguation(text):
                continue
            result[key] = {'status': 'found', 'abstract': text, 'source': 'MKG-W/entity_texts.tsv',
                           'source_uri': uri, 'wikidata_uri': row.get('wikidata_uri', ''),
                           'dbpedia_uri': uri, 'matched_title': key, 'reused_at': now()}
    return result


def parse_pages(titles, payload):
    if payload.get('error'):
        raise RuntimeError('Wikipedia API error: ' + str(payload['error'].get('code')))
    query = payload.get('query')
    if not isinstance(query, dict) or not isinstance(query.get('pages'), dict):
        raise ValueError('Wikipedia response lacks pages')
    if payload.get('continue'):
        raise ValueError('Unexpected pagination; lower the request batch size')
    aliases = {}
    for section in ('normalized', 'converted', 'redirects'):
        for alias in query.get(section, []):
            aliases[alias['from']] = alias['to']
    pages = {page['title']: page for page in query['pages'].values() if 'title' in page}
    result = {}
    for key in titles:
        title = resolve_title_alias(key, aliases)
        page = pages.get(title)
        if page is None:
            raise ValueError('Response omits requested title: ' + key)
        props = page.get('pageprops', {})
        text = normalize_abstract(page.get('extract') or '')
        status = ('missing_page' if 'missing' in page or 'invalid' in page else
                  'disambiguation' if 'disambiguation' in props else
                  'found' if text else 'empty_extract')
        if status != 'found':
            text = ''
        qid = props.get('wikibase_item', '')
        result[key] = {'status': status, 'abstract': text, 'source': 'wikipedia_intro',
                       'source_uri': 'https://en.wikipedia.org/wiki/' + parse.quote(title.replace(' ', '_'), safe=URI_SAFE),
                       'wikidata_uri': 'http://www.wikidata.org/entity/' + qid if qid else '',
                       'dbpedia_uri': 'http://dbpedia.org/resource/' + parse.quote(key, safe=URI_SAFE),
                       'matched_title': title, 'page_id': page.get('pageid'), 'fetched_at': now()}
    return result


def throttle():
    global NEXT_REQUEST_AT
    with RATE_LOCK:
        delay = max(0, NEXT_REQUEST_AT - time.monotonic())
        NEXT_REQUEST_AT = max(time.monotonic(), NEXT_REQUEST_AT) + 1.0
    if delay:
        time.sleep(delay)


def fetch_batch(titles, timeout=25, retries=2):
    params = {'action': 'query', 'format': 'json', 'prop': 'extracts|pageprops',
              'ppprop': 'wikibase_item|disambiguation', 'exintro': 1, 'explaintext': 1,
              'exlimit': 20, 'redirects': 1, 'maxlag': 5, 'titles': '|'.join(titles)}
    url = API + '?' + parse.urlencode(params)
    for attempt in range(retries + 1):
        try:
            throttle()
            req = request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json',
                                                'Accept-Encoding': 'gzip'})
            with request.urlopen(req, timeout=timeout) as response:
                if response.headers.get('Content-Encoding', '').lower() == 'gzip':
                    with gzip.GzipFile(fileobj=response) as decoded:
                        payload = json.load(decoded)
                else:
                    payload = json.load(response)
            return parse_pages(titles, payload)
        except (error.URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            last_error = f'{type(exc).__name__}: {str(exc)[:240]}'
            if isinstance(exc, error.HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                break
            if attempt < retries:
                delay = min(30, 2 ** (attempt + 1))
                if isinstance(exc, error.HTTPError):
                    value = exc.headers.get('Retry-After', '')
                    if value.isdigit():
                        delay = min(60, max(delay, int(value)))
                time.sleep(delay)
    # A failed HTTP request is kept distinct from an absent encyclopedia page.
    message = locals().get('last_error', 'HTTP request rejected')
    return {key: {'status': 'fetch_error', 'abstract': '', 'source': 'wikipedia_intro',
                  'error': message, 'attempted_at': now()} for key in titles}


def export(dataset, rows, entries, links=None):
    output = dataset / 'entity_texts.tsv'
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            key = row['entity_key']
            entry = entries.get(key, {})
            text = entry.get('abstract', '') if entry.get('status') == 'found' else ''
            writer.writerow({'entity_id': row['entity_id'], 'entity_key': key,
                             'wikidata_uri': entry.get('wikidata_uri', ''),
                             'dbpedia_uri': (links or {}).get(key) or entry.get('dbpedia_uri') or 'http://dbpedia.org/resource/' + parse.quote(key, safe=URI_SAFE),
                             'abstract_status': 'found' if text else 'missing', 'abstract': text})
    temporary.replace(output)
    stats = Counter(entries.get(row['entity_key'], {}).get('status', 'pending') for row in rows)
    keys = {row['entity_key'] for row in rows}
    test = {token for line in (dataset / 'test.txt').read_text(encoding='utf-8').splitlines()
            for token in line.split()[:3:2]}
    test_found = sum(entries.get(key, {}).get('status') == 'found' for key in test)
    summary = {'updated_at': now(), 'entities': len(rows), 'with_abstract': stats['found'],
               'missing_abstract': len(rows) - stats['found'], 'statuses': dict(stats),
               'sources': dict(Counter(entry.get('source', '') for key, entry in entries.items()
                                       if key in keys and entry.get('status') == 'found')),
               'test_entities': len(test), 'test_with_abstract': test_found,
               'test_missing_abstract': len(test) - test_found, 'output': str(output.resolve()),
               'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'alignment': 'Exact entity2id key to English encyclopedia title; normalized titles and redirects allowed; no fuzzy matching.',
               'provenance': str((dataset / 'entity_texts.provenance.json').resolve()),
               'short_descriptions_used_as_abstracts': False, 'llm_generated': False}
    save(dataset / 'entity_texts.summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--reuse-tsv', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 20 or not 1 <= args.workers <= 3:
        parser.error('batch size must be 1–20, workers 1–3')
    dataset = args.dataset_dir.resolve()
    rows = entities(dataset)
    keys = {row['entity_key'] for row in rows}
    links, link_path = entity_links(dataset, keys)
    titles = {key: parse.unquote(uri.split('/resource/', 1)[1]) for key, uri in links.items()}
    if any(title_key(titles[key]) != title_key(key) for key in keys):
        raise ValueError('A mapped DBpedia title differs from the entity key; explicit title mapping required')
    provenance = dataset / 'entity_texts.provenance.json'
    entries = json.loads(provenance.read_text()) if provenance.exists() else {}
    entries = {key: entry for key, entry in entries.items() if key in keys}
    for entry in entries.values():
        if entry.get('source') == 'MKG-W/entity_texts.tsv' and looks_like_disambiguation(entry.get('abstract', '')):
            entry.update(status='pending', abstract='', recheck_reason='Cached text resembles a disambiguation introduction')
    for key, entry in reuse_texts(args.reuse_tsv, keys).items():
        if entries.get(key, {}).get('status') != 'found':
            entries[key] = entry
    save(provenance, entries)
    summary = export(dataset, rows, entries, links)
    summary['entity_links'] = str(link_path)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    missing = [row['entity_key'] for row in rows if entries.get(row['entity_key'], {}).get('status') in {None, 'fetch_error', 'pending'}]
    if args.offline or not missing:
        return
    batches = [missing[start:start + args.batch_size] for start in range(0, len(missing), args.batch_size)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_batch, batch) for batch in batches]
        for index, future in enumerate(as_completed(futures), 1):
            entries.update(future.result())
            if index % 5 == 0 or index == len(batches):
                save(provenance, entries)
            if index % 15 == 0 or index == len(batches):
                summary = export(dataset, rows, entries, links)
                print(json.dumps({'batch': index, 'batches': len(batches), **summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
