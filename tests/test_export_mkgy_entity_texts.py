import csv
import importlib.util
from pathlib import Path
import sys
import io
import gzip
import json
from urllib.error import HTTPError

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('mkgy_texts', SCRIPTS / 'export_mkgy_entity_texts.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_exact_titles_and_redirects_keep_the_requested_entity_key():
    value = {'query': {'normalized': [{'from': 'Old_Name', 'to': 'Old Name'}],
                       'redirects': [{'from': 'Old Name', 'to': 'New Name'}],
                       'pages': {'1': {'pageid': 1, 'title': 'New Name', 'extract': 'A real\n\nintroduction.',
                                       'pageprops': {'wikibase_item': 'Q1'}}}}}
    rows = exporter.parse_pages(['Old_Name'], value)
    assert set(rows) == {'Old_Name'}
    assert rows['Old_Name']['abstract'] == 'A real introduction.'
    assert rows['Old_Name']['wikidata_uri'].endswith('/Q1')
    assert rows['Old_Name']['source_uri'].endswith('/New_Name')
    assert rows['Old_Name']['status'] == 'found'


def test_missing_pages_and_disambiguations_do_not_become_descriptions():
    value = {'query': {'pages': {
        '-1': {'title': 'Absent', 'missing': ''},
        '2': {'title': 'Ambiguous', 'extract': 'May refer to several people.', 'pageprops': {'disambiguation': ''}},
        '3': {'title': 'Empty'},
    }}}
    rows = exporter.parse_pages(['Absent', 'Ambiguous', 'Empty'], value)
    assert [rows[key]['status'] for key in rows] == ['missing_page', 'disambiguation', 'empty_extract']
    assert not any(row['abstract'] for row in rows.values())


@pytest.mark.parametrize('payload', [{'error': {'code': 'maxlag'}}, {'query': {'pages': {}}, 'continue': {'excontinue': 1}}, {}])
def test_api_failures_and_unconsumed_continuation_are_not_missing_pages(payload):
    with pytest.raises((ValueError, RuntimeError)):
        exporter.parse_pages(['Entity'], payload)


def test_reuse_is_exact_and_tsv_matches_loader_and_id_order(tmp_path):
    source = tmp_path / 'source.tsv'
    source.write_text('entity_id\tentity_key\twikidata_uri\tdbpedia_uri\tabstract_status\tabstract\n'
                      '0\tQ7\thttp://www.wikidata.org/entity/Q7\thttp://dbpedia.org/resource/AC/DC\tfound\tA real music abstract.\n'
                      '1\tQ8\t\thttp://dbpedia.org/resource/Similar_Name\tfound\tWrong entity.\n')
    entries = exporter.reuse_texts(source, {'AC/DC', 'Similar_Name_(film)'})
    assert set(entries) == {'AC/DC'}
    (tmp_path / 'entity2id.txt').write_text('Similar_Name_(film) 1\nAC/DC 0\n')
    (tmp_path / 'test.txt').write_text('AC/DC relation Similar_Name_(film) .\n')
    rows = exporter.entities(tmp_path)
    summary = exporter.export(tmp_path, rows, entries)
    assert summary['with_abstract'] == 1 and summary['test_missing_abstract'] == 1
    with (tmp_path / 'entity_texts.tsv').open(newline='') as stream:
        exported = list(csv.DictReader(stream, delimiter='\t'))
    assert [row['entity_id'] for row in exported] == ['0', '1']
    assert exported[0]['abstract_status'] == 'found'
    assert exported[1]['abstract_status'] == 'missing' and exported[1]['abstract'] == ''
    assert exported[0]['dbpedia_uri'].endswith('/AC/DC')


def test_entity_links_preserve_provided_uris_and_reject_partial_mapping(tmp_path):
    (tmp_path / 'ent_links.tsv').write_text('http://dbpedia.org/resource/Entity_One\tEntity_One\n')
    links, path = exporter.entity_links(tmp_path, {'Entity_One'})
    assert path.name == 'ent_links.tsv'
    assert links == {'Entity_One': 'http://dbpedia.org/resource/Entity_One'}
    with pytest.raises(ValueError, match='match entity2id'):
        exporter.entity_links(tmp_path, {'Entity_One', 'Entity_Two'})


def test_duplicate_entity_links_are_rejected(tmp_path):
    (tmp_path / 'ent_links.tsv').write_text('http://dbpedia.org/resource/A\tA\nhttp://dbpedia.org/resource/B\tA\n')
    with pytest.raises(ValueError, match='Duplicate entity link'):
        exporter.entity_links(tmp_path, {'A'})


@pytest.mark.parametrize('intro', ['The Example may refer to: A film, a song.', 'Example is a term. Example may also refer to:'])
def test_cached_disambiguation_texts_are_rechecked_instead_of_reused(tmp_path, intro):
    source = tmp_path / 'source.tsv'
    source.write_text('entity_id\tentity_key\twikidata_uri\tdbpedia_uri\tabstract_status\tabstract\n'
                      '0\tQ1\t\thttp://dbpedia.org/resource/Example\tfound\t' + intro + '\n')
    assert exporter.reuse_texts(source, {'Example'}) == {}


def test_fetch_accepts_compressed_json_and_keeps_api_errors_distinct(monkeypatch):
    monkeypatch.setattr(exporter, 'throttle', lambda: None)
    payload = {'query': {'pages': {'1': {
        'title': 'Example', 'extract': 'An actual encyclopedia introduction.'}}}}

    def response(req, timeout):
        assert 'exlimit=20' in req.full_url
        stream = io.BytesIO(gzip.compress(json.dumps(payload).encode()))
        stream.headers = {'Content-Encoding': 'gzip'}
        return stream

    monkeypatch.setattr(exporter.request, 'urlopen', response)
    values = exporter.fetch_batch(['Example'], retries=0)
    assert values['Example']['status'] == 'found'

    def forbidden(req, timeout):
        raise HTTPError(req.full_url, 403, 'Forbidden', {}, None)

    monkeypatch.setattr(exporter.request, 'urlopen', forbidden)
    values = exporter.fetch_batch(['Example'], retries=0)
    assert values['Example']['status'] == 'fetch_error'
    assert '403' in values['Example']['error']
