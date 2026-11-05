"""Remote configuration and CPU-only preflight gate before spending API/GPU resources."""
import json
from types import SimpleNamespace

from scripts.run_wn18rr_api_preflight import resolve_configuration, api_ranker
from mmkgc.ranking.profiles import required_scorers_for_mode
import scripts.run_wn18rr_api_preflight as api


def config_file(tmp_path):
    path = tmp_path/'api.yaml'
    path.write_text('lmm:\n  model_name: api-model\n  base_url_env: TAOR_TEST_ENDPOINT\n'
                    '  api_key_env: TAOR_TEST_KEY\ninference:\n  max_token: 4096\n')
    return path


def test_missing_api_resources_are_reported_together_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv('TAOR_TEST_ENDPOINT', raising=False)
    monkeypatch.delenv('TAOR_TEST_KEY', raising=False)
    config, status = resolve_configuration(config_file(tmp_path))
    assert config is None
    assert status == {'status': 'missing_configuration', 'missing': ['TAOR_TEST_ENDPOINT', 'TAOR_TEST_KEY']}


def test_ready_status_never_serializes_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv('TAOR_TEST_ENDPOINT', 'https://api.example.test/v1')
    monkeypatch.setenv('TAOR_TEST_KEY', 'private-test-api-key')
    config, status = resolve_configuration(config_file(tmp_path), model='vision-api')
    assert config.model_name == 'vision-api'
    assert config.api_key == 'private-test-api-key'
    assert 'private-test-api-key' not in json.dumps(status)
    assert status['api_host'] == 'api.example.test'


def test_preflight_ranker_has_no_local_models_or_gpu_scorers():
    config = SimpleNamespace(model_name='api-model', max_token=4096)
    ranker, judge = api_ranker(config, 120)
    assert ranker.scorers == {'gpt4omini_judge': judge}
    for key in ('sentence_bert', 'bertscore', 'clip', 'blip_itm'):
        local = getattr(ranker.profile, key)
        assert not local.enabled and not local.required and local.model_dir is None
    for mode in ('text', 'image'):
        assert required_scorers_for_mode(ranker.profile, mode, has_existing_image=True,
                                         has_existing_text=True) == {'gpt4omini_judge'}


def test_service_interval_paces_real_transport_wrapper_without_changing_payloads(monkeypatch):
    clock = [0.0]
    seen = []
    monkeypatch.setattr(api, 'time', SimpleNamespace(monotonic=lambda: clock[0],
                                                   sleep=lambda delay: clock.__setitem__(0, clock[0]+delay)))
    def transport(req, **kwargs):
        seen.append((clock[0],req,kwargs))
        return b'unchanged response'
    monkeypatch.setattr(api.request, 'urlopen', transport)
    restore = api.pace_api_requests('api.example.test', 1000)
    try:
        for url in ('https://api.example.test/a','https://other.example.test/a','https://api.example.test/b'):
            assert api.request.urlopen(url, timeout=12) == b'unchanged response'
    finally:
        restore()
    assert [entry[0] for entry in seen] == [0,0,1]
    assert all(entry[2] == {'timeout':12} for entry in seen)
