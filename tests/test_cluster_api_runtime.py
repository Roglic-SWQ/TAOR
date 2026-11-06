import json
import os
import sys

import pytest

import scripts.run_with_cluster_api as runtime


def settings(tmp_path,monkeypatch):
    for key in ('OPENAI_MODEL','OPENAI_BASE_URL','OPENAI_API_KEY','HTTPS_PROXY','NO_PROXY','no_proxy'):
        monkeypatch.setenv(key,os.environ.get(key,''))
        if key.startswith('OPENAI_'):
            monkeypatch.delenv(key,raising=False)
    env=tmp_path/'api.env'
    env.write_text('OPENAI_MODEL=independent-test\nOPENAI_BASE_URL=https://review.example.test/v1\n'
                   'OPENAI_API_KEY=private-api-test-key\n')
    proxy=tmp_path/'proxy.json'
    proxy.write_text(json.dumps({'HTTPS_PROXY':'http://test-user:private-proxy-test-password@proxy.example.test:8888'}))
    return env,proxy


def test_check_only_does_not_print_private_values(tmp_path,monkeypatch,capsys):
    env,proxy=settings(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['runner','--env-file',str(env),'--proxy-settings',str(proxy),'--check-only'])
    runtime.main()
    output=capsys.readouterr().out
    assert json.loads(output)['status']=='configuration_ready'
    assert 'private-api-test-key' not in output and 'private-proxy-test-password' not in output
    assert os.environ['NO_PROXY']=='127.0.0.1,localhost'


def test_module_args_and_stdout_remain_owned_by_target_module(tmp_path,monkeypatch,capsys):
    env,proxy=settings(tmp_path,monkeypatch)
    args=['runner','--env-file',str(env),'--proxy-settings',str(proxy),
          '--module','mmkgc.doctor','--','--missing-mode','text']
    monkeypatch.setattr(sys,'argv',args)
    restored=[]
    monkeypatch.setattr(runtime,'pace_api_requests',lambda *a:lambda:restored.append(True))
    def execute(name,run_name):
        assert name=='mmkgc.doctor' and run_name=='__main__'
        assert sys.argv==['mmkgc.doctor','--missing-mode','text']
        assert os.environ['OPENAI_API_KEY']=='private-api-test-key'
        print('{"overall_status":"ok"}')
    monkeypatch.setattr(runtime.runpy,'run_module',execute)
    runtime.main()
    assert json.loads(capsys.readouterr().out)=={'overall_status':'ok'}
    assert restored==[True] and sys.argv==args


def test_proxy_file_does_not_set_unrelated_environment(tmp_path):
    path=tmp_path/'proxy.json';path.write_text(json.dumps({'OPENAI_API_KEY':'untrusted'}))
    with pytest.raises(ValueError):runtime.load_proxy_settings(path)
