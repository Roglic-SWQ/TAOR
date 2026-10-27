from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmkgc import defaults
from mmkgc.run import build_parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("missing_mode", ["text", "image"])
def test_default_runtime_uses_gpt_for_all_language_stages(monkeypatch, tmp_path, missing_mode):
    (tmp_path / ".env").write_text(
        "OPENAI_BASE_URL=https://gpt.example.test/v1\n"
        "OPENAI_API_KEY=project-gpt-key\n"
        "DEFAULT_MODEL=unrelated-default-model\n"
    )
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "DEFAULT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("gpt4omini-mainline.yaml", "openai-gpt4o-mini-judge.yaml"):
        shutil.copyfile(PROJECT_ROOT / "configs" / name, configs / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(defaults, "MMKGC_ROOT", tmp_path)
    reranker = object()
    monkeypatch.setattr(defaults, "Qwen3EvidenceReranker", lambda *args, **kwargs: reranker)
    monkeypatch.setattr(defaults, "MMKGCPipeline", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(defaults, "validate_sdxl_model_dir", lambda path: Path(path))
    monkeypatch.setattr(defaults, "SDXLImageGenerator", lambda config: SimpleNamespace(config=config))

    runtime = defaults.build_default_mainline_runtime(missing_mode=missing_mode)

    components = [runtime.pipeline.semantic_planner, runtime.pipeline.knowledge_planner,
                  runtime.pipeline.relation_planner, runtime.pipeline.text_generator]
    if missing_mode == "image":
        components.append(runtime.pipeline.image_prompt_generator)
    for component in components:
        assert component.config.model_name == "gpt-4o-mini"
        assert component.config.base_url == "https://gpt.example.test/v1"
        assert component.config.api_key == "project-gpt-key"
    assert runtime.pipeline.evidence_reranker is reranker
    assert runtime.pipeline.semantic_planner.planner_name == "gpt4omini-stage-semantic-planner"
    assert runtime.pipeline.knowledge_planner.planner_name == "gpt4omini-stage-semantic-planner"

    args = build_parser().parse_args([])
    assert args.model_config == "configs/gpt4omini-mainline.yaml"
    assert args.stage1_model_config == "configs/gpt4omini-mainline.yaml"
    assert args.stage2_model_config == "configs/openai-gpt4o-mini-judge.yaml"


@pytest.mark.parametrize("size", [4, 10, 50])
@pytest.mark.parametrize("explicit_credentials", [False, True])
def test_launchers_use_gpt_without_a_qwen_service(tmp_path, size, explicit_credentials):
    project = tmp_path / "project"
    scripts = project / "scripts"
    configs = project / "configs"
    binaries = tmp_path / "bin"
    conda_root = tmp_path / "conda"
    for directory in (scripts, configs, binaries, conda_root / "etc/profile.d"):
        directory.mkdir(parents=True)
    script = scripts / f"run_mkgw_image_test_{size}.sh"
    shutil.copyfile(PROJECT_ROOT / "scripts" / script.name, script)
    for name in ("gpt4omini-mainline.yaml", "openai-gpt4o-mini-judge.yaml"):
        shutil.copyfile(PROJECT_ROOT / "configs" / name, configs / name)
    (configs / "stage4-model-fusion-gpu1-visible.yaml").write_text("{}")
    (configs / "mkgw-image-test-10.json").write_text("[]")
    (configs / "mkgw-image-test-50.json").write_text("[]")
    (project / "tmp_mkgw_four_entities_manifest.jsonl").write_text("{}\n")
    (project / "datasets/MKG-W").mkdir(parents=True)
    (project / "datasets/MKG-W bak").mkdir(parents=True)
    (project / "Qwen3-Reranker-4B").mkdir()
    sdxl = project / "SDXL/stable-diffusion-xl-base-1.0"
    sdxl.mkdir(parents=True)
    (sdxl / "model_index.json").write_text("{}")
    (project / ".env").write_text(
        "OPENAI_BASE_URL=https://project-gpt.example.test/v1\n"
        "OPENAI_API_KEY=project-gpt-key\n"
        "DEFAULT_MODEL=unrelated-project-default\n"
    )

    (conda_root / "etc/profile.d/conda.sh").write_text("conda() { :; }\n")
    (binaries / "conda").write_text(
        "#!/bin/sh\nprintf '%s\\n' " + shlex.quote(str(conda_root)) + "\n"
    )
    (binaries / "curl").write_text("#!/bin/sh\nexit 89\n")
    (binaries / "python").write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "from mmkgc.adapters.qwen import load_qwen_config\n"
        "from mmkgc.defaults import load_runtime_environment\n"
        "if sys.argv[1:3] == ['-m', 'mmkgc.run']:\n"
        "    args = sys.argv[3:]\n"
        "    load_runtime_environment(Path.cwd())\n"
        "    result = {}\n"
        "    for flag in ['--model-config', '--stage1-model-config', '--stage2-model-config']:\n"
        "        config = load_qwen_config(args[args.index(flag) + 1])\n"
        "        result[flag] = [config.model_name, config.base_url, config.api_key]\n"
        "    print('ROUTING_JSON=' + json.dumps(result))\n"
    )
    for executable in binaries.iterdir():
        executable.chmod(0o755)

    env = os.environ.copy()
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "DEFAULT_MODEL", "MODEL_CONFIG",
                 "STAGE1_MODEL_CONFIG", "STAGE2_MODEL_CONFIG", "STAGE4_CONFIG",
                 "DATASET_DIR", "MANIFEST_PATH", "SDXL_MODEL_DIR", "STAGE1_RERANKER_MODEL_DIR"):
        env.pop(name, None)
    env.update(
        PATH=str(binaries) + os.pathsep + env.get("PATH", ""),
        PYTHONPATH=str(PROJECT_ROOT),
        QWEN_MODEL_NAME="test-local-qwen",
        QWEN_HOST="127.0.0.2",
        QWEN_PORT="23456",
        QWEN_API_KEY="local-qwen-key",
    )
    if explicit_credentials:
        env.update(
            DEFAULT_MODEL="unrelated-shell-default",
            OPENAI_BASE_URL="https://shell-gpt.example.test/v1",
            OPENAI_API_KEY="shell-gpt-key",
        )
    completed = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=env, text=True,
        capture_output=True, check=True, timeout=15,
    )
    routing = json.loads(next(
        line.removeprefix("ROUTING_JSON=")
        for line in completed.stdout.splitlines() if line.startswith("ROUTING_JSON=")
    ))
    source = "shell" if explicit_credentials else "project"
    for flag in ("--model-config", "--stage1-model-config", "--stage2-model-config"):
        assert routing[flag] == [
            "gpt-4o-mini", f"https://{source}-gpt.example.test/v1", f"{source}-gpt-key",
        ]
