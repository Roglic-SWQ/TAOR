"""Exercise launcher routing without starting model services or GPU workloads."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_mkgw_image_completion_4580.sh"
BACKEND = ROOT / "scripts/complete_mkgw_missing_images.py"


@pytest.fixture
def shell_fixture(tmp_path):
    project = tmp_path / "project with spaces"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / LAUNCHER.name
    shutil.copyfile(LAUNCHER, launcher)
    (scripts / BACKEND.name).write_text("# Python backend is replaced by a process probe in this fixture.\n")
    dataset = project / "datasets/data with spaces"
    dataset.mkdir(parents=True)
    manifest = project / "manifest.json"
    manifest.write_text("[]")
    journal = tmp_path / "calls.jsonl"
    python_probe = tmp_path / "python_probe"
    python_probe.write_text("#!" + sys.executable + "\n" + '''
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with Path(os.environ['PROBE_JOURNAL']).open('a') as f:
    f.write(json.dumps({'args':args,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'pipeline_gpu':os.environ.get('PIPELINE_GPU')})+'\\n')
if '--job-dir' in args:
    root=Path(args[args.index('--job-dir')+1])
    root.mkdir(parents=True)
    (root/'job_config.json').write_text('{}')
    snapshots=root/'config_snapshots'
    snapshots.mkdir()
    (snapshots/'completion_runner.py').write_text('# frozen probe')
if '-' in args and os.environ.get('PROBE_FAIL_HEALTH')=='1' and args[-1]!='1':
    raise SystemExit(7)
if '--worker' in args:
    raise SystemExit(int(os.environ.get('PROBE_WORKER_EXIT','0')))
''')
    python_probe.chmod(0o755)
    job = project / "outputs/test job"
    env = {**os.environ, "MMKGC_PYTHON": str(python_probe), "PROBE_JOURNAL": str(journal),
           "DATASET_DIR": str(dataset), "MANIFEST_PATH": str(manifest), "JOB_DIR": str(job),
           "PIPELINE_GPU": "2", "CANDIDATE_BUDGET": "3"}
    return launcher, job, dataset, manifest, journal, env


def calls(journal):
    return [json.loads(line) for line in journal.read_text().splitlines()]


def test_help_does_not_require_conda_python_or_models():
    result = subprocess.run(["bash", str(LAUNCHER), "--help"], capture_output=True, text=True,
                            env={**os.environ, "MMKGC_PYTHON": "/nonexistent/python"})
    assert result.returncode == 0
    assert "4580" in result.stdout and "--resume" in result.stdout and "--prepare-only" in result.stdout


def test_new_job_preserves_spaced_paths_budget_scope_and_background_gpu(shell_fixture):
    launcher, job, dataset, manifest, journal, env = shell_fixture
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    recorded = calls(journal)
    prepared = recorded[0]["args"]
    assert prepared[prepared.index("--dataset-dir") + 1] == str(dataset)
    assert prepared[prepared.index("--entity-manifest") + 1] == str(manifest)
    assert prepared[prepared.index("--candidate-budget") + 1] == "3"
    assert prepared[prepared.index("--expected-entities") + 1] == "4580"
    assert recorded[1]["args"][-1] == "0"  # Health check is enabled for a real start.
    started = recorded[2]
    assert started["args"][started["args"].index("--worker") + 1] == str(job)
    assert "--launch" in started["args"]
    assert started["gpu"] == "2" and started["pipeline_gpu"] == "2"


def test_prepare_only_never_starts_worker_or_checks_live_service(shell_fixture):
    launcher, job, _, _, journal, env = shell_fixture
    result = subprocess.run(["bash", str(launcher), "--prepare-only"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    recorded = calls(journal)
    assert len(recorded) == 2
    assert recorded[-1]["args"][-1] == "1"
    assert all("--worker" not in row["args"] for row in recorded)
    assert (job / "job_config.json").is_file()


def test_unhealthy_service_preserves_prepared_job_and_does_not_launch(shell_fixture):
    launcher, job, _, _, journal, env = shell_fixture
    env["PROBE_FAIL_HEALTH"] = "1"
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == 7
    assert (job / "job_config.json").is_file()
    assert all("--worker" not in row["args"] for row in calls(journal))


def test_resume_foreground_uses_frozen_worker_retries_and_propagates_exit(shell_fixture):
    launcher, job, _, _, journal, env = shell_fixture
    prepared = subprocess.run(["bash", str(launcher), "--prepare-only"], env=env, capture_output=True, text=True)
    assert prepared.returncode == 0
    journal.write_text("")
    env["PROBE_WORKER_EXIT"] = "6"
    result = subprocess.run(["bash", str(launcher), "--resume", str(job), "--foreground", "--retry-failed"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 6
    recorded = calls(journal)
    assert len(recorded) == 2  # Resume does not prepare a new job.
    command = recorded[-1]["args"]
    assert str(job / "config_snapshots/completion_runner.py") in command
    assert "--retry-failed" in command and "--launch" not in command
    assert (launcher.parent.parent / "outputs/_logs" / (job.name + ".log")).is_file()


def test_detached_resume_reaches_frozen_worker_with_retry_and_gpu(tmp_path):
    root = tmp_path / "job"
    snapshot = root / "config_snapshots"
    snapshot.mkdir(parents=True)
    (root / "job_config.json").write_text(json.dumps({"project_root": str(tmp_path), "prerequisite": None,
                                                    "runtime_kwargs": {"candidate_budget": 3}}))
    (root / "entity_manifest.json").write_text('[{"entity_key":"Q1","entity_id":1}]')
    marker = root / "worker_probe.json"
    (snapshot / "completion_runner.py").write_text('''
import json,os,sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--worker')+1])
(root/'worker_probe.json').write_text(json.dumps({'argv':sys.argv,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}))
''')
    result = subprocess.run([sys.executable, "-B", str(BACKEND), "--worker", str(root), "--launch", "--retry-failed"],
                            capture_output=True, text=True, env={**os.environ, "PIPELINE_GPU": "2"})
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.is_file()
    observed = json.loads(marker.read_text())
    assert "--retry-failed" in observed["argv"]
    assert observed["gpu"] == "2"
    assert json.loads((root / "launch.json").read_text())["pid"] == receipt["pid"]


def test_detached_launch_rejects_existing_worker(tmp_path):
    import fcntl

    root = tmp_path / "job"
    root.mkdir()
    (root / "job_config.json").write_text(json.dumps({"project_root": str(tmp_path), "prerequisite": None}))
    (root / "entity_manifest.json").write_text('[]')
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, "-B", str(BACKEND), "--worker", str(root), "--launch"],
                                capture_output=True, text=True)
    assert result.returncode != 0
    assert "already has a running worker" in result.stderr
    assert not (root / "launch.json").exists()
