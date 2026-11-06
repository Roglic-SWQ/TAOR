#!/usr/bin/env bash
# Execute only in a newly allocated one-GPU job. Never inherit a dev CUDA mask.
set -euo pipefail
PROJECT_ROOT=/medisco/linjiashi/Code/TAOR
ART="${TAOR_REAL_ARTIFACT_ROOT:-${PROJECT_ROOT}/outputs/wn18rr-real-test-20261106}"
ATTEMPT="${TAOR_REAL_ATTEMPT:-$(date -u +%Y%m%dT%H%M%S)}"
LOG_DIR="${ART}/attempt-${ATTEMPT}"
PY=/medisco/linjiashi/venvs/taor-inference/bin/python
SERVER_PY=/medisco/linjiashi/venvs/taor-qwen7b/bin/python
MODEL_LABEL="${TAOR_REAL_MODEL_LABEL:-qwen7b}"
[[ "${MODEL_LABEL}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid model label'; exit 2; }
export TAOR_REAL_MODEL_LABEL="${MODEL_LABEL}"
mkdir -p "${ART}"
mkdir "${LOG_DIR}"
export TAOR_REAL_LOG_DIR="${LOG_DIR}"
exec >"${LOG_DIR}/cluster.log" 2>&1
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi
"${PY}" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
root=Path(os.environ['TAOR_REAL_LOG_DIR'])
files=[p for folder in ('mmkgc','scripts','configs') for p in Path(folder).rglob('*')
       if p.is_file() and p.suffix in {'.py','.yaml','.sh'}]
(root/'execution.json').write_text(json.dumps({'git_head':subprocess.check_output(
    ['/usr/bin/git','rev-parse','HEAD'],text=True).strip(), 'gpu':torch.cuda.get_device_name(0),
    'visible_devices':torch.cuda.device_count(),'source_sha256':{
        str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
PY
if [[ -z "${TAOR_REAL_LANGUAGE_MODEL_CONFIG:-}" ]]; then
"${SERVER_PY}" -m vllm.entrypoints.openai.api_server \
  --model /medisco/linjiashi/models/Qwen/Qwen2-VL-7B-Instruct \
  --served-model-name Qwen/Qwen2-VL-7B-Instruct \
  --host 127.0.0.1 --port 12345 --dtype bfloat16 \
  --middleware mmkgc.adapters.request_deadline.RequestDeadlineMiddleware \
  --gpu-memory-utilization 0.45 --max-model-len 24576 \
  --limit-mm-per-prompt '{"image":4}' --max-num-seqs 2 --enforce-eager \
  >"${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT
for ((attempt=0; attempt<180; attempt++)); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo 'Qwen server exited'; exit 3; fi
  if "${PY}" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:12345/v1/models",timeout=2).read()' 2>/dev/null; then break; fi
  if ((attempt == 179)); then echo 'Qwen startup timed out'; exit 3; fi
  sleep 5
done
fi
RUNNER=("${PY}" -m mmkgc.run)
DOCTOR=("${PY}" -m mmkgc.doctor)
if [[ -n "${TAOR_REAL_API_ENV_FILE:-}" ]]; then
  API_WRAPPER=("${PY}" scripts/run_with_cluster_api.py --env-file "${TAOR_REAL_API_ENV_FILE}"
               --proxy-settings "${TAOR_REAL_PROXY_SETTINGS:?private proxy settings required}")
  RUNNER=("${API_WRAPPER[@]}" --module mmkgc.run --)
  DOCTOR=("${API_WRAPPER[@]}" --module mmkgc.doctor --)
fi
COMMON=(--dataset-type wn18rr --dataset-dir datasets/wn18rr-import-20261106/WN18RR
  --graph-split train --entity-manifest "${ART}/pilot-entity.json" --limit 1
  --stage1-reranker-model-dir /medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B
  --stage1-reranker-device cuda:0 --sdxl-device cuda:0 --candidate-budget 1
  --model-call-timeout-seconds 120 --stage4-config "${TAOR_REAL_STAGE4_CONFIG:-configs/stage4-qwen7b-local.yaml}"
  --output-root "${LOG_DIR}/runs")
if [[ -n "${TAOR_REAL_VISUAL_VERIFIER_CONFIG:-}" ]]; then
  COMMON+=(--stage1-visual-verifier-config "${TAOR_REAL_VISUAL_VERIFIER_CONFIG}")
fi
if [[ -n "${TAOR_REAL_LANGUAGE_MODEL_CONFIG:-}" ]]; then
  COMMON+=(--model-config "${TAOR_REAL_LANGUAGE_MODEL_CONFIG}"
           --stage1-model-config "${TAOR_REAL_LANGUAGE_MODEL_CONFIG}"
           --stage2-model-config "${TAOR_REAL_LANGUAGE_MODEL_CONFIG}")
fi
"${DOCTOR[@]}" "${COMMON[@]}" --qwen-timeout-seconds 20 --missing-mode image >"${LOG_DIR}/doctor.json"
if [[ -n "${TAOR_REAL_FROZEN_FAILURE:-}" ]]; then
  timeout 1200 "${PY}" scripts/check_wn18rr_native_contract.py \
    --failure-file "${TAOR_REAL_FROZEN_FAILURE}" --output-root "${LOG_DIR}/frozen-control"
  exit $?
fi
set +e
timeout 1200 "${RUNNER[@]}" "${COMMON[@]}" --missing-mode text --run-name "text-${MODEL_LABEL}"
TEXT_STATUS=$?
timeout 1200 "${RUNNER[@]}" "${COMMON[@]}" --missing-mode image --run-name "image-${MODEL_LABEL}-sdxl" \
  --sdxl-model-dir /medisco/linjiashi/models/SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 --sdxl-height 1024 --sdxl-steps 20 --sdxl-seed 20260708
IMAGE_STATUS=$?
set -e
export TAOR_REAL_TEXT_STATUS="${TEXT_STATUS}" TAOR_REAL_IMAGE_STATUS="${IMAGE_STATUS}"
"${PY}" - <<'PY'
from pathlib import Path
import hashlib,json,os
from PIL import Image
root=Path(os.environ['TAOR_REAL_LOG_DIR'])
report={'text_exit_code':int(os.environ['TAOR_REAL_TEXT_STATUS']),
        'image_exit_code':int(os.environ['TAOR_REAL_IMAGE_STATUS']),'runs':{}}
label=os.environ['TAOR_REAL_MODEL_LABEL']
for name in (f'text-{label}',f'image-{label}-sdxl'):
    entity=root/'runs'/name/'entities/02436645'
    final=entity/'final.json'
    result=json.loads(final.read_text()) if final.exists() else {}
    report['runs'][name]={key:result.get(key) for key in ('ranking_status','best_text_candidate_id',
                                                       'best_image_candidate_id','ground_truth_used')}
    images=[]
    for path in sorted((entity/'generated_images').glob('*.png')):
        with Image.open(path) as image:
            image.load(); assert image.size==(1024,1024)
        images.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report['runs'][name]['new_sdxl_images']=images
report['completed']=(report['text_exit_code']==report['image_exit_code']==0 and
    all(r['ranking_status']=='completed' for r in report['runs'].values()) and
    bool(report['runs'][f'image-{label}-sdxl']['new_sdxl_images']))
(root/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False),flush=True)
assert report['completed'], 'Real execution did not satisfy all generation and Stage 4 checks'
PY
nvidia-smi
