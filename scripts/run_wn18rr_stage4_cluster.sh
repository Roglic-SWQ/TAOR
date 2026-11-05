#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/medisco/linjiashi/Code/TAOR
ART="${TAOR_STAGE4_ARTIFACT_ROOT:-${PROJECT_ROOT}/outputs/wn18rr-stage4-20261106}"
ATTEMPT="${TAOR_STAGE4_ATTEMPT:-$(date -u +%Y%m%dT%H%M%S)}"
LOG_DIR="${ART}/attempt-${ATTEMPT}"
PY=/medisco/linjiashi/venvs/taor-inference/bin/python
SERVER_PY=/medisco/linjiashi/venvs/taor-qwen7b/bin/python
mkdir -p "${ART}"
mkdir "${LOG_DIR}"
exec >"${LOG_DIR}/cluster.log" 2>&1
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi
"${SERVER_PY}" -m vllm.entrypoints.openai.api_server \
  --model /medisco/linjiashi/models/Qwen/Qwen2-VL-7B-Instruct \
  --served-model-name Qwen/Qwen2-VL-7B-Instruct \
  --host 127.0.0.1 --port 12345 --dtype bfloat16 \
  --middleware mmkgc.adapters.request_deadline.RequestDeadlineMiddleware \
  --structured-outputs-config '{"backend":"lm-format-enforcer"}' \
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
"${PY}" -m mmkgc.doctor --dataset-type wn18rr \
  --dataset-dir datasets/wn18rr-import-20261106/WN18RR --graph-split train --missing-mode image \
  --stage1-reranker-model-dir /medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B \
  --stage1-reranker-device cuda:0 --sdxl-device cuda:0 \
  --stage4-config configs/stage4-qwen7b-local.yaml >"${LOG_DIR}/doctor.json"
timeout 2400 "${PY}" scripts/run_wn18rr_reflection.py \
  --source-root "${TAOR_STAGE4_SOURCE_ROOT:-outputs/wn18rr-onboarding-20261106/attempt-r3/runs}" \
  --dataset-dir datasets/wn18rr-import-20261106/WN18RR \
  --stage4-config configs/stage4-qwen7b-local.yaml --output-root "${LOG_DIR}/closure"
nvidia-smi
