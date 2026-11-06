#!/usr/bin/env bash
# Run only inside a newly allocated, single-GPU ClusterX job.
set -euo pipefail
PROJECT_ROOT=/medisco/linjiashi/Code/TAOR
BATCH_ROOT="${TAOR_COMPLETION_ROOT:?Set the frozen completion batch root}"
PY=/medisco/linjiashi/venvs/taor-inference/bin/python
SERVER_PY=/medisco/linjiashi/venvs/taor-qwen7b/bin/python
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
LOG_DIR=$(mktemp -d "${BATCH_ROOT}/job-XXXXXXXX")
exec >"${LOG_DIR}/cluster.log" 2>&1
nvidia-smi
"${PY}" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print('Single GPU:', torch.cuda.get_device_name(0), flush=True)
PY
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
"${PY}" scripts/run_with_cluster_api.py \
  --env-file /medisco/linjiashi/Code/.env.standardization.local \
  --proxy-settings /medisco/linjiashi/.config/taor-runtime/proxy-hybrid-20261106.json \
  --module scripts.run_wn18rr_completion_batch -- --root "${BATCH_ROOT}" "${@}"
nvidia-smi
