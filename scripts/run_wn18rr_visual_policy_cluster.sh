#!/usr/bin/env bash
# Single allocated GPU; fresh artifacts, original failing runs stay immutable.
set -euo pipefail
PROJECT_ROOT=/medisco/linjiashi/Code/TAOR
ART="${PROJECT_ROOT}/outputs/wn18rr-visual-policy-20261106"
ATTEMPT="${TAOR_VISUAL_ATTEMPT:-$(date -u +%Y%m%dT%H%M%S)}"
LOG_DIR="${ART}/attempt-${ATTEMPT}"
PY=/medisco/linjiashi/venvs/taor-inference/bin/python
SERVER_PY=/medisco/linjiashi/venvs/taor-qwen7b/bin/python
mkdir -p "${ART}"
mkdir "${LOG_DIR}"
exec >"${LOG_DIR}/cluster.log" 2>&1
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}" TAOR_VISUAL_LOG_DIR="${LOG_DIR}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi
"${PY}" - <<'PY'
import hashlib, json, os
from pathlib import Path
files = sorted(p for root in ('mmkgc', 'scripts', 'configs') for p in Path(root).rglob('*')
               if p.suffix in {'.py', '.yaml', '.txt', '.sh'} and p.is_file())
Path(os.environ['TAOR_VISUAL_LOG_DIR'], 'execution-sha256.json').write_text(json.dumps(
    {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))
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
FAILED=0
timeout 240 "${PY}" scripts/check_wn18rr_visual_extraction.py --output-root "${LOG_DIR}/fixed-caption-control" || FAILED=1
COMMON=(--dataset-type wn18rr --dataset-dir datasets/wn18rr-import-20261106/WN18RR
  --graph-split train --entity-manifest outputs/wn18rr-onboarding-20261106/one-entity.json --limit 1
  --stage1-reranker-model-dir /medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B
  --stage1-reranker-device cuda:0 --sdxl-device cuda:0 --candidate-budget 1
  --model-call-timeout-seconds 120 --stage4-config configs/stage4-qwen7b-local.yaml
  --output-root "${LOG_DIR}/runs")
timeout 1200 "${PY}" -m mmkgc.run "${COMMON[@]}" --missing-mode text --run-name text-qwen7b || FAILED=1
timeout 1200 "${PY}" -m mmkgc.run "${COMMON[@]}" --missing-mode image --run-name image-qwen7b-sdxl \
  --sdxl-width 1024 --sdxl-height 1024 --sdxl-steps 20 --sdxl-seed 20260708 || FAILED=1
nvidia-smi
exit "${FAILED}"
