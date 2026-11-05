#!/usr/bin/env bash
# Run inside an allocated one-GPU SSP job; preserve the scheduler's CUDA mask.
set -euo pipefail
PROJECT_ROOT="/medisco/linjiashi/Code/TAOR"
ART="${PROJECT_ROOT}/outputs/wn18rr-onboarding-20261106"
PY="/medisco/linjiashi/venvs/taor-inference/bin/python"
SERVER_PY="/medisco/linjiashi/venvs/taor-qwen7b/bin/python"
mkdir -p "${ART}"
ATTEMPT="${TAOR_SMOKE_ATTEMPT:-$(date -u +%Y%m%dT%H%M%S)}"
LOG_DIR="${ART}/attempt-${ATTEMPT}"
mkdir "${LOG_DIR}"
export TAOR_SMOKE_LOG_DIR="${LOG_DIR}"
exec >"${LOG_DIR}/gpu-smoke.log" 2>&1
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="${NO_PROXY}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-unset}"
# Model transfer and isolated environment setup run outside the allocated GPU job.
for ((attempt=0; attempt<180; attempt++)); do
    if [[ -f "${ART}/download-completed.json" && -f "${ART}/smoke-ready.json" ]]; then break; fi
    if ((attempt == 179)); then echo 'Timed out waiting for prepared inputs'; exit 2; fi
    sleep 10
done
"${PY}" -c 'import torch,transformers,diffusers; print(torch.__version__, transformers.__version__, diffusers.__version__); assert torch.cuda.is_available()'
"${SERVER_PY}" -m vllm.entrypoints.openai.api_server \
    --model /medisco/linjiashi/models/Qwen/Qwen2-VL-7B-Instruct \
    --served-model-name Qwen/Qwen2-VL-7B-Instruct \
    --host 127.0.0.1 --port 12345 --dtype bfloat16 \
    --middleware mmkgc.adapters.request_deadline.RequestDeadlineMiddleware \
    --gpu-memory-utilization 0.45 --max-model-len 12288 \
    --limit-mm-per-prompt '{"image":4}' --max-num-seqs 2 --enforce-eager \
    >"${LOG_DIR}/qwen-server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT
for ((attempt=0; attempt<180; attempt++)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo 'Qwen server exited'; exit 3; fi
    if "${PY}" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:12345/v1/models",timeout=2).read()' 2>/dev/null; then break; fi
    if ((attempt == 179)); then echo 'Qwen server startup timed out'; exit 3; fi
    sleep 5
done
COMMON=(--dataset-type wn18rr --dataset-dir datasets/wn18rr-import-20261106/WN18RR
    --graph-split train --entity-manifest "${ART}/one-entity.json" --limit 1
    --stage1-reranker-model-dir /medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B
    --stage1-reranker-device cuda:0 --sdxl-device cuda:0 --candidate-budget 1
    --model-call-timeout-seconds 120 --skip-stage4 --output-root "${LOG_DIR}/runs")
"${PY}" -m mmkgc.doctor "${COMMON[@]}" --missing-mode image >"${LOG_DIR}/doctor-gpu.json"
timeout 1200 "${PY}" -m mmkgc.run "${COMMON[@]}" --missing-mode text --run-name text-qwen7b
# Real SDXL pipeline call; a prompt-only result does not satisfy this check.
timeout 1200 "${PY}" -m mmkgc.run "${COMMON[@]}" --missing-mode image --run-name image-qwen7b-sdxl \
    --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
    --sdxl-width 1024 --sdxl-height 1024 --sdxl-steps 20 --sdxl-seed 20260708
"${PY}" - <<'PY'
from pathlib import Path
import json, os
from PIL import Image
root=Path(os.environ['TAOR_SMOKE_LOG_DIR'])
images=list((root/'runs/image-qwen7b-sdxl').glob('entities/**/generated_images/*.png'))
if not images:
    images=list((root/'runs/image-qwen7b-sdxl').glob('entities/**/*.png'))
assert images, 'No generated SDXL pixels'
for path in images:
    with Image.open(path) as image:
        image.load()
        assert image.size == (1024,1024)
(root/'gpu-smoke-completed.json').write_text(json.dumps({'generated_images':[str(p) for p in images], 'stage4':'deferred'},indent=2))
PY
nvidia-smi
