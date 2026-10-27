#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-kb}"
QWEN_PYTHON="${QWEN_PYTHON:-${PROJECT_ROOT}/.venv-qwen7b/bin/python}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-${PROJECT_ROOT}/Qwen/Qwen2-VL-7B-Instruct}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-12345}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen2-VL-7B-Instruct}"
QWEN_GPU="${QWEN_GPU:-0}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.85}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-12288}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found in PATH" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "conda activation script was not found under ${CONDA_BASE}" >&2
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ ! -d "${QWEN_MODEL_DIR}" ]]; then
    echo "Qwen model directory does not exist: ${QWEN_MODEL_DIR}" >&2
    exit 1
fi

if [[ ! -x "${QWEN_PYTHON}" ]]; then
    echo "Qwen service Python was not found: ${QWEN_PYTHON}; see docs/stage1-stage2-runtime.zh.md" >&2
    exit 1
fi

echo "Starting Qwen service"
echo "  environment: ${CONDA_ENV}"
echo "  python:      ${QWEN_PYTHON}"
echo "  model:       ${QWEN_MODEL_DIR}"
echo "  endpoint:    http://${QWEN_HOST}:${QWEN_PORT}/v1"
echo "  model name:  ${QWEN_MODEL_NAME}"
echo "  attention:   ${VLLM_ATTENTION_BACKEND}"

export VLLM_ATTENTION_BACKEND
CUDA_VISIBLE_DEVICES="${QWEN_GPU}" exec "${QWEN_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${QWEN_MODEL_DIR}" \
    --served-model-name "${QWEN_MODEL_NAME}" \
    --host "${QWEN_HOST}" \
    --port "${QWEN_PORT}" \
    --dtype "${QWEN_DTYPE}" \
    --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${QWEN_MAX_MODEL_LEN}"
