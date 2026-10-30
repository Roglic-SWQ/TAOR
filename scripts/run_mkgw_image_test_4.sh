#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-kb}"
PIPELINE_GPU="${PIPELINE_GPU:-1}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/datasets/MKG-W}"
MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/tmp_mkgw_four_entities_manifest.jsonl}"
SDXL_MODEL_DIR="${SDXL_MODEL_DIR:-${PROJECT_ROOT}/SDXL/stable-diffusion-xl-base-1.0}"
STAGE1_RERANKER_MODEL_DIR="${STAGE1_RERANKER_MODEL_DIR:-${PROJECT_ROOT}/Qwen3-Reranker-4B}"
STAGE4_CONFIG="${STAGE4_CONFIG:-${PROJECT_ROOT}/configs/stage4-model-fusion-gpu1-visible.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_ROOT}/configs/qwen7b-mainline.yaml}"
STAGE1_MODEL_CONFIG="${STAGE1_MODEL_CONFIG:-${PROJECT_ROOT}/configs/qwen7b-stage1.yaml}"
STAGE2_MODEL_CONFIG="${STAGE2_MODEL_CONFIG:-${PROJECT_ROOT}/configs/qwen7b-stage2.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
RUN_NAME="${RUN_NAME:-qwen7b_sdxl_four_$(date +%Y%m%d_%H%M%S)}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if ! command -v conda >/dev/null 2>&1; then
    fail "conda was not found in PATH"
fi

CONDA_BASE="$(conda info --base)"
[[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] || \
    fail "conda activation script was not found under ${CONDA_BASE}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

[[ -d "${DATASET_DIR}" ]] || fail "dataset directory does not exist: ${DATASET_DIR}"
[[ -f "${MANIFEST_PATH}" ]] || fail "entity manifest does not exist: ${MANIFEST_PATH}"
[[ -d "${SDXL_MODEL_DIR}" ]] || fail "SDXL model directory does not exist: ${SDXL_MODEL_DIR}"
[[ -f "${SDXL_MODEL_DIR}/model_index.json" ]] || \
    fail "SDXL model_index.json is missing: ${SDXL_MODEL_DIR}/model_index.json"
[[ -d "${STAGE1_RERANKER_MODEL_DIR}" ]] || \
    fail "Stage 1 reranker model directory does not exist: ${STAGE1_RERANKER_MODEL_DIR}"
[[ -f "${MODEL_CONFIG}" ]] || fail "model config does not exist: ${MODEL_CONFIG}"
[[ -f "${STAGE1_MODEL_CONFIG}" ]] || fail "Stage 1 model config does not exist: ${STAGE1_MODEL_CONFIG}"
[[ -f "${STAGE2_MODEL_CONFIG}" ]] || fail "Stage 2 model config does not exist: ${STAGE2_MODEL_CONFIG}"
[[ -f "${STAGE4_CONFIG}" ]] || fail "Stage 4 config does not exist: ${STAGE4_CONFIG}"

command -v python >/dev/null 2>&1 || fail "python was not found in conda environment: ${CONDA_ENV}"
echo "Running MKG-W four-entity image completion test"
echo "  environment: ${CONDA_ENV}"
echo "  model config: ${MODEL_CONFIG}"
echo "  stage1 config: ${STAGE1_MODEL_CONFIG}"
echo "  stage2 config: ${STAGE2_MODEL_CONFIG}"
echo "  dataset:     ${DATASET_DIR}"
echo "  manifest:    ${MANIFEST_PATH}"
echo "  sdxl:        ${SDXL_MODEL_DIR}"
echo "  gpu:         ${PIPELINE_GPU}"
echo "  output:      ${OUTPUT_ROOT}/${RUN_NAME}"

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES="${PIPELINE_GPU}" exec python -m mmkgc.run \
    --dataset-dir "${DATASET_DIR}" \
    --graph-split train \
    --missing-mode image \
    --entity-manifest "${MANIFEST_PATH}" \
    --limit 4 \
    --candidate-budget 5 \
    --stage1-reranker-model-dir "${STAGE1_RERANKER_MODEL_DIR}" \
    --model-config "${MODEL_CONFIG}" \
    --stage1-model-config "${STAGE1_MODEL_CONFIG}" \
    --stage2-model-config "${STAGE2_MODEL_CONFIG}" \
    --stage4-config "${STAGE4_CONFIG}" \
    --sdxl-model-dir "${SDXL_MODEL_DIR}" \
    --sdxl-width 1024 \
    --sdxl-height 1024 \
    --sdxl-steps 20 \
    --sdxl-seed 20260708 \
    --sdxl-guidance-scale 7.5 \
    --sdxl-device cuda \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}"
