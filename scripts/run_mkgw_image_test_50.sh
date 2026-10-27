#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-kb}"
PIPELINE_GPU="${PIPELINE_GPU:-1}"

DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/datasets/MKG-W bak}"
MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/configs/mkgw-image-test-50.json}"
STAGE1_RERANKER_MODEL_DIR="${STAGE1_RERANKER_MODEL_DIR:-${PROJECT_ROOT}/Qwen3-Reranker-4B}"
SDXL_MODEL_DIR="${SDXL_MODEL_DIR:-${PROJECT_ROOT}/SDXL/stable-diffusion-xl-base-1.0}"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_ROOT}/configs/gpt4omini-mainline.yaml}"
STAGE4_CONFIG="${STAGE4_CONFIG:-${PROJECT_ROOT}/configs/stage4-model-fusion-gpu1-visible.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
RUN_NAME="${RUN_NAME:-mkgw_image_test_50_$(date +%Y%m%d_%H%M%S)}"
CANDIDATE_BUDGET="${CANDIDATE_BUDGET:-5}"
SDXL_SEED="${SDXL_SEED:-20261012}"

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
[[ -f "${STAGE4_CONFIG}" ]] || fail "Stage 4 config does not exist: ${STAGE4_CONFIG}"

command -v python >/dev/null 2>&1 || fail "python was not found in conda environment: ${CONDA_ENV}"

python - "${DATASET_DIR}" "${MANIFEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

dataset_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(payload, list) or len(payload) != 50:
    raise SystemExit("manifest must contain exactly 50 entity rows")

info = json.loads((dataset_dir / "entity_info.json").read_text(encoding="utf-8"))
image_root = dataset_dir / "images"
extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
missing = []
for row in payload:
    key = str(row.get("entity_key") or "").strip()
    label = str(info.get(key, {}).get("label") or row.get("label") or key)
    variants = [label, label.replace(" ", "_"), key, key.replace(" ", "_")]
    found = False
    for variant in dict.fromkeys(variants):
        directory = image_root / variant
        if directory.is_dir() and any(
            item.is_file() and item.suffix.lower() in extensions
            for item in directory.iterdir()
        ):
            found = True
            break
    if not found:
        missing.append(f"{key} ({label})")
if missing:
    raise SystemExit("entities without readable image files: " + ", ".join(missing))
print("validated 50 manifest entities with readable image files")
PY

echo "Running MKG-W 50-entity image completion test"
echo "  environment: ${CONDA_ENV}"
echo "  model config: ${MODEL_CONFIG}"
echo "  dataset:     ${DATASET_DIR}"
echo "  manifest:    ${MANIFEST_PATH}"
echo "  sdxl:        ${SDXL_MODEL_DIR}"
echo "  gpu:         ${PIPELINE_GPU}"
echo "  candidates:  ${CANDIDATE_BUDGET} per entity"
echo "  output:      ${OUTPUT_ROOT}/${RUN_NAME}"

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES="${PIPELINE_GPU}" exec python -m mmkgc.run \
    --dataset-dir "${DATASET_DIR}" \
    --graph-split train \
    --missing-mode image \
    --entity-manifest "${MANIFEST_PATH}" \
    --limit 50 \
    --candidate-budget "${CANDIDATE_BUDGET}" \
    --stage1-reranker-model-dir "${STAGE1_RERANKER_MODEL_DIR}" \
    --model-config "${MODEL_CONFIG}" \
    --stage1-model-config "${STAGE1_MODEL_CONFIG:-${PROJECT_ROOT}/configs/gpt4omini-mainline.yaml}" \
    --stage2-model-config "${STAGE2_MODEL_CONFIG:-${PROJECT_ROOT}/configs/openai-gpt4o-mini-judge.yaml}" \
    --stage4-config "${STAGE4_CONFIG}" \
    --sdxl-model-dir "${SDXL_MODEL_DIR}" \
    --sdxl-width 1024 \
    --sdxl-height 1024 \
    --sdxl-steps 20 \
    --sdxl-seed "${SDXL_SEED}" \
    --sdxl-guidance-scale 7.5 \
    --sdxl-device cuda \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}"
