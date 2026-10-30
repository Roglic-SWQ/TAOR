#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'HELP'
用法：bash scripts/run_mkgw_image_completion_4580.sh [选项]

默认：原缺图名单 4580 个实体，每实体 3 张候选图，后台运行。
新任务默认在 GPT 评分等待期间准备下一实体，最多两个实体在途，本地模型顺序执行。
请先在另一个终端执行 bash scripts/run_qwen7b.sh 并等待服务就绪。

选项：
  --resume DIR     从已有任务目录恢复，使用该目录保存的配置和代码快照
  --retry-failed   恢复时也重试失败或没有合格图片的实体
  --foreground    在当前终端运行，日志同时写入 outputs/_logs/
  --prepare-only  仅准备或校验任务，不连接模型服务、不开始生成
  -h, --help      显示帮助

环境变量：
  CONDA_ENV        流水线 conda 环境，默认 kb
  MMKGC_PYTHON     显式指定 Python 时跳过 conda 激活
  PIPELINE_GPU     流水线使用的物理 GPU，默认 1；千问脚本默认使用 GPU 0
  CANDIDATE_BUDGET 新任务每实体候选图数量，默认 3；恢复时使用已保存配置
  DATASET_DIR      默认 datasets/MKG-W bak
  MANIFEST_PATH    默认 configs/mkgw-test-original-missing-images-4580.json
  RUN_NAME         新任务名称，默认包含候选图数量和启动时间
  JOB_DIR          输出目录，须位于项目 outputs/ 内；已有任务会自动恢复

示例：
  bash scripts/run_mkgw_image_completion_4580.sh
  bash scripts/run_mkgw_image_completion_4580.sh --resume outputs/<任务名>
  bash scripts/run_mkgw_image_completion_4580.sh --resume outputs/<任务名> --retry-failed
HELP
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

RESUME_DIR=""
RETRY_FAILED=0
FOREGROUND=0
PREPARE_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "--resume 需要任务目录"
            RESUME_DIR="$2"
            shift 2
            ;;
        --retry-failed) RETRY_FAILED=1; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        --prepare-only) PREPARE_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "未知选项：$1（使用 --help 查看用法）" ;;
    esac
done

CONDA_ENV="${CONDA_ENV:-kb}"
PIPELINE_GPU="${PIPELINE_GPU:-1}"
CANDIDATE_BUDGET="${CANDIDATE_BUDGET:-3}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/datasets/MKG-W bak}"
MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/configs/mkgw-test-original-missing-images-4580.json}"
RUN_NAME="${RUN_NAME:-mkgw_image_completion_4580_qwen7b_${CANDIDATE_BUDGET}img_$(date +%Y%m%d_%H%M%S)}"
JOB_DIR="${RESUME_DIR:-${JOB_DIR:-${PROJECT_ROOT}/outputs/${RUN_NAME}}}"
RUNNER="${SCRIPT_DIR}/complete_mkgw_missing_images.py"

if [[ -z "${MMKGC_PYTHON:-}" ]]; then
    command -v conda >/dev/null 2>&1 || fail "找不到 conda；请初始化 conda 或设置 MMKGC_PYTHON"
    CONDA_BASE="$(conda info --base)"
    [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] || fail "找不到 conda 激活脚本"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    MMKGC_PYTHON="$(command -v python)"
fi
command -v "${MMKGC_PYTHON}" >/dev/null 2>&1 || fail "Python 不可用：${MMKGC_PYTHON}"
[[ -f "${RUNNER}" ]] || fail "找不到补全程序：${RUNNER}"

cd "${PROJECT_ROOT}"
if [[ -n "${RESUME_DIR}" && ! -f "${JOB_DIR}/job_config.json" ]]; then
    fail "恢复目录没有 job_config.json：${JOB_DIR}"
fi
if [[ ! -f "${JOB_DIR}/job_config.json" ]]; then
    [[ "${RETRY_FAILED}" -eq 0 ]] || fail "--retry-failed 用于恢复已有任务"
    [[ ! -e "${JOB_DIR}" ]] || fail "目录已存在但没有任务配置，不能覆盖：${JOB_DIR}"
    [[ -d "${DATASET_DIR}" ]] || fail "数据集目录不存在：${DATASET_DIR}"
    [[ -f "${MANIFEST_PATH}" ]] || fail "实体名单不存在：${MANIFEST_PATH}"
    "${MMKGC_PYTHON}" -B "${RUNNER}" \
        --dataset-dir "${DATASET_DIR}" \
        --entity-manifest "${MANIFEST_PATH}" \
        --expected-entities 4580 \
        --candidate-budget "${CANDIDATE_BUDGET}" \
        --job-dir "${JOB_DIR}"
fi

# Reuse the frozen runner's validation and chat health check without loading GPU models.
"${MMKGC_PYTHON}" -B - "${JOB_DIR}" "${PREPARE_ONLY}" <<'PY'
import fcntl
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
runner = root / 'config_snapshots/completion_runner.py'
if not runner.is_file():
    raise SystemExit('任务代码快照缺失：' + str(runner))
module = runpy.run_path(str(runner))
job = module['Job'](root)
if len(job.manifest) != 4580 or job.config.get('expected_entities') != 4580:
    raise SystemExit('该脚本仅处理原名单的 4580 个实体')
module['verify_inputs'](job.config)
with (root / 'worker.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('任务已在运行，请查看 progress.json 或现有日志')
print(json.dumps({'job_dir': str(root), 'entities': len(job.manifest),
                  'candidate_budget': job.config['runtime_kwargs']['candidate_budget'],
                  'completed_entities': job.state.get('completed_entities', 0)}, ensure_ascii=False), flush=True)
if sys.argv[2] != '1':
    try:
        module['qwen_health'](job.config['qwen_endpoint'])
    except Exception as exc:
        raise SystemExit(f'千问服务尚未就绪（{type(exc).__name__}）；请先运行 bash scripts/run_qwen7b.sh，服务就绪后用 --resume 恢复此任务。')
PY

echo "任务目录：${JOB_DIR}"
echo "阶段结果：${JOB_DIR}/entities/<QID>/"
echo "耗时报告：${JOB_DIR}/timing_report.html"
echo "默认每实体 3 张候选；恢复任务以保存的配置为准。保留五阶段结果和日志，不复制原图。"
echo "继续此任务：bash scripts/run_mkgw_image_completion_4580.sh --resume '${JOB_DIR}'"

if [[ "${PREPARE_ONLY}" -eq 1 ]]; then
    echo "准备和校验完成。"
    exit 0
fi

export PIPELINE_GPU
export CUDA_VISIBLE_DEVICES="${PIPELINE_GPU}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

WORKER_ARGS=(--worker "${JOB_DIR}")
if [[ "${RETRY_FAILED}" -eq 1 ]]; then
    WORKER_ARGS+=(--retry-failed)
fi
if [[ "${FOREGROUND}" -eq 1 ]]; then
    LOG_DIR="${PROJECT_ROOT}/outputs/_logs"
    mkdir -p "${LOG_DIR}"
    JOB_NAME="$(basename -- "${JOB_DIR}")"
    echo "前台运行，日志：${LOG_DIR}/${JOB_NAME}.log"
    "${MMKGC_PYTHON}" -B "${JOB_DIR}/config_snapshots/completion_runner.py" "${WORKER_ARGS[@]}" \
        2>&1 | tee -a "${LOG_DIR}/${JOB_NAME}.log"
else
    "${MMKGC_PYTHON}" -B "${RUNNER}" "${WORKER_ARGS[@]}" --launch
fi
