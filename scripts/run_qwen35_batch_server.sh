#!/usr/bin/env bash
# One Qwen model instance; vLLM continuously batches up to ten requests.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="/medisco/linjiashi/venvs/taor-qwen35-batch/bin:${PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1
export VLLM_LOG_STATS_INTERVAL=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
exec /medisco/linjiashi/venvs/taor-qwen35-batch/bin/python -m vllm.entrypoints.openai.api_server \
  --model /medisco/linjiashi/models/Qwen/Qwen3.5-9B \
  --served-model-name Qwen/Qwen3.5-9B --host 127.0.0.1 --port 12345 \
  --dtype bfloat16 --max-model-len 24576 --max-num-seqs 10 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.45 \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --mm-encoder-attn-backend TORCH_SDPA --skip-mm-profiling \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --middleware mmkgc.adapters.request_deadline.RequestDeadlineMiddleware \
  --no-enable-prefix-caching --enforce-eager
