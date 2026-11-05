#!/usr/bin/env bash
# Submit with gpus_per_task=0; no local neural model inference.
set -euo pipefail
PROJECT_ROOT=/medisco/linjiashi/Code/TAOR
PY=/medisco/linjiashi/venvs/taor-inference/bin/python
ART="${PROJECT_ROOT}/outputs/wn18rr-api-preflight-20261106"
ATTEMPT="${TAOR_API_ATTEMPT:-$(date -u +%Y%m%dT%H%M%S)}"
LOG_DIR="${ART}/attempt-${ATTEMPT}"
mkdir -p "${ART}"
mkdir "${LOG_DIR}"
exec >"${LOG_DIR}/cluster.log" 2>&1
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}" CUDA_VISIBLE_DEVICES=-1
export NO_PROXY=api.tensoris.ai,127.0.0.1,localhost no_proxy=api.tensoris.ai,127.0.0.1,localhost
"${PY}" - <<'PY'
import httpx, json
from pathlib import Path
from dotenv import dotenv_values
from urllib.parse import urlsplit
values=dotenv_values('/medisco/linjiashi/Code/.env.standardization.local')
url=values['OPENAI_BASE_URL'].rstrip('/')+'/models'
try:
    with httpx.Client(trust_env=False,timeout=15) as client:
        response=client.get(url)
        print(json.dumps({'api_host':urlsplit(url).hostname,'connectivity_status':response.status_code}),flush=True)
except Exception as exc:
    print(json.dumps({'connectivity_error':type(exc).__name__}),flush=True)
PY
timeout 1200 "${PY}" scripts/run_wn18rr_api_preflight.py \
  --env-file /medisco/linjiashi/Code/.env.standardization.local \
  --output-root "${LOG_DIR}/run"
