"""Preflight checks for MMKGC runtime dependencies."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any
from urllib import error, request

from mmkgc.adapters import QwenConfig, load_qwen_config
from mmkgc.data import DATASET_STORE_TYPES, build_dataset_store
from mmkgc.ranking.profiles import load_stage4_profile, preflight_stage4_profile
from mmkgc.defaults import load_runtime_environment, validate_sdxl_model_dir
from mmkgc.adapters.evidence_reranker import Qwen3EvidenceReranker
from mmkgc.run import build_parser as build_run_parser, validate_dataset_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the doctor CLI parser from the run parser plus doctor options."""
    parser = build_run_parser()
    parser.prog = "python -m mmkgc.doctor"
    parser.description = "Check MMKGC runtime prerequisites before launching a run."
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root used to resolve Stage 4 profile paths.",
    )
    parser.add_argument(
        "--qwen-timeout-seconds",
        type=float,
        default=2.0,
        help="HTTP timeout for configured model service reachability checks.",
    )
    parser.add_argument(
        "--skip-qwen-service",
        action="store_true",
        help="Validate model configs without probing HTTP services.",
    )
    parser.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format. JSON is currently the stable contract.",
    )
    return parser


def run_doctor(argv: list[str] | None = None) -> dict[str, Any]:
    """Run preflight checks and return a structured report."""
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    load_runtime_environment(project_root)
    checks: list[dict[str, Any]] = []

    dataset_dir = _check_dataset(args, checks)
    _check_output_root(args.output_root, checks)

    qwen_config = _check_qwen_config(args.model_config, checks)
    if qwen_config is not None:
        if args.skip_qwen_service:
            checks.append(_check("qwen_service", "skipped", "skipped by --skip-qwen-service"))
        else:
            ok, message = probe_qwen_service(qwen_config, args.qwen_timeout_seconds)
            checks.append(_check("qwen_service", "ok" if ok else "error", message))

    for stage, path in (("stage1", args.stage1_model_config), ("stage2", args.stage2_model_config)):
        try:
            config = load_qwen_config(path)
            checks.append(_check(f"{stage}_model_config", "ok", f"loaded {config.model_name}"))
            if args.skip_qwen_service:
                checks.append(_check(f"{stage}_service", "skipped", "service probing skipped"))
            else:
                ok, message = probe_qwen_service(config, args.qwen_timeout_seconds)
                checks.append(_check(f"{stage}_service", "ok" if ok else "error", message))
        except Exception as exc:
            checks.append(_check(f"{stage}_model_config", "error", f"{type(exc).__name__}: {exc}"))
    try:
        reranker_path = Path(args.stage1_reranker_model_dir)
        if not reranker_path.is_absolute():
            reranker_path = project_root / reranker_path
        Qwen3EvidenceReranker(reranker_path, device=args.stage1_reranker_device, max_length=args.stage1_reranker_max_length)
        checks.append(_check("stage1_reranker", "ok", f"local model files found at {reranker_path}"))
    except Exception as exc:
        checks.append(_check("stage1_reranker", "error", f"{type(exc).__name__}: {exc}"))

    cuda_devices: list[str] = [args.stage1_reranker_device]
    if args.missing_mode == "image":
        checks.append(_check("image_generator_mode", "ok", "fixed to sdxl-local"))
        _check_sdxl_model_dir(args.sdxl_model_dir, checks)
        cuda_devices.append(args.sdxl_device)
    else:
        checks.append(_check("image_generator_mode", "skipped", "image generation is not required"))
        checks.append(_check("sdxl_model_dir", "skipped", "text-only run"))

    stage4_devices = _check_stage4_profile(args, project_root, checks)
    cuda_devices.extend(stage4_devices)
    _check_cuda_devices(cuda_devices, checks)

    return {
        "overall_status": _overall_status(checks),
        "project_root": str(project_root),
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else str(args.dataset_dir),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    """Print a JSON doctor report and return non-zero when errors are present."""
    result = run_doctor(argv)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["overall_status"] != "error" else 1


def probe_qwen_service(config: QwenConfig, timeout_seconds: float) -> tuple[bool, str]:
    """Check that the configured OpenAI-compatible service responds to /models."""
    endpoint = f"{config.base_url.rstrip('/')}/models"
    headers = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    req = request.Request(endpoint, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
    except error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {endpoint}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if 200 <= int(status) < 300:
        return True, f"model service reachable at {endpoint}"
    return False, f"unexpected HTTP status {status} from {endpoint}"


def _check(name: str, status: str, message: str) -> dict[str, str]:
    return {"name": name, "status": status, "message": message}


def _check_dataset(args: argparse.Namespace, checks: list[dict[str, Any]]) -> Path | None:
    try:
        dataset_dir = validate_dataset_dir(args.dataset_dir)
        store = build_dataset_store(
            args.dataset_type,
            dataset_dir,
            graph_split=args.graph_split,
        )
        sample = store.select_records(args.missing_mode, limit=1)
    except Exception as exc:
        checks.append(_check("dataset_dir", "error", f"{type(exc).__name__}: {exc}"))
        return None
    if not sample:
        checks.append(_check("dataset_dir", "error", "dataset selection returned no records"))
        return dataset_dir
    checks.append(_check("dataset_dir", "ok", f"{args.dataset_type} dataset is readable"))
    return dataset_dir


def _check_output_root(output_root: str | Path, checks: list[dict[str, Any]]) -> None:
    target = Path(output_root)
    probe_dir = target if target.exists() else _nearest_existing_parent(target)
    try:
        with tempfile.NamedTemporaryFile(prefix=".mmkgc-doctor-", dir=probe_dir) as handle:
            handle.write(b"ok")
            handle.flush()
    except Exception as exc:
        checks.append(_check("output_root", "error", f"{type(exc).__name__}: {exc}"))
        return
    checks.append(_check("output_root", "ok", f"writable via {probe_dir}"))


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return current
        current = parent
    return current


def _check_qwen_config(
    model_config: str | Path,
    checks: list[dict[str, Any]],
) -> QwenConfig | None:
    try:
        config = load_qwen_config(model_config)
    except Exception as exc:
        checks.append(_check("qwen_config", "error", f"{type(exc).__name__}: {exc}"))
        return None
    checks.append(_check("qwen_config", "ok", f"loaded model config for {config.model_name}"))
    return config


def _check_sdxl_model_dir(model_dir: str | Path, checks: list[dict[str, Any]]) -> None:
    try:
        path = validate_sdxl_model_dir(model_dir)
    except Exception as exc:
        checks.append(_check("sdxl_model_dir", "error", f"{type(exc).__name__}: {exc}"))
        return
    checks.append(_check("sdxl_model_dir", "ok", f"found SDXL model at {path}"))


def _check_stage4_profile(
    args: argparse.Namespace,
    project_root: Path,
    checks: list[dict[str, Any]],
) -> list[str]:
    try:
        profile = load_stage4_profile(args.stage4_config, project_root=project_root)
        failures = preflight_stage4_profile(profile, project_root=project_root)
    except Exception as exc:
        checks.append(_check("stage4_profile", "error", f"{type(exc).__name__}: {exc}"))
        return []
    devices = _stage4_cuda_devices(profile)
    if failures:
        checks.append(_check("stage4_profile", "error", "; ".join(failures)))
        return devices
    checks.append(_check("stage4_profile", "ok", f"profile {profile.profile_name} preflight passed"))
    return devices


def _stage4_cuda_devices(profile: Any) -> list[str]:
    devices = []
    for field_name in ("sentence_bert", "bertscore", "clip", "blip_itm"):
        scorer = getattr(profile, field_name)
        if scorer.enabled and scorer.device:
            devices.append(str(scorer.device))
    return devices


def _check_cuda_devices(devices: list[str], checks: list[dict[str, Any]]) -> None:
    cuda_devices = sorted({device for device in devices if str(device).startswith("cuda")})
    if not cuda_devices:
        checks.append(_check("cuda_devices", "skipped", "no CUDA devices requested"))
        return
    try:
        import torch
    except Exception as exc:
        checks.append(_check("cuda_devices", "warning", f"torch unavailable: {type(exc).__name__}"))
        return
    if not torch.cuda.is_available():
        checks.append(_check("cuda_devices", "error", "CUDA is not available to torch"))
        return
    count = torch.cuda.device_count()
    missing = []
    for device in cuda_devices:
        parts = device.split(":", maxsplit=1)
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= count:
            missing.append(device)
    if missing:
        checks.append(
            _check(
                "cuda_devices",
                "error",
                f"requested CUDA devices not available ({count} visible): {', '.join(missing)}",
            )
        )
        return
    checks.append(_check("cuda_devices", "ok", f"{count} CUDA device(s) visible to torch"))


def _overall_status(checks: list[dict[str, Any]]) -> str:
    statuses = {check.get("status") for check in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


if __name__ == "__main__":
    raise SystemExit(main())
