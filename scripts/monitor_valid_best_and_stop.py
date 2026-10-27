#!/usr/bin/env python3
"""Stop a tmux generation job once enough entities have valid best images."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--tmux-session", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    return parser.parse_args()


def count_valid_best(run_dirs: list[Path]) -> tuple[int, int]:
    best_qids: set[str] = set()
    final_count = 0
    for run_dir in run_dirs:
        entities_dir = run_dir / "entities"
        if not entities_dir.exists():
            continue
        for final_json in entities_dir.glob("*/final.json"):
            final_count += 1
            try:
                data = json.loads(final_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("best_image_candidate_id"):
                best_qids.add(final_json.parent.name)
    return len(best_qids), final_count


def tmux_session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def append_log(log_file: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def main() -> int:
    args = parse_args()
    run_dirs = [Path(p) for p in args.run_dir]
    log_file = Path(args.log_file)

    append_log(
        log_file,
        f"monitor_start threshold={args.threshold} session={args.tmux_session}",
    )
    while True:
        best_count, final_count = count_valid_best(run_dirs)
        append_log(log_file, f"valid_best={best_count} final_json={final_count}")

        if best_count >= args.threshold:
            append_log(log_file, f"threshold_reached killing={args.tmux_session}")
            subprocess.run(["tmux", "kill-session", "-t", args.tmux_session], check=False)
            return 0

        if not tmux_session_exists(args.tmux_session):
            append_log(log_file, f"session_missing stopping_monitor={args.tmux_session}")
            return 1

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
