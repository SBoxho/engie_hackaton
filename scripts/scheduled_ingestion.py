"""Run bounded ingestion on a schedule.

Use `--once` for CI or release verification, and `--interval-seconds` for a
small long-running deployment. Live fetching is opt-in through APP_MODE=live;
demo mode runs the offline cache path only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.config import settings


def _update_command(hours: int) -> list[str]:
    command = [sys.executable, "-m", "scripts.update_data", "--hours", str(hours)]
    if settings.is_demo_mode and not settings.demo_allow_external_api:
        command.append("--offline")
    return command


def _run_once(hours: int) -> int:
    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if settings.is_demo_mode and not settings.demo_allow_external_api:
        print(f"[{started}] APP_MODE=demo with external APIs disabled; scheduled live ingestion skipped.", flush=True)
        return 0
    command = _update_command(hours)
    print(f"[{started}] running ingestion: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    finished = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    print(f"[{finished}] ingestion exit_code={completed.returncode}", flush=True)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=settings.history_hours)
    parser.add_argument("--once", action="store_true", help="Run one ingestion attempt and exit.")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--max-runs", type=int, help="Stop after this many attempts.")
    args = parser.parse_args()
    if args.hours < 1:
        parser.error("--hours must be positive")
    if args.interval_seconds < 60:
        parser.error("--interval-seconds must be at least 60")

    runs = 0
    last_status = 0
    while True:
        runs += 1
        last_status = _run_once(args.hours)
        if args.once or (args.max_runs is not None and runs >= args.max_runs):
            return last_status
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
