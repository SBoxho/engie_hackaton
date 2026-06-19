"""Initialize local directories and verify demo artifacts.

This command is intentionally offline. It prepares cache/database directories
and confirms that the fixed replay bundle needed for a clean demo checkout is
present, without creating synthetic electricity values.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import settings


REQUIRED_DEMO_ARTIFACTS = (
    "energy_recent.parquet",
    "manifest.json",
    "quality_report.json",
    "baseline_backtest.json",
    "demand_model_evaluation.json",
    "mood_calibration.json",
)


def _ensure_dirs() -> list[Path]:
    paths = [
        settings.raw_dir,
        settings.processed_dir,
        settings.energy_store_dir,
        settings.ecowatt_cache_dir,
        settings.processed_dir / "demand_forecast",
        settings.processed_dir / "usual_demand",
        settings.project_root / "data" / "interim",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _demo_manifest() -> dict:
    path = settings.demo_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    created = _ensure_dirs()
    missing = [name for name in REQUIRED_DEMO_ARTIFACTS if not (settings.demo_dir / name).exists()]
    manifest = _demo_manifest()
    print("Initialized local data/cache directories:")
    for path in created:
        print(f"- {path.relative_to(settings.project_root).as_posix()}")
    if missing:
        print("Missing required demo artifacts:")
        for name in missing:
            print(f"- demo_data/{name}")
        return 2
    print(
        "Demo bundle ready: "
        f"{manifest.get('window_start_utc', 'unknown')} to {manifest.get('window_end_utc', 'unknown')}"
    )
    model_path = settings.processed_dir / "demand_forecast" / "demand_residual_quantile_model.pkl"
    if not model_path.exists():
        print("No champion probabilistic model artifact found; inference will use the documented baseline fallback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
