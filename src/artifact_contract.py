"""Canonical demo artifact readiness contract.

Small, file-based checks shared by the Streamlit app, deployment health, and
bundle exporter. The contract is intentionally lightweight for hackathon demos:
required artifacts fail clearly; optional artifacts degrade to status messages.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.config import settings

LOGGER = logging.getLogger(__name__)

Status = Literal["ok", "missing", "empty", "unreadable", "invalid", "stale"]
Kind = Literal["parquet", "json"]


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    label: str
    path: Path
    kind: Kind
    required: bool
    min_rows: int = 0
    required_keys: tuple[str, ...] = ()
    timestamp_column: str | None = None
    max_age_days: int | None = None


@dataclass(frozen=True)
class ArtifactCheck:
    spec: ArtifactSpec
    status: Status
    detail: str
    rows: int | None = None
    generated_at: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def blocks_demo(self) -> bool:
        return self.spec.required and not self.ok

    @property
    def health_status(self) -> str:
        if self.ok:
            return "ok"
        if self.blocks_demo:
            return "missing"
        return self.status


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def file_readable(path: Path) -> bool:
    try:
        with path.open("rb"):
            return True
    except OSError:
        return False


def parquet_row_count(path: Path) -> tuple[int | None, str | None]:
    if not file_exists(path):
        return None, "missing"
    if not file_readable(path):
        return None, "unreadable"
    try:
        return len(pd.read_parquet(path)), None
    except (OSError, ValueError, ImportError) as exc:
        return None, f"unreadable: {exc}"


def has_minimum_rows(path: Path, minimum: int) -> tuple[bool, int | None, str | None]:
    rows, error = parquet_row_count(path)
    if error:
        return False, rows, error
    return (rows or 0) >= minimum, rows, None


def read_json_object(path: Path) -> tuple[dict[str, Any], str | None]:
    if not file_exists(path):
        return {}, "missing"
    if not file_readable(path):
        return {}, "unreadable"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, f"unreadable: {exc}"
    if not isinstance(payload, dict):
        return {}, "invalid: JSON root must be an object"
    return payload, None


def _is_stale(value: str | None, max_age_days: int | None) -> bool:
    if not value or max_age_days is None:
        return False
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return False
    age = pd.Timestamp.now(tz=timezone.utc) - timestamp
    return age > pd.Timedelta(days=max_age_days)


def demo_artifact_specs(root: Path | None = None) -> list[ArtifactSpec]:
    base = root or settings.demo_dir
    return [
        ArtifactSpec(
            "manifest",
            "Demo manifest",
            base / "manifest.json",
            "json",
            True,
            required_keys=("schema_version", "generated_at", "artifacts"),
            max_age_days=90,
        ),
        ArtifactSpec(
            "energy",
            "Demo energy",
            base / "energy_recent.parquet",
            "parquet",
            True,
            min_rows=1,
            timestamp_column="timestamp",
        ),
        ArtifactSpec(
            "weather",
            "Demo weather",
            base / "weather_national.parquet",
            "parquet",
            False,
            min_rows=1,
            timestamp_column="timestamp",
        ),
        ArtifactSpec(
            "ecowatt",
            "Demo EcoWatt",
            base / "ecowatt.parquet",
            "parquet",
            False,
            min_rows=1,
            timestamp_column="timestamp",
        ),
        ArtifactSpec(
            "quality",
            "Quality report",
            base / "quality_report.json",
            "json",
            True,
            required_keys=("findings", "passed"),
        ),
        ArtifactSpec(
            "demand_evaluation",
            "Demand model evaluation",
            base / "demand_model_evaluation.json",
            "json",
            True,
            required_keys=("predictions", "metrics"),
        ),
        ArtifactSpec(
            "model_forecast",
            "Model forecast",
            base / "model_forecast.json",
            "json",
            False,
            required_keys=("schema_version", "forecasts"),
        ),
        ArtifactSpec(
            "baseline",
            "Baseline backtest",
            base / "baseline_backtest.json",
            "json",
            True,
            required_keys=("predictions", "metrics"),
        ),
        ArtifactSpec(
            "mood",
            "Mood calibration",
            base / "mood_calibration.json",
            "json",
            True,
            required_keys=("segments", "fixed_thresholds"),
        ),
    ]


def validate_artifact(spec: ArtifactSpec) -> ArtifactCheck:
    if spec.kind == "parquet":
        rows, error = parquet_row_count(spec.path)
        if error == "missing":
            return ArtifactCheck(spec, "missing", "file is missing")
        if error:
            return ArtifactCheck(spec, "unreadable", error)
        if (rows or 0) < spec.min_rows:
            return ArtifactCheck(
                spec,
                "empty",
                f"{rows or 0:,} rows; expected at least {spec.min_rows:,}",
                rows,
            )
        detail = f"{rows:,} rows"
        if spec.timestamp_column:
            try:
                frame = pd.read_parquet(spec.path, columns=[spec.timestamp_column])
                latest = pd.to_datetime(frame[spec.timestamp_column], utc=True, errors="coerce").max()
                if pd.notna(latest):
                    detail += f", latest {latest.strftime('%Y-%m-%d %H:%M UTC')}"
            except Exception as exc:  # defensive UI status only
                LOGGER.info("Could not inspect timestamp for %s: %s", spec.path, exc)
        return ArtifactCheck(spec, "ok", detail, rows)

    payload, error = read_json_object(spec.path)
    if error == "missing":
        return ArtifactCheck(spec, "missing", "file is missing")
    if error:
        status: Status = "invalid" if error.startswith("invalid") else "unreadable"
        return ArtifactCheck(spec, status, error)
    missing_keys = [key for key in spec.required_keys if key not in payload]
    if missing_keys:
        return ArtifactCheck(spec, "invalid", "missing key(s): " + ", ".join(missing_keys))
    generated = payload.get("generated_at") if isinstance(payload.get("generated_at"), str) else None
    if _is_stale(generated, spec.max_age_days):
        return ArtifactCheck(
            spec,
            "stale",
            f"generated {generated}; older than {spec.max_age_days} days",
            generated_at=generated,
        )
    return ArtifactCheck(
        spec,
        "ok",
        f"generated {generated}" if generated else "available",
        generated_at=generated,
    )


def validate_demo_bundle(root: Path | None = None, *, log: bool = True) -> list[ArtifactCheck]:
    checks = [validate_artifact(spec) for spec in demo_artifact_specs(root)]
    if log:
        for check in checks:
            level = logging.ERROR if check.blocks_demo else (logging.WARNING if not check.ok else logging.INFO)
            LOGGER.log(level, "Demo artifact %s: %s (%s)", check.spec.key, check.status, check.detail)
    return checks


def demo_blocking_message(checks: list[ArtifactCheck]) -> str | None:
    failures = [check for check in checks if check.blocks_demo]
    if not failures:
        return None
    return "; ".join(f"{check.spec.label}: {check.detail}" for check in failures)
