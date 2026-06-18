from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.artifact_contract import ArtifactCheck, validate_demo_bundle
from src.config import settings
from src.demo_mode import external_api_enabled


@dataclass(frozen=True)
class HealthCheck:
    label: str
    status: str
    detail: str


def _has_json(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False, "unreadable"
    if isinstance(payload, dict):
        generated = payload.get("generated_at")
        return True, f"generated {generated}" if generated else "available"
    return False, "invalid"


def _contract_to_health(check: ArtifactCheck) -> HealthCheck:
    return HealthCheck(check.spec.label, check.health_status, check.detail)


def artifact_checks() -> list[HealthCheck]:
    """Return deployment artifact checks using the canonical readiness contract."""
    if settings.is_demo_mode:
        return [_contract_to_health(check) for check in validate_demo_bundle(log=True)]

    checks: list[HealthCheck] = []
    files = list(settings.energy_store_dir.rglob("*.parquet")) if settings.energy_store_dir.exists() else []
    checks.append(
        HealthCheck(
            "Processed energy store",
            "ok" if files else "missing",
            f"{len(files):,} parquet partitions" if files else "missing",
        )
    )
    if settings.weather_features_path.exists():
        try:
            rows = len(pd.read_parquet(settings.weather_features_path))
            checks.append(HealthCheck("Weather features", "ok" if rows else "empty", f"{rows:,} rows"))
        except (OSError, ValueError, ImportError) as exc:
            checks.append(HealthCheck("Weather features", "unreadable", str(exc)))
    else:
        checks.append(HealthCheck("Weather features", "optional", "missing"))
    for label, path in [
        ("Baseline backtest", settings.baseline_artifact_path),
        ("Mood calibration", settings.mood_artifact_path),
        ("Demand model evaluation", settings.processed_dir / "demand_model" / "evaluation.json"),
    ]:
        ok, detail = _has_json(path)
        checks.append(HealthCheck(label, "ok" if ok else "optional", detail))
    return checks


def data_check(data: pd.DataFrame, source_status: str) -> HealthCheck:
    if data.empty:
        return HealthCheck("Data loaded", "missing", "no rows available")
    latest = pd.to_datetime(data["timestamp"].max(), utc=True, errors="coerce")
    latest_text = latest.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(latest) else "unknown timestamp"
    return HealthCheck("Data loaded", "ok", f"{len(data):,} rows, latest {latest_text}; {source_status}")


def mode_check() -> HealthCheck:
    if settings.is_demo_mode:
        api_text = "external APIs disabled" if not external_api_enabled() else "external APIs allowed"
        return HealthCheck("Run mode", "demo", f"APP_MODE=demo, {api_text}")
    return HealthCheck("Run mode", "live", "APP_MODE=live, live fetch/cache fallbacks enabled")


def runtime_checks(
    *,
    data: pd.DataFrame,
    source_status: str,
    weather: pd.DataFrame,
    ecowatt: pd.DataFrame,
    model_payload: dict[str, Any],
    calibration_status: str,
) -> list[HealthCheck]:
    weather_detail = "available" if not weather.empty else "optional artifact missing, empty, or outside this window"
    ecowatt_detail = "available" if not ecowatt.empty else "optional EcoWatt artifact missing, empty, or outside this window"
    model_detail = "available" if model_payload else "not available"
    return [
        mode_check(),
        data_check(data, source_status),
        HealthCheck("Weather context", "ok" if not weather.empty else "optional", weather_detail),
        HealthCheck("EcoWatt signal", "ok" if not ecowatt.empty else "empty", ecowatt_detail),
        HealthCheck("Model evaluation", "ok" if model_payload else "optional", model_detail),
        HealthCheck("Mood thresholds", "ok", calibration_status),
        *artifact_checks(),
    ]


def render_deployment_health(
    *,
    data: pd.DataFrame,
    source_status: str,
    weather: pd.DataFrame,
    ecowatt: pd.DataFrame,
    model_payload: dict[str, Any],
    calibration_status: str,
) -> None:
    checks = runtime_checks(
        data=data,
        source_status=source_status,
        weather=weather,
        ecowatt=ecowatt,
        model_payload=model_payload,
        calibration_status=calibration_status,
    )
    failures = [check for check in checks if check.status == "missing"]
    warnings = [check for check in checks if check.status in {"optional", "empty", "stale", "invalid", "unreadable"}]

    with st.sidebar:
        st.markdown("### Deployment health")
        if failures:
            st.error(f"{len(failures)} required check(s) need attention.")
        elif warnings:
            st.warning(f"Ready with {len(warnings)} artifact warning(s).")
        else:
            st.success("Ready for public demo.")

        for check in checks:
            icon = {
                "ok": "[ok]",
                "demo": "[demo]",
                "live": "[live]",
                "optional": "[optional]",
                "missing": "[missing]",
                "empty": "[empty]",
                "stale": "[stale]",
                "invalid": "[invalid]",
                "unreadable": "[unreadable]",
            }.get(check.status, "[info]")
            st.caption(f"{icon} **{check.label}** - {check.detail}")
