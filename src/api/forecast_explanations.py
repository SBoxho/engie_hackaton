"""Forecast and explanation API service.

This service is framework-neutral so it can be used by the minimal WSGI server,
tests, or future app integration without introducing a new backend framework.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.config import settings
from src.demo_mode import demo_energy, read_demo_parquet
from src.observability import record_forecast_run
from src.models.forecast_explainability import (
    CAUSAL_CAVEAT,
    DEFAULT_RECONCILIATION_TOLERANCE_GW,
    build_hourly_explanations,
    explain_forecast_changes,
)
from src.models.probabilistic_demand import (
    MODEL_FILENAME,
    build_inference_feature_rows,
    forecast_with_artifact,
    generated_at_utc,
    load_model_artifact,
    utc_iso,
)
from src.models.usual_demand import BaselineConfig, build_hourly_analytical_dataset, combine_public_inputs


DEFAULT_FORECAST_MODEL_PATH = settings.processed_dir / "demand_forecast" / MODEL_FILENAME


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ForecastExplanationApiService:
    """Build, cache, and retrieve explained 48-hour demand forecasts."""

    def __init__(
        self,
        *,
        hourly_loader: Callable[[], pd.DataFrame] | None = None,
        artifact: Mapping[str, Any] | None = None,
        artifact_path: str | Path | None = DEFAULT_FORECAST_MODEL_PATH,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._hourly_loader = hourly_loader or load_default_hourly_features
        self._now = now
        self._runs: dict[str, dict[str, Any]] = {}
        self.artifact: dict[str, Any] | None = dict(artifact) if artifact is not None else None
        if self.artifact is None and artifact_path is not None:
            path = Path(artifact_path)
            if path.exists():
                try:
                    self.artifact = load_model_artifact(path)
                except Exception:
                    self.artifact = None

    def create_forecast(self, *, scope: str = "france", hours: int = 48) -> dict[str, Any]:
        normalized_scope = _normalize_scope(scope)
        horizon_hours = tuple(range(1, max(0, min(int(hours), 48)) + 1))
        if not horizon_hours:
            raise ValueError("hours must be between 1 and 48")
        hourly = self._hourly_loader()
        if hourly.empty:
            raise ValueError("No hourly demand features are available for forecast inference.")
        origin = _latest_origin(hourly)
        config_payload = (self.artifact or {}).get("train_config", {})
        baseline_config = BaselineConfig(
            min_samples=int(config_payload.get("baseline_min_samples", 5)),
            recent_days=int(config_payload.get("baseline_recent_days", 28)),
            max_history_days=config_payload.get("baseline_max_history_days"),
        )
        rows = build_inference_feature_rows(
            hourly,
            origin,
            horizons_hours=horizon_hours,
            timezone_name=settings.timezone,
            baseline_config=baseline_config,
        )
        forecast = forecast_with_artifact(self.artifact, rows)
        generated_at = generated_at_utc()
        source_freshness = source_freshness_for_hourly(hourly, now=self._now())
        run_id = _run_id(normalized_scope, origin, forecast, self.artifact)
        points = build_hourly_explanations(
            artifact=self.artifact,
            feature_rows=rows,
            forecast_frame=forecast,
            run_id=run_id,
            generated_at=generated_at,
            source_freshness=source_freshness,
            tolerance_gw=DEFAULT_RECONCILIATION_TOLERANCE_GW,
        )
        route = "validated_model" if all(point.get("route") == "validated_model" for point in points) else "baseline_fallback"
        fallback_reasons = sorted({str(point.get("fallback_reason")) for point in points if point.get("fallback_reason")})
        run = {
            "run_id": run_id,
            "scope": normalized_scope,
            "origin": utc_iso(origin),
            "generated_at": generated_at,
            "horizon_hours": len(points),
            "route": route,
            "fallback_reason": fallback_reasons[0] if fallback_reasons else None,
            "model_version": (self.artifact or {}).get("model_version") if route == "validated_model" else None,
            "source_freshness": source_freshness,
            "model_explanation": {
                "forecast_equation": "p50 = usual_demand_baseline + P50 residual_tree_model",
                "method": "SHAP TreeExplainer for the residual correction when a champion P50 tree model is available.",
                "causal_caveat": CAUSAL_CAVEAT,
                "reconciliation_tolerance_gw": DEFAULT_RECONCILIATION_TOLERANCE_GW,
            },
            "points": points,
        }
        self._runs[run_id] = run
        record_forecast_run(run_id)
        return run

    def get_forecast(self, run_id: str) -> dict[str, Any]:
        try:
            return self._runs[str(run_id)]
        except KeyError as exc:
            raise KeyError(f"unknown forecast run id: {run_id}") from exc

    def get_explanation(self, *, run_id: str, timestamp: str) -> dict[str, Any]:
        run = self.get_forecast(run_id)
        requested = _canonical_timestamp(timestamp)
        for point in run.get("points", []):
            if _canonical_timestamp(str(point.get("timestamp"))) == requested:
                return {
                    "run_id": run_id,
                    "timestamp": point.get("timestamp"),
                    "casual": {
                        key: point.get(key)
                        for key in (
                            "expected_demand_gw",
                            "p10_gw",
                            "p90_gw",
                            "usual_demand_baseline_gw",
                            "difference_vs_usual_gw",
                            "top_positive_concept_drivers",
                            "top_negative_concept_drivers",
                            "explanation",
                            "confidence_level",
                            "confidence_reasons",
                            "source_freshness",
                            "model_version",
                        )
                    },
                    "technical": point.get("technical"),
                    "caveats": point.get("caveats"),
                }
        raise KeyError(f"timestamp {timestamp} is not present in run {run_id}")

    def get_model_card(self) -> dict[str, Any]:
        if self.artifact is None:
            return fallback_model_card()
        card = dict(self.artifact.get("model_card") or {})
        limitations = list(card.get("limitations") or [])
        limitations = [item for item in limitations if "No SHAP explanations" not in str(item)]
        limitations.append(
            "SHAP is applied to the P50 residual tree model only; usual demand is shown separately."
        )
        limitations.append(CAUSAL_CAVEAT)
        card.update(
            {
                "model_version": self.artifact.get("model_version"),
                "status": self.artifact.get("status"),
                "explainability": {
                    "method": "SHAP TreeExplainer on P50 residual model",
                    "display_policy": "Grouped concept drivers for casual users; raw SHAP values in technical payloads.",
                    "reconciliation_tolerance_gw": DEFAULT_RECONCILIATION_TOLERANCE_GW,
                    "causal_caveat": CAUSAL_CAVEAT,
                },
                "limitations": limitations,
            }
        )
        return card

    def get_forecast_changes(self, *, current: str, previous: str) -> dict[str, Any]:
        return explain_forecast_changes(self.get_forecast(current), self.get_forecast(previous))

    def store_run(self, run: Mapping[str, Any]) -> None:
        run_id = str(run.get("run_id"))
        if not run_id:
            raise ValueError("stored forecast run requires run_id")
        self._runs[run_id] = dict(run)


def load_default_hourly_features() -> pd.DataFrame:
    """Load inference features from demo artifacts by default."""

    if settings.is_demo_mode:
        energy = demo_energy()
        weather = read_demo_parquet(settings.demo_weather_path)
        records = combine_public_inputs(energy=energy, weather=weather)
        return build_hourly_analytical_dataset(records, timezone=settings.timezone)

    from src.data_processing.storage import PartitionedParquetStore

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(hours=max(settings.history_hours, 24 * 35))
    energy = PartitionedParquetStore(settings.energy_store_dir).read(start=start.to_pydatetime(), end=end.to_pydatetime())
    weather = pd.read_parquet(settings.weather_features_path) if settings.weather_features_path.exists() else pd.DataFrame()
    records = combine_public_inputs(energy=energy, weather=weather)
    return build_hourly_analytical_dataset(records, timezone=settings.timezone)


def source_freshness_for_hourly(hourly: pd.DataFrame, *, now: datetime) -> dict[str, Any]:
    latest = _latest_origin(hourly)
    now_ts = pd.Timestamp(now)
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    age_seconds = max((now_ts - latest).total_seconds(), 0.0)
    if settings.is_demo_mode:
        state = "historical_replay"
        reason = "Bundled demo replay sample anchored to the presentation date."
    elif age_seconds <= 45 * 60:
        state = "fresh"
        reason = "Latest source record is within the freshness window."
    elif age_seconds <= 6 * 3600:
        state = "delayed"
        reason = "Latest source record is delayed but usable."
    else:
        state = "stale"
        reason = "Latest source record is stale."
    return {
        "state": state,
        "latest_timestamp": utc_iso(latest),
        "age_seconds": round(age_seconds, 3),
        "reason": reason,
    }


def fallback_model_card() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_version": None,
        "model_family": "usual_demand_plus_residual_quantile",
        "status": "baseline_fallback",
        "intended_use": "48-hour national French electricity-demand forecasting with p10/p50/p90 uncertainty.",
        "forecast_equation": "p50 = usual_demand_baseline when no champion residual model is available",
        "fallback": "The transparent usual-demand baseline is used for every forecast hour.",
        "explainability": {
            "method": "No residual SHAP values are available without a champion P50 tree model.",
            "display_policy": "Usual demand and fallback reason are shown separately.",
            "reconciliation_tolerance_gw": DEFAULT_RECONCILIATION_TOLERANCE_GW,
            "causal_caveat": CAUSAL_CAVEAT,
        },
        "limitations": [
            "Baseline fallback does not decompose a residual tree model.",
            "Confidence is based on measurable diagnostics, not SHAP values.",
            CAUSAL_CAVEAT,
        ],
    }


def _latest_origin(hourly: pd.DataFrame) -> pd.Timestamp:
    if hourly.empty or "timestamp" not in hourly:
        raise ValueError("Hourly feature table requires timestamp values.")
    times = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce").dropna()
    if times.empty:
        raise ValueError("Hourly feature table has no valid timestamps.")
    return pd.Timestamp(times.max())


def _normalize_scope(scope: str) -> str:
    normalized = str(scope or "france").strip().lower()
    if normalized not in {"france", "national"}:
        raise ValueError("Only scope=france is supported by the national 48-hour forecast endpoint.")
    return "france"


def _run_id(scope: str, origin: pd.Timestamp, forecast: pd.DataFrame, artifact: Mapping[str, Any] | None) -> str:
    payload = {
        "scope": scope,
        "origin": utc_iso(origin),
        "model_version": (artifact or {}).get("model_version"),
        "p50": [round(float(value), 6) for value in forecast.get("p50", pd.Series(dtype=float)).tolist()],
    }
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]
    return f"forecast-{digest}"


def _canonical_timestamp(value: str) -> str:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return utc_iso(timestamp) or str(timestamp)


default_forecast_service = ForecastExplanationApiService()
