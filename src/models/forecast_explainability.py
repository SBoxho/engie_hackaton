"""Explain 48-hour probabilistic demand forecasts.

The forecast model is structured as:

    p50 demand = usual-demand baseline + P50 residual tree model

This module keeps the usual-demand baseline visible and applies SHAP
TreeExplainer only to the residual correction. Public/casual payloads expose
grouped concept drivers. Raw feature SHAP values stay in the technical payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.models.probabilistic_demand import feature_matrix, quantile_key, utc_iso


CONCEPT_WEATHER = "Weather"
CONCEPT_CALENDAR = "Calendar"
CONCEPT_TIME = "Time of day"
CONCEPT_RECENT_DEMAND = "Recent demand"
CONCEPT_REGIONAL = "Regional or national pattern"
CONCEPT_GENERATION = "Generation and exchange context"
CONCEPT_DATA_QUALITY = "Data-quality fallback"

CONCEPTS: tuple[str, ...] = (
    CONCEPT_WEATHER,
    CONCEPT_CALENDAR,
    CONCEPT_TIME,
    CONCEPT_RECENT_DEMAND,
    CONCEPT_REGIONAL,
    CONCEPT_GENERATION,
    CONCEPT_DATA_QUALITY,
)

DEFAULT_RECONCILIATION_TOLERANCE_GW = 0.001
CAUSAL_CAVEAT = "Feature attributions are model explanations, not causal proof."
SHAP_METHOD = "SHAP TreeExplainer on the P50 residual tree model"


class ShapUnavailable(RuntimeError):
    """Raised when the configured P50 tree model cannot be explained by SHAP."""


@dataclass(frozen=True)
class ConfidenceResult:
    level: str
    score: int
    reasons: list[str]
    inputs: dict[str, Any]


def concept_for_feature(feature_name: str) -> str:
    """Map raw features to stable user-facing concepts."""

    name = str(feature_name).lower()
    if any(token in name for token in ("missing", "fallback", "coverage", "quality", "schema_failure")):
        return CONCEPT_DATA_QUALITY
    if any(token in name for token in ("incomplete", "observation_count", "location_count", "sample_count")):
        return CONCEPT_DATA_QUALITY
    if "fallback_level" in name or "imputation" in name:
        return CONCEPT_DATA_QUALITY

    weather_tokens = (
        "weather",
        "temperature",
        "apparent_temperature",
        "humidity",
        "wind_speed",
        "cloud",
        "radiation",
        "heating_degree",
        "cooling_degree",
        "meteo",
    )
    if any(token in name for token in weather_tokens):
        return CONCEPT_WEATHER

    recent_tokens = (
        "origin_demand",
        "usual_demand",
        "demand_lag",
        "demand_roll",
        "demand_change",
        "recent_demand",
    )
    if any(token in name for token in recent_tokens):
        return CONCEPT_RECENT_DEMAND

    generation_tokens = (
        "generation",
        "production",
        "imports",
        "exports",
        "net_import",
        "exchange",
        "physical_balance",
        "supply",
        "nuclear",
        "solar_mw",
        "wind_mw",
        "hydro",
        "gas",
        "coal",
        "oil",
        "bioenergy",
    )
    if any(token in name for token in generation_tokens):
        return CONCEPT_GENERATION

    time_tokens = ("hour_of_day", "local_hour", "target_hour", "horizon_hours")
    if any(token in name for token in time_tokens):
        return CONCEPT_TIME

    calendar_tokens = (
        "weekday",
        "weekend",
        "holiday",
        "school",
        "season",
        "month",
        "dst",
        "utc_offset",
        "day_type",
    )
    if any(token in name for token in calendar_tokens):
        return CONCEPT_CALENDAR

    regional_tokens = ("region", "geographic_scope", "national", "derived_national", "rte_public")
    if any(token in name for token in regional_tokens):
        return CONCEPT_REGIONAL

    return CONCEPT_REGIONAL


def grouped_contributions_from_raw(
    feature_columns: Iterable[str],
    shap_values_mw: Iterable[float],
    *,
    postprocessing_adjustment_mw: float = 0.0,
) -> dict[str, float]:
    """Sum raw feature SHAP values into concepts, preserving additivity."""

    grouped = {concept: 0.0 for concept in CONCEPTS}
    for feature, value in zip(feature_columns, shap_values_mw):
        grouped[concept_for_feature(str(feature))] += _finite_float(value, default=0.0)
    if abs(postprocessing_adjustment_mw) > 0:
        grouped[CONCEPT_DATA_QUALITY] += float(postprocessing_adjustment_mw)
    return grouped


def compute_p50_tree_shap(
    model: Any,
    matrix: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Return SHAP values and expected residual value for a tree model."""

    try:
        import shap  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ShapUnavailable("The shap package is not installed.") from exc

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(matrix)
    except Exception as exc:  # pragma: no cover - exercised with unsupported model types.
        raise ShapUnavailable(f"SHAP TreeExplainer could not explain the P50 model: {exc}") from exc

    if hasattr(values, "values"):
        values = values.values
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.shape[0] != len(matrix):
        raise ShapUnavailable("SHAP returned a row count that does not match the forecast rows.")
    expected = _scalar_expected_value(getattr(explainer, "expected_value", 0.0))
    return array, expected


def build_hourly_explanations(
    *,
    artifact: Mapping[str, Any] | None,
    feature_rows: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    run_id: str,
    generated_at: str,
    source_freshness: Mapping[str, Any] | None = None,
    tolerance_gw: float = DEFAULT_RECONCILIATION_TOLERANCE_GW,
) -> list[dict[str, Any]]:
    """Build casual and technical explanations for every forecast hour."""

    rows = feature_rows.reset_index(drop=True).copy()
    forecast = forecast_frame.reset_index(drop=True).copy()
    source_freshness = dict(source_freshness or {})
    shap_values: np.ndarray | None = None
    shap_expected_mw = 0.0
    shap_error: str | None = None
    x_values = pd.DataFrame(index=rows.index)
    feature_columns = list((artifact or {}).get("feature_columns") or [])

    p50_key = quantile_key(0.50)
    can_try_shap = (
        artifact is not None
        and artifact.get("status") == "champion"
        and isinstance(artifact.get("models"), Mapping)
        and p50_key in artifact.get("models", {})
        and bool(feature_columns)
    )
    if can_try_shap:
        try:
            x_values = feature_matrix(rows, feature_columns, artifact.get("feature_imputation_values", {}))
            shap_values, shap_expected_mw = compute_p50_tree_shap(artifact["models"][p50_key], x_values)
        except ShapUnavailable as exc:
            shap_error = str(exc)
            shap_values = None

    points: list[dict[str, Any]] = []
    for index, row in rows.iterrows():
        point = forecast.iloc[index]
        if shap_values is None:
            points.append(
                _fallback_point_explanation(
                    row=row,
                    point=point,
                    run_id=run_id,
                    generated_at=generated_at,
                    source_freshness=source_freshness,
                    artifact=artifact,
                    shap_error=shap_error,
                    tolerance_gw=tolerance_gw,
                )
            )
            continue
        points.append(
            _shap_point_explanation(
                row=row,
                point=point,
                run_id=run_id,
                generated_at=generated_at,
                source_freshness=source_freshness,
                artifact=artifact,
                feature_columns=feature_columns,
                feature_values=x_values.iloc[index],
                shap_values_mw=shap_values[index],
                shap_expected_mw=shap_expected_mw,
                tolerance_gw=tolerance_gw,
            )
        )
    return points


def assess_forecast_confidence(
    row: Mapping[str, Any],
    point: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any] | None = None,
    source_freshness: Mapping[str, Any] | None = None,
) -> ConfidenceResult:
    """Assess confidence from measurable factors, never from SHAP values."""

    horizon = int(_finite_float(point.get("horizon_hours", row.get("horizon_hours")), default=0.0))
    p10 = _finite_float(point.get("p10"), default=np.nan)
    p90 = _finite_float(point.get("p90"), default=np.nan)
    interval_width_gw = (p90 - p10) / 1000.0 if np.isfinite(p10) and np.isfinite(p90) else np.nan
    weather_disagreement = _first_finite(
        row,
        (
            "weather_model_disagreement_c",
            "weather_forecast_spread_c",
            "temperature_ensemble_spread_c",
        ),
    )
    missing_count = _missing_or_fallback_count(row)
    ood_score = _first_finite(row, ("ood_score", "out_of_distribution_score", "distribution_shift_score"))
    recent_error_gw = _recent_error_gw(artifact, horizon)
    freshness_state = str((source_freshness or {}).get("state") or "")
    route = str(point.get("route") or "")

    score = 100
    reasons: list[str] = []

    if horizon > 36:
        score -= 20
        reasons.append(f"Long horizon: {horizon}h ahead.")
    elif horizon > 24:
        score -= 10
        reasons.append(f"Medium-long horizon: {horizon}h ahead.")
    elif horizon > 12:
        score -= 5
        reasons.append(f"Medium horizon: {horizon}h ahead.")
    else:
        reasons.append(f"Short horizon: {horizon}h ahead.")

    if np.isfinite(interval_width_gw):
        if interval_width_gw > 6.0:
            score -= 25
        elif interval_width_gw > 4.0:
            score -= 15
        elif interval_width_gw > 2.5:
            score -= 5
        reasons.append(f"P10-P90 interval width is {interval_width_gw:.2f} GW.")
    else:
        score -= 15
        reasons.append("Prediction interval width is unavailable.")

    if weather_disagreement is None:
        reasons.append("No weather-model disagreement diagnostic is available.")
    elif weather_disagreement > 4.0:
        score -= 15
        reasons.append(f"Weather-model disagreement is high at {weather_disagreement:.1f} C.")
    elif weather_disagreement > 2.0:
        score -= 8
        reasons.append(f"Weather-model disagreement is moderate at {weather_disagreement:.1f} C.")
    else:
        reasons.append(f"Weather-model disagreement is low at {weather_disagreement:.1f} C.")

    if missing_count > 4:
        score -= 20
        reasons.append(f"{missing_count} missing or fallback feature flags are active.")
    elif missing_count > 0:
        score -= 10
        reasons.append(f"{missing_count} missing or fallback feature flags are active.")
    else:
        reasons.append("No missing or fallback feature flags are active.")

    if ood_score is None:
        reasons.append("No out-of-distribution diagnostic is available.")
    elif ood_score > 0.8:
        score -= 25
        reasons.append(f"Out-of-distribution score is high at {ood_score:.2f}.")
    elif ood_score > 0.5:
        score -= 15
        reasons.append(f"Out-of-distribution score is elevated at {ood_score:.2f}.")
    else:
        reasons.append(f"Out-of-distribution score is low at {ood_score:.2f}.")

    if recent_error_gw is None:
        reasons.append("Recent model-error diagnostic is unavailable.")
    elif recent_error_gw >= 2.0:
        score -= 20
        reasons.append(f"Recent validation MAE is high at {recent_error_gw:.2f} GW.")
    elif recent_error_gw >= 1.2:
        score -= 10
        reasons.append(f"Recent validation MAE is moderate at {recent_error_gw:.2f} GW.")
    else:
        reasons.append(f"Recent validation MAE is {recent_error_gw:.2f} GW.")

    if route == "baseline_fallback":
        score = min(score, 55)
        reasons.append("Forecast uses the usual-demand fallback route.")
    if "stale" in freshness_state or "unavailable" in freshness_state:
        score -= 15
        reasons.append(f"Source freshness state is {freshness_state}.")

    score = int(max(0, min(100, score)))
    if score >= 80:
        level = "high"
    elif score >= 55:
        level = "medium"
    else:
        level = "low"

    inputs = {
        "horizon_hours": horizon,
        "interval_width_gw": _round_or_none(interval_width_gw),
        "weather_model_disagreement_c": weather_disagreement,
        "missing_or_fallback_feature_count": missing_count,
        "out_of_distribution_score": ood_score,
        "recent_model_error_gw": recent_error_gw,
        "route": route,
        "source_freshness_state": freshness_state or None,
    }
    return ConfidenceResult(level=level, score=score, reasons=reasons, inputs=inputs)


def render_hour_explanation(
    *,
    timestamp: str,
    expected_demand_gw: float,
    usual_demand_gw: float,
    difference_vs_usual_gw: float,
    top_positive: list[dict[str, Any]],
    top_negative: list[dict[str, Any]],
    route: str,
    model_explanation_available: bool,
) -> str:
    """Render deterministic plain-language text without generative AI."""

    if not model_explanation_available or route == "baseline_fallback":
        return (
            f"At {timestamp}, expected demand is {expected_demand_gw:.2f} GW. "
            f"The forecast follows the usual-demand baseline of {usual_demand_gw:.2f} GW "
            f"with a {difference_vs_usual_gw:+.2f} GW residual correction. "
            f"This is a fallback model explanation, not causal proof."
        )

    positive_text = _driver_text(top_positive, "No positive concept driver is material")
    negative_text = _driver_text(top_negative, "No negative concept driver is material")
    return (
        f"At {timestamp}, expected demand is {expected_demand_gw:.2f} GW. "
        f"Usual demand is {usual_demand_gw:.2f} GW and the residual model adjusts it by "
        f"{difference_vs_usual_gw:+.2f} GW. "
        f"Model drivers associated with a higher residual: {positive_text}. "
        f"Model drivers associated with a lower residual: {negative_text}. "
        f"These are model explanations, not causal proof."
    )


def explain_forecast_changes(
    current_run: Mapping[str, Any],
    previous_run: Mapping[str, Any],
    *,
    tolerance_gw: float = DEFAULT_RECONCILIATION_TOLERANCE_GW,
) -> dict[str, Any]:
    """Explain forecast changes by decomposing baseline and concept deltas."""

    previous_by_time = {str(point["timestamp"]): point for point in previous_run.get("points", [])}
    changes: list[dict[str, Any]] = []
    for current in current_run.get("points", []):
        timestamp = str(current.get("timestamp"))
        previous = previous_by_time.get(timestamp)
        if previous is None:
            continue
        delta_gw = _finite_float(current.get("expected_demand_gw"), default=0.0) - _finite_float(
            previous.get("expected_demand_gw"),
            default=0.0,
        )
        components = _change_components(current, previous)
        dominant = _dominant_component(delta_gw, components)
        text = _change_text(delta_gw, dominant, current, previous, tolerance_gw=tolerance_gw)
        changes.append(
            {
                "timestamp": timestamp,
                "current_expected_demand_gw": current.get("expected_demand_gw"),
                "previous_expected_demand_gw": previous.get("expected_demand_gw"),
                "delta_gw": round(delta_gw, 6),
                "dominant_component": dominant,
                "components": components,
                "explanation": text,
                "caveat": CAUSAL_CAVEAT,
            }
        )
    largest = max(changes, key=lambda item: abs(float(item["delta_gw"])), default=None)
    return {
        "current_run_id": current_run.get("run_id"),
        "previous_run_id": previous_run.get("run_id"),
        "method": "component comparison of usual baseline, residual expected value, and grouped SHAP concepts",
        "caveat": CAUSAL_CAVEAT,
        "largest_change": largest,
        "changes": changes,
    }


def _shap_point_explanation(
    *,
    row: pd.Series,
    point: pd.Series,
    run_id: str,
    generated_at: str,
    source_freshness: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
    feature_columns: list[str],
    feature_values: pd.Series,
    shap_values_mw: np.ndarray,
    shap_expected_mw: float,
    tolerance_gw: float,
) -> dict[str, Any]:
    usual_mw = _finite_float(point.get("usual_demand_mw", row.get("usual_demand_mw")), default=0.0)
    p50_mw = _finite_float(point.get("p50"), default=usual_mw)
    raw_residual_prediction_mw = usual_mw + shap_expected_mw + float(np.sum(shap_values_mw))
    postprocessing_adjustment_mw = p50_mw - raw_residual_prediction_mw
    grouped_mw = grouped_contributions_from_raw(
        feature_columns,
        shap_values_mw,
        postprocessing_adjustment_mw=postprocessing_adjustment_mw,
    )
    return _point_payload(
        row=row,
        point=point,
        run_id=run_id,
        generated_at=generated_at,
        source_freshness=source_freshness,
        artifact=artifact,
        method=SHAP_METHOD,
        model_explanation_available=True,
        residual_expected_mw=shap_expected_mw,
        grouped_mw=grouped_mw,
        raw_shap_values=[
            {
                "feature": feature,
                "concept": concept_for_feature(feature),
                "value_mw": round(float(value), 6),
                "value_gw": round(float(value) / 1000.0, 9),
                "feature_value": _json_scalar(feature_values.get(feature)),
            }
            for feature, value in zip(feature_columns, shap_values_mw)
        ],
        feature_values={feature: _json_scalar(feature_values.get(feature)) for feature in feature_columns},
        postprocessing_adjustment_mw=postprocessing_adjustment_mw,
        tolerance_gw=tolerance_gw,
    )


def _fallback_point_explanation(
    *,
    row: pd.Series,
    point: pd.Series,
    run_id: str,
    generated_at: str,
    source_freshness: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
    shap_error: str | None,
    tolerance_gw: float,
) -> dict[str, Any]:
    usual_mw = _finite_float(point.get("usual_demand_mw", row.get("usual_demand_mw")), default=0.0)
    p50_mw = _finite_float(point.get("p50"), default=usual_mw)
    grouped_mw = {concept: 0.0 for concept in CONCEPTS}
    grouped_mw[CONCEPT_DATA_QUALITY] = p50_mw - usual_mw
    method = "usual-demand fallback with residual decomposition unavailable"
    if shap_error:
        method = f"{method}: {shap_error}"
    return _point_payload(
        row=row,
        point=point,
        run_id=run_id,
        generated_at=generated_at,
        source_freshness=source_freshness,
        artifact=artifact,
        method=method,
        model_explanation_available=False,
        residual_expected_mw=0.0,
        grouped_mw=grouped_mw,
        raw_shap_values=[],
        feature_values={},
        postprocessing_adjustment_mw=0.0,
        tolerance_gw=tolerance_gw,
    )


def _point_payload(
    *,
    row: pd.Series,
    point: pd.Series,
    run_id: str,
    generated_at: str,
    source_freshness: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
    method: str,
    model_explanation_available: bool,
    residual_expected_mw: float,
    grouped_mw: Mapping[str, float],
    raw_shap_values: list[dict[str, Any]],
    feature_values: Mapping[str, Any],
    postprocessing_adjustment_mw: float,
    tolerance_gw: float,
) -> dict[str, Any]:
    timestamp = utc_iso(point.get("target_timestamp")) or str(point.get("target_timestamp"))
    usual_mw = _finite_float(point.get("usual_demand_mw", row.get("usual_demand_mw")), default=0.0)
    p50_mw = _finite_float(point.get("p50"), default=usual_mw)
    p10_mw = _finite_float(point.get("p10"), default=p50_mw)
    p90_mw = _finite_float(point.get("p90"), default=p50_mw)
    residual_correction_mw = p50_mw - usual_mw
    concept_contributions = {
        concept: {
            "concept": concept,
            "contribution_gw": round(float(grouped_mw.get(concept, 0.0)) / 1000.0, 9),
            "direction": _direction(float(grouped_mw.get(concept, 0.0))),
        }
        for concept in CONCEPTS
    }
    grouped_sum_mw = sum(float(grouped_mw.get(concept, 0.0)) for concept in CONCEPTS)
    reconstructed_gw = (usual_mw + residual_expected_mw + grouped_sum_mw) / 1000.0
    expected_demand_gw = p50_mw / 1000.0
    error_gw = reconstructed_gw - expected_demand_gw
    positives = _top_drivers(concept_contributions, positive=True)
    negatives = _top_drivers(concept_contributions, positive=False)
    confidence = assess_forecast_confidence(row, point, artifact=artifact, source_freshness=source_freshness)
    explanation_text = render_hour_explanation(
        timestamp=timestamp,
        expected_demand_gw=expected_demand_gw,
        usual_demand_gw=usual_mw / 1000.0,
        difference_vs_usual_gw=residual_correction_mw / 1000.0,
        top_positive=positives,
        top_negative=negatives,
        route=str(point.get("route") or ""),
        model_explanation_available=model_explanation_available,
    )
    technical = {
        "method": method,
        "raw_shap_values": raw_shap_values,
        "feature_values": dict(feature_values),
        "feature_concepts": {feature: concept_for_feature(feature) for feature in feature_values},
        "residual_expected_value_gw": round(residual_expected_mw / 1000.0, 9),
        "residual_correction_gw": round(residual_correction_mw / 1000.0, 9),
        "postprocessing_adjustment_gw": round(postprocessing_adjustment_mw / 1000.0, 9),
        "reconciliation": {
            "equation": (
                "p50_gw = usual_demand_baseline_gw + residual_expected_value_gw "
                "+ sum(grouped_contribution_gw)"
            ),
            "tolerance_gw": tolerance_gw,
            "reconstructed_p50_gw": round(reconstructed_gw, 9),
            "reported_p50_gw": round(expected_demand_gw, 9),
            "error_gw": round(error_gw, 12),
            "within_tolerance": abs(error_gw) <= tolerance_gw,
        },
        "confidence_inputs": confidence.inputs,
        "caveats": [
            CAUSAL_CAVEAT,
            "The usual-demand baseline is shown separately; SHAP explains only the residual correction.",
        ],
    }
    return {
        "explanation_id": f"{run_id}:{timestamp}",
        "timestamp": timestamp,
        "expected_demand_gw": round(expected_demand_gw, 6),
        "p10_gw": round(p10_mw / 1000.0, 6),
        "p90_gw": round(p90_mw / 1000.0, 6),
        "usual_demand_baseline_gw": round(usual_mw / 1000.0, 6),
        "difference_vs_usual_gw": round(residual_correction_mw / 1000.0, 6),
        "residual_model_expected_value_gw": round(residual_expected_mw / 1000.0, 6),
        "concept_contributions": list(concept_contributions.values()),
        "top_positive_concept_drivers": positives,
        "top_negative_concept_drivers": negatives,
        "explanation": explanation_text,
        "confidence_level": confidence.level,
        "confidence_score": confidence.score,
        "confidence_reasons": confidence.reasons,
        "source_freshness": dict(source_freshness),
        "model_version": point.get("model_version") or (artifact or {}).get("model_version"),
        "route": point.get("route"),
        "fallback_reason": point.get("fallback_reason"),
        "model_explanation_available": model_explanation_available,
        "technical": technical,
        "caveats": [CAUSAL_CAVEAT],
    }


def _top_drivers(concept_contributions: Mapping[str, Mapping[str, Any]], *, positive: bool) -> list[dict[str, Any]]:
    rows = []
    for item in concept_contributions.values():
        value = float(item["contribution_gw"])
        if positive and value <= 1e-9:
            continue
        if not positive and value >= -1e-9:
            continue
        rows.append(
            {
                "concept": item["concept"],
                "contribution_gw": value,
                "direction": item["direction"],
            }
        )
    if positive:
        rows.sort(key=lambda item: (-float(item["contribution_gw"]), str(item["concept"])))
    else:
        rows.sort(key=lambda item: (float(item["contribution_gw"]), str(item["concept"])))
    return rows[:3]


def _driver_text(drivers: list[dict[str, Any]], empty: str) -> str:
    if not drivers:
        return empty
    return ", ".join(f"{item['concept']} ({float(item['contribution_gw']):+.2f} GW)" for item in drivers)


def _direction(value_mw: float) -> str:
    if value_mw > 1e-6:
        return "raises_forecast"
    if value_mw < -1e-6:
        return "lowers_forecast"
    return "neutral"


def _missing_or_fallback_count(row: Mapping[str, Any]) -> int:
    count = 0
    for key, value in row.items():
        name = str(key).lower()
        if any(token in name for token in ("missing", "fallback", "incomplete")):
            number = _finite_float(value, default=0.0)
            if number > 0:
                count += 1
    fallback_level = _finite_float(row.get("usual_demand_fallback_level"), default=0.0)
    if fallback_level >= 4:
        count += 1
    return int(count)


def _recent_error_gw(artifact: Mapping[str, Any] | None, horizon: int) -> float | None:
    if not artifact:
        return None
    metrics = artifact.get("metrics") or {}
    for item in metrics.get("by_horizon") or []:
        if int(_finite_float(item.get("horizon_hours"), default=-1.0)) == int(horizon):
            value = _finite_float(item.get("mae_gw"), default=np.nan)
            return None if not np.isfinite(value) else float(value)
    overall = metrics.get("overall") or {}
    value = _finite_float(overall.get("mae_gw"), default=np.nan)
    return None if not np.isfinite(value) else float(value)


def _first_finite(row: Mapping[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        if name in row:
            value = _finite_float(row.get(name), default=np.nan)
            if np.isfinite(value):
                return float(value)
    return None


def _change_components(current: Mapping[str, Any], previous: Mapping[str, Any]) -> list[dict[str, Any]]:
    current_concepts = {item["concept"]: float(item["contribution_gw"]) for item in current.get("concept_contributions", [])}
    previous_concepts = {item["concept"]: float(item["contribution_gw"]) for item in previous.get("concept_contributions", [])}
    components = [
        {
            "component": "Usual-demand baseline",
            "delta_gw": round(
                _finite_float(current.get("usual_demand_baseline_gw"), default=0.0)
                - _finite_float(previous.get("usual_demand_baseline_gw"), default=0.0),
                6,
            ),
        },
        {
            "component": "Residual model expected value",
            "delta_gw": round(
                _finite_float(current.get("residual_model_expected_value_gw"), default=0.0)
                - _finite_float(previous.get("residual_model_expected_value_gw"), default=0.0),
                6,
            ),
        },
    ]
    for concept in CONCEPTS:
        components.append(
            {
                "component": concept,
                "delta_gw": round(current_concepts.get(concept, 0.0) - previous_concepts.get(concept, 0.0), 6),
            }
        )
    return components


def _dominant_component(delta_gw: float, components: list[dict[str, Any]]) -> dict[str, Any]:
    if abs(delta_gw) <= 1e-9:
        return {"component": "No material change", "delta_gw": 0.0}
    same_direction = [
        item
        for item in components
        if (float(item["delta_gw"]) > 0 and delta_gw > 0) or (float(item["delta_gw"]) < 0 and delta_gw < 0)
    ]
    candidates = same_direction or components
    return max(candidates, key=lambda item: (abs(float(item["delta_gw"])), str(item["component"])))


def _change_text(
    delta_gw: float,
    dominant: Mapping[str, Any],
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    *,
    tolerance_gw: float,
) -> str:
    if abs(delta_gw) <= tolerance_gw:
        return (
            f"The forecast changed by {delta_gw:+.2f} GW, which is within the "
            f"{tolerance_gw:.3f} GW explanation tolerance."
        )
    direction = "increased" if delta_gw > 0 else "decreased"
    component = str(dominant.get("component"))
    detail = _weather_change_detail(component, current, previous, delta_gw)
    if detail:
        return (
            f"The forecast {direction} by {abs(delta_gw):.2f} GW; as a model explanation, "
            f"the largest change is {component} ({float(dominant.get('delta_gw', 0.0)):+.2f} GW), "
            f"with {detail}. This is not causal proof."
        )
    return (
        f"The forecast {direction} by {abs(delta_gw):.2f} GW; as a model explanation, "
        f"the largest change is {component} ({float(dominant.get('delta_gw', 0.0)):+.2f} GW). "
        f"This is not causal proof."
    )


def _weather_change_detail(
    component: str,
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    delta_gw: float,
) -> str | None:
    if component != CONCEPT_WEATHER:
        return None
    current_features = ((current.get("technical") or {}).get("feature_values") or {})
    previous_features = ((previous.get("technical") or {}).get("feature_values") or {})
    for feature in ("weather_temperature_c", "temperature_c", "weather_apparent_temperature_c", "heating_degree_c"):
        if feature not in current_features or feature not in previous_features:
            continue
        current_value = _finite_float(current_features.get(feature), default=np.nan)
        previous_value = _finite_float(previous_features.get(feature), default=np.nan)
        if not np.isfinite(current_value) or not np.isfinite(previous_value):
            continue
        if "heating_degree" in feature and current_value > previous_value and delta_gw > 0:
            return "a higher heating-degree feature"
        if "temperature" in feature and current_value < previous_value and delta_gw > 0:
            return "the newest temperature feature colder"
        if "temperature" in feature and current_value > previous_value and delta_gw < 0:
            return "the newest temperature feature milder"
    return None


def _scalar_expected_value(value: Any) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(array[0])


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return number


def _round_or_none(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 6)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return utc_iso(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
