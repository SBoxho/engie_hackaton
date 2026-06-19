from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from app.components.charts import MIX_COLUMNS
from app.formatting import (
    format_mw as _format_mw,
    format_percentage as _format_percentage,
    format_signed_mw as _format_signed_mw,
)
from src.contracts.status_thresholds import balance_status_for_ratio, score_status, status_label
from src.models.mood_calibration import season_for_month

Mode = Literal["LIVE", "REPLAY", "SIMULATION"]


@dataclass(frozen=True)
class GridSnapshotView:
    as_of: pd.Timestamp
    mode: Mode
    demand: dict[str, Any]
    generation_by_source: dict[str, float]
    regional_state: pd.DataFrame
    weather: dict[str, Any]
    availability: dict[str, Any]
    official_signals: dict[str, Any]
    freshness: dict[str, Any]


@dataclass(frozen=True)
class ForecastPointView:
    timestamp: pd.Timestamp
    p10: float
    p50: float
    p90: float
    source: str
    pressure_label: str
    drivers: list[dict[str, Any]]
    backtest_error: float
    horizon_hours: int = 0
    usual_demand_mw: float | None = None
    route_label: str | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ScenarioResultView:
    baseline: pd.DataFrame
    modified: pd.DataFrame
    deltas: dict[str, Any]
    assumptions: list[str]
    explanation: str


SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "cold_snap": {
        "label": "3°C cold snap",
        "concept": "Demand",
        "default": 3.0,
        "unit": "°C",
        "detail": "Heating demand rises; supply is unchanged unless pressure forces imports.",
    },
    "low_wind": {
        "label": "Low-wind evening",
        "concept": "Supply",
        "default": 2.5,
        "unit": "GW",
        "detail": "Wind availability falls in the evening; demand is unchanged.",
    },
    "generation_outage": {
        "label": "1.3 GW generation unavailable",
        "concept": "Supply",
        "default": 1.3,
        "unit": "GW",
        "detail": "Firm generation is unavailable; demand is unchanged.",
    },
    "ev_shift": {
        "label": "100,000 EVs shifted overnight",
        "concept": "Demand timing",
        "default": 100_000,
        "unit": "EVs",
        "detail": "The same energy moves from evening to overnight hours.",
    },
    "solar_above": {
        "label": "Solar above forecast",
        "concept": "Renewable supply",
        "default": 1.2,
        "unit": "GW",
        "detail": "Midday renewable generation rises and carbon intensity falls.",
    },
}


def format_mw(value: float | int | None) -> str:
    return _format_mw(value)


def format_signed_mw(value: float | int | None) -> str:
    return _format_signed_mw(value)


def format_percent(value: float | int | None, *, signed: bool = False) -> str:
    return _format_percentage(value, signed=signed)


def ui_mode(source_mode: str, *, simulation: bool = False) -> Mode:
    if simulation:
        return "SIMULATION"
    return "LIVE" if source_mode.upper() == "LIVE" else "REPLAY"


def _as_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _with_comparable_keys(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result = result.dropna(subset=["timestamp"])
    local = result["timestamp"].dt.tz_convert(timezone)
    result["season"] = local.dt.month.map(season_for_month)
    result["day_type"] = local.dt.dayofweek.map(lambda day: "weekend" if day >= 5 else "weekday")
    result["local_hour"] = local.dt.hour.astype(int)
    return result


def demand_anomaly_label(percent: float | None) -> str:
    if percent is None or pd.isna(percent):
        return "Comparable history unavailable"
    if abs(float(percent)) < 0.07:
        return "Normal"
    if percent < 0:
        return f"{abs(percent):.0%} below usual"
    return f"{percent:.0%} above usual"


def add_regional_anomalies(
    current: pd.DataFrame,
    history: pd.DataFrame,
    *,
    timezone: str = "Europe/Paris",
    min_comparables: int = 4,
) -> pd.DataFrame:
    """Compare each regional demand value with season/day/hour peers."""
    if current.empty:
        return current.copy()
    required = {"timestamp", "region_code", "consumption_mw"}
    missing = required.difference(current.columns)
    if missing:
        raise ValueError(f"regional frame missing required columns: {sorted(missing)}")

    result = _with_comparable_keys(current, timezone)
    comparable = _with_comparable_keys(history, timezone) if not history.empty else result.copy()
    comparable = comparable.dropna(subset=["region_code", "consumption_mw"]).copy()
    if comparable.empty:
        comparable = result.copy()

    means: list[float] = []
    percentiles: list[float] = []
    counts: list[int] = []
    anomalies: list[float] = []
    labels: list[str] = []

    for row in result.itertuples(index=False):
        pool = comparable.loc[
            comparable["region_code"].eq(row.region_code)
            & comparable["season"].eq(row.season)
            & comparable["day_type"].eq(row.day_type)
            & comparable["local_hour"].eq(row.local_hour)
        ]
        if len(pool) < min_comparables:
            pool = comparable.loc[
                comparable["region_code"].eq(row.region_code)
                & comparable["season"].eq(row.season)
                & comparable["local_hour"].eq(row.local_hour)
            ]
        if len(pool) < min_comparables:
            pool = comparable.loc[comparable["region_code"].eq(row.region_code)]
        if pool.empty:
            pool = result.loc[result["region_code"].eq(row.region_code)]

        values = pd.to_numeric(pool["consumption_mw"], errors="coerce").dropna()
        current_value = float(row.consumption_mw)
        usual = float(values.median()) if not values.empty else current_value
        anomaly = 0.0 if usual == 0 else (current_value - usual) / usual
        percentile = float((values <= current_value).mean()) if not values.empty else 0.5
        means.append(usual)
        counts.append(int(len(values)))
        anomalies.append(anomaly)
        percentiles.append(min(max(percentile, 0.0), 1.0))
        labels.append(demand_anomaly_label(anomaly))

    result["usual_demand_mw"] = means
    result["comparable_count"] = counts
    result["demand_anomaly_pct"] = anomalies
    result["demand_anomaly_label"] = labels
    result["demand_anomaly_percentile"] = percentiles
    result["demand_anomaly_score"] = (0.5 + result["demand_anomaly_pct"] / 0.36).clip(0, 1)
    return result


def synthesize_regional_history(
    current: pd.DataFrame,
    *,
    timezone: str = "Europe/Paris",
    days: int = 28,
) -> pd.DataFrame:
    """Build a replay-only comparable context when no regional history is bundled."""
    if current.empty:
        return current.copy()
    latest = _as_utc(current["timestamp"].max()).floor("h")
    timestamps = pd.date_range(latest - pd.Timedelta(days=days), latest, freq="h")
    records: list[dict[str, Any]] = []
    ranked = current[["region_code", "consumption_mw"]].copy()
    ranked["consumption_mw"] = pd.to_numeric(ranked["consumption_mw"], errors="coerce")
    ranked["rank_pct"] = ranked["consumption_mw"].rank(pct=True, method="first")
    anomaly_targets = {
        str(row.region_code): float((row.rank_pct - 0.5) * 0.36)
        for row in ranked.itertuples(index=False)
    }
    for row in current.itertuples(index=False):
        current_local = _as_utc(row.timestamp).tz_convert(timezone)
        current_factor = _hour_factor(current_local.hour) * _day_factor(current_local.dayofweek)
        anomaly_target = anomaly_targets.get(str(row.region_code), 0.0)
        usual_current_hour = float(row.consumption_mw) / max(1 + anomaly_target, 0.2)
        base = usual_current_hour / current_factor
        for index, ts in enumerate(timestamps):
            local = ts.tz_convert(timezone)
            drift = 1 + ((index % 13) - 6) * 0.006
            records.append(
                {
                    **row._asdict(),
                    "timestamp": ts,
                    "consumption_mw": max(base * _hour_factor(local.hour) * _day_factor(local.dayofweek) * drift, 0),
                }
            )
    return pd.DataFrame.from_records(records)


def _hour_factor(hour: int) -> float:
    if 18 <= hour <= 21:
        return 1.13
    if 7 <= hour <= 9:
        return 1.06
    if 1 <= hour <= 5:
        return 0.86
    if 11 <= hour <= 15:
        return 0.96
    return 1.0


def _day_factor(day_of_week: int) -> float:
    return 0.94 if day_of_week >= 5 else 1.0


def pressure_label(demand_mw: float, available_mw: float, *, uncertainty_mw: float = 0.0) -> str:
    _ = uncertainty_mw
    if available_mw <= 0 or pd.isna(available_mw):
        return status_label("unknown")
    ratio = float(demand_mw) / float(available_mw)
    return status_label(balance_status_for_ratio(ratio))


def pressure_score(label: str) -> float:
    return {
        "Normal": 0.35,
        "Watch": 0.72,
        "High": 0.95,
        "Unknown": 0.5,
        "No recommendation": 0.35,
    }.get(label, 0.5)


def build_grid_snapshot(
    energy: pd.DataFrame,
    *,
    mode: Mode,
    regional_state: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    ecowatt: pd.DataFrame | None = None,
    source_label: str = "",
    timezone: str = "Europe/Paris",
) -> GridSnapshotView:
    if energy.empty:
        now = pd.Timestamp.now(tz="UTC")
        return GridSnapshotView(
            as_of=now,
            mode=mode,
            demand={"current_mw": 0.0, "anomaly_label": "Unavailable", "pressure_label": "Unknown"},
            generation_by_source={label: 0.0 for label in MIX_COLUMNS.values()},
            regional_state=regional_state,
            weather={"headline": "Weather unavailable", "detail": "No weather context is available."},
            availability={"available_mw": 0.0, "supply_margin_mw": 0.0, "import_requirement_mw": 0.0},
            official_signals={"label": "Unavailable", "status": "unknown"},
            freshness={"label": "Unavailable", "source": source_label},
        )

    frame = energy.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    latest = frame.iloc[-1]
    as_of = _as_utc(latest["timestamp"])

    demand = float(latest.get("consumption_mw", 0) or 0)
    total_production = float(latest.get("total_production_mw", 0) or 0)
    imports = max(float(latest.get("imports_mw", 0) or 0), float(latest.get("net_imports_mw", 0) or 0), 0.0)
    available = total_production + imports
    margin = available - demand

    comparable = _national_comparable_anomaly(frame, as_of, demand, timezone)
    pressure = pressure_label(demand, available)
    generation = {
        label: max(float(latest.get(column, 0) or 0), 0.0)
        for column, label in MIX_COLUMNS.items()
    }
    weather_summary = _weather_summary(latest, weather)
    signal = _official_signal(ecowatt, as_of)

    local = as_of.tz_convert(timezone)
    if mode == "LIVE":
        freshness_label = f"Updated {local:%H:%M}"
    else:
        freshness_label = f"Historical sample {local:%d %b %Y %H:%M}"

    return GridSnapshotView(
        as_of=as_of,
        mode=mode,
        demand={
            "current_mw": demand,
            "usual_mw": comparable["usual_mw"],
            "anomaly_pct": comparable["anomaly_pct"],
            "anomaly_label": demand_anomaly_label(comparable["anomaly_pct"]),
            "pressure_label": pressure,
        },
        generation_by_source=generation,
        regional_state=regional_state,
        weather=weather_summary,
        availability={
            "available_mw": available,
            "production_mw": total_production,
            "imports_mw": imports,
            "supply_margin_mw": margin,
            "import_requirement_mw": max(demand - total_production, 0.0),
            "pressure_label": pressure,
        },
        official_signals=signal,
        freshness={"label": freshness_label, "source": source_label},
    )


def _national_comparable_anomaly(
    frame: pd.DataFrame,
    as_of: pd.Timestamp,
    demand: float,
    timezone: str,
) -> dict[str, float]:
    keyed = _with_comparable_keys(frame, timezone)
    local = as_of.tz_convert(timezone)
    season = season_for_month(local.month)
    day_type = "weekend" if local.dayofweek >= 5 else "weekday"
    pool = keyed.loc[
        keyed["season"].eq(season)
        & keyed["day_type"].eq(day_type)
        & keyed["local_hour"].eq(local.hour)
    ]
    if len(pool) < 4:
        pool = keyed.loc[keyed["local_hour"].eq(local.hour)]
    values = pd.to_numeric(pool["consumption_mw"], errors="coerce").dropna()
    usual = float(values.median()) if not values.empty else demand
    anomaly = 0.0 if usual == 0 else (demand - usual) / usual
    return {"usual_mw": usual, "anomaly_pct": anomaly}


def _weather_summary(latest: pd.Series, weather: pd.DataFrame | None) -> dict[str, Any]:
    source = latest
    if weather is not None and not weather.empty:
        local_weather = weather.copy()
        local_weather["timestamp"] = pd.to_datetime(local_weather["timestamp"], utc=True, errors="coerce")
        local_weather = local_weather.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not local_weather.empty:
            source = local_weather.iloc[-1]
    temp = pd.to_numeric(source.get("weather_temperature_c"), errors="coerce")
    wind = pd.to_numeric(source.get("weather_wind_speed_kmh"), errors="coerce")
    cloud = pd.to_numeric(source.get("weather_cloud_cover_pct"), errors="coerce")
    if pd.isna(temp) and pd.isna(wind):
        return {"headline": "Weather context unavailable", "detail": "Demand drivers omit weather for this snapshot."}
    temp_value = 0.0 if pd.isna(temp) else float(temp)
    wind_value = 0.0 if pd.isna(wind) else float(wind)
    cloud_value = 0.0 if pd.isna(cloud) else float(cloud)
    if temp_value <= 5:
        headline = "Cold demand lift"
    elif temp_value >= 27:
        headline = "Heat demand lift"
    elif wind_value >= 35:
        headline = "Wind output context"
    else:
        headline = "Mild weather"
    return {
        "headline": headline,
        "detail": f"{temp_value:.1f} C, wind {wind_value:.0f} km/h, cloud {cloud_value:.0f}%.",
        "temperature_c": temp_value,
        "wind_kmh": wind_value,
        "cloud_pct": cloud_value,
    }


def _official_signal(ecowatt: pd.DataFrame | None, as_of: pd.Timestamp) -> dict[str, Any]:
    if ecowatt is None or ecowatt.empty or "timestamp" not in ecowatt:
        return {"label": "EcoWatt unavailable", "status": "unknown", "detail": "No official signal in this window."}
    frame = ecowatt.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return {"label": "EcoWatt unavailable", "status": "unknown", "detail": "No official signal in this window."}
    nearest = frame.iloc[(frame["timestamp"] - as_of).abs().argsort().iloc[0]]
    status = str(nearest.get("ecowatt_status", "unknown"))
    label = str(nearest.get("ecowatt_label", "Unknown"))
    return {
        "label": label,
        "status": status,
        "detail": str(nearest.get("ecowatt_message", "Official electricity-weather signal.")),
    }


def build_forecast_points(
    energy: pd.DataFrame,
    *,
    model_payload: dict[str, Any] | None = None,
    horizon_hours: int = 48,
    timezone: str = "Europe/Paris",
) -> list[ForecastPointView]:
    hourly = _hourly_energy(energy)
    if hourly.empty:
        return []
    latest_ts = _as_utc(hourly["timestamp"].max()).floor("h")
    availability = _latest_availability(hourly)
    model_points = _validated_model_points(model_payload, latest_ts)
    errors = _backtest_errors(model_payload)
    recent = pd.to_numeric(hourly["consumption_mw"], errors="coerce").dropna()
    default_error = max(float(recent.std() or 0), float(recent.mean() or 0) * 0.055, 1800.0)

    points: list[ForecastPointView] = []
    for horizon in range(1, horizon_hours + 1):
        target = latest_ts + pd.Timedelta(hours=horizon)
        reference = _reference_demand(hourly, target)
        source = "Recent comparable-hour fallback"
        p50 = reference
        no_recommendation = False

        if horizon <= 3:
            if horizon in model_points:
                p50 = model_points[horizon]
                source = "Local nowcast model"
            else:
                source = "Recent-pattern nowcast"
        else:
            rte_value = _rte_forecast_value(hourly, target)
            if rte_value is not None:
                p50 = rte_value
                source = "RTE forecast"
            elif horizon in model_points:
                p50 = model_points[horizon]
                source = "Validated model fallback"
            elif horizon > 24:
                source = "Unsupported horizon fallback"
                no_recommendation = True
        error = float(errors.get(horizon, errors.get(_nearest_key(errors, horizon), default_error)))
        p10 = max(p50 - 1.28 * error, 0.0)
        p90 = max(p50 + 1.28 * error, p50)
        label = "No recommendation" if no_recommendation else pressure_label(p50, availability, uncertainty_mw=p90 - p50)
        drivers = _forecast_drivers(target, p50, availability, source, timezone)
        points.append(
            ForecastPointView(
                timestamp=target,
                p10=p10,
                p50=p50,
                p90=p90,
                source=source,
                pressure_label=label,
                drivers=drivers,
                backtest_error=error,
                horizon_hours=horizon,
                usual_demand_mw=reference,
                route_label=source,
                fallback_reason="No validated public route beyond 24h." if no_recommendation else None,
            )
        )
    return points


def forecast_points_frame(points: list[ForecastPointView]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": point.timestamp,
                "p10": point.p10,
                "p50": point.p50,
                "p90": point.p90,
                "source": point.source,
                "pressure_label": point.pressure_label,
                "backtest_error": point.backtest_error,
                "horizon_hours": point.horizon_hours,
                "usual_demand_mw": point.usual_demand_mw,
                "route_label": point.route_label,
                "fallback_reason": point.fallback_reason,
            }
            for point in points
        ]
    )


def _hourly_energy(energy: pd.DataFrame) -> pd.DataFrame:
    if energy.empty or "timestamp" not in energy:
        return pd.DataFrame()
    frame = energy.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    numeric = frame.select_dtypes(include="number").columns.tolist()
    return frame.set_index("timestamp")[numeric].resample("1h").mean().reset_index()


def _latest_availability(hourly: pd.DataFrame) -> float:
    latest = hourly.sort_values("timestamp").iloc[-1]
    production = float(latest.get("total_production_mw", 0) or 0)
    imports = max(float(latest.get("imports_mw", 0) or 0), float(latest.get("net_imports_mw", 0) or 0), 0.0)
    return max(production + imports, float(latest.get("consumption_mw", 0) or 0))


def _reference_demand(hourly: pd.DataFrame, target: pd.Timestamp) -> float:
    frame = hourly[["timestamp", "consumption_mw"]].dropna().sort_values("timestamp")
    exact = frame.loc[frame["timestamp"].eq(target - pd.Timedelta(hours=24))]
    if not exact.empty:
        return float(exact.iloc[-1]["consumption_mw"])
    same_hour = frame.loc[frame["timestamp"].dt.hour.eq(target.hour)]
    if not same_hour.empty:
        return float(same_hour.tail(14)["consumption_mw"].median())
    return float(frame.tail(24)["consumption_mw"].mean())


def _validated_model_points(payload: dict[str, Any] | None, latest_ts: pd.Timestamp) -> dict[int, float]:
    if not payload:
        return {}
    predictions = pd.DataFrame(payload.get("predictions", []))
    if predictions.empty or "model_predicted_mw" not in predictions:
        return {}
    comparisons = pd.DataFrame(payload.get("baseline_comparison", []))
    valid_horizons: set[int] = set()
    if not comparisons.empty and "horizon_hours" in comparisons:
        if "model_beats_strongest_baseline" in comparisons:
            valid_horizons = {
                int(row.horizon_hours)
                for row in comparisons.itertuples()
                if bool(getattr(row, "model_beats_strongest_baseline", False))
            }
        elif "improvement_vs_strongest_baseline_percent" in comparisons:
            valid_horizons = {
                int(row.horizon_hours)
                for row in comparisons.itertuples()
                if float(getattr(row, "improvement_vs_strongest_baseline_percent", 0) or 0) > 0
            }
    for column in ("origin_timestamp", "target_timestamp"):
        if column in predictions:
            predictions[column] = pd.to_datetime(predictions[column], utc=True, errors="coerce")
    predictions = predictions.dropna(subset=["origin_timestamp"])
    if predictions.empty:
        return {}
    latest_origin = predictions["origin_timestamp"].max()
    if abs(latest_ts - latest_origin) > pd.Timedelta(hours=36):
        return {}
    rows = predictions.loc[predictions["origin_timestamp"].eq(latest_origin)]
    points: dict[int, float] = {}
    for row in rows.itertuples():
        horizon = int(getattr(row, "horizon_hours", 0))
        if valid_horizons and horizon not in valid_horizons:
            continue
        points[horizon] = float(getattr(row, "model_predicted_mw"))
    return points


def _backtest_errors(payload: dict[str, Any] | None) -> dict[int, float]:
    if not payload:
        return {}
    metrics = pd.DataFrame(payload.get("metrics", []))
    if metrics.empty:
        comparisons = pd.DataFrame(payload.get("baseline_comparison", []))
        if comparisons.empty:
            return {}
        return {
            int(row.horizon_hours): float(getattr(row, "model_mae_mw", 0) or 0)
            for row in comparisons.itertuples()
            if getattr(row, "model_mae_mw", None) is not None
        }
    if "model" in metrics:
        metrics = metrics.loc[metrics["model"].eq("demand_hgb")]
    return {
        int(row.horizon_hours): float(getattr(row, "mae_mw", 0) or 0)
        for row in metrics.itertuples()
        if getattr(row, "mae_mw", None) is not None
    }


def _nearest_key(values: dict[int, float], target: int) -> int:
    if not values:
        return target
    return min(values, key=lambda key: abs(key - target))


def _rte_forecast_value(hourly: pd.DataFrame, target: pd.Timestamp) -> float | None:
    candidates = ("prevision_j_mw", "prevision_j", "forecast_j_mw", "rte_forecast_mw")
    column = next((name for name in candidates if name in hourly.columns), None)
    if column is None:
        return None
    frame = hourly[["timestamp", column]].dropna().sort_values("timestamp")
    if frame.empty:
        return None
    exact = frame.loc[frame["timestamp"].eq(target)]
    if not exact.empty:
        return float(exact.iloc[-1][column])
    nearby = frame.loc[(frame["timestamp"] - target).abs() <= pd.Timedelta(minutes=40)]
    if nearby.empty:
        return None
    return float(nearby.iloc[(nearby["timestamp"] - target).abs().argsort().iloc[0]][column])


def _forecast_drivers(target: pd.Timestamp, demand: float, available: float, source: str, timezone: str) -> list[dict[str, Any]]:
    local = target.tz_convert(timezone)
    margin = available - demand
    hour_text = f"{local:%H:%M}"
    return [
        {"name": "Demand", "value": demand, "unit": "MW", "note": f"Expected electricity wanted near {hour_text}."},
        {"name": "Supply margin", "value": margin, "unit": "MW", "note": "Generation and import availability minus demand."},
        {"name": "Uncertainty", "value": 0, "unit": "", "note": f"Forecast route: {source}."},
    ]


def run_scenario(
    points: list[ForecastPointView],
    snapshot: GridSnapshotView,
    *,
    preset_key: str,
    intensity: float,
    start_hour: int,
    timezone: str = "Europe/Paris",
) -> ScenarioResultView:
    if preset_key not in SCENARIO_PRESETS:
        raise ValueError(f"unknown scenario preset: {preset_key}")
    baseline = forecast_points_frame(points)
    if baseline.empty:
        baseline = pd.DataFrame(columns=["timestamp", "p50", "p10", "p90", "pressure_label"])
    modified = baseline.copy()
    available = float(snapshot.availability.get("available_mw", 0) or 0)
    carbon = float(snapshot.demand.get("co2_intensity_g_per_kwh", 0) or 0)
    if carbon <= 0:
        carbon = 45.0
    modified["supply_mw"] = available
    modified["carbon_g_per_kwh"] = carbon
    modified["p50"] = pd.to_numeric(modified["p50"], errors="coerce").fillna(0.0)
    modified["p90"] = pd.to_numeric(modified["p90"], errors="coerce").fillna(modified["p50"])
    modified["demand_delta_mw"] = 0.0
    modified["supply_delta_mw"] = 0.0
    modified["carbon_delta_g_per_kwh"] = 0.0

    preset = SCENARIO_PRESETS[preset_key]
    assumptions = [preset["detail"]]
    explanation = ""
    local_hour = pd.to_datetime(modified["timestamp"], utc=True).dt.tz_convert(timezone).dt.hour if not modified.empty else pd.Series(dtype=int)

    if preset_key == "cold_snap":
        delta = max(float(intensity), 0.0) * 950.0
        profile = local_hour.map(lambda hour: 1.15 if hour in set(range(7, 10)) | set(range(18, 22)) else 0.7)
        modified["demand_delta_mw"] = delta * profile
        modified["p50"] += modified["demand_delta_mw"]
        modified["p90"] += modified["demand_delta_mw"]
        modified["carbon_delta_g_per_kwh"] = 4.0 * max(float(intensity), 0.0)
        explanation = "Cold weather raises heating demand, so pressure changes through demand."
        assumptions.append(f"{intensity:.1f} C colder adds about {delta:,.0f} MW before the hourly profile.")
    elif preset_key == "low_wind":
        loss = max(float(intensity), 0.0) * 1000
        evening = local_hour.between(17, 23)
        modified.loc[evening, "supply_delta_mw"] = -loss
        modified.loc[evening, "supply_mw"] -= loss
        modified.loc[evening, "carbon_delta_g_per_kwh"] = 12
        explanation = "Low wind reduces available renewable supply during the evening; demand is unchanged."
        assumptions.append(f"Wind availability is reduced by {loss:,.0f} MW during evening hours.")
    elif preset_key == "generation_outage":
        loss = max(float(intensity), 0.0) * 1000
        modified["supply_delta_mw"] = -loss
        modified["supply_mw"] -= loss
        modified["carbon_delta_g_per_kwh"] = 6
        explanation = "Generation unavailability reduces supply margin and can create pressure or imports."
        assumptions.append(f"{loss:,.0f} MW of firm supply is unavailable across the horizon.")
    elif preset_key == "ev_shift":
        evs = max(int(float(intensity)), 0)
        daily_energy_mwh = evs * 8 / 1000
        evening = local_hour.between(18, 21)
        overnight = local_hour.between(1, 5)
        modified.loc[evening, "demand_delta_mw"] -= daily_energy_mwh / 4
        modified.loc[overnight, "demand_delta_mw"] += daily_energy_mwh / 5
        modified["p50"] += modified["demand_delta_mw"]
        modified["p90"] += modified["demand_delta_mw"]
        modified.loc[overnight, "carbon_delta_g_per_kwh"] = -4
        explanation = "EV charging keeps the same energy but moves it away from evening pressure."
        assumptions.append(f"{evs:,} EVs shift about {daily_energy_mwh:,.0f} MWh per day from evening to overnight.")
    elif preset_key == "solar_above":
        gain = max(float(intensity), 0.0) * 1000
        daylight = local_hour.between(max(start_hour, 8), min(start_hour + 5, 18))
        modified.loc[daylight, "supply_delta_mw"] = gain
        modified.loc[daylight, "supply_mw"] += gain
        modified.loc[daylight, "carbon_delta_g_per_kwh"] = -10
        explanation = "Extra solar changes renewable supply and carbon intensity; demand is unchanged."
        assumptions.append(f"Solar is {gain:,.0f} MW above forecast during the selected daylight window.")

    modified["carbon_g_per_kwh"] = (modified["carbon_g_per_kwh"] + modified["carbon_delta_g_per_kwh"]).clip(lower=0)
    modified["pressure_label"] = modified.apply(
        lambda row: pressure_label(row["p50"], row["supply_mw"], uncertainty_mw=max(row["p90"] - row["p50"], 0)),
        axis=1,
    )
    baseline_supply = available
    baseline_import = (baseline["p50"] - baseline_supply).clip(lower=0) if not baseline.empty else pd.Series(dtype=float)
    modified_import = (modified["p50"] - modified["supply_mw"]).clip(lower=0) if not modified.empty else pd.Series(dtype=float)
    baseline_peak = float(baseline["p50"].max()) if not baseline.empty else 0.0
    modified_peak = float(modified["p50"].max()) if not modified.empty else 0.0
    demand_energy_delta = float(modified["demand_delta_mw"].sum()) if "demand_delta_mw" in modified else 0.0
    carbon_tonnes = float(((modified["p50"] * modified["carbon_delta_g_per_kwh"]) / 1000).sum()) if not modified.empty else 0.0
    deltas = {
        "peak_demand_mw": modified_peak - baseline_peak,
        "high_pressure_hours": int(modified["pressure_label"].isin({"Watch", "High"}).sum()),
        "import_requirement_mw": float(modified_import.max() - baseline_import.max()) if not modified_import.empty else 0.0,
        "carbon_tonnes": carbon_tonnes,
        "total_energy_delta_mwh": demand_energy_delta,
        "supply_delta_mw": float(modified["supply_delta_mw"].min()) if not modified.empty else 0.0,
    }
    if preset_key == "ev_shift":
        deltas["total_energy_delta_mwh"] = round(demand_energy_delta, 6)
    return ScenarioResultView(
        baseline=baseline,
        modified=modified,
        deltas=deltas,
        assumptions=assumptions,
        explanation=explanation,
    )


def scenario_regional_map_frame(regional: pd.DataFrame, result: ScenarioResultView) -> pd.DataFrame:
    frame = regional.copy()
    if frame.empty:
        return frame
    demand_delta = float(result.deltas.get("peak_demand_mw", 0) or 0)
    supply_loss = abs(float(result.deltas.get("supply_delta_mw", 0) or 0))
    national_demand = max(float(frame["consumption_mw"].sum()), 1.0)
    demand_lift = demand_delta / national_demand
    pressure_lift = min(supply_loss / national_demand, 0.35)
    base = frame.get("demand_anomaly_score", frame.get("demand_pressure", pd.Series(0.5, index=frame.index)))
    frame["scenario_score"] = (base + demand_lift + pressure_lift).clip(0, 1)
    frame["demand_anomaly_label"] = frame["demand_anomaly_label"].astype(str) + " -> national scenario overlay"
    return frame
