from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go

from src.models.mood_calibration import FIXED_THRESHOLDS, season_for_month

STATUS_CODES = {
    "Unknown": 0,
    "Comfortable": 1,
    "Watch": 2,
    "Tense": 3,
    "Low-carbon opportunity": 4,
}
STATUS_COLORS = {
    "Unknown": "#9ca3af",
    "Comfortable": "#10b981",
    "Watch": "#f59e0b",
    "Tense": "#ef4444",
    "Low-carbon opportunity": "#0284c7",
}
STATUS_SHORT_LABELS = {
    "Unknown": "?",
    "Comfortable": "OK",
    "Watch": "Watch",
    "Tense": "Tense",
    "Low-carbon opportunity": "Low CO2",
}
ECOWATT_STATUS_CODES = {
    "unknown": STATUS_CODES["Unknown"],
    "green": STATUS_CODES["Comfortable"],
    "orange": STATUS_CODES["Watch"],
    "red": STATUS_CODES["Tense"],
}
ECOWATT_SHORT_LABELS = {
    "unknown": "?",
    "green": "Eco OK",
    "orange": "Orange",
    "red": "Red",
}
MODEL_HORIZONS = (1, 3, 6, 24)
FRESH_MODEL_MAX_AGE = pd.Timedelta(hours=36)
RTE_FORECAST_COLUMNS = (
    "prevision_j_mw",
    "prevision_j",
    "previsionj",
    "forecast_j_mw",
    "j_forecast_mw",
    "prevision_j1_mw",
    "prevision_j_1_mw",
    "prevision_j1",
    "prevision_j_1",
    "previsionj1",
    "forecast_j_1_mw",
)


@dataclass(frozen=True)
class EnergyWeatherBuild:
    timeline: pd.DataFrame
    metadata: dict[str, Any]


def _as_utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _hourly_history(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "timestamp" not in data:
        return pd.DataFrame(columns=["timestamp"])
    frame = data.copy()
    frame["timestamp"] = _as_utc_series(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return pd.DataFrame(columns=["timestamp"])
    numeric = frame.select_dtypes(include="number").columns.tolist()
    hourly = (
        frame.set_index("timestamp")[numeric]
        .resample("1h")
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )
    return hourly


def _reference_by_timestamp(
    timeline: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    target_column: str,
    value_column: str,
    output_column: str,
    tolerance: pd.Timedelta = pd.Timedelta(minutes=40),
) -> pd.DataFrame:
    if hourly.empty or value_column not in hourly:
        timeline[output_column] = float("nan")
        return timeline
    reference = hourly[["timestamp", value_column]].rename(
        columns={"timestamp": target_column, value_column: output_column}
    )
    merged = pd.merge_asof(
        timeline.sort_values(target_column),
        reference.sort_values(target_column),
        on=target_column,
        direction="nearest",
        tolerance=tolerance,
    )
    return merged.sort_values("target").reset_index(drop=True)


def _repeat_recent_hourly(
    timeline: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    if output_column not in timeline:
        timeline[output_column] = float("nan")
    if hourly.empty or value_column not in hourly or timeline[output_column].notna().all():
        return timeline
    recent = hourly[value_column].dropna().tail(24).tolist()
    if not recent:
        return timeline
    repeated = (recent * ((len(timeline) // len(recent)) + 1))[: len(timeline)]
    timeline[output_column] = timeline[output_column].fillna(pd.Series(repeated, index=timeline.index))
    return timeline


def _find_first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_lookup = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    return None


def _add_rte_forecast(timeline: pd.DataFrame, hourly: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    column = _find_first_column(hourly, RTE_FORECAST_COLUMNS)
    timeline["rte_forecast_mw"] = float("nan")
    if column is None:
        return timeline, "unavailable"
    forecast = hourly[["timestamp", column]].rename(columns={column: "rte_forecast_mw"})
    timeline = pd.merge_asof(
        timeline.sort_values("target"),
        forecast.sort_values("timestamp"),
        left_on="target",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=40),
    ).drop(columns=["timestamp_y"], errors="ignore")
    timeline = timeline.rename(columns={"timestamp_x": "timestamp"})
    return timeline.sort_values("target").reset_index(drop=True), column


def _add_model_predictions(
    timeline: pd.DataFrame,
    payload: Mapping[str, Any] | None,
    latest_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    timeline["model_predicted_mw"] = float("nan")
    if not payload:
        return timeline, {"status": "unavailable"}

    predictions = pd.DataFrame(payload.get("predictions", []))
    if predictions.empty or "model_predicted_mw" not in predictions:
        return timeline, {"status": "unavailable"}

    for column in ("origin_timestamp", "target_timestamp"):
        if column in predictions:
            predictions[column] = pd.to_datetime(predictions[column], utc=True, errors="coerce")
    predictions = predictions.dropna(subset=["origin_timestamp", "target_timestamp"])
    if predictions.empty:
        return timeline, {"status": "unavailable"}

    latest_origin = predictions["origin_timestamp"].max()
    age = abs(latest_ts - latest_origin)
    available_horizons = sorted(
        int(value) for value in predictions["horizon_hours"].dropna().unique()
        if int(value) in MODEL_HORIZONS
    )
    metadata = {
        "status": "fresh" if age <= FRESH_MODEL_MAX_AGE else "stale",
        "latest_origin": latest_origin.isoformat(),
        "available_horizons": available_horizons,
        "age_hours": age / pd.Timedelta(hours=1),
    }
    if metadata["status"] != "fresh":
        return timeline, metadata

    latest = predictions.loc[
        predictions["origin_timestamp"].eq(latest_origin)
        & predictions["horizon_hours"].isin(MODEL_HORIZONS)
    ].copy()
    if latest.empty:
        return timeline, metadata

    latest["target"] = latest_origin + pd.to_timedelta(latest["horizon_hours"], unit="h")
    latest = latest[["target", "model_predicted_mw", "horizon_hours"]].sort_values("target")
    timeline = timeline.merge(latest, on="target", how="left")
    if "model_predicted_mw_y" in timeline:
        timeline["model_predicted_mw"] = timeline["model_predicted_mw_y"].combine_first(
            timeline["model_predicted_mw_x"]
        )
        timeline = timeline.drop(columns=["model_predicted_mw_x", "model_predicted_mw_y"])
    return timeline, metadata


def _add_ecowatt_overlay(
    timeline: pd.DataFrame,
    ecowatt: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    defaults = {
        "ecowatt_status": "unknown",
        "ecowatt_label": "Unknown",
        "ecowatt_severity": 0,
        "ecowatt_message": "No EcoWatt signal is available for this hour.",
        "ecowatt_source": "Unavailable",
        "ecowatt_source_url": "",
    }
    for column, value in defaults.items():
        timeline[column] = value

    if ecowatt is None or ecowatt.empty or "timestamp" not in ecowatt:
        timeline["ecowatt_status_code"] = timeline["ecowatt_status"].map(ECOWATT_STATUS_CODES)
        timeline["ecowatt_short_label"] = timeline["ecowatt_status"].map(ECOWATT_SHORT_LABELS)
        return timeline, {"status": "unavailable"}

    source = ecowatt.copy()
    source["timestamp"] = _as_utc_series(source["timestamp"])
    source = source.dropna(subset=["timestamp"]).sort_values("timestamp")
    if source.empty:
        timeline["ecowatt_status_code"] = timeline["ecowatt_status"].map(ECOWATT_STATUS_CODES)
        timeline["ecowatt_short_label"] = timeline["ecowatt_status"].map(ECOWATT_SHORT_LABELS)
        return timeline, {"status": "unavailable"}

    columns = ["timestamp"] + [column for column in defaults if column in source]
    merged = pd.merge_asof(
        timeline.sort_values("target"),
        source[columns].sort_values("timestamp").rename(columns={"timestamp": "ecowatt_timestamp"}),
        left_on="target",
        right_on="ecowatt_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=35),
        suffixes=("", "_official"),
    )
    for column, value in defaults.items():
        official = f"{column}_official"
        if official in merged:
            merged[column] = merged[official].combine_first(merged[column])
            merged = merged.drop(columns=[official])
        merged[column] = merged[column].fillna(value)

    merged["ecowatt_status_code"] = merged["ecowatt_status"].map(ECOWATT_STATUS_CODES).fillna(0)
    merged["ecowatt_short_label"] = merged["ecowatt_status"].map(ECOWATT_SHORT_LABELS).fillna("?")
    sources = sorted(
        str(value)
        for value in merged["ecowatt_source"].dropna().unique()
        if str(value) != "Unavailable"
    )
    return merged.sort_values("target").reset_index(drop=True), {
        "status": "available" if sources else "unavailable",
        "source_counts": merged["ecowatt_status"].value_counts().to_dict(),
        "sources": sources,
    }


def _thresholds_for_target(
    target: pd.Timestamp,
    artifact: Mapping[str, Any] | None,
    timezone: str,
) -> tuple[dict[str, float], str]:
    if not artifact:
        return FIXED_THRESHOLDS.copy(), "fixed"
    local = target.tz_convert(str(artifact.get("timezone", timezone)))
    season = season_for_month(local.month)
    hour = int(local.hour)
    lookup = {
        (item.get("level"), item.get("season"), item.get("local_hour")): item
        for item in artifact.get("segments", [])
    }
    minimum = int(artifact.get("min_sample", 1))
    for key in (
        ("season_hour", season, hour),
        ("season", season, None),
        ("local_hour", None, hour),
        ("global", None, None),
    ):
        segment = lookup.get(key)
        if segment is not None and int(segment.get("sample", 0)) >= minimum:
            return dict(segment.get("thresholds", {})), str(segment.get("level", "unknown"))
    return dict(artifact.get("fixed_thresholds", FIXED_THRESHOLDS)), "fixed"


def _status_for_row(row: pd.Series, recent_quantiles: Mapping[str, float]) -> str:
    demand = pd.to_numeric(row.get("demand_signal_mw"), errors="coerce")
    co2 = pd.to_numeric(row.get("co2_intensity_g_per_kwh"), errors="coerce")
    demand_missing = pd.isna(demand)
    co2_missing = pd.isna(co2)
    if demand_missing and co2_missing:
        return "Unknown"

    high_demand = float(row.get("demand_high_threshold_mw") or recent_quantiles["demand_q85"])
    watch_demand = float(recent_quantiles["demand_q60"])
    low_co2 = float(row.get("co2_low_threshold") or recent_quantiles["co2_q25"])
    high_co2 = float(row.get("co2_high_threshold") or recent_quantiles["co2_q75"])

    if not demand_missing and float(demand) >= high_demand:
        return "Tense"
    if not co2_missing and float(co2) >= high_co2:
        return "Watch"
    if (
        not co2_missing
        and float(co2) <= low_co2
        and (demand_missing or float(demand) < watch_demand)
    ):
        return "Low-carbon opportunity"
    if not demand_missing and float(demand) >= watch_demand:
        return "Watch"
    return "Comfortable"


def build_energy_weather_timeline(
    data: pd.DataFrame,
    *,
    latest_ts: pd.Timestamp | None = None,
    model_payload: Mapping[str, Any] | None = None,
    mood_artifact: Mapping[str, Any] | None = None,
    ecowatt: pd.DataFrame | None = None,
    timezone: str = "Europe/Paris",
) -> EnergyWeatherBuild:
    """Build the public 24h status timeline from the best available artifacts."""
    hourly = _hourly_history(data)
    if latest_ts is None:
        if hourly.empty:
            latest_ts = pd.Timestamp.now(tz="UTC")
        else:
            latest_ts = hourly["timestamp"].max()
    latest_ts = pd.Timestamp(latest_ts)
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.tz_localize("UTC")
    else:
        latest_ts = latest_ts.tz_convert("UTC")

    start = latest_ts.floor("h") + pd.Timedelta(hours=1)
    timeline = pd.DataFrame({"target": pd.date_range(start=start, periods=24, freq="h")})
    timeline["reference_timestamp"] = timeline["target"] - pd.Timedelta(hours=24)

    timeline = _reference_by_timestamp(
        timeline,
        hourly,
        target_column="reference_timestamp",
        value_column="consumption_mw",
        output_column="reference_consumption_mw",
    )
    timeline = _repeat_recent_hourly(
        timeline,
        hourly,
        value_column="consumption_mw",
        output_column="reference_consumption_mw",
    )
    timeline = _reference_by_timestamp(
        timeline,
        hourly,
        target_column="reference_timestamp",
        value_column="co2_intensity_g_per_kwh",
        output_column="co2_intensity_g_per_kwh",
    )
    timeline = _repeat_recent_hourly(
        timeline,
        hourly,
        value_column="co2_intensity_g_per_kwh",
        output_column="co2_intensity_g_per_kwh",
    )
    timeline, rte_source = _add_rte_forecast(timeline, hourly)
    timeline, model_metadata = _add_model_predictions(timeline, model_payload, latest_ts)

    timeline["demand_signal_mw"] = timeline["model_predicted_mw"].combine_first(
        timeline["rte_forecast_mw"]
    ).combine_first(timeline["reference_consumption_mw"])
    timeline["demand_source"] = "Recent same-hour pattern"
    timeline.loc[timeline["rte_forecast_mw"].notna(), "demand_source"] = "RTE J/J-1 forecast"
    timeline.loc[timeline["model_predicted_mw"].notna(), "demand_source"] = "Demand model"
    timeline.loc[timeline["demand_signal_mw"].isna(), "demand_source"] = "Unavailable"
    timeline["co2_source"] = "Recent same-hour pattern"
    timeline.loc[timeline["co2_intensity_g_per_kwh"].isna(), "co2_source"] = "Unavailable"

    demand_series = hourly.get("consumption_mw", pd.Series(dtype="float64")).dropna()
    co2_series = hourly.get("co2_intensity_g_per_kwh", pd.Series(dtype="float64")).dropna()
    recent_quantiles = {
        "demand_q60": float(demand_series.quantile(0.60)) if not demand_series.empty else FIXED_THRESHOLDS["consumption_high"] * 0.85,
        "demand_q85": float(demand_series.quantile(0.85)) if not demand_series.empty else FIXED_THRESHOLDS["consumption_high"],
        "co2_q25": float(co2_series.quantile(0.25)) if not co2_series.empty else FIXED_THRESHOLDS["co2_low"],
        "co2_q75": float(co2_series.quantile(0.75)) if not co2_series.empty else FIXED_THRESHOLDS["co2_high"],
    }

    threshold_rows = timeline["target"].apply(
        lambda target: _thresholds_for_target(target, mood_artifact, timezone)
    )
    timeline["threshold_source"] = [item[1] for item in threshold_rows]
    timeline["demand_high_threshold_mw"] = [
        item[0].get("consumption_high", recent_quantiles["demand_q85"]) for item in threshold_rows
    ]
    timeline["co2_low_threshold"] = [
        item[0].get("co2_low", recent_quantiles["co2_q25"]) for item in threshold_rows
    ]
    timeline["co2_high_threshold"] = [
        item[0].get("co2_high", recent_quantiles["co2_q75"]) for item in threshold_rows
    ]
    timeline["status"] = timeline.apply(_status_for_row, axis=1, recent_quantiles=recent_quantiles)
    timeline["status_code"] = timeline["status"].map(STATUS_CODES)
    timeline["status_label"] = timeline["status"].map(STATUS_SHORT_LABELS)
    timeline, ecowatt_metadata = _add_ecowatt_overlay(timeline, ecowatt)
    timeline["local_time"] = timeline["target"].dt.tz_convert(timezone)
    timeline["hour_label"] = timeline["local_time"].dt.strftime("%H:%M")

    metadata = {
        "origin_timestamp": latest_ts.isoformat(),
        "timeline_start": start.isoformat(),
        "demand_source_counts": timeline["demand_source"].value_counts().to_dict(),
        "model": model_metadata,
        "rte_forecast_source": rte_source,
        "ecowatt": ecowatt_metadata,
        "recent_quantiles": recent_quantiles,
    }
    return EnergyWeatherBuild(timeline=timeline, metadata=metadata)


def _hour_ranges(frame: pd.DataFrame, statuses: set[str], *, limit: int = 4) -> str:
    selected = frame.loc[frame["status"].isin(statuses), "hour_label"].head(limit).tolist()
    return ", ".join(selected) if selected else "None visible in the next 24h"


def summarize_energy_weather(timeline: pd.DataFrame, metadata: Mapping[str, Any]) -> dict[str, str]:
    if timeline.empty:
        return {
            "shift": "No reliable window yet",
            "avoid": "No reliable warning yet",
            "confidence": "Unknown: no hourly signals are available.",
        }
    shift = _hour_ranges(timeline, {"Low-carbon opportunity", "Comfortable"})
    official_tense = (
        timeline.loc[timeline["ecowatt_status"].isin({"orange", "red"}), "hour_label"].head(4).tolist()
        if "ecowatt_status" in timeline
        else []
    )
    avoid = ", ".join(official_tense) if official_tense else _hour_ranges(timeline, {"Tense"})
    if avoid == "None visible in the next 24h":
        avoid = _hour_ranges(timeline, {"Watch"})

    model = dict(metadata.get("model", {}))
    model_status = model.get("status", "unavailable")
    source_counts = dict(metadata.get("demand_source_counts", {}))
    if model_status == "fresh":
        horizons = ", ".join(f"{h}h" for h in model.get("available_horizons", [])) or "selected"
        confidence = f"Medium: fresh model points are available at {horizons}; other hours use measured context."
    elif model_status == "stale":
        confidence = "Low: the model artifact is historical, so the timeline uses recent measured patterns instead."
    elif source_counts.get("RTE J/J-1 forecast"):
        confidence = "Medium: RTE forecast values are present, with measured CO2 context where available."
    elif source_counts.get("Recent same-hour pattern"):
        confidence = "Low to medium: this is a recent-pattern outlook, not a live operational forecast."
    else:
        confidence = "Unknown: demand and CO2 signals are missing."
    return {"shift": shift, "avoid": avoid, "confidence": confidence}


def energy_weather_heatmap(timeline: pd.DataFrame) -> go.Figure:
    if timeline.empty:
        timeline = pd.DataFrame(
            {
                "hour_label": ["--:--"],
                "status": ["Unknown"],
                "status_code": [STATUS_CODES["Unknown"]],
                "status_label": [STATUS_SHORT_LABELS["Unknown"]],
                "demand_signal_mw": [pd.NA],
                "demand_source": ["Unavailable"],
                "co2_intensity_g_per_kwh": [pd.NA],
                "co2_source": ["Unavailable"],
                "threshold_source": ["Unavailable"],
                "ecowatt_status": ["unknown"],
                "ecowatt_status_code": [ECOWATT_STATUS_CODES["unknown"]],
                "ecowatt_short_label": [ECOWATT_SHORT_LABELS["unknown"]],
                "ecowatt_label": ["Unknown"],
                "ecowatt_message": ["No EcoWatt signal is available."],
                "ecowatt_source": ["Unavailable"],
            }
        )
    color_values = [STATUS_CODES[label] for label in STATUS_CODES]
    colorscale = []
    max_code = max(color_values)
    for label, code in STATUS_CODES.items():
        start = max((code - 0.5) / max_code, 0)
        end = min((code + 0.5) / max_code, 1)
        colorscale.extend([(start, STATUS_COLORS[label]), (end, STATUS_COLORS[label])])

    for column, value in (
        ("threshold_source", "Unavailable"),
        ("ecowatt_status", "unknown"),
        ("ecowatt_status_code", ECOWATT_STATUS_CODES["unknown"]),
        ("ecowatt_short_label", ECOWATT_SHORT_LABELS["unknown"]),
        ("ecowatt_label", "Unknown"),
        ("ecowatt_message", "No EcoWatt signal is available."),
        ("ecowatt_source", "Unavailable"),
    ):
        if column not in timeline:
            timeline[column] = value

    app_custom = [
        [
            row.status,
            f"Demand signal: {row.demand_signal_mw:,.0f} MW"
            if pd.notna(row.demand_signal_mw)
            else "Demand signal: unavailable",
            f"CO2 intensity: {row.co2_intensity_g_per_kwh:,.0f} g/kWh"
            if pd.notna(row.co2_intensity_g_per_kwh)
            else "CO2 intensity: unavailable",
            f"Demand source: {row.demand_source}; thresholds: {row.threshold_source}",
        ]
        for row in timeline.itertuples()
    ]
    ecowatt_custom = [
        [
            row.ecowatt_label,
            "EcoWatt is the official electricity weather signal.",
            str(row.ecowatt_message or "No official message for this hour."),
            f"Source: {row.ecowatt_source}",
        ]
        for row in timeline.itertuples()
    ]
    fig = go.Figure(
        go.Heatmap(
            z=[timeline["status_code"].tolist(), timeline["ecowatt_status_code"].tolist()],
            x=timeline["hour_label"].tolist(),
            y=["App outlook", "EcoWatt"],
            text=[timeline["status_label"].tolist(), timeline["ecowatt_short_label"].tolist()],
            texttemplate="%{text}",
            textfont=dict(color="white", size=12),
            customdata=[app_custom, ecowatt_custom],
            colorscale=colorscale,
            zmin=0,
            zmax=max_code,
            colorbar=dict(
                title=None,
                tickmode="array",
                tickvals=list(STATUS_CODES.values()),
                ticktext=list(STATUS_CODES.keys()),
                len=0.78,
            ),
            hovertemplate=(
                "<b>%{y} - %{x}</b><br>"
                "Status: %{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                "%{customdata[2]}<br>"
                "%{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,17,29,0.35)",
        font=dict(color="#dbeafe"),
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title=None, tickangle=0),
        yaxis=dict(title=None, showticklabels=False),
        hovermode="closest",
    )
    return fig
