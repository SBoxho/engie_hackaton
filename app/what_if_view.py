from __future__ import annotations

from dataclasses import dataclass
import html
from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go

from app.components.charts import dark_chart_layout
from app.components.regional_map import (
    DIVERGING_COLORSCALE,
    clipped_diverging_values,
    diverging_metric_range,
    signed_mw_tick,
)
from app.formatting import format_mw, format_signed_mw, format_timestamp, format_uncertainty_range
from src.data_sources.rte_eco2mix_regional import REGION_NAMES


SCENARIO_UI_VERSION = "what-if-builder.v1"

SCENARIO_PRESETS: dict[str, dict[str, str]] = {
    "cold_snap": {
        "label": "Colder weather",
        "short": "Cold snap",
        "concept": "Demand",
        "detail": "Heating-sensitive demand changes during the selected window.",
    },
    "generation_unavailability": {
        "label": "Generation unavailable",
        "short": "Unit unavailable",
        "concept": "Supply",
        "detail": "Available generation is reduced during the selected window.",
    },
    "ev_charging_shift": {
        "label": "EV charging shift",
        "short": "EV shift",
        "concept": "Demand timing",
        "detail": "Evening charging is moved into the overnight target window.",
    },
}

SCENARIO_ALIASES = {
    "generation_outage": "generation_unavailability",
    "outage": "generation_unavailability",
    "ev_shift": "ev_charging_shift",
}


@dataclass(frozen=True)
class ScenarioControls:
    scenario_type: str
    scope_type: str
    region: str
    magnitude: float
    start_offset_hours: int
    end_offset_hours: int
    asset_name: str
    ev_energy_kwh: float
    ev_participation: float
    original_window: tuple[str, str]
    target_window: tuple[str, str]

    @property
    def duration_hours(self) -> int:
        return self.end_offset_hours - self.start_offset_hours


@dataclass(frozen=True)
class ScenarioSummary:
    peak_demand_delta_mw: float
    min_balance_score_delta: float
    watch_high_hour_delta: int
    changed_watch_high_hours: int
    import_export_delta_mwh_range: tuple[float, float]
    carbon_delta_tonnes_range: tuple[float, float]


def normalize_scenario_key(value: Any) -> str:
    key = str(value or "cold_snap").strip().lower()
    key = SCENARIO_ALIASES.get(key, key)
    return key if key in SCENARIO_PRESETS else "cold_snap"


def restore_scenario_controls(
    query_params: Mapping[str, Any],
    *,
    default_scenario: str | None = None,
    default_region: str = "11",
) -> ScenarioControls:
    scenario_type = normalize_scenario_key(_first(query_params, "scenario") or default_scenario)
    start = _bounded_int(_first(query_params, "start"), default=18, minimum=0, maximum=47)
    duration = _bounded_int(_first(query_params, "duration"), default=_default_duration(scenario_type), minimum=1, maximum=48)
    end = min(start + duration, 48)
    if end <= start:
        end = min(start + 1, 48)
        start = max(0, end - 1)
    scope_type = str(_first(query_params, "scope") or "national").strip().lower()
    if scope_type not in {"national", "regional"} or scenario_type == "generation_unavailability":
        scope_type = "national"
    return ScenarioControls(
        scenario_type=scenario_type,
        scope_type=scope_type,
        region=str(_first(query_params, "region") or default_region),
        magnitude=_defaulted_float(_first(query_params, "mag"), default=_default_magnitude(scenario_type)),
        start_offset_hours=start,
        end_offset_hours=end,
        asset_name=str(_first(query_params, "asset") or "Unit A"),
        ev_energy_kwh=_defaulted_float(_first(query_params, "ev_kwh"), default=8.0),
        ev_participation=_bounded_float(_first(query_params, "participation"), default=0.5, minimum=0.05, maximum=1.0),
        original_window=_window_from_query(_first(query_params, "source_window"), default=("18:00", "22:00")),
        target_window=_window_from_query(_first(query_params, "target_window"), default=("01:00", "05:00")),
    )


def scenario_query_params(controls: ScenarioControls) -> dict[str, str]:
    params = {
        "scenario": controls.scenario_type,
        "scope": controls.scope_type,
        "region": controls.region,
        "mag": _compact_number(controls.magnitude),
        "start": str(controls.start_offset_hours),
        "duration": str(controls.duration_hours),
    }
    if controls.scenario_type == "generation_unavailability":
        params["asset"] = controls.asset_name
    if controls.scenario_type == "ev_charging_shift":
        params.update(
            {
                "ev_kwh": _compact_number(controls.ev_energy_kwh),
                "participation": _compact_number(controls.ev_participation),
                "source_window": "-".join(controls.original_window),
                "target_window": "-".join(controls.target_window),
            }
        )
    return params


def build_scenario_request(
    controls: ScenarioControls,
    *,
    baseline_from_time: pd.Timestamp,
    timezone_name: str,
) -> dict[str, Any]:
    baseline = _as_utc(baseline_from_time).floor("h")
    start = baseline + pd.Timedelta(hours=controls.start_offset_hours)
    end = baseline + pd.Timedelta(hours=controls.end_offset_hours)
    scenario_type = normalize_scenario_key(controls.scenario_type)
    scope: str | dict[str, Any]
    if controls.scope_type == "regional" and scenario_type != "generation_unavailability":
        scope = {"type": "regional", "regions": [controls.region]}
    else:
        scope = "national"

    if scenario_type == "cold_snap":
        magnitude: dict[str, Any] = {"temperature_delta_c": -abs(float(controls.magnitude))}
    elif scenario_type == "generation_unavailability":
        magnitude = {
            "unavailable_capacity_mw": abs(float(controls.magnitude)) * 1000.0,
            "asset_name": controls.asset_name or None,
        }
        scope = "national"
    else:
        magnitude = {
            "vehicles": max(int(round(controls.magnitude)), 1),
            "average_energy_per_vehicle_kwh": max(float(controls.ev_energy_kwh), 0.1),
            "participation_rate": min(max(float(controls.ev_participation), 0.01), 1.0),
            "original_charging_window": {"start": controls.original_window[0], "end": controls.original_window[1]},
            "target_charging_window": {"start": controls.target_window[0], "end": controls.target_window[1]},
        }
    return {
        "scenario_type": scenario_type,
        "magnitude": magnitude,
        "scope": scope,
        "start_time": start.isoformat().replace("+00:00", "Z"),
        "end_time": end.isoformat().replace("+00:00", "Z"),
        "baseline_from_time": baseline.isoformat().replace("+00:00", "Z"),
        "hours": 48,
        "timezone": timezone_name,
        "assumptions": {
            "interface": SCENARIO_UI_VERSION,
            "estimate": "educational directional scenario",
            "single_active_scenario": True,
        },
        "user_label": SCENARIO_PRESETS[scenario_type]["label"],
    }


def validate_controls(controls: ScenarioControls) -> str | None:
    if controls.end_offset_hours <= controls.start_offset_hours:
        return "Scenario end time must be after start time."
    if controls.scenario_type == "ev_charging_shift" and controls.original_window == controls.target_window:
        return "EV source and target charging windows must be different."
    if controls.scenario_type == "ev_charging_shift" and controls.ev_participation <= 0:
        return "EV participation must be greater than zero."
    if controls.magnitude <= 0:
        return "Scenario magnitude must be greater than zero."
    return None


def scenario_frames(result: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.DataFrame(result.get("baseline_series") or [])
    scenario = pd.DataFrame(result.get("scenario_series") or [])
    for frame in (baseline, scenario):
        if not frame.empty and "timestamp" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame.dropna(subset=["timestamp"], inplace=True)
            frame.sort_values("timestamp", inplace=True, kind="stable")
    return baseline, scenario


def scenario_summary(result: Mapping[str, Any]) -> ScenarioSummary:
    baseline, scenario = scenario_frames(result)
    min_balance_delta = 0.0
    watch_high_delta = 0
    if not baseline.empty and not scenario.empty:
        baseline_score = pd.to_numeric(baseline.get("balance_score"), errors="coerce")
        scenario_score = pd.to_numeric(scenario.get("balance_score"), errors="coerce")
        if baseline_score.notna().any() and scenario_score.notna().any():
            min_balance_delta = float(scenario_score.min() - baseline_score.min())
        baseline_status = baseline.get("balance_status", pd.Series(dtype=str)).astype(str).str.lower()
        scenario_status = scenario.get("balance_status", pd.Series(dtype=str)).astype(str).str.lower()
        watch_high_delta = int(scenario_status.isin({"watch", "high"}).sum() - baseline_status.isin({"watch", "high"}).sum())
    import_range = _range_tuple(
        result.get("estimated_import_export_delta", {}).get("net_import_delta_mwh_range")
    )
    carbon_range = _range_tuple(
        result.get("estimated_carbon_range", {}).get("total_tonnes_co2_delta_range")
    )
    return ScenarioSummary(
        peak_demand_delta_mw=float(result.get("peak_demand_delta_mw") or 0.0),
        min_balance_score_delta=min_balance_delta,
        watch_high_hour_delta=watch_high_delta,
        changed_watch_high_hours=int(result.get("changed_watch_or_high_hour_count") or 0),
        import_export_delta_mwh_range=import_range,
        carbon_delta_tonnes_range=carbon_range,
    )


def magnitude_label_from_request(result: Mapping[str, Any]) -> str:
    request = result.get("scenario_request") or {}
    scenario_type = str(request.get("scenario_type") or "")
    magnitude = request.get("magnitude") or {}
    if scenario_type == "cold_snap":
        value = abs(float(magnitude.get("temperature_delta_c") or 0.0))
        return f"{value:g} deg C colder"
    if scenario_type == "generation_unavailability":
        value = float(magnitude.get("unavailable_capacity_mw") or 0.0)
        asset = magnitude.get("asset_name")
        label = format_mw(value)
        return f"{label} unavailable" if not asset else f"{asset}: {label} unavailable"
    if scenario_type == "ev_charging_shift":
        vehicles = int(float(magnitude.get("vehicles") or 0))
        energy = float(magnitude.get("average_energy_per_vehicle_kwh") or 0.0)
        participation = float(magnitude.get("participation_rate") or 0.0)
        return f"{vehicles:,} EVs, {energy:g} kWh each, {participation:.0%} participating"
    return "Scenario magnitude"


def scenario_window_label(result: Mapping[str, Any], *, timezone_name: str) -> str:
    request = result.get("scenario_request") or {}
    start = _as_utc(request.get("start_time"))
    end = _as_utc(request.get("end_time"))
    duration = float(request.get("duration_hours") or 0.0)
    return (
        f"{format_timestamp(start, timezone_name=timezone_name, include_date=False)} to "
        f"{format_timestamp(end, timezone_name=timezone_name, include_date=False)} ({duration:g}h window)"
    )


def causal_chain_steps(result: Mapping[str, Any]) -> list[tuple[str, str]]:
    chain = result.get("causal_chain") or {}
    scenario_type = str(chain.get("scenario_type") or result.get("scenario_request", {}).get("scenario_type") or "")
    if scenario_type == "cold_snap":
        return [
            ("Colder weather", "Temperature input changes"),
            ("Heating demand rises", "Demand P50 is rerun for active hours"),
            ("Residual load rises", "More demand remains after weather-sensitive supply"),
            ("Flexible generation/imports respond", "Estimated response ranges absorb part of the residual"),
            ("Balance tightens", "App balance status can move"),
        ]
    if scenario_type == "generation_unavailability":
        return [
            ("Unit unavailable", "Capacity is removed from availability"),
            ("Available generation falls", "Demand is unchanged"),
            ("Residual load rises", "The same demand has less available supply"),
            ("Flexible supply/imports respond", "Estimated response ranges absorb part of the residual"),
            ("Balance context changes", "App balance status can move"),
        ]
    return [
        ("EV charging shifts", "The same energy moves windows"),
        ("Evening demand falls", "Source-window demand delta is negative"),
        ("Overnight demand rises", "Target-window demand delta is positive"),
        ("Peak may move", "Rebound hours are flagged by the scenario API"),
        ("Carbon range changes", "Response-mix emissions are estimated as a range"),
    ]


def selected_timestamp_from_chart_event(event: object) -> pd.Timestamp | None:
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    points = getattr(selection, "points", None) if selection is not None else None
    if points is None and isinstance(event, Mapping):
        points = event.get("selection", {}).get("points", [])
    if not points:
        return None
    point = points[0]
    value = point.get("customdata") if isinstance(point, Mapping) else None
    if isinstance(value, list | tuple) and value:
        return _safe_timestamp(value[0])
    if isinstance(point, Mapping):
        return _safe_timestamp(point.get("x"))
    return None


def closest_scenario_timestamp(result: Mapping[str, Any], selected: Any) -> pd.Timestamp | None:
    _, scenario = scenario_frames(result)
    if scenario.empty:
        return None
    target = _safe_timestamp(selected)
    if target is None:
        return pd.Timestamp(scenario.iloc[0]["timestamp"])
    distances = (scenario["timestamp"] - target).abs()
    return pd.Timestamp(scenario.loc[int(distances.idxmin()), "timestamp"])


def selected_hour_row(result: Mapping[str, Any], selected: Any) -> dict[str, Any]:
    baseline, scenario = scenario_frames(result)
    target = closest_scenario_timestamp(result, selected)
    if target is None or scenario.empty:
        return {}
    distances = (scenario["timestamp"] - target).abs()
    index = int(distances.idxmin())
    scenario_row = scenario.iloc[index].to_dict()
    baseline_row = baseline.iloc[index].to_dict() if not baseline.empty and index < len(baseline) else {}
    return {"baseline": baseline_row, "scenario": scenario_row}


def event_timeline_chart(
    result: Mapping[str, Any],
    *,
    selected_timestamp: Any = None,
    timezone_name: str = "Europe/Paris",
) -> go.Figure:
    baseline, scenario = scenario_frames(result)
    request = result.get("scenario_request") or {}
    start = _as_utc(request.get("start_time"))
    end = _as_utc(request.get("end_time"))
    fig = go.Figure()
    if not baseline.empty:
        fig.add_trace(
            go.Scatter(
                x=baseline["timestamp"],
                y=baseline["demand_mw"],
                mode="lines",
                name="Baseline demand context",
                line=dict(color="rgba(226,232,240,.78)", width=2),
                fill="tozeroy",
                fillcolor="rgba(148,163,184,.12)",
                hovertemplate="Baseline: %{y:,.0f} MW<extra></extra>",
            )
        )
        top = float(pd.to_numeric(baseline["demand_mw"], errors="coerce").max())
        bottom = float(pd.to_numeric(baseline["demand_mw"], errors="coerce").min())
        pad = max((top - bottom) * 0.12, 100.0)
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=bottom - pad,
            y1=top + pad,
            fillcolor="rgba(66,214,199,.18)",
            line=dict(color="rgba(66,214,199,.72)", width=2),
            layer="below",
        )
        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[top + pad * 0.5, top + pad * 0.5],
                mode="markers",
                name="Start and end handles",
                marker=dict(size=14, color="#f8fafc", symbol="line-ns-open", line=dict(color="#42d6c7", width=3)),
                hovertemplate="%{x|%a %H:%M}<extra></extra>",
            )
        )
        fig.update_yaxes(range=[bottom - pad, top + pad])
    if selected_timestamp is not None:
        selected = _safe_timestamp(selected_timestamp)
        if selected is not None:
            fig.add_vline(x=selected, line=dict(color="#f8fafc", width=1.5, dash="dot"))
    _add_day_separators(fig, baseline, timezone_name)
    fig.update_layout(
        **dark_chart_layout(
            height=250,
            margin=dict(l=8, r=8, t=16, b=8),
            xaxis_title=None,
            yaxis_title="MW",
            hovermode="x unified",
            legend_title=None,
        )
    )
    return fig


def baseline_scenario_chart(
    result: Mapping[str, Any],
    *,
    selected_timestamp: Any = None,
    timezone_name: str = "Europe/Paris",
) -> go.Figure:
    baseline, scenario = scenario_frames(result)
    fig = go.Figure()
    if not baseline.empty:
        fig.add_trace(
            go.Scatter(
                x=baseline["timestamp"],
                y=baseline["demand_p90_mw"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Baseline p90",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=baseline["timestamp"],
                y=baseline["demand_p10_mw"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(148,163,184,.16)",
                line=dict(width=0),
                name="Baseline uncertainty",
                hovertemplate="P10-P90: %{y:,.0f} MW<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=baseline["timestamp"],
                y=baseline["demand_mw"],
                mode="lines",
                name="Baseline P50",
                line=dict(color="#f8fafc", width=2.2),
                hovertemplate="Baseline P50: %{y:,.0f} MW<extra></extra>",
            )
        )
    if not baseline.empty and not scenario.empty:
        fig.add_trace(
            go.Scatter(
                x=scenario["timestamp"],
                y=scenario["demand_mw"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Scenario delta upper",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=baseline["timestamp"],
                y=baseline["demand_mw"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(66,214,199,.20)",
                line=dict(width=0),
                name="Delta area",
                hoverinfo="skip",
            )
        )
    if not scenario.empty:
        customdata = [
            [
                pd.Timestamp(row.timestamp).isoformat(),
                row.balance_status,
                row.demand_delta_mw,
                row.import_export_delta_mw,
            ]
            for row in scenario.itertuples(index=False)
        ]
        fig.add_trace(
            go.Scatter(
                x=scenario["timestamp"],
                y=scenario["demand_mw"],
                mode="lines+markers",
                name="Scenario P50",
                line=dict(color="#42d6c7", width=2.7),
                marker=dict(size=7, color="#42d6c7", line=dict(width=1, color="#07111d")),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x|%a %d %b %H:%M}</b><br>"
                    "Scenario P50: %{y:,.0f} MW<br>"
                    "Status: %{customdata[1]}<br>"
                    "Demand delta: %{customdata[2]:+,.0f} MW<br>"
                    "Import/export response: %{customdata[3]:+,.0f} MW<extra></extra>"
                ),
            )
        )
    for item in result.get("changed_watch_or_high_hours") or []:
        timestamp = _safe_timestamp(item.get("timestamp"))
        if timestamp is None:
            continue
        fig.add_vrect(
            x0=timestamp - pd.Timedelta(minutes=30),
            x1=timestamp + pd.Timedelta(minutes=30),
            fillcolor="rgba(245,158,11,.13)",
            line_width=0,
            layer="below",
        )
    selected = closest_scenario_timestamp(result, selected_timestamp)
    if selected is not None and not scenario.empty:
        row = scenario.loc[(scenario["timestamp"] - selected).abs().idxmin()]
        fig.add_trace(
            go.Scatter(
                x=[row["timestamp"]],
                y=[row["demand_mw"]],
                mode="markers",
                name="Selected hour",
                marker=dict(size=17, color="#f8fafc", symbol="circle-open", line=dict(color="#42d6c7", width=3)),
                hoverinfo="skip",
            )
        )
    _add_day_separators(fig, scenario if not scenario.empty else baseline, timezone_name)
    fig.update_layout(
        **dark_chart_layout(
            height=430,
            margin=dict(l=8, r=8, t=18, b=8),
            xaxis_title=None,
            yaxis_title="Demand MW",
            hovermode="x unified",
            legend_title=None,
            clickmode="event+select",
        )
    )
    return fig


def demand_delta_chart(result: Mapping[str, Any]) -> go.Figure:
    _, scenario = scenario_frames(result)
    fig = go.Figure()
    if not scenario.empty:
        deltas = pd.to_numeric(scenario["demand_delta_mw"], errors="coerce").fillna(0.0)
        colors = ["#ef4444" if value > 0 else "#38bdf8" if value < 0 else "rgba(148,163,184,.5)" for value in deltas]
        fig.add_bar(
            x=scenario["timestamp"],
            y=deltas,
            marker_color=colors,
            name="Demand delta",
            hovertemplate="Demand delta: %{y:+,.0f} MW<extra></extra>",
        )
    fig.add_hline(y=0, line=dict(color="rgba(226,232,240,.55)", width=1))
    fig.update_layout(
        **dark_chart_layout(
            height=230,
            margin=dict(l=8, r=8, t=12, b=8),
            xaxis_title=None,
            yaxis_title="MW",
            showlegend=False,
        )
    )
    return fig


def generation_response_chart(result: Mapping[str, Any]) -> go.Figure:
    _, scenario = scenario_frames(result)
    generation_range = _range_tuple(
        result.get("estimated_generation_response_range", {}).get("flexible_generation_delta_mwh_range")
    )
    import_range = _range_tuple(
        result.get("estimated_import_export_delta", {}).get("net_import_delta_mwh_range")
    )
    unresolved = 0.0
    if not scenario.empty and "unresolved_residual_mw" in scenario:
        unresolved = float(pd.to_numeric(scenario["unresolved_residual_mw"], errors="coerce").fillna(0.0).sum())
    rows = [
        ("Flexible generation", generation_range[0], generation_range[1], "#42d6c7"),
        ("Imports / exports", import_range[0], import_range[1], "#93c5fd"),
        ("Unresolved residual", min(0.0, unresolved), max(0.0, unresolved), "#f59e0b"),
    ]
    fig = go.Figure()
    for label, low, high, color in rows:
        fig.add_bar(
            y=[label],
            x=[high - low],
            base=[low],
            orientation="h",
            marker_color=color,
            name=label,
            hovertemplate=f"{html.escape(label)}: {low:+,.0f} to {high:+,.0f} MWh<extra></extra>",
        )
    fig.add_vline(x=0, line=dict(color="rgba(226,232,240,.55)", width=1))
    fig.update_layout(
        **dark_chart_layout(
            height=260,
            margin=dict(l=8, r=8, t=16, b=8),
            xaxis_title="Estimated MWh delta",
            yaxis_title=None,
            showlegend=False,
        )
    )
    return fig


def regional_delta_frame(regional: pd.DataFrame, result: Mapping[str, Any]) -> pd.DataFrame:
    source = regional.copy()
    if source.empty:
        rows = [
            {"region_code": code, "region_display": name, "consumption_mw": 0.0}
            for code, name in REGION_NAMES.items()
        ]
        source = pd.DataFrame(rows)
    if "region_code" not in source:
        source["region_code"] = source.index.astype(str)
    if "region_display" not in source:
        source["region_display"] = source["region_code"].map(lambda code: REGION_NAMES.get(str(code), str(code)))
    deltas = {
        str(item.get("region_code")): item
        for item in (result.get("regional_deltas") or {}).get("regions", [])
    }
    supported = bool((result.get("regional_deltas") or {}).get("supported", True))
    unavailable_reason = str(
        (result.get("regional_deltas") or {}).get("reason")
        or "Regional scenario demand-delta data is unavailable."
    )
    source["region_code"] = source["region_code"].astype(str)
    source["total_delta_mwh"] = source["region_code"].map(lambda code: float(deltas.get(code, {}).get("total_delta_mwh") or 0.0))
    source["peak_delta_mw"] = source["region_code"].map(lambda code: float(deltas.get(code, {}).get("peak_delta_mw") or 0.0))
    source["delta_available"] = supported
    source["unavailable_reason"] = "" if supported else unavailable_reason
    source["changed"] = (
        source["delta_available"]
        & (source["peak_delta_mw"].abs().gt(1e-6) | source["total_delta_mwh"].abs().gt(1e-6))
    )
    source["delta_label"] = source.apply(
        lambda row: (
            f"{format_signed_mw(row['peak_delta_mw'])} peak, {_format_signed_energy(row['total_delta_mwh'])}"
            if row["delta_available"] and row["changed"]
            else "Regional demand delta unavailable"
            if not row["delta_available"]
            else "No demand delta"
        ),
        axis=1,
    )
    return source


def regional_delta_choropleth(frame: pd.DataFrame, geojson: dict) -> go.Figure:
    work = frame.copy()
    if work.empty:
        work = pd.DataFrame(columns=["region_code", "region_display", "peak_delta_mw", "total_delta_mwh", "changed"])
    work["region_code"] = work.get("region_code", pd.Series(dtype=str)).astype(str)
    work["region_display"] = work.get("region_display", work["region_code"]).map(
        lambda value: REGION_NAMES.get(str(value), str(value))
    )
    work["peak_delta_mw"] = pd.to_numeric(work.get("peak_delta_mw"), errors="coerce")
    work["total_delta_mwh"] = pd.to_numeric(work.get("total_delta_mwh"), errors="coerce")
    available = (
        work["delta_available"].fillna(False).astype(bool)
        if "delta_available" in work
        else pd.Series(True, index=work.index)
    )
    finite_delta = work["peak_delta_mw"].notna() & work["total_delta_mwh"].notna()
    work["delta_available"] = available & finite_delta
    changed_raw = work["changed"] if "changed" in work else pd.Series(False, index=work.index)
    work["changed"] = changed_raw.fillna(False).astype(bool) & work["delta_available"]
    unavailable_reason = (
        work["unavailable_reason"]
        if "unavailable_reason" in work
        else pd.Series("Regional scenario demand-delta data is unavailable.", index=work.index)
    )
    work["unavailable_reason"] = unavailable_reason.fillna("Regional scenario demand-delta data is unavailable.")

    changed = work.loc[work["changed"]].copy()
    unchanged = work.loc[work["delta_available"] & ~work["changed"]].copy()
    unavailable = work.loc[~work["delta_available"]].copy()
    max_abs = diverging_metric_range(changed["peak_delta_mw"] if not changed.empty else work["peak_delta_mw"], floor=1.0)

    fig = go.Figure()
    if not changed.empty:
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=changed["region_code"],
                z=clipped_diverging_values(changed["peak_delta_mw"], max_abs),
                zmin=-max_abs,
                zmid=0,
                zmax=max_abs,
                featureidkey="properties.code",
                colorscale=DIVERGING_COLORSCALE,
                marker_line_color="rgba(248,250,252,.82)",
                marker_line_width=0.95,
                colorbar=dict(
                    title="Peak delta",
                    tickmode="array",
                    tickvals=[-max_abs, 0, max_abs],
                    ticktext=[signed_mw_tick(-max_abs), "0 MW", signed_mw_tick(max_abs)],
                    outlinewidth=0,
                    thickness=12,
                    len=0.72,
                ),
                customdata=changed[["region_display", "peak_delta_mw", "total_delta_mwh", "changed"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Peak delta: %{customdata[1]:+,.0f} MW<br>"
                    "Total delta: %{customdata[2]:+,.0f} MWh<br>"
                    "Changed: %{customdata[3]}<extra></extra>"
                ),
                name="Changed regional demand delta",
            )
        )
    if not unchanged.empty:
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=unchanged["region_code"],
                z=[0] * len(unchanged),
                zmin=0,
                zmax=1,
                featureidkey="properties.code",
                colorscale=[[0, "#94a3b8"], [1, "#94a3b8"]],
                showscale=False,
                marker_line_color="rgba(226,232,240,.62)",
                marker_line_width=0.75,
                customdata=unchanged[["region_display", "peak_delta_mw", "total_delta_mwh", "changed"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Peak delta: %{customdata[1]:+,.0f} MW<br>"
                    "Total delta: %{customdata[2]:+,.0f} MWh<br>"
                    "Changed: %{customdata[3]}<extra></extra>"
                ),
                name="Unchanged regional demand delta",
            )
        )
    if not unavailable.empty:
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=unavailable["region_code"],
                z=[0] * len(unavailable),
                zmin=0,
                zmax=1,
                featureidkey="properties.code",
                colorscale=[[0, "#475569"], [1, "#475569"]],
                showscale=False,
                marker_line_color="rgba(226,232,240,.72)",
                marker_line_width=0.85,
                customdata=unavailable[["region_display", "unavailable_reason"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Demand delta: Unavailable<br>"
                    "Reason: %{customdata[1]}<extra></extra>"
                ),
                name="Unavailable regional demand delta",
            )
        )
    fig.update_geos(
        scope="europe",
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        projection_type="mercator",
    )
    fig.update_layout(
        **dark_chart_layout(
            height=520,
            margin=dict(l=8, r=8, t=16, b=8),
            geo=dict(
                bgcolor="rgba(0,0,0,0)",
                lakecolor="rgba(14, 165, 233, .18)",
                landcolor="rgba(15, 28, 44, .92)",
            ),
            showlegend=False,
        )
    )
    return fig


def _short_region_label(value: Any) -> str:
    text = str(value)
    aliases = {
        "Auvergne-Rhone-Alpes": "AURA",
        "Bourgogne-Franche-Comte": "BFC",
        "Bourgogne-Franche-Comté": "BFC",
        "Centre-Val de Loire": "Centre",
        "Grand Est": "Grand Est",
        "Hauts-de-France": "Hauts",
        "Ile-de-France": "IDF",
        "Île-de-France": "IDF",
        "Nouvelle-Aquitaine": "N. Aquitaine",
        "Occitanie": "Occitanie",
        "Pays de la Loire": "Pays Loire",
        "Provence-Alpes-Cote d'Azur": "PACA",
        "Provence-Alpes-Côte d'Azur": "PACA",
    }
    return aliases.get(text, text)


def changed_region_table(frame: pd.DataFrame) -> pd.DataFrame:
    changed = frame.loc[frame["changed"]].copy()
    if changed.empty:
        return pd.DataFrame(columns=["Region", "Peak demand delta", "Total energy delta"])
    changed["abs_peak"] = changed["peak_delta_mw"].abs()
    changed = changed.sort_values("abs_peak", ascending=False)
    return pd.DataFrame(
        {
            "Region": changed["region_display"],
            "Peak demand delta": changed["peak_delta_mw"].map(format_signed_mw),
            "Total energy delta": changed["total_delta_mwh"].map(_format_signed_energy),
        }
    )


def format_watch_high_delta(value: int) -> str:
    if value > 0:
        return f"+{value} Watch/High h"
    if value < 0:
        return f"{value} Watch/High h"
    return "0 Watch/High h"


def format_score_delta(value: float) -> str:
    return f"{value:+.2f} score"


def format_signed_range(values: tuple[float, float], *, unit: str) -> str:
    low, high = values
    if unit == "MWh":
        return f"{_format_signed_energy(low)} to {_format_signed_energy(high)}"
    return f"{low:+,.0f} to {high:+,.0f} {unit}"


def format_carbon_range(values: tuple[float, float]) -> str:
    low, high = values
    return f"{low:+,.0f} to {high:+,.0f} t CO2"


def _add_day_separators(fig: go.Figure, frame: pd.DataFrame, timezone_name: str) -> None:
    if frame.empty or "timestamp" not in frame:
        return
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return
    start_local = timestamps.min().tz_convert(timezone_name)
    end_local = timestamps.max().tz_convert(timezone_name)
    separators = pd.date_range(
        start_local.normalize() + pd.Timedelta(days=1),
        end_local.normalize(),
        freq="D",
        tz=timezone_name,
    )
    for separator in separators:
        fig.add_vline(
            x=separator.tz_convert("UTC"),
            line=dict(color="rgba(148,163,184,.34)", width=1, dash="dot"),
        )


def _default_magnitude(scenario_type: str) -> float:
    if scenario_type == "generation_unavailability":
        return 1.3
    if scenario_type == "ev_charging_shift":
        return 100_000.0
    return 3.0


def _default_duration(scenario_type: str) -> int:
    return 12 if scenario_type == "ev_charging_shift" else 6


def _first(values: Mapping[str, Any], key: str) -> str | None:
    if key not in values:
        return None
    value = values[key]
    if isinstance(value, list | tuple):
        return None if not value else str(value[0])
    if value is None:
        return None
    return str(value)


def _as_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _safe_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        return _as_utc(value)
    except (TypeError, ValueError):
        return None


def _defaulted_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if pd.notna(number) else default


def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    return min(max(_defaulted_float(value, default=default), minimum), maximum)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return min(max(number, minimum), maximum)


def _window_from_query(value: Any, *, default: tuple[str, str]) -> tuple[str, str]:
    text = str(value or "").strip()
    if "-" not in text:
        return default
    start, end = [part.strip() for part in text.split("-", 1)]
    if _valid_hhmm(start) and _valid_hhmm(end) and start != end:
        return start, end
    return default


def _valid_hhmm(value: str) -> bool:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (ValueError, TypeError):
        return False
    return 0 <= hour <= 23 and 0 <= minute <= 59


def _compact_number(value: float) -> str:
    return f"{value:g}"


def _range_tuple(value: Any) -> tuple[float, float]:
    if isinstance(value, list | tuple) and len(value) >= 2:
        return float(value[0] or 0.0), float(value[1] or 0.0)
    return 0.0, 0.0


def _format_signed_energy(value_mwh: float) -> str:
    value = float(value_mwh or 0.0)
    if abs(value) >= 1000:
        return f"{value / 1000.0:+,.1f} GWh"
    return f"{value:+,.0f} MWh"


def chain_markup(steps: list[tuple[str, str]]) -> str:
    items = []
    for index, (title, detail) in enumerate(steps):
        arrow = '<div class="ep-chain-arrow" aria-hidden="true">&rarr;</div>' if index < len(steps) - 1 else ""
        items.append(
            '<div class="ep-chain-node">'
            f'<div class="ep-chain-index">{index + 1}</div>'
            f'<div class="ep-title">{html.escape(title)}</div>'
            f'<div class="ep-detail">{html.escape(detail)}</div>'
            f"</div>{arrow}"
        )
    return f'<div class="ep-chain" aria-label="Causal chain">{"".join(items)}</div>'
