from __future__ import annotations

from dataclasses import dataclass
import html
import json
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
from app.formatting import (
    format_mw,
    format_number,
    format_percentage,
    format_signed_mw,
    format_temperature,
    format_timestamp,
)
from app.i18n import normalize_locale, t
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

API_TEXT_TRANSLATION_KEYS = {
    "Generating-unit unavailability is represented as national supply availability only in v1.": "api_text.generation_unavailability_regional_reason",
    "Regional demand deltas are allocated from the reconciled regional demand shares; this is not regional adequacy.": "api_text.regional_delta_method",
    "Regional scenario demand-delta data is unavailable.": "api_text.regional_delta_unavailable",
    "This scenario changes national supply context only.": "api_text.national_supply_only",
    "This scenario type has no regional demand allocation in v1.": "api_text.no_regional_allocation",
    "Scenario response estimate.": "api_text.scenario_response_estimate",
    "After flexible generation response, a configured fraction of the remaining residual is represented as higher net imports, lower exports, lower net imports, or higher exports.": "api_text.import_export_method",
    "Flexible generation absorbs a configured portion of residual change; wind and solar follow baseline forecasts.": "api_text.generation_response_method",
    "Each positive residual MWh is multiplied by a plausible response-mix intensity range; each negative residual MWh is treated as avoided response-mix output.": "api_text.carbon_method",
    "This is a directional public-decision scenario, not an operator dispatch forecast.": "api_text.caveat_directional",
    "Imports, exports, flexible generation, and unresolved residual are represented as ranges to avoid false precision.": "api_text.caveat_ranges",
    "The engine does not make import requirement automatically equal to peak-demand change.": "api_text.caveat_import_requirement",
    "Carbon is a plausible response-mix range, not verified emissions accounting.": "api_text.caveat_carbon",
    "Wind and solar follow their baseline adjusted forecasts.": "api_text.assumption_wind_solar",
    "Nuclear follows availability context; generating-unit unavailability reduces available supply, not demand.": "api_text.assumption_nuclear",
    "Cold-snap demand rerun uses heating_sensitivity_mw_per_c and deterministic local-hour multipliers.": "api_text.assumption_cold_snap",
    "EV charging shift conserves total modelled scenario energy across complete source and target windows.": "api_text.assumption_ev_shift",
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


def scenario_preset_display(scenario_type: str, *, locale: str = "en") -> dict[str, str]:
    key = normalize_scenario_key(scenario_type)
    defaults = SCENARIO_PRESETS[key]
    return {
        "label": t(f"what_if.scenarios.{key}.label", locale=locale, default=defaults["label"]),
        "short": t(f"what_if.scenarios.{key}.short", locale=locale, default=defaults["short"]),
        "concept": t(f"what_if.scenarios.{key}.concept", locale=locale, default=defaults["concept"]),
        "detail": t(f"what_if.scenarios.{key}.detail", locale=locale, default=defaults["detail"]),
    }


def scenario_preset_label(scenario_type: str, *, locale: str = "en") -> str:
    return scenario_preset_display(scenario_type, locale=locale)["label"]


def status_display_label(value: Any, *, locale: str = "en") -> str:
    key = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    return t(f"what_if.status.{key}", locale=locale, default=str(value or "unknown").title())


def card_status_display(status: str, *, locale: str = "en") -> str:
    if status == "watch":
        return t("what_if.card_status.watch", locale=locale, default="Watch")
    if status == "green":
        return t("what_if.card_status.green", locale=locale, default="Normal")
    if status == "info":
        return t("what_if.card_status.info", locale=locale, default="Info")
    return status


def translate_api_text(text: Any, *, locale: str = "en") -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if value.startswith("Request assumptions: "):
        return _translate_request_assumptions(value, locale=locale)
    if value.startswith("Scenario assumptions artifact: "):
        version = value.removeprefix("Scenario assumptions artifact: ").rstrip(".")
        return t("what_if.api_text.assumption_artifact", locale=locale, default=value, version=version)
    if value.startswith("Flexible generation response fraction range: "):
        fraction = value.removeprefix("Flexible generation response fraction range: ").removesuffix(" of residual change.")
        return t("what_if.api_text.assumption_generation_fraction", locale=locale, default=value, fraction=fraction)
    if value.startswith("Import/export response fraction range after flexible response: "):
        fraction = value.removeprefix("Import/export response fraction range after flexible response: ").rstrip(".")
        return t("what_if.api_text.assumption_import_fraction", locale=locale, default=value, fraction=fraction)
    key = API_TEXT_TRANSLATION_KEYS.get(value)
    if key:
        return t(f"what_if.{key}", locale=locale, default=value)
    return value


def restore_scenario_controls(
    query_params: Mapping[str, Any],
    *,
    default_scenario: str | None = None,
    default_region: str = "11",
) -> ScenarioControls:
    scenario_type = normalize_scenario_key(_first(query_params, "scenario") or default_scenario)
    start = _bounded_int(_first(query_params, "start"), default=12, minimum=0, maximum=47)
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


def validate_controls(controls: ScenarioControls, *, locale: str = "en") -> str | None:
    if controls.end_offset_hours <= controls.start_offset_hours:
        return t(
            "what_if.validation.end_after_start",
            locale=locale,
            default="Scenario end time must be after start time.",
        )
    if controls.scenario_type == "ev_charging_shift" and controls.original_window == controls.target_window:
        return t(
            "what_if.validation.ev_windows_different",
            locale=locale,
            default="EV source and target charging windows must be different.",
        )
    if controls.scenario_type == "ev_charging_shift" and controls.ev_participation <= 0:
        return t(
            "what_if.validation.ev_participation_positive",
            locale=locale,
            default="EV participation must be greater than zero.",
        )
    if controls.magnitude <= 0:
        return t(
            "what_if.validation.magnitude_positive",
            locale=locale,
            default="Scenario magnitude must be greater than zero.",
        )
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


def magnitude_label_from_request(result: Mapping[str, Any], *, locale: str = "en") -> str:
    request = result.get("scenario_request") or {}
    scenario_type = str(request.get("scenario_type") or "")
    magnitude = request.get("magnitude") or {}
    if scenario_type == "cold_snap":
        value = abs(float(magnitude.get("temperature_delta_c") or 0.0))
        return t(
            "what_if.magnitude.cold_snap",
            locale=locale,
            default="{value} colder",
            value=format_temperature(value, locale=locale),
        )
    if scenario_type == "generation_unavailability":
        value = float(magnitude.get("unavailable_capacity_mw") or 0.0)
        asset = magnitude.get("asset_name")
        label = format_mw(value, locale=locale)
        if not asset:
            return t("what_if.magnitude.generation_unavailability", locale=locale, default="{value} unavailable", value=label)
        return t(
            "what_if.magnitude.generation_unavailability_asset",
            locale=locale,
            default="{asset}: {value} unavailable",
            asset=asset,
            value=label,
        )
    if scenario_type == "ev_charging_shift":
        vehicles = int(float(magnitude.get("vehicles") or 0))
        energy = float(magnitude.get("average_energy_per_vehicle_kwh") or 0.0)
        participation = float(magnitude.get("participation_rate") or 0.0)
        return t(
            "what_if.magnitude.ev_charging_shift",
            locale=locale,
            default="{vehicles} EVs, {energy} kWh each, {participation} participating",
            vehicles=format_number(vehicles, locale=locale),
            energy=_format_compact_number(energy, locale=locale),
            participation=format_percentage(participation, locale=locale),
        )
    return t("what_if.magnitude.generic", locale=locale, default="Scenario magnitude")


def scenario_window_label(result: Mapping[str, Any], *, timezone_name: str, locale: str = "en") -> str:
    request = result.get("scenario_request") or {}
    start = _as_utc(request.get("start_time"))
    end = _as_utc(request.get("end_time"))
    duration = float(request.get("duration_hours") or 0.0)
    return t(
        "what_if.window.label",
        locale=locale,
        default="{start} to {end} ({duration}h window)",
        start=format_timestamp(start, timezone_name=timezone_name, include_date=False, locale=locale),
        end=format_timestamp(end, timezone_name=timezone_name, include_date=False, locale=locale),
        duration=_format_compact_number(duration, locale=locale),
    )


def causal_chain_steps(result: Mapping[str, Any], *, locale: str = "en") -> list[tuple[str, str]]:
    chain = result.get("causal_chain") or {}
    scenario_type = str(chain.get("scenario_type") or result.get("scenario_request", {}).get("scenario_type") or "")
    if scenario_type == "cold_snap":
        return [
            (_chain_title("cold_snap", 1, locale=locale), _chain_detail("cold_snap", 1, locale=locale)),
            (_chain_title("cold_snap", 2, locale=locale), _chain_detail("cold_snap", 2, locale=locale)),
            (_chain_title("cold_snap", 3, locale=locale), _chain_detail("cold_snap", 3, locale=locale)),
            (_chain_title("cold_snap", 4, locale=locale), _chain_detail("cold_snap", 4, locale=locale)),
            (_chain_title("cold_snap", 5, locale=locale), _chain_detail("cold_snap", 5, locale=locale)),
        ]
    if scenario_type == "generation_unavailability":
        return [
            (_chain_title("generation_unavailability", 1, locale=locale), _chain_detail("generation_unavailability", 1, locale=locale)),
            (_chain_title("generation_unavailability", 2, locale=locale), _chain_detail("generation_unavailability", 2, locale=locale)),
            (_chain_title("generation_unavailability", 3, locale=locale), _chain_detail("generation_unavailability", 3, locale=locale)),
            (_chain_title("generation_unavailability", 4, locale=locale), _chain_detail("generation_unavailability", 4, locale=locale)),
            (_chain_title("generation_unavailability", 5, locale=locale), _chain_detail("generation_unavailability", 5, locale=locale)),
        ]
    return [
        (_chain_title("ev_charging_shift", 1, locale=locale), _chain_detail("ev_charging_shift", 1, locale=locale)),
        (_chain_title("ev_charging_shift", 2, locale=locale), _chain_detail("ev_charging_shift", 2, locale=locale)),
        (_chain_title("ev_charging_shift", 3, locale=locale), _chain_detail("ev_charging_shift", 3, locale=locale)),
        (_chain_title("ev_charging_shift", 4, locale=locale), _chain_detail("ev_charging_shift", 4, locale=locale)),
        (_chain_title("ev_charging_shift", 5, locale=locale), _chain_detail("ev_charging_shift", 5, locale=locale)),
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
    locale: str = "en",
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
                name=t("what_if.charts.event_timeline.baseline_context", locale=locale, default="Baseline demand context"),
                line=dict(color="rgba(226,232,240,.78)", width=2),
                fill="tozeroy",
                fillcolor="rgba(148,163,184,.12)",
                hovertemplate=t("what_if.charts.event_timeline.baseline_hover", locale=locale, default="Baseline: %{y:,.0f} MW<extra></extra>"),
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
                name=t("what_if.charts.event_timeline.handles", locale=locale, default="Start and end handles"),
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
    locale: str = "en",
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
                name=t("what_if.charts.baseline_scenario.baseline_p90", locale=locale, default="Baseline p90"),
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
                name=t("what_if.charts.baseline_scenario.baseline_uncertainty", locale=locale, default="Baseline uncertainty"),
                hovertemplate=t("what_if.charts.baseline_scenario.uncertainty_hover", locale=locale, default="P10-P90: %{y:,.0f} MW<extra></extra>"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=baseline["timestamp"],
                y=baseline["demand_mw"],
                mode="lines",
                name=t("what_if.charts.baseline_scenario.baseline_p50", locale=locale, default="Baseline P50"),
                line=dict(color="#f8fafc", width=2.2),
                hovertemplate=t("what_if.charts.baseline_scenario.baseline_p50_hover", locale=locale, default="Baseline P50: %{y:,.0f} MW<extra></extra>"),
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
                name=t("what_if.charts.baseline_scenario.delta_upper", locale=locale, default="Scenario delta upper"),
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
                name=t("what_if.charts.baseline_scenario.delta_area", locale=locale, default="Delta area"),
                hoverinfo="skip",
            )
        )
    if not scenario.empty:
        customdata = [
            [
                pd.Timestamp(row.timestamp).isoformat(),
                status_display_label(row.balance_status, locale=locale),
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
                name=t("what_if.charts.baseline_scenario.scenario_p50", locale=locale, default="Scenario P50"),
                line=dict(color="#42d6c7", width=2.7),
                marker=dict(size=7, color="#42d6c7", line=dict(width=1, color="#07111d")),
                customdata=customdata,
                hovertemplate=t(
                    "what_if.charts.baseline_scenario.scenario_hover",
                    locale=locale,
                    default=(
                        "<b>%{x|%a %d %b %H:%M}</b><br>"
                        "Scenario P50: %{y:,.0f} MW<br>"
                        "Status: %{customdata[1]}<br>"
                        "Demand delta: %{customdata[2]:+,.0f} MW<br>"
                        "Import/export response: %{customdata[3]:+,.0f} MW<extra></extra>"
                    ),
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
                name=t("what_if.charts.baseline_scenario.selected_hour", locale=locale, default="Selected hour"),
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
            yaxis_title=t("what_if.charts.baseline_scenario.yaxis", locale=locale, default="Demand MW"),
            hovermode="x unified",
            legend_title=None,
            clickmode="event+select",
        )
    )
    return fig


def demand_delta_chart(result: Mapping[str, Any], *, locale: str = "en") -> go.Figure:
    _, scenario = scenario_frames(result)
    fig = go.Figure()
    if not scenario.empty:
        deltas = pd.to_numeric(scenario["demand_delta_mw"], errors="coerce").fillna(0.0)
        colors = ["#ef4444" if value > 0 else "#38bdf8" if value < 0 else "rgba(148,163,184,.5)" for value in deltas]
        fig.add_bar(
            x=scenario["timestamp"],
            y=deltas,
            marker_color=colors,
            name=t("what_if.charts.demand_delta.name", locale=locale, default="Demand delta"),
            hovertemplate=t("what_if.charts.demand_delta.hover", locale=locale, default="Demand delta: %{y:+,.0f} MW<extra></extra>"),
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


def generation_response_chart(result: Mapping[str, Any], *, locale: str = "en") -> go.Figure:
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
        (t("what_if.charts.generation_response.flexible_generation", locale=locale, default="Flexible generation"), generation_range[0], generation_range[1], "#42d6c7"),
        (t("what_if.charts.generation_response.imports_exports", locale=locale, default="Imports / exports"), import_range[0], import_range[1], "#93c5fd"),
        (t("what_if.charts.generation_response.unresolved_residual", locale=locale, default="Unresolved residual"), min(0.0, unresolved), max(0.0, unresolved), "#f59e0b"),
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
            hovertemplate=t(
                "what_if.charts.generation_response.hover",
                locale=locale,
                default="{label}: {low} to {high} MWh<extra></extra>",
                label=html.escape(label),
                low=format_number(low, signed=True, locale=locale),
                high=format_number(high, signed=True, locale=locale),
            ),
        )
    fig.add_vline(x=0, line=dict(color="rgba(226,232,240,.55)", width=1))
    fig.update_layout(
        **dark_chart_layout(
            height=260,
            margin=dict(l=8, r=8, t=16, b=8),
            xaxis_title=t("what_if.charts.generation_response.xaxis", locale=locale, default="Estimated MWh delta"),
            yaxis_title=None,
            showlegend=False,
        )
    )
    return fig


def regional_delta_frame(regional: pd.DataFrame, result: Mapping[str, Any], *, locale: str = "en") -> pd.DataFrame:
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
    unavailable_reason = translate_api_text(unavailable_reason, locale=locale)
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
            t(
                "what_if.regional.delta_label",
                locale=locale,
                default="{peak} peak, {total}",
                peak=format_signed_mw(row["peak_delta_mw"], locale=locale),
                total=_format_signed_energy(row["total_delta_mwh"], locale=locale),
            )
            if row["delta_available"] and row["changed"]
            else t(
                "what_if.regional.delta_unavailable",
                locale=locale,
                default="Regional demand delta unavailable",
            )
            if not row["delta_available"]
            else t("what_if.regional.no_delta", locale=locale, default="No demand delta")
        ),
        axis=1,
    )
    return source


def regional_delta_choropleth(frame: pd.DataFrame, geojson: dict, *, locale: str = "en") -> go.Figure:
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
    work["unavailable_reason"] = unavailable_reason.fillna("Regional scenario demand-delta data is unavailable.").map(
        lambda value: translate_api_text(value, locale=locale)
    )
    changed_display_column = "changed"
    if normalize_locale(locale, default="en") != "en":
        changed_display_column = "changed_label"
        work[changed_display_column] = work["changed"].map(
            lambda value: t("what_if.common.yes", locale=locale) if value else t("what_if.common.no", locale=locale)
        )

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
                    title=t("what_if.charts.regional_delta.colorbar", locale=locale, default="Peak delta"),
                    tickmode="array",
                    tickvals=[-max_abs, 0, max_abs],
                    ticktext=[signed_mw_tick(-max_abs), "0 MW", signed_mw_tick(max_abs)],
                    outlinewidth=0,
                    thickness=12,
                    len=0.72,
                ),
                customdata=changed[["region_display", "peak_delta_mw", "total_delta_mwh", changed_display_column]],
                hovertemplate=t(
                    "what_if.charts.regional_delta.changed_hover",
                    locale=locale,
                    default=(
                        "<b>%{customdata[0]}</b><br>"
                        "Peak delta: %{customdata[1]:+,.0f} MW<br>"
                        "Total delta: %{customdata[2]:+,.0f} MWh<br>"
                        "Changed: %{customdata[3]}<extra></extra>"
                    ),
                ),
                name=t("what_if.charts.regional_delta.changed_name", locale=locale, default="Changed regional demand delta"),
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
                customdata=unchanged[["region_display", "peak_delta_mw", "total_delta_mwh", changed_display_column]],
                hovertemplate=t(
                    "what_if.charts.regional_delta.changed_hover",
                    locale=locale,
                    default=(
                        "<b>%{customdata[0]}</b><br>"
                        "Peak delta: %{customdata[1]:+,.0f} MW<br>"
                        "Total delta: %{customdata[2]:+,.0f} MWh<br>"
                        "Changed: %{customdata[3]}<extra></extra>"
                    ),
                ),
                name=t("what_if.charts.regional_delta.unchanged_name", locale=locale, default="Unchanged regional demand delta"),
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
                hovertemplate=t(
                    "what_if.charts.regional_delta.unavailable_hover",
                    locale=locale,
                    default=(
                        "<b>%{customdata[0]}</b><br>"
                        "Demand delta: Unavailable<br>"
                        "Reason: %{customdata[1]}<extra></extra>"
                    ),
                ),
                name=t("what_if.charts.regional_delta.unavailable_name", locale=locale, default="Unavailable regional demand delta"),
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


def changed_region_table(frame: pd.DataFrame, *, locale: str = "en") -> pd.DataFrame:
    changed = frame.loc[frame["changed"]].copy()
    columns = [
        t("what_if.tables.changed_regions.region", locale=locale, default="Region"),
        t("what_if.tables.changed_regions.peak_demand_delta", locale=locale, default="Peak demand delta"),
        t("what_if.tables.changed_regions.total_energy_delta", locale=locale, default="Total energy delta"),
    ]
    if changed.empty:
        return pd.DataFrame(columns=columns)
    changed["abs_peak"] = changed["peak_delta_mw"].abs()
    changed = changed.sort_values("abs_peak", ascending=False)
    return pd.DataFrame(
        {
            columns[0]: changed["region_display"],
            columns[1]: changed["peak_delta_mw"].map(lambda value: format_signed_mw(value, locale=locale)),
            columns[2]: changed["total_delta_mwh"].map(lambda value: _format_signed_energy(value, locale=locale)),
        }
    )


def format_watch_high_delta(value: int, *, locale: str = "en") -> str:
    if value > 0:
        return t(
            "what_if.formats.watch_high_delta",
            locale=locale,
            default="{value} Watch/High h",
            value=f"+{format_number(value, locale=locale)}",
        )
    if value < 0:
        return t(
            "what_if.formats.watch_high_delta",
            locale=locale,
            default="{value} Watch/High h",
            value=f"-{format_number(abs(value), locale=locale)}",
        )
    return t("what_if.formats.watch_high_delta", locale=locale, default="{value} Watch/High h", value=format_number(0, locale=locale))


def format_score_delta(value: float, *, locale: str = "en") -> str:
    return t(
        "what_if.formats.score_delta",
        locale=locale,
        default="{value} score",
        value=format_number(value, decimals=2, signed=True, locale=locale),
    )


def format_signed_range(values: tuple[float, float], *, unit: str, locale: str = "en") -> str:
    low, high = values
    if unit == "MWh":
        return t(
            "what_if.formats.range",
            locale=locale,
            default="{low} to {high}",
            low=_format_signed_energy(low, locale=locale),
            high=_format_signed_energy(high, locale=locale),
        )
    return t(
        "what_if.formats.range",
        locale=locale,
        default="{low} to {high}",
        low=f"{format_number(low, signed=True, locale=locale)} {unit}",
        high=f"{format_number(high, signed=True, locale=locale)} {unit}",
    )


def format_carbon_range(values: tuple[float, float], *, locale: str = "en") -> str:
    low, high = values
    return t(
        "what_if.formats.range",
        locale=locale,
        default="{low} to {high}",
        low=f"{format_number(low, signed=True, locale=locale)} t CO2",
        high=f"{format_number(high, signed=True, locale=locale)} t CO2",
    )


def _chain_title(scenario_type: str, index: int, *, locale: str) -> str:
    return t(
        f"what_if.chain.{scenario_type}.{index}.title",
        locale=locale,
        default=_CHAIN_DEFAULTS[scenario_type][index - 1][0],
    )


def _chain_detail(scenario_type: str, index: int, *, locale: str) -> str:
    return t(
        f"what_if.chain.{scenario_type}.{index}.detail",
        locale=locale,
        default=_CHAIN_DEFAULTS[scenario_type][index - 1][1],
    )


_CHAIN_DEFAULTS = {
    "cold_snap": [
        ("Colder weather", "Temperature input changes"),
        ("Heating demand rises", "Demand P50 is rerun for active hours"),
        ("Residual load rises", "More demand remains after weather-sensitive supply"),
        ("Flexible generation/imports respond", "Estimated response ranges absorb part of the residual"),
        ("Balance tightens", "App balance status can move"),
    ],
    "generation_unavailability": [
        ("Unit unavailable", "Capacity is removed from availability"),
        ("Available generation falls", "Demand is unchanged"),
        ("Residual load rises", "The same demand has less available supply"),
        ("Flexible supply/imports respond", "Estimated response ranges absorb part of the residual"),
        ("Balance context changes", "App balance status can move"),
    ],
    "ev_charging_shift": [
        ("EV charging shifts", "The same energy moves windows"),
        ("Evening demand falls", "Source-window demand delta is negative"),
        ("Overnight demand rises", "Target-window demand delta is positive"),
        ("Peak may move", "Rebound hours are flagged by the scenario API"),
        ("Carbon range changes", "Response-mix emissions are estimated as a range"),
    ],
}


def _translate_request_assumptions(value: str, *, locale: str) -> str:
    raw_payload = value.removeprefix("Request assumptions: ").strip()
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return t("what_if.api_text.request_assumptions_raw", locale=locale, default=value, payload=raw_payload)
    if not isinstance(payload, Mapping):
        return t("what_if.api_text.request_assumptions_raw", locale=locale, default=value, payload=raw_payload)

    estimate = str(payload.get("estimate") or "")
    estimate_label = (
        t("what_if.api_text.estimate_educational_directional", locale=locale, default=estimate)
        if estimate == "educational directional scenario"
        else estimate
    )
    single_active = bool(payload.get("single_active_scenario"))
    single_active_label = t(
        "what_if.api_text.single_active_yes" if single_active else "what_if.api_text.single_active_no",
        locale=locale,
        default="yes" if single_active else "no",
    )
    return t(
        "what_if.api_text.request_assumptions",
        locale=locale,
        default="Request assumptions: interface {interface}; estimate {estimate}; single active scenario: {single_active}.",
        interface=str(payload.get("interface") or ""),
        estimate=estimate_label,
        single_active=single_active_label,
    )


def _format_compact_number(value: float | int, *, locale: str) -> str:
    number = float(value)
    decimals = 0 if number.is_integer() else 1
    return format_number(number, decimals=decimals, locale=locale)


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
    return 12


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


def _format_signed_energy(value_mwh: float, *, locale: str = "en") -> str:
    value = float(value_mwh or 0.0)
    if abs(value) >= 1000:
        return f"{format_number(value / 1000.0, decimals=1, signed=True, locale=locale)} GWh"
    return f"{format_number(value, signed=True, locale=locale)} MWh"


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
