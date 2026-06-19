from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.cards import driver_card, explanation_card, metric_card, status_badge_html
from app.components.charts import COLORS, dark_chart_layout
from app.components.foundation import provenance_badge_html, provenance_kind_for_source
from app.formatting import (
    UNAVAILABLE,
    format_age_seconds,
    format_carbon,
    format_gw,
    format_mw,
    format_percentage,
    format_signed_mw,
    format_timestamp,
)
from app.state import select_timestamp_from_options
from app.view_models import ForecastPointView, GridSnapshotView
from src.contracts.energy_twin import (
    CurrentGenerationMix,
    CurrentStateResponse,
    DomainMode,
    EstimateProvenanceKind,
    Freshness,
    ModelledBalanceContext,
    NullableMetric,
    OfficialSignal,
    OperatingState,
    QuantifiedValue,
    Status,
    TwinResponse,
    TwinSnapshot,
)
from src.data_sources.rte_eco2mix_regional import REGION_NAMES


@dataclass(frozen=True)
class HeroSummary:
    demand_gw: str
    unusual_text: str
    difference_gw: str
    main_driver: str
    direction: str
    last_update: str
    freshness: str


def selected_twin_snapshot(
    twin: TwinResponse | None,
    selected: datetime | None,
) -> TwinSnapshot | None:
    if twin is None or not twin.snapshots:
        return None
    if selected is None:
        return twin.snapshots[0]
    target = pd.Timestamp(selected)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    target = target.tz_convert("UTC")
    return min(
        twin.snapshots,
        key=lambda item: abs(pd.Timestamp(item.event_time).tz_convert("UTC") - target),
    )


def next_12_snapshots(twin: TwinResponse | None) -> list[TwinSnapshot]:
    if twin is None:
        return []
    return list(twin.snapshots[1:13])


def build_hero_summary(
    current_state: CurrentStateResponse | None,
    twin: TwinResponse | None,
    snapshot: GridSnapshotView,
    *,
    timezone_name: str,
    forecast_points: list[ForecastPointView] | None = None,
) -> HeroSummary:
    demand = _current_demand(current_state, snapshot)
    pct = _current_anomaly_pct(current_state, snapshot)
    diff_gw = _current_difference_gw(current_state, snapshot)
    freshness = current_state.national_context.freshness if current_state is not None else None
    update_time = freshness.timestamp if freshness is not None and freshness.timestamp is not None else snapshot.as_of
    age = format_age_seconds(freshness.age_seconds) if freshness is not None else snapshot.freshness.get("label", "")
    return HeroSummary(
        demand_gw=format_gw(demand),
        unusual_text=demand_difference_text(pct),
        difference_gw=_format_gw_delta(diff_gw),
        main_driver=main_driver_text(current_state, twin, snapshot),
        direction=near_term_direction(twin, demand, forecast_points=forecast_points),
        last_update=format_timestamp(update_time, timezone_name=timezone_name),
        freshness=age,
    )


def render_hero_summary(summary: HeroSummary) -> None:
    st.markdown(
        f"""
        <section class="ep-now-hero" aria-label="Plain-language current electricity summary">
          <div>
            <div class="ep-eyebrow">What is happening now?</div>
            <h1>{html.escape(summary.demand_gw)} national demand</h1>
            <p>{html.escape(summary.unusual_text)} ({html.escape(summary.difference_gw)} versus usual). {html.escape(summary.main_driver)}</p>
          </div>
          <div class="ep-now-hero-grid">
            <div><span>Near-term direction</span><strong>{html.escape(summary.direction)}</strong></div>
            <div><span>Last update</span><strong>{html.escape(summary.last_update)}</strong><small>{html.escape(summary.freshness)}</small></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_rows(
    current_state: CurrentStateResponse | None,
    selected_snapshot: TwinSnapshot | None,
    *,
    timezone_name: str,
) -> None:
    official = _official_signal(current_state, selected_snapshot)
    balance = _modelled_balance(current_state, selected_snapshot)
    official_label, official_detail, official_status, official_source = _official_row_parts(official, timezone_name)
    balance_label, balance_detail, balance_status = _balance_row_parts(balance)
    st.markdown(
        f"""
        <div class="ep-status-row-wrap" aria-label="Official signal and app balance context rows">
          <div class="ep-source-status-row">
            <div>
              <div class="ep-label">Official EcoWatt signal</div>
              <div class="ep-title">{html.escape(official_label)}</div>
              <div class="ep-detail">{html.escape(official_detail)}</div>
            </div>
            <div class="ep-status-actions">
              {status_badge_html(official_status, official_status)}
              {provenance_badge_html("official" if official is not None and _official_available(official) else "unavailable")}
            </div>
            <div class="ep-status-source">{html.escape(official_source)}</div>
          </div>
          <div class="ep-source-status-row ep-source-status-modelled">
            <div>
              <div class="ep-label">Modelled national balance context</div>
              <div class="ep-title">{html.escape(balance_label)}</div>
              <div class="ep-detail">{html.escape(balance_detail)}</div>
            </div>
            <div class="ep-status-actions">
              {status_badge_html(balance_status, balance_status)}
              {provenance_badge_html("modelled" if balance is not None else "unavailable")}
            </div>
            <div class="ep-status-source">Separate from EcoWatt; not an official warning.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_next_12h_ribbon(
    twin: TwinResponse | None,
    selected: datetime | None,
    *,
    timezone_name: str,
) -> datetime | None:
    snapshots = next_12_snapshots(twin)
    if not snapshots:
        st.info("Next-12-hours forecast context is unavailable.")
        return selected

    options = [pd.Timestamp(item.event_time) for item in snapshots]
    index = select_timestamp_from_options(options, selected)
    selected_ts = options[index]
    selected_day: str | None = None
    st.markdown('<div class="ep-next12" role="group" aria-label="Next 12 hours">', unsafe_allow_html=True)
    for day, group in _snapshot_groups_by_local_day(snapshots, timezone_name):
        if day != selected_day:
            selected_day = day
            st.markdown(f'<div class="ep-next12-date">{html.escape(day)}</div>', unsafe_allow_html=True)
        columns = st.columns(len(group), gap="small")
        for column, item in zip(columns, group):
            local = pd.Timestamp(item.event_time).tz_convert(timezone_name)
            demand = _quantified_value(item.demand_forecast.p50 if item.demand_forecast else None)
            status_text = _status_display(
                item.modelled_national_balance_context.status
                if item.modelled_national_balance_context is not None
                else Status.UNKNOWN
            )
            label = f"{local:%H:%M}\n{status_text}\n{format_gw(demand)}"
            with column:
                if st.button(
                    label,
                    key=f"next12_{local:%Y%m%d%H}",
                    use_container_width=True,
                    type="primary" if pd.Timestamp(item.event_time) == selected_ts else "secondary",
                ):
                    selected_ts = pd.Timestamp(item.event_time)
    st.markdown("</div>", unsafe_allow_html=True)
    return selected_ts.to_pydatetime()


def render_forecast_point_ribbon(
    points: list[ForecastPointView],
    selected: datetime | None,
    *,
    timezone_name: str,
) -> datetime | None:
    if not points:
        st.info("Next-12-hours forecast context is unavailable.")
        return selected
    visible = points[:12]
    options = [pd.Timestamp(point.timestamp) for point in visible]
    index = select_timestamp_from_options(options, selected)
    selected_ts = options[index]
    selected_day: str | None = None
    st.markdown('<div class="ep-next12" role="group" aria-label="Next 12 hours">', unsafe_allow_html=True)
    for day, group in _forecast_groups_by_local_day(visible, timezone_name):
        if day != selected_day:
            selected_day = day
            st.markdown(f'<div class="ep-next12-date">{html.escape(day)}</div>', unsafe_allow_html=True)
        columns = st.columns(len(group), gap="small")
        for column, point in zip(columns, group):
            local = point.timestamp.tz_convert(timezone_name)
            label = f"{local:%H:%M}\n{point.pressure_label}\n{format_gw(point.p50)}"
            with column:
                if st.button(
                    label,
                    key=f"next12_{local:%Y%m%d%H}",
                    use_container_width=True,
                    type="primary" if pd.Timestamp(point.timestamp) == selected_ts else "secondary",
                ):
                    selected_ts = pd.Timestamp(point.timestamp)
    st.markdown("</div>", unsafe_allow_html=True)
    return selected_ts.to_pydatetime()


def selected_forecast_point(
    points: list[ForecastPointView],
    selected: datetime | None,
) -> ForecastPointView | None:
    if not points:
        return None
    if selected is None:
        return points[0]
    target = pd.Timestamp(selected)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    target = target.tz_convert("UTC")
    return min(points, key=lambda point: abs(pd.Timestamp(point.timestamp).tz_convert("UTC") - target))


def render_selected_forecast_context(point: ForecastPointView | None, *, timezone_name: str) -> None:
    if point is None:
        explanation_card(
            "Selected-hour forecast unavailable",
            "No forecast point is available for the selected hour.",
            label="Selected hour",
            status="unknown",
            provenance="unavailable",
        )
        return
    local = point.timestamp.tz_convert(timezone_name)
    explanation_card(
        f"{local:%a %d %b %H:%M}: {format_gw(point.p50)}",
        f"{point.pressure_label}. Forecast route: {point.source}. This context is modelled and is not an official warning.",
        label="Selected-hour forecast context",
        status=point.pressure_label,
        provenance="modelled",
    )


def build_current_state_map_frame(current_state: CurrentStateResponse | None) -> pd.DataFrame:
    if current_state is None:
        return pd.DataFrame()
    freshness = current_state.selected_region_context.freshness
    freshness_label = freshness_label_text(freshness.state, freshness.age_seconds)
    records: list[dict[str, Any]] = []
    for region in current_state.map:
        observed = region.observed_demand.value
        usual = region.usual_demand.value
        anomaly = region.demand_anomaly_pct.value
        difference = None if observed is None or usual is None else observed - usual
        records.append(
            {
                "region_code": region.region_id,
                "region_display": region.region_name,
                "demand_anomaly_pct": anomaly,
                "demand_anomaly_label": demand_difference_text(anomaly, already_percent=True),
                "consumption_mw": observed,
                "usual_demand_mw": usual,
                "difference_mw": difference,
                "demand_label": format_mw(observed),
                "usual_label": format_mw(usual),
                "difference_label": format_signed_mw(difference),
                "freshness_label": freshness_label,
                "source_label": source_quality_label(region.source_quality),
                "availability_flag": bool(region.availability_flag),
                "unavailable_reason": _metric_reason(
                    region.observed_demand,
                    region.usual_demand,
                    region.demand_anomaly_pct,
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def fallback_map_frame(regional: pd.DataFrame, *, freshness_label: str, source_label: str) -> pd.DataFrame:
    if regional.empty:
        return pd.DataFrame()
    frame = regional.copy()
    frame["demand_anomaly_pct"] = pd.to_numeric(frame.get("demand_anomaly_pct"), errors="coerce") * 100
    if "usual_demand_mw" not in frame:
        frame["usual_demand_mw"] = pd.NA
    frame["difference_mw"] = pd.to_numeric(frame.get("consumption_mw"), errors="coerce") - pd.to_numeric(
        frame.get("usual_demand_mw"),
        errors="coerce",
    )
    frame["demand_label"] = frame["consumption_mw"].map(format_mw)
    frame["usual_label"] = frame["usual_demand_mw"].map(format_mw)
    frame["difference_label"] = frame["difference_mw"].map(format_signed_mw)
    frame["freshness_label"] = freshness_label
    frame["source_label"] = source_label
    frame["availability_flag"] = frame["demand_anomaly_pct"].notna()
    frame["unavailable_reason"] = "Regional demand anomaly is unavailable."
    return frame


def render_region_selector(selected_region: str) -> str:
    codes = list(REGION_NAMES)
    index = codes.index(selected_region) if selected_region in codes else 0
    return st.selectbox(
        "Selected region",
        codes,
        index=index,
        format_func=lambda code: REGION_NAMES.get(code, code),
        help="Select a region to update the regional contribution panel.",
    )


def render_selected_region_panel(
    current_state: CurrentStateResponse | None,
    map_frame: pd.DataFrame,
    selected_region: str,
) -> None:
    selected_row = _map_row(map_frame, selected_region)
    context = current_state.selected_region_context if current_state and current_state.region == selected_region else None
    region_name = REGION_NAMES.get(selected_region, selected_region)
    if context is not None:
        region_name = context.region_name
    demand = context.demand.current.value if context is not None else _row_value(selected_row, "consumption_mw")
    anomaly_pct = (
        context.demand.difference_vs_usual_pct.value
        if context is not None
        else _row_value(selected_row, "demand_anomaly_pct")
    )
    local_generation = context.local_generation.total.value if context is not None else None
    net_flow = context.net_flow.value if context is not None else None
    physical_balance = context.physical_balance.value if context is not None else None
    flow_text = (
        f"Physical balance {format_signed_mw(physical_balance)}"
        if physical_balance is not None
        else f"Net flow {format_signed_mw(net_flow)}"
    )
    metric_card(
        "Regional demand",
        format_mw(demand),
        demand_difference_text(anomaly_pct, already_percent=True),
        icon="Demand",
        status=_status_for_anomaly_pct(anomaly_pct),
        provenance=_provenance_for_current_state(current_state),
    )
    metric_card(
        "Local generation",
        format_mw(local_generation),
        "Generation measured in the region, within the connected French grid.",
        icon="Gen",
        status="info",
        provenance=_provenance_for_current_state(current_state),
    )
    metric_card(
        "Physical flow context",
        flow_text,
        "Imports and exports are physical flow context when available.",
        icon="Flow",
        status="info",
        provenance=_provenance_for_current_state(current_state),
    )
    note = (
        context.connected_grid_note
        if context is not None
        else "Regional context is not an independent shortage warning; unavailable fields stay labelled unavailable."
    )
    explanation_card(
        region_name,
        note,
        label="Selected-region context",
        status="info",
        provenance="modelled" if context is not None else "unavailable",
    )


def render_driver_cards(
    current_state: CurrentStateResponse | None,
    selected_snapshot: TwinSnapshot | None,
    snapshot: GridSnapshotView,
) -> None:
    demand = _current_demand(current_state, snapshot)
    usual = _metric_value(current_state.national_context.demand.usual if current_state is not None else None)
    anomaly = _current_anomaly_pct(current_state, snapshot)
    balance = _modelled_balance(current_state, selected_snapshot)
    official = _official_signal(current_state, selected_snapshot)
    generation_detail = _generation_exchange_driver_detail(current_state, selected_snapshot)
    weather = snapshot.weather
    columns = st.columns(4)
    with columns[0]:
        driver_card(
            "Demand",
            demand_difference_text(anomaly, already_percent=True),
            f"France is using {format_gw(demand)}; usual for this context is {format_gw(usual)}.",
            provenance=_provenance_for_current_state(current_state),
        )
    with columns[1]:
        driver_card(
            "Gen",
            _balance_label(balance),
            generation_detail,
            label="Generation and exchange",
            provenance="modelled" if balance is not None else "unavailable",
        )
    with columns[2]:
        driver_card(
            "Weather",
            str(weather.get("headline", "Weather unavailable")),
            str(weather.get("detail", "Weather context is unavailable for this snapshot.")),
            provenance="replay" if snapshot.mode == "REPLAY" else "observed",
        )
    with columns[3]:
        driver_card(
            "Eco",
            _official_title(official),
            _official_detail(official),
            label="Official signal",
            provenance="official" if official is not None and _official_available(official) else "unavailable",
        )


def generation_mix_figure(
    mix: CurrentGenerationMix | None,
    *,
    demand_mw: float | None,
    net_imports_mw: float | None,
) -> go.Figure:
    items = generation_mix_items(mix)
    labels = [item["label"] for item in items]
    values = [float(item["value_mw"]) for item in items]
    total = sum(values)
    max_x = max([total, float(demand_mw or 0), 1.0]) * 1.12
    fig = go.Figure()
    for item in items:
        fig.add_bar(
            x=[item["value_mw"]],
            y=["Generation"],
            orientation="h",
            name=item["label"],
            marker_color=item["color"],
            text=[item["direct_label"]],
            textposition="inside" if item["share"] >= 0.08 else "none",
            insidetextanchor="middle",
            hovertemplate=f"{html.escape(item['label'])}: %{{x:,.0f}} MW<extra></extra>",
        )
    if demand_mw is not None:
        fig.add_vline(
            x=float(demand_mw),
            line_color="#f8fafc",
            line_width=2,
            annotation_text="Demand",
            annotation_position="top",
        )
    flow = flow_context_text(net_imports_mw)
    fig.add_annotation(
        x=max_x,
        y="Generation",
        text=flow,
        showarrow=False,
        xanchor="right",
        yshift=34,
        font=dict(color="#bae6fd", size=12),
    )
    fig.update_layout(
        **dark_chart_layout(
            barmode="stack",
            height=230,
            margin=dict(l=8, r=28, t=42, b=34),
            xaxis_title="MW",
            yaxis_title=None,
            showlegend=False,
            hovermode="closest",
        )
    )
    fig.update_xaxes(range=[0, max_x])
    return fig


def generation_mix_items(mix: CurrentGenerationMix | None) -> list[dict[str, Any]]:
    if mix is None:
        return [{"label": "Unavailable", "value_mw": 0.0, "share": 1.0, "direct_label": "Unavailable", "color": "#64748b"}]
    raw = [
        {
            "label": technology_label(item.technology),
            "value_mw": float(item.power.value or 0.0),
            "share": float(item.share.value or 0.0) / 100.0,
            "color": COLORS.get(technology_label(item.technology), "#94a3b8"),
        }
        for item in mix.technologies
        if item.power.value is not None and float(item.power.value) > 0
    ]
    if not raw:
        return [{"label": "Unavailable", "value_mw": 0.0, "share": 1.0, "direct_label": "Unavailable", "color": "#64748b"}]
    raw = sorted(raw, key=lambda item: item["value_mw"], reverse=True)
    keep: list[dict[str, Any]] = []
    other_value = 0.0
    total = sum(float(item["value_mw"]) for item in raw)
    for index, item in enumerate(raw):
        if index >= 5 or (len(raw) > 6 and item["value_mw"] / max(total, 1.0) < 0.03):
            other_value += item["value_mw"]
        else:
            keep.append(item)
    if other_value > 0:
        keep.append(
            {
                "label": "Other",
                "value_mw": other_value,
                "share": other_value / max(total, 1.0),
                "color": "#94a3b8",
            }
        )
    for item in keep:
        item["direct_label"] = f"{item['label']} {format_gw(item['value_mw'])}"
    return keep


def render_carbon_context(current_state: CurrentStateResponse | None, selected_snapshot: TwinSnapshot | None) -> None:
    carbon = None
    provenance = "unavailable"
    if selected_snapshot is not None and selected_snapshot.carbon_estimate is not None:
        carbon = selected_snapshot.carbon_estimate.intensity.value
        provenance = provenance_kind_for_source(selected_snapshot.carbon_estimate.source)
    elif current_state is not None:
        carbon = current_state.national_context.carbon_estimate.estimate.value
        provenance = _provenance_for_current_state(current_state)
    metric_card(
        "Carbon context",
        format_carbon(carbon),
        "Displayed separately; not used in official EcoWatt or modelled balance status.",
        icon="CO2",
        status="info",
        provenance=provenance,
    )


def demand_difference_text(value: float | None, *, already_percent: bool = True) -> str:
    if value is None or pd.isna(value):
        return "Demand versus usual unavailable"
    percent = float(value) if already_percent else float(value) * 100
    if abs(percent) < 7:
        return "Close to usual"
    if percent < 0:
        return f"{abs(percent):.0f}% below usual"
    return f"{percent:.0f}% above usual"


def freshness_label_text(state: OperatingState | Freshness | None, age_seconds: float | None) -> str:
    if state is None:
        return "Freshness unavailable"
    state_text = str(getattr(state, "value", state)).replace("_", " ").title()
    age = format_age_seconds(age_seconds)
    return state_text if age == UNAVAILABLE else f"{state_text}, {age}"


def source_quality_label(value: str | None) -> str:
    if not value:
        return "Source unavailable"
    return str(value).replace("_", " ").replace("-", " ").title()


def technology_label(value: str) -> str:
    lookup = {
        "nuclear": "Nuclear",
        "wind": "Wind",
        "solar": "Solar",
        "hydro": "Hydro",
        "gas": "Gas",
        "coal": "Coal",
        "oil": "Oil",
        "bioenergy": "Bioenergy",
    }
    key = str(value).strip().lower()
    return lookup.get(key, key.replace("_", " ").title())


def flow_context_text(net_imports_mw: float | None) -> str:
    if net_imports_mw is None:
        return "Net exchange unavailable"
    if abs(float(net_imports_mw)) < 1:
        return "Net exchange near zero"
    if net_imports_mw > 0:
        return f"Net imports {format_gw(net_imports_mw, signed=True)}"
    return f"Net exports {format_gw(abs(net_imports_mw), signed=False)}"


def near_term_direction(
    twin: TwinResponse | None,
    current_demand_mw: float | None,
    *,
    forecast_points: list[ForecastPointView] | None = None,
) -> str:
    point_values = [point.p50 for point in (forecast_points or [])[:3]]
    if point_values:
        values = [float(value) for value in point_values if value is not None]
    else:
        snapshots = next_12_snapshots(twin)[:3]
        values = [
            _quantified_value(item.demand_forecast.p50 if item.demand_forecast else None)
            for item in snapshots
        ]
        values = [value for value in values if value is not None]
    if not values or current_demand_mw is None:
        return "Near-term direction unavailable"
    delta = values[-1] - float(current_demand_mw)
    threshold = max(abs(float(current_demand_mw)) * 0.01, 500.0)
    if delta > threshold:
        return f"Rising over the next few hours ({format_gw(delta, signed=True)})"
    if delta < -threshold:
        return f"Falling over the next few hours ({format_gw(delta, signed=True)})"
    return "Broadly steady over the next few hours"


def main_driver_text(
    current_state: CurrentStateResponse | None,
    twin: TwinResponse | None,
    snapshot: GridSnapshotView,
) -> str:
    anomaly = _current_anomaly_pct(current_state, snapshot)
    if anomaly is not None and abs(float(anomaly)) >= 7:
        return "The clearest driver is demand compared with usual."
    current_snapshot = twin.snapshots[0] if twin is not None and twin.snapshots else None
    if current_snapshot is not None and current_snapshot.modelled_balance_contributions:
        contribution = max(current_snapshot.modelled_balance_contributions, key=lambda item: item.contribution)
        if contribution.component == "announced_unavailability_ratio" and contribution.contribution > 0.08:
            return "Announced generation unavailability is the main modelled driver."
        if contribution.component == "residual_load_percentile" and contribution.contribution > 0.55:
            return "Demand left after wind and solar is the main modelled driver."
    headline = str(snapshot.weather.get("headline", ""))
    if headline and "unavailable" not in headline.lower():
        return f"Weather context: {headline.lower()}."
    return "No single driver is dominant from the available sources."


def _snapshot_groups_by_local_day(snapshots: list[TwinSnapshot], timezone_name: str) -> list[tuple[str, list[TwinSnapshot]]]:
    groups: list[tuple[str, list[TwinSnapshot]]] = []
    for item in snapshots:
        local = pd.Timestamp(item.event_time).tz_convert(timezone_name)
        label = f"{local:%a %d %b}"
        if groups and groups[-1][0] == label:
            groups[-1][1].append(item)
        else:
            groups.append((label, [item]))
    return groups


def _forecast_groups_by_local_day(
    points: list[ForecastPointView],
    timezone_name: str,
) -> list[tuple[str, list[ForecastPointView]]]:
    groups: list[tuple[str, list[ForecastPointView]]] = []
    for point in points:
        local = point.timestamp.tz_convert(timezone_name)
        label = f"{local:%a %d %b}"
        if groups and groups[-1][0] == label:
            groups[-1][1].append(point)
        else:
            groups.append((label, [point]))
    return groups


def _official_signal(
    current_state: CurrentStateResponse | None,
    selected_snapshot: TwinSnapshot | None,
) -> CurrentStateResponse | OfficialSignal | Any:
    if selected_snapshot is not None and selected_snapshot.official_signal_context is not None:
        return selected_snapshot.official_signal_context
    if current_state is not None:
        return current_state.national_context.official_ecowatt_signal
    return None


def _modelled_balance(
    current_state: CurrentStateResponse | None,
    selected_snapshot: TwinSnapshot | None,
) -> ModelledBalanceContext | Any | None:
    if selected_snapshot is not None and selected_snapshot.modelled_national_balance_context is not None:
        return selected_snapshot.modelled_national_balance_context
    if current_state is not None:
        return current_state.national_context.modelled_status
    return None


def _official_row_parts(official: Any, timezone_name: str) -> tuple[str, str, str, str]:
    if official is None:
        return "EcoWatt unavailable", "No official signal is available in the current typed response.", "Unavailable", ""
    title = _official_title(official)
    detail = _official_detail(official)
    time_value = getattr(official, "signal_time", None) or getattr(official, "timestamp", None)
    if time_value is not None:
        detail = f"{detail} Signal time: {format_timestamp(time_value, timezone_name=timezone_name)}."
    status = _official_status(official)
    raw_source = getattr(official, "source", "") or ""
    source = str(getattr(raw_source, "name", raw_source))
    return title, detail, status, source


def _balance_row_parts(balance: Any | None) -> tuple[str, str, str]:
    if balance is None:
        return "Balance context unavailable", "The modelled balance context is unavailable for the selected hour.", "Unavailable"
    label = _balance_label(balance)
    margin = getattr(balance, "supply_margin", None)
    imports = getattr(balance, "net_imports", None)
    if margin is not None:
        detail = f"Generation plus exchange leaves {format_signed_mw(_quantified_value(margin))} versus demand."
    else:
        detail = str(getattr(balance, "reason", "") or "Documented threshold context from demand and generation inputs.")
    if imports is not None:
        detail = f"{detail} {flow_context_text(_quantified_value(imports))}."
    return label, detail, label


def _official_title(official: Any | None) -> str:
    if official is None:
        return "EcoWatt unavailable"
    if hasattr(official, "available") and not bool(official.available):
        return "EcoWatt unavailable"
    label = getattr(official, "label", None)
    return str(label or "EcoWatt unavailable")


def _official_detail(official: Any | None) -> str:
    if official is None:
        return "No official EcoWatt signal is available."
    if hasattr(official, "available") and not bool(official.available):
        return str(getattr(official, "reason", None) or "No official EcoWatt signal is available.")
    return str(getattr(official, "detail", None) or getattr(official, "reason", None) or "Official national signal.")


def _official_available(official: Any) -> bool:
    if hasattr(official, "available"):
        return bool(official.available)
    status = getattr(official, "status", Status.UNKNOWN)
    return status not in {Status.UNKNOWN, "unknown", None}


def _official_status(official: Any) -> str:
    if not _official_available(official):
        return "Unavailable"
    status = getattr(official, "status", None)
    if status is None:
        return _official_title(official)
    return _status_display(status)


def _balance_label(balance: Any | None) -> str:
    if balance is None:
        return "Unavailable"
    return _status_display(getattr(balance, "status", getattr(balance, "label", "Unknown")))


def _status_display(value: Any) -> str:
    if isinstance(value, Status):
        return {
            Status.NORMAL: "Normal",
            Status.WATCH: "Watch",
            Status.HIGH: "High",
            Status.UNKNOWN: "Unknown",
        }[value]
    text = str(value or "Unknown")
    lowered = text.lower()
    return {
        "normal": "Normal",
        "watch": "Watch",
        "high": "High",
        "unknown": "Unknown",
        "green": "Normal",
        "orange": "Watch",
        "red": "High",
    }.get(lowered, text[:1].upper() + text[1:])


def _generation_exchange_driver_detail(
    current_state: CurrentStateResponse | None,
    selected_snapshot: TwinSnapshot | None,
) -> str:
    balance = _modelled_balance(current_state, selected_snapshot)
    if balance is not None and hasattr(balance, "supply_margin"):
        margin = _quantified_value(balance.supply_margin)
        imports = _quantified_value(balance.net_imports)
        return f"Generation and exchange context is {format_signed_mw(margin)} versus demand. {flow_context_text(imports)}."
    if current_state is not None:
        imports = current_state.national_context.net_imports.value
        generation = current_state.national_context.generation_mix.total.value
        return f"Observed generation is {format_gw(generation)}. {flow_context_text(imports)}."
    return "Generation and exchange context is unavailable."


def _current_demand(current_state: CurrentStateResponse | None, snapshot: GridSnapshotView) -> float | None:
    if current_state is not None:
        return current_state.national_context.demand.current.value
    return float(snapshot.demand.get("current_mw", 0) or 0)


def _current_anomaly_pct(current_state: CurrentStateResponse | None, snapshot: GridSnapshotView) -> float | None:
    if current_state is not None:
        return current_state.national_context.demand.difference_vs_usual_pct.value
    value = snapshot.demand.get("anomaly_pct")
    return None if value is None else float(value) * 100


def _current_difference_gw(current_state: CurrentStateResponse | None, snapshot: GridSnapshotView) -> float | None:
    if current_state is not None:
        return current_state.national_context.demand.difference_vs_usual_gw.value
    current = snapshot.demand.get("current_mw")
    usual = snapshot.demand.get("usual_mw")
    if current is None or usual is None:
        return None
    return (float(current) - float(usual)) / 1000.0


def _format_gw_delta(value_gw: float | None) -> str:
    if value_gw is None:
        return UNAVAILABLE
    return f"{value_gw:+,.1f} GW"


def _metric_value(metric: NullableMetric | QuantifiedValue | None) -> float | None:
    return None if metric is None else metric.value


def _quantified_value(value: QuantifiedValue | NullableMetric | None) -> float | None:
    return None if value is None else value.value


def _metric_reason(*metrics: NullableMetric) -> str:
    for metric in metrics:
        if metric.reason:
            return metric.reason
    return "Regional value is unavailable."


def _map_row(frame: pd.DataFrame, selected_region: str) -> pd.Series | None:
    if frame.empty or "region_code" not in frame:
        return None
    rows = frame[frame["region_code"].astype(str).eq(str(selected_region))]
    if rows.empty:
        return None
    return rows.iloc[0]


def _row_value(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row:
        return None
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _status_for_anomaly_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    if abs(float(value)) < 7:
        return "normal"
    return "watch" if abs(float(value)) < 15 else "high"


def _provenance_for_current_state(current_state: CurrentStateResponse | None) -> str:
    if current_state is None:
        return "unavailable"
    if current_state.operating_state == OperatingState.HISTORICAL_REPLAY:
        return "replay"
    if current_state.operating_state in {
        OperatingState.SOURCE_UNAVAILABLE,
    }:
        return "unavailable"
    if current_state.operating_state == OperatingState.LAST_KNOWN_GOOD_FALLBACK:
        return "fallback"
    return "observed"
