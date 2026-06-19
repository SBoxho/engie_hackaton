from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.cards import driver_card, explanation_card, metric_card
from app.components.charts import COLORS, dark_chart_layout
from app.components.foundation import provenance_kind_for_source
from app.formatting import (
    UNAVAILABLE,
    format_age_seconds,
    format_carbon,
    format_date,
    format_gw,
    format_mw,
    format_number,
    format_percentage,
    format_signed_mw,
    format_temperature,
    format_timestamp,
)
from app.i18n import t
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
    locale: str = "en",
) -> HeroSummary:
    demand = _current_demand(current_state, snapshot)
    pct = _current_anomaly_pct(current_state, snapshot)
    diff_gw = _current_difference_gw(current_state, snapshot)
    freshness = current_state.national_context.freshness if current_state is not None else None
    update_time = freshness.timestamp if freshness is not None and freshness.timestamp is not None else snapshot.as_of
    age = (
        _localize_unavailable(format_age_seconds(freshness.age_seconds, locale=locale), locale)
        if freshness is not None
        else str(snapshot.freshness.get("label", ""))
    )
    return HeroSummary(
        demand_gw=_format_gw_text(demand, locale=locale),
        unusual_text=demand_difference_text(pct, locale=locale),
        difference_gw=_format_gw_delta(diff_gw, locale=locale),
        main_driver=main_driver_text(current_state, twin, snapshot, locale=locale),
        direction=near_term_direction(twin, demand, forecast_points=forecast_points, locale=locale),
        last_update=_localize_unavailable(format_timestamp(update_time, timezone_name=timezone_name, locale=locale), locale),
        freshness=age,
    )


def render_hero_summary(summary: HeroSummary, *, locale: str = "en") -> None:
    aria_label = t("now.hero.aria", locale=locale)
    eyebrow = t("now.hero.eyebrow", locale=locale)
    title = t("now.hero.title", locale=locale, demand=summary.demand_gw)
    body = t(
        "now.hero.body",
        locale=locale,
        unusual=summary.unusual_text,
        difference=summary.difference_gw,
        driver=summary.main_driver,
    )
    near_term_label = t("now.hero.near_term_direction", locale=locale)
    last_update_label = t("now.hero.last_update", locale=locale)
    st.markdown(
        f"""
        <section class="ep-now-hero" aria-label="{html.escape(aria_label)}">
          <div>
            <div class="ep-eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(body)}</p>
          </div>
          <div class="ep-now-hero-grid">
            <div><span>{html.escape(near_term_label)}</span><strong>{html.escape(summary.direction)}</strong></div>
            <div><span>{html.escape(last_update_label)}</span><strong>{html.escape(summary.last_update)}</strong><small>{html.escape(summary.freshness)}</small></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_next_12h_ribbon(
    twin: TwinResponse | None,
    selected: datetime | None,
    *,
    timezone_name: str,
    locale: str = "en",
) -> datetime | None:
    snapshots = next_12_snapshots(twin)
    if not snapshots:
        st.info(t("now.forecast.unavailable_info", locale=locale))
        return selected

    options = [pd.Timestamp(item.event_time) for item in snapshots]
    index = select_timestamp_from_options(options, selected)
    selected_ts = options[index]
    selected_day: str | None = None
    aria_label = t("now.forecast.aria", locale=locale)
    st.markdown(
        f'<div class="ep-next12" role="group" aria-label="{html.escape(aria_label)}">',
        unsafe_allow_html=True,
    )
    for day, group in _snapshot_groups_by_local_day(snapshots, timezone_name, locale=locale):
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
                else Status.UNKNOWN,
                locale=locale,
            )
            label = f"{local:%H:%M}\n{status_text}\n{_format_gw_text(demand, locale=locale)}"
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
    locale: str = "en",
) -> datetime | None:
    if not points:
        st.info(t("now.forecast.unavailable_info", locale=locale))
        return selected
    visible = points[:12]
    options = [pd.Timestamp(point.timestamp) for point in visible]
    index = select_timestamp_from_options(options, selected)
    selected_ts = options[index]
    selected_day: str | None = None
    aria_label = t("now.forecast.aria", locale=locale)
    st.markdown(
        f'<div class="ep-next12" role="group" aria-label="{html.escape(aria_label)}">',
        unsafe_allow_html=True,
    )
    for day, group in _forecast_groups_by_local_day(visible, timezone_name, locale=locale):
        if day != selected_day:
            selected_day = day
            st.markdown(f'<div class="ep-next12-date">{html.escape(day)}</div>', unsafe_allow_html=True)
        columns = st.columns(len(group), gap="small")
        for column, point in zip(columns, group):
            local = point.timestamp.tz_convert(timezone_name)
            label = f"{local:%H:%M}\n{_status_display(point.pressure_label, locale=locale)}\n{_format_gw_text(point.p50, locale=locale)}"
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


def current_weather_summary(
    live_payload: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
    *,
    locale: str = "en",
) -> dict[str, Any]:
    """Merge a live Open-Meteo current payload with the snapshot's fallback weather.

    The live payload (when available) wins on raw values; the fallback supplies
    the headline-classification text whenever we can compute one. Returns a dict
    with at minimum ``headline`` and ``detail`` keys so downstream renderers
    never have to None-check.
    """
    base = dict(fallback or {})
    if live_payload:
        for key in ("temperature_c", "wind_kmh", "cloud_pct"):
            value = live_payload.get(key)
            if value is not None:
                base[key] = value
        base["source"] = live_payload.get("source", base.get("source"))
        base["location"] = live_payload.get("location", base.get("location"))
        base["observed_at"] = live_payload.get("observed_at", base.get("observed_at"))
        base["is_live"] = True
    else:
        base.setdefault("is_live", False)

    temp = base.get("temperature_c")
    wind = base.get("wind_kmh")
    if temp is None and wind is None:
        base["headline"] = t("now.weather.unavailable_headline", locale=locale)
        base["detail"] = t("now.weather.unavailable_detail", locale=locale)
        return base

    temp_value = 0.0 if temp is None else float(temp)
    wind_value = 0.0 if wind is None else float(wind)
    cloud_value = float(base.get("cloud_pct") or 0.0)
    base["headline"] = _weather_headline_for_values(temp_value, wind_value, locale=locale)
    base["detail"] = _weather_detail_for_values(temp_value, wind_value, cloud_value, locale=locale)
    base["temperature_c"] = temp_value
    base["wind_kmh"] = wind_value
    base["cloud_pct"] = cloud_value
    return base


def render_selected_forecast_context(point: ForecastPointView | None, *, timezone_name: str, locale: str = "en") -> None:
    if point is None:
        explanation_card(
            t("now.forecast.unavailable_title", locale=locale),
            t("now.forecast.unavailable_body", locale=locale),
            label=t("now.forecast.unavailable_label", locale=locale),
            provenance="unavailable",
        )
        return
    timestamp_text = _localize_unavailable(
        format_timestamp(point.timestamp, timezone_name=timezone_name, locale=locale),
        locale,
    )
    explanation_card(
        t(
            "now.forecast.title",
            locale=locale,
            timestamp=timestamp_text,
            demand=_format_gw_text(point.p50, locale=locale),
        ),
        t(
            "now.forecast.detail",
            locale=locale,
            status=_status_display(point.pressure_label, locale=locale),
            source=_route_label(point.source, locale=locale),
        ),
        label=t("now.forecast.label", locale=locale),
        provenance="modelled",
    )


def build_current_state_map_frame(current_state: CurrentStateResponse | None, *, locale: str = "en") -> pd.DataFrame:
    if current_state is None:
        return pd.DataFrame()
    freshness = current_state.selected_region_context.freshness
    freshness_label = freshness_label_text(freshness.state, freshness.age_seconds, locale=locale)
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
                "demand_anomaly_label": demand_difference_text(anomaly, already_percent=True, locale=locale),
                "consumption_mw": observed,
                "usual_demand_mw": usual,
                "difference_mw": difference,
                "demand_label": _format_mw_text(observed, locale=locale),
                "usual_label": _format_mw_text(usual, locale=locale),
                "difference_label": _format_signed_mw_text(difference, locale=locale),
                "freshness_label": freshness_label,
                "source_label": source_quality_label(region.source_quality, locale=locale),
                "availability_flag": bool(region.availability_flag),
                "unavailable_reason": _metric_reason(
                    region.observed_demand,
                    region.usual_demand,
                    region.demand_anomaly_pct,
                    locale=locale,
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def fallback_map_frame(regional: pd.DataFrame, *, freshness_label: str, source_label: str, locale: str = "en") -> pd.DataFrame:
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
    frame["demand_label"] = frame["consumption_mw"].map(lambda value: _format_mw_text(value, locale=locale))
    frame["usual_label"] = frame["usual_demand_mw"].map(lambda value: _format_mw_text(value, locale=locale))
    frame["difference_label"] = frame["difference_mw"].map(lambda value: _format_signed_mw_text(value, locale=locale))
    frame["freshness_label"] = freshness_label
    frame["source_label"] = source_label
    frame["availability_flag"] = frame["demand_anomaly_pct"].notna()
    frame["unavailable_reason"] = t("now.states.regional_anomaly_unavailable", locale=locale)
    return frame


def render_region_selector(selected_region: str, *, locale: str = "en") -> str:
    codes = list(REGION_NAMES)
    index = codes.index(selected_region) if selected_region in codes else 0
    return st.selectbox(
        t("now.regions.selector_label", locale=locale),
        codes,
        index=index,
        format_func=lambda code: REGION_NAMES.get(code, code),
        help=t("now.regions.selector_help", locale=locale),
    )


def render_selected_region_panel(
    current_state: CurrentStateResponse | None,
    map_frame: pd.DataFrame,
    selected_region: str,
    *,
    hide_replay_badge: bool = False,
    locale: str = "en",
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
        t("now.regions.flow_balance", locale=locale, value=_format_signed_mw_text(physical_balance, locale=locale))
        if physical_balance is not None
        else t("now.regions.flow_net", locale=locale, value=_format_signed_mw_text(net_flow, locale=locale))
    )
    provenance = _provenance_for_current_state(current_state, hide_replay=hide_replay_badge)
    metric_card(
        t("now.regions.demand_label", locale=locale),
        _format_mw_text(demand, locale=locale),
        demand_difference_text(anomaly_pct, already_percent=True, locale=locale),
        icon=t("now.regions.demand_icon", locale=locale),
        provenance=provenance,
    )
    metric_card(
        t("now.regions.generation_label", locale=locale),
        _format_mw_text(local_generation, locale=locale),
        t("now.regions.generation_detail", locale=locale),
        icon=t("now.regions.generation_icon", locale=locale),
        provenance=provenance,
    )
    metric_card(
        t("now.regions.flow_label", locale=locale),
        flow_text,
        t("now.regions.flow_detail", locale=locale),
        icon=t("now.regions.flow_icon", locale=locale),
        provenance=provenance,
    )
    note = (
        t("now.regions.connected_grid_note", locale=locale)
        if context is not None
        else t("now.regions.note_unavailable", locale=locale)
    )
    explanation_card(
        region_name,
        note,
        label=t("now.regions.selected_label", locale=locale),
        provenance="modelled" if context is not None else "unavailable",
    )


def render_driver_cards(
    current_state: CurrentStateResponse | None,
    selected_snapshot: TwinSnapshot | None,
    snapshot: GridSnapshotView,
    *,
    weather_override: dict[str, Any] | None = None,
    hide_replay_badge: bool = False,
    locale: str = "en",
) -> None:
    demand = _current_demand(current_state, snapshot)
    usual = _metric_value(current_state.national_context.demand.usual if current_state is not None else None)
    anomaly = _current_anomaly_pct(current_state, snapshot)
    balance = _modelled_balance(current_state, selected_snapshot)
    official = _official_signal(current_state, selected_snapshot)
    generation_detail = _generation_exchange_driver_detail(current_state, selected_snapshot, locale=locale)
    weather = weather_override if weather_override is not None else snapshot.weather
    weather_is_live = bool(weather_override and weather_override.get("is_live"))
    columns = st.columns(4)
    with columns[0]:
        driver_card(
            t("now.drivers.demand_icon", locale=locale),
            demand_difference_text(anomaly, already_percent=True, locale=locale),
            t(
                "now.drivers.demand_detail",
                locale=locale,
                demand=_format_gw_text(demand, locale=locale),
                usual=_format_gw_text(usual, locale=locale),
            ),
            label=t("now.drivers.demand_label", locale=locale),
            provenance=_provenance_for_current_state(current_state, hide_replay=hide_replay_badge),
        )
    with columns[1]:
        driver_card(
            t("now.drivers.generation_icon", locale=locale),
            _balance_label(balance, locale=locale),
            generation_detail,
            label=t("now.drivers.generation_label", locale=locale),
            provenance="modelled" if balance is not None else "unavailable",
        )
    with columns[2]:
        weather_headline, weather_detail = _weather_summary_display(weather, locale=locale)
        weather_live_base_detail = weather_detail
        if weather_is_live:
            location = weather.get("location") or "Paris"
            source = weather.get("source") or "Open-Meteo"
            weather_detail = t(
                "now.weather.live_suffix",
                locale=locale,
                detail=weather_live_base_detail,
                location=location,
                source=source,
            )
        if weather_is_live:
            weather_provenance: str | None = "observed"
        elif snapshot.mode == "REPLAY":
            weather_provenance = None if hide_replay_badge else "replay"
        else:
            weather_provenance = "observed"
        driver_card(
            t("now.drivers.weather_icon", locale=locale),
            weather_headline,
            weather_detail,
            provenance=weather_provenance,
        )
    with columns[3]:
        driver_card(
            t("now.official.icon", locale=locale),
            _official_title(official, locale=locale),
            _official_detail(official, locale=locale),
            label=t("now.official.label", locale=locale),
            provenance="official" if official is not None and _official_available(official) else "unavailable",
        )


def generation_mix_figure(
    mix: CurrentGenerationMix | None,
    *,
    demand_mw: float | None,
    net_imports_mw: float | None,
    locale: str = "en",
) -> go.Figure:
    items = generation_mix_items(mix, locale=locale)
    values = [float(item["value_mw"]) for item in items]
    total = sum(values)
    max_x = max([total, float(demand_mw or 0), 1.0]) * 1.12
    stack_label = t("now.generation_mix.stack_label", locale=locale)
    fig = go.Figure()
    for item in items:
        fig.add_bar(
            x=[item["value_mw"]],
            y=[stack_label],
            orientation="h",
            name=item["label"],
            marker_color=item["color"],
            text=[item["direct_label"]],
            textposition="inside" if item["share"] >= 0.08 else "none",
            insidetextanchor="middle",
            customdata=[[item["value_label"]]],
            hovertemplate=f"{html.escape(item['label'])}: %{{customdata[0]}}<extra></extra>",
        )
    if demand_mw is not None:
        fig.add_vline(
            x=float(demand_mw),
            line_color="#f8fafc",
            line_width=2,
            annotation_text=t("now.generation_mix.demand_marker", locale=locale),
            annotation_position="top",
        )
    flow = flow_context_text(net_imports_mw, locale=locale)
    fig.add_annotation(
        x=max_x,
        y=stack_label,
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


def generation_mix_items(mix: CurrentGenerationMix | None, *, locale: str = "en") -> list[dict[str, Any]]:
    if mix is None:
        unavailable = _unavailable_text(locale)
        return [
            {
                "label": unavailable,
                "value_mw": 0.0,
                "value_label": _format_mw_text(0.0, locale=locale),
                "share": 1.0,
                "direct_label": unavailable,
                "color": "#64748b",
            }
        ]
    raw = [
        {
            "label": technology_label(item.technology, locale=locale),
            "value_mw": float(item.power.value or 0.0),
            "share": float(item.share.value or 0.0) / 100.0,
            "color": COLORS.get(technology_label(item.technology, locale="en"), "#94a3b8"),
        }
        for item in mix.technologies
        if item.power.value is not None and float(item.power.value) > 0
    ]
    if not raw:
        unavailable = _unavailable_text(locale)
        return [
            {
                "label": unavailable,
                "value_mw": 0.0,
                "value_label": _format_mw_text(0.0, locale=locale),
                "share": 1.0,
                "direct_label": unavailable,
                "color": "#64748b",
            }
        ]
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
                "label": t("now.technology.other", locale=locale),
                "value_mw": other_value,
                "share": other_value / max(total, 1.0),
                "color": "#94a3b8",
            }
        )
    for item in keep:
        item["value_label"] = _format_mw_text(item["value_mw"], locale=locale)
        item["direct_label"] = f"{item['label']} {_format_gw_text(item['value_mw'], locale=locale)}"
    return keep


def render_carbon_context(
    current_state: CurrentStateResponse | None,
    selected_snapshot: TwinSnapshot | None,
    *,
    hide_replay_badge: bool = False,
    locale: str = "en",
) -> None:
    carbon = None
    provenance: str | None = "unavailable"
    if selected_snapshot is not None and selected_snapshot.carbon_estimate is not None:
        carbon = selected_snapshot.carbon_estimate.intensity.value
        provenance = provenance_kind_for_source(selected_snapshot.carbon_estimate.source)
        if hide_replay_badge and provenance == "replay":
            provenance = None
    elif current_state is not None:
        carbon = current_state.national_context.carbon_estimate.estimate.value
        provenance = _provenance_for_current_state(current_state, hide_replay=hide_replay_badge)
    metric_card(
        t("now.carbon.label", locale=locale),
        _format_carbon_text(carbon, locale=locale),
        t("now.carbon.detail", locale=locale),
        icon="CO2",
        provenance=provenance,
    )


def demand_difference_text(value: float | None, *, already_percent: bool = True, locale: str = "en") -> str:
    if value is None or pd.isna(value):
        return t("now.demand_difference.unavailable", locale=locale)
    percent = float(value) if already_percent else float(value) * 100
    if abs(percent) < 7:
        return t("now.demand_difference.close_to_usual", locale=locale)
    percent_text = format_percentage(abs(percent), already_percent=True, locale=locale)
    if percent < 0:
        return t("now.demand_difference.below_usual", locale=locale, percent=percent_text)
    return t("now.demand_difference.above_usual", locale=locale, percent=percent_text)


def freshness_label_text(state: OperatingState | Freshness | None, age_seconds: float | None, locale: str = "en") -> str:
    if state is None:
        return t("now.freshness.unavailable", locale=locale)
    state_text = _freshness_state_text(state, locale=locale)
    age = _localize_unavailable(format_age_seconds(age_seconds, locale=locale), locale)
    return state_text if age == _unavailable_text(locale) else f"{state_text}, {age}"


def source_quality_label(value: str | None, *, locale: str = "en") -> str:
    if not value:
        return t("now.source_quality.unknown", locale=locale)
    key = _keyify(value)
    default = str(value).replace("_", " ").replace("-", " ").title()
    return t(f"now.source_quality.{key}", locale=locale, default=default)


def technology_label(value: str, *, locale: str = "en") -> str:
    key = str(value).strip().lower()
    default = key.replace("_", " ").title()
    return t(f"now.technology.{key}", locale=locale, default=default)


def flow_context_text(net_imports_mw: float | None, *, locale: str = "en") -> str:
    if net_imports_mw is None:
        return t("now.flow.net_exchange_unavailable", locale=locale)
    if abs(float(net_imports_mw)) < 1:
        return t("now.flow.net_exchange_near_zero", locale=locale)
    if net_imports_mw > 0:
        return t("now.flow.net_imports", locale=locale, value=_format_gw_text(net_imports_mw, signed=True, locale=locale))
    return t("now.flow.net_exports", locale=locale, value=_format_gw_text(abs(net_imports_mw), locale=locale))


def near_term_direction(
    twin: TwinResponse | None,
    current_demand_mw: float | None,
    *,
    forecast_points: list[ForecastPointView] | None = None,
    locale: str = "en",
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
        return t("now.near_term.unavailable", locale=locale)
    delta = values[-1] - float(current_demand_mw)
    threshold = max(abs(float(current_demand_mw)) * 0.01, 500.0)
    if delta > threshold:
        return t("now.near_term.rising", locale=locale, delta=_format_gw_text(delta, signed=True, locale=locale))
    if delta < -threshold:
        return t("now.near_term.falling", locale=locale, delta=_format_gw_text(delta, signed=True, locale=locale))
    return t("now.near_term.steady", locale=locale)


def main_driver_text(
    current_state: CurrentStateResponse | None,
    twin: TwinResponse | None,
    snapshot: GridSnapshotView,
    *,
    locale: str = "en",
) -> str:
    anomaly = _current_anomaly_pct(current_state, snapshot)
    if anomaly is not None and abs(float(anomaly)) >= 7:
        return t("now.main_driver.demand", locale=locale)
    current_snapshot = twin.snapshots[0] if twin is not None and twin.snapshots else None
    if current_snapshot is not None and current_snapshot.modelled_balance_contributions:
        contribution = max(current_snapshot.modelled_balance_contributions, key=lambda item: item.contribution)
        if contribution.component == "announced_unavailability_ratio" and contribution.contribution > 0.08:
            return t("now.main_driver.unavailability", locale=locale)
        if contribution.component == "residual_load_percentile" and contribution.contribution > 0.55:
            return t("now.main_driver.residual_load", locale=locale)
    headline = str(snapshot.weather.get("headline", ""))
    if headline and "unavailable" not in headline.lower():
        return t("now.main_driver.weather", locale=locale, headline=_weather_headline_display(headline, locale=locale).lower())
    return t("now.main_driver.no_single_driver", locale=locale)


def _snapshot_groups_by_local_day(
    snapshots: list[TwinSnapshot],
    timezone_name: str,
    *,
    locale: str = "en",
) -> list[tuple[str, list[TwinSnapshot]]]:
    groups: list[tuple[str, list[TwinSnapshot]]] = []
    for item in snapshots:
        local = pd.Timestamp(item.event_time).tz_convert(timezone_name)
        label = format_date(local, timezone_name=timezone_name, locale=locale, abbreviated=True)
        if groups and groups[-1][0] == label:
            groups[-1][1].append(item)
        else:
            groups.append((label, [item]))
    return groups


def _forecast_groups_by_local_day(
    points: list[ForecastPointView],
    timezone_name: str,
    *,
    locale: str = "en",
) -> list[tuple[str, list[ForecastPointView]]]:
    groups: list[tuple[str, list[ForecastPointView]]] = []
    for point in points:
        local = point.timestamp.tz_convert(timezone_name)
        label = format_date(local, timezone_name=timezone_name, locale=locale, abbreviated=True)
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


def _official_title(official: Any | None, *, locale: str = "en") -> str:
    if official is None:
        return t("now.official.title_unavailable", locale=locale)
    if hasattr(official, "available") and not bool(official.available):
        return t("now.official.title_unavailable", locale=locale)
    status = getattr(official, "status", None)
    if status is not None:
        return _status_display(status, locale=locale)
    label = getattr(official, "label", None)
    return _signal_label_text(str(label), locale=locale) if label else t("now.official.title_unavailable", locale=locale)


def _official_detail(official: Any | None, *, locale: str = "en") -> str:
    if official is None:
        return t("now.official.detail_unavailable", locale=locale)
    if hasattr(official, "available") and not bool(official.available):
        if locale == "en":
            return str(getattr(official, "reason", None) or t("now.official.detail_unavailable", locale=locale))
        return t("now.official.detail_unavailable", locale=locale)
    if locale == "en":
        return str(getattr(official, "detail", None) or getattr(official, "reason", None) or t("now.official.detail_available", locale=locale))
    return t("now.official.detail_available", locale=locale)


def _official_available(official: Any) -> bool:
    if hasattr(official, "available"):
        return bool(official.available)
    status = getattr(official, "status", Status.UNKNOWN)
    return status not in {Status.UNKNOWN, "unknown", None}


def _official_status(official: Any, *, locale: str = "en") -> str:
    if not _official_available(official):
        return _unavailable_text(locale)
    status = getattr(official, "status", None)
    if status is None:
        return _official_title(official, locale=locale)
    return _status_display(status, locale=locale)


def _balance_label(balance: Any | None, *, locale: str = "en") -> str:
    if balance is None:
        return _unavailable_text(locale)
    return _status_display(getattr(balance, "status", getattr(balance, "label", "Unknown")), locale=locale)


def _status_display(value: Any, *, locale: str = "en") -> str:
    if isinstance(value, Status):
        return t(f"now.status.{value.value}", locale=locale)
    text = str(value or "Unknown")
    lowered = text.lower()
    aliases = {
        "normal": "normal",
        "watch": "watch",
        "high": "high",
        "unknown": "unknown",
        "green": "normal",
        "orange": "watch",
        "red": "high",
        "no recommendation": "no_recommendation",
    }
    key = aliases.get(lowered)
    return t(f"now.status.{key}", locale=locale) if key else text[:1].upper() + text[1:]


def _generation_exchange_driver_detail(
    current_state: CurrentStateResponse | None,
    selected_snapshot: TwinSnapshot | None,
    *,
    locale: str = "en",
) -> str:
    balance = _modelled_balance(current_state, selected_snapshot)
    if balance is not None and hasattr(balance, "supply_margin"):
        margin = _quantified_value(balance.supply_margin)
        imports = _quantified_value(balance.net_imports)
        return t(
            "now.drivers.generation_detail.modelled",
            locale=locale,
            margin=_format_signed_mw_text(margin, locale=locale),
            flow=flow_context_text(imports, locale=locale),
        )
    if current_state is not None:
        imports = current_state.national_context.net_imports.value
        generation = current_state.national_context.generation_mix.total.value
        return t(
            "now.drivers.generation_detail.current_state",
            locale=locale,
            generation=_format_gw_text(generation, locale=locale),
            flow=flow_context_text(imports, locale=locale),
        )
    return t("now.drivers.generation_detail.unavailable", locale=locale)


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


def _format_gw_delta(value_gw: float | None, *, locale: str = "en") -> str:
    if value_gw is None:
        return _unavailable_text(locale)
    return _format_gw_text(value_gw * 1000.0, signed=True, locale=locale)


def _format_mw_text(value: float | int | None, *, locale: str = "en") -> str:
    return _localize_unavailable(format_mw(value, locale=locale), locale)


def _format_signed_mw_text(value: float | int | None, *, locale: str = "en") -> str:
    return _localize_unavailable(format_signed_mw(value, locale=locale), locale)


def _format_gw_text(value: float | int | None, *, signed: bool = False, locale: str = "en") -> str:
    return _localize_unavailable(format_gw(value, signed=signed, locale=locale), locale)


def _format_carbon_text(value: float | int | None, *, locale: str = "en") -> str:
    return _localize_unavailable(format_carbon(value, locale=locale), locale)


def _unavailable_text(locale: str) -> str:
    return t("shared.format.unavailable", locale=locale, default=UNAVAILABLE)


def _localize_unavailable(value: str, locale: str) -> str:
    return _unavailable_text(locale) if value == UNAVAILABLE else value


def _freshness_state_text(state: OperatingState | Freshness, *, locale: str = "en") -> str:
    key = _keyify(getattr(state, "value", state))
    default = str(getattr(state, "value", state)).replace("_", " ").title()
    return t(f"now.freshness.{key}", locale=locale, default=default)


def _signal_label_text(value: str, *, locale: str = "en") -> str:
    return _status_display(value, locale=locale)


def _route_label(value: str | None, *, locale: str = "en") -> str:
    if not value:
        return _unavailable_text(locale)
    key = _keyify(value)
    return t(f"now.routes.{key}", locale=locale, default=str(value))


def _weather_headline_for_values(temperature_c: float, wind_kmh: float, *, locale: str = "en") -> str:
    if temperature_c <= 5:
        return t("now.weather.cold_demand_lift", locale=locale)
    if temperature_c >= 27:
        return t("now.weather.heat_demand_lift", locale=locale)
    if wind_kmh >= 35:
        return t("now.weather.wind_output_context", locale=locale)
    return t("now.weather.mild_weather", locale=locale)


def _weather_detail_for_values(temperature_c: float, wind_kmh: float, cloud_pct: float, *, locale: str = "en") -> str:
    return t(
        "now.weather.detail",
        locale=locale,
        temperature=_localize_unavailable(format_temperature(temperature_c, locale=locale), locale),
        wind=format_number(wind_kmh, locale=locale),
        cloud=format_percentage(cloud_pct, already_percent=True, locale=locale),
    )


def _weather_summary_display(weather: dict[str, Any], *, locale: str = "en") -> tuple[str, str]:
    temp = weather.get("temperature_c")
    wind = weather.get("wind_kmh")
    if temp is not None or wind is not None:
        temp_value = 0.0 if temp is None else float(temp)
        wind_value = 0.0 if wind is None else float(wind)
        cloud_value = float(weather.get("cloud_pct") or 0.0)
        return (
            _weather_headline_for_values(temp_value, wind_value, locale=locale),
            _weather_detail_for_values(temp_value, wind_value, cloud_value, locale=locale),
        )
    return (
        _weather_headline_display(str(weather.get("headline", "")), locale=locale),
        _weather_detail_display(str(weather.get("detail", "")), locale=locale),
    )


def _weather_headline_display(value: str, *, locale: str = "en") -> str:
    key = _keyify(value)
    if key in {"cold_demand_lift", "heat_demand_lift", "wind_output_context", "mild_weather"}:
        return t(f"now.weather.{key}", locale=locale)
    if not value or "unavailable" in value.lower():
        return t("now.weather.unavailable_headline", locale=locale)
    return value


def _weather_detail_display(value: str, *, locale: str = "en") -> str:
    if not value or "unavailable" in value.lower() or "omit weather" in value.lower():
        return t("now.weather.unavailable_detail", locale=locale)
    return value


def _translate_unavailable_reason(value: str | None, *, locale: str = "en") -> str:
    if not value:
        return t("now.states.regional_value_unavailable", locale=locale)
    text = str(value)
    lowered = text.lower()
    if "no regional demand record" in lowered:
        return t("now.unavailable_reasons.no_regional_demand_record", locale=locale)
    if lowered.startswith("no regional record is available for "):
        region = text.rstrip(".").split(" for ", maxsplit=1)[-1]
        return t("now.unavailable_reasons.no_regional_record", locale=locale, region=region)
    if "usual-demand baseline is unavailable" in lowered:
        return t("now.unavailable_reasons.usual_demand_geography", locale=locale)
    if "demand anomaly is unavailable" in lowered:
        return t("now.unavailable_reasons.demand_anomaly_missing", locale=locale)
    if "difference versus usual is unavailable" in lowered:
        return t("now.unavailable_reasons.difference_missing", locale=locale)
    if "map demand anomaly unavailable" in lowered:
        return t("now.unavailable_reasons.map_demand_anomaly", locale=locale)
    if lowered == "value unavailable.":
        return t("now.unavailable_reasons.value_unavailable", locale=locale)
    return text if locale == "en" else t("now.states.regional_value_unavailable", locale=locale)


def _keyify(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _metric_value(metric: NullableMetric | QuantifiedValue | None) -> float | None:
    return None if metric is None else metric.value


def _quantified_value(value: QuantifiedValue | NullableMetric | None) -> float | None:
    return None if value is None else value.value


def _metric_reason(*metrics: NullableMetric, locale: str = "en") -> str:
    for metric in metrics:
        if metric.reason:
            return _translate_unavailable_reason(metric.reason, locale=locale)
    return t("now.states.regional_value_unavailable", locale=locale)


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


def _provenance_for_current_state(
    current_state: CurrentStateResponse | None,
    *,
    hide_replay: bool = False,
) -> str | None:
    if current_state is None:
        return "unavailable"
    if current_state.operating_state == OperatingState.HISTORICAL_REPLAY:
        # When the caller has already conveyed replay context elsewhere
        # (e.g. the "Demo context" mode pill), omit the per-card badge to
        # avoid repeating ourselves on every metric/driver tile.
        return None if hide_replay else "replay"
    if current_state.operating_state in {
        OperatingState.SOURCE_UNAVAILABLE,
    }:
        return "unavailable"
    if current_state.operating_state == OperatingState.LAST_KNOWN_GOOD_FALLBACK:
        return "fallback"
    return "observed"
