from __future__ import annotations

import html
from collections import Counter

import pandas as pd
import streamlit as st

from app.api_client import load_typed_public_context
from app.components.cards import explanation_card, metric_card, section_header, viz_note
from app.components.foundation import (
    error_state,
    notices_from_contracts,
    render_context_bar,
    render_trust_notices,
    responsive_columns,
)
from app.components.now_dashboard import current_weather_summary
from app.components.public import forecast_chart, render_public_header, selected_location
from app.components.regional_map import has_meaningful_anomaly_signal, regional_anomaly_choropleth
from app.data_loader import load_live_current_weather, load_live_weather_forecast
from app.formatting import (
    format_carbon,
    format_gw,
    format_mw,
    format_number,
    format_percentage,
    format_signed_mw,
    format_temperature,
    format_timestamp,
    format_uncertainty_range,
)
from app.i18n import nav_label, t
from app.next48h_view import (
    BEST_WINDOW_OBJECTIVES,
    best_window_objective_label,
    choose_best_window,
    confidence_summary,
    enrich_forecast_frame_with_twin,
    forecast_display_table,
    forecast_points_from_twin,
    future_regional_map_frame,
    generation_mix_rows,
    local_hour_label,
    peak_forecast_row,
    projected_future_regional_map_frame,
    selected_forecast_point,
    selected_hour_explanation,
    selected_timestamp_from_chart_event,
    selected_twin_snapshot,
    time_delta_text,
    twin_aligned_to_reference,
    weather_for_hour,
    window_label,
)
from app.state import (
    mode_for_page,
    persist_app_state,
    read_app_state,
    select_timestamp_from_options,
    with_updates,
)
from app.view_models import (
    add_regional_anomalies,
    build_forecast_points,
    build_grid_snapshot,
    forecast_points_frame,
    ui_mode,
)
from src.contracts.energy_twin import DomainMode, TwinSnapshot
from src.config import settings
from src.data_sources.rte_eco2mix_regional import REGION_NAMES


def _select_hour(index: int, points: list) -> None:
    point = points[max(0, min(index, len(points) - 1))]
    persist_app_state(
        with_updates(
            read_app_state(default_mode=DomainMode.REPLAY),
            selected_timestamp=point.timestamp.to_pydatetime(),
        )
    )
    st.rerun()


def _balance_status(snapshot: TwinSnapshot | None) -> str:
    if snapshot is None:
        return "unknown"
    balance = snapshot.modelled_national_balance_context or snapshot.national.balance_context
    return balance.status.value if balance else "unknown"


def _status_text(value: str | None, *, locale: str) -> str:
    key = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    return t(f"next48h.status.{key}", locale=locale, default=str(value or "Unknown"))


def _card_status(value: str | None, *, locale: str) -> str | None:
    return value if locale == "en" else None


def _card_provenance(value: str | None, *, locale: str) -> str | None:
    return value if locale == "en" else None


def _localized_term_tooltip_html(term_key: str, *, locale: str) -> str:
    term = t(f"next48h.terms.{term_key}.term", locale=locale)
    definition = t(f"next48h.terms.{term_key}.definition", locale=locale)
    return (
        f'<span class="ep-term" tabindex="0" role="note" '
        f'aria-label="{html.escape(term)}: {html.escape(definition)}">'
        f'<abbr title="{html.escape(definition)}">{html.escape(term)}</abbr>'
        f'<span class="ep-term-popover" aria-hidden="true">{html.escape(definition)}</span></span>'
    )


def _weather_copy(weather: dict, *, locale: str) -> tuple[str, str]:
    temp = weather.get("temperature_c")
    wind = weather.get("wind_kmh")
    if temp is None and wind is None:
        return (
            t("next48h.weather.unavailable_headline", locale=locale),
            t("next48h.weather.unavailable_detail", locale=locale),
        )
    temp_value = None if temp is None else float(temp)
    wind_value = 0.0 if wind is None else float(wind)
    cloud_value = float(weather.get("cloud_pct") or 0.0)
    if temp_value is not None and temp_value <= 5:
        headline_key = "cold"
    elif temp_value is not None and temp_value >= 27:
        headline_key = "heat"
    elif wind_value >= 35:
        headline_key = "wind"
    else:
        headline_key = "mild"
    detail = t(
        "next48h.weather.detail",
        locale=locale,
        temperature=(
            format_temperature(temp_value, locale=locale)
            if temp_value is not None
            else t("next48h.tables.forecast.unavailable", locale=locale)
        ),
        wind=f"{format_number(wind_value, locale=locale)} km/h",
        cloud=format_percentage(cloud_value, already_percent=True, locale=locale),
    )
    return t(f"next48h.weather.headline.{headline_key}", locale=locale), detail


def _source_text(value: str, *, locale: str) -> str:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return t("next48h.tables.forecast.unavailable", locale=locale)
    if "usual-demand baseline fallback" in lowered or "usual demand baseline fallback" in lowered:
        return t("next48h.sources.usual_fallback", locale=locale)
    if "unsupported" in lowered:
        return t("next48h.sources.unsupported_route", locale=locale)
    if "fallback" in lowered:
        return t("next48h.sources.fallback_route", locale=locale)
    if lowered == "demand forecast":
        return t("next48h.sources.demand_forecast", locale=locale)
    return str(value)


state = read_app_state(default_mode=DomainMode.REPLAY)
app_context = load_typed_public_context(state)
context = app_context.legacy
energy: pd.DataFrame = context["energy"]
if energy.empty:
    error_state(
        t("next48h.errors.context_title", locale=state.locale),
        t("next48h.errors.context_body", locale=state.locale),
    )
    st.stop()

regional = add_regional_anomalies(context["regional"], context["regional_history"], timezone=settings.timezone)
snapshot = build_grid_snapshot(
    energy,
    mode=ui_mode(context["mode"]),
    regional_state=regional,
    weather=context["weather"],
    ecowatt=context["ecowatt"],
    source_label=context["national_source"],
    timezone=settings.timezone,
)
live_weather = current_weather_summary(load_live_current_weather(), snapshot.weather)
legacy_points = build_forecast_points(
    energy,
    model_payload=context["model_payload"],
    horizon_hours=48,
    timezone=settings.timezone,
)
aligned_twin = app_context.twin if twin_aligned_to_reference(app_context.twin, snapshot.as_of) else None
typed_points = forecast_points_from_twin(aligned_twin, legacy_points=legacy_points)
points = typed_points or legacy_points
if not points:
    error_state(
        t("next48h.errors.forecast_title", locale=state.locale),
        t("next48h.errors.forecast_body", locale=state.locale),
    )
    st.stop()

timestamps = [point.timestamp for point in points]
default_timestamp = timestamps[0].to_pydatetime()
state = with_updates(
    state,
    mode=mode_for_page(replay=app_context.is_replay, page="next_48h"),
    selected_timestamp=state.selected_timestamp or default_timestamp,
    selected_forecast_run=app_context.forecast_run_id if aligned_twin else None,
)
selected_index = select_timestamp_from_options(timestamps, state.selected_timestamp)
selected_point = points[selected_index]
state = with_updates(state, selected_timestamp=selected_point.timestamp.to_pydatetime())
persist_app_state(state)
selected_snapshot = selected_twin_snapshot(aligned_twin, selected_point.timestamp)

forecast = enrich_forecast_frame_with_twin(forecast_points_frame(points), aligned_twin)
peak = peak_forecast_row(forecast)
selected_confidence = confidence_summary(selected_point, selected_snapshot, locale=state.locale)

render_public_header(
    nav_label("next_48h", locale=state.locale),
    t("next48h.header.subtitle", locale=state.locale),
    state.mode.value.upper(),
)
render_context_bar(
    state,
    twin=aligned_twin,
    current_state=app_context.current_state,
    timezone_name=settings.timezone,
    weather=live_weather,
    hide_replay_badge=True,
    hide_fallback_badge=True,
)
render_trust_notices(
    notice
    for notice in notices_from_contracts(aligned_twin, app_context.current_state)
    if notice.title not in {"Demo fixture mode", "Optional source unavailable"}
)

section_header(
    t("next48h.sections.forecast.kicker", locale=state.locale),
    t("next48h.sections.forecast.title", locale=state.locale),
    t("next48h.sections.forecast.copy", locale=state.locale),
)
viz_note(
    t("next48h.viz.title", locale=state.locale),
    t("next48h.viz.detail", locale=state.locale),
    source=t("next48h.viz.source", locale=state.locale),
)
chart_event = st.plotly_chart(
    forecast_chart(
        energy,
        forecast,
        selected_timestamp=selected_point.timestamp,
        now_timestamp=snapshot.as_of,
        timezone_name=settings.timezone,
        locale=state.locale,
    ),
    key="next48_forecast_chart",
    width="stretch",
    on_select="rerun",
    selection_mode="points",
)
chart_timestamp = selected_timestamp_from_chart_event(chart_event)
if chart_timestamp is not None:
    chart_point = selected_forecast_point(points, chart_timestamp)
    if chart_point is not None and chart_point.timestamp != selected_point.timestamp:
        persist_app_state(with_updates(state, selected_timestamp=chart_point.timestamp.to_pydatetime()))
        st.rerun()

hour_labels = [
    f"{local_hour_label(point.timestamp, timezone_name=settings.timezone, locale=state.locale)} - "
    f"{_status_text(point.pressure_label, locale=state.locale)}"
    for point in points
]
selected_label = st.selectbox(t("next48h.controls.forecast_hour", locale=state.locale), hour_labels, index=selected_index)
selected_label_index = hour_labels.index(selected_label)
if selected_label_index != selected_index:
    persist_app_state(with_updates(state, selected_timestamp=points[selected_label_index].timestamp.to_pydatetime()))
    st.rerun()

nav_cols = st.columns([1, 1, 3], gap="small")
with nav_cols[0]:
    if st.button(t("next48h.controls.previous_hour", locale=state.locale), disabled=selected_index <= 0, width="stretch"):
        _select_hour(selected_index - 1, points)
with nav_cols[1]:
    if st.button(t("next48h.controls.next_hour", locale=state.locale), disabled=selected_index >= len(points) - 1, width="stretch"):
        _select_hour(selected_index + 1, points)
with nav_cols[2]:
    st.caption(
        t(
            "next48h.controls.selected_forecast_timestamp",
            locale=state.locale,
            timestamp=format_timestamp(selected_point.timestamp, timezone_name=settings.timezone, locale=state.locale),
        )
    )

with st.expander(t("next48h.controls.forecast_data", locale=state.locale), expanded=False):
    st.dataframe(
        forecast_display_table(forecast, timezone_name=settings.timezone, locale=state.locale),
        width="stretch",
        hide_index=True,
    )

objective_label = st.radio(
    t("next48h.controls.best_window_objective", locale=state.locale),
    list(BEST_WINDOW_OBJECTIVES),
    index=2,
    horizontal=True,
    format_func=lambda key: best_window_objective_label(key, locale=state.locale),
)
objective_key = str(objective_label)
best = choose_best_window(forecast, objective_key, locale=state.locale)
st.caption(t("next48h.controls.objective_caption", locale=state.locale))

summary = st.columns(3)
with summary[0]:
    if peak is not None:
        peak_time = pd.Timestamp(peak["timestamp"])
        metric_card(
            t("next48h.summary.peak.label", locale=state.locale),
            local_hour_label(peak_time, timezone_name=settings.timezone, locale=state.locale),
            t(
                "next48h.summary.peak.detail",
                locale=state.locale,
                range=format_uncertainty_range(peak["p10"], peak["p90"], locale=state.locale),
                delta=time_delta_text(selected_point.timestamp, peak_time, locale=state.locale),
            ),
            icon="P",
            status=_card_status(str(peak.get("pressure_label", "unknown")), locale=state.locale),
            provenance=_card_provenance("modelled", locale=state.locale),
        )
    else:
        metric_card(
            t("next48h.summary.peak.label", locale=state.locale),
            t("next48h.summary.peak.unavailable_value", locale=state.locale),
            t("next48h.summary.peak.unavailable_detail", locale=state.locale),
            icon="P",
            status=_card_status("unknown", locale=state.locale),
            provenance=_card_provenance("unavailable", locale=state.locale),
        )
with summary[1]:
    if best is not None:
        metric_card(
            t("next48h.summary.best_window.label", locale=state.locale),
            window_label(best.start, best.end, timezone_name=settings.timezone, locale=state.locale),
            t(
                "next48h.summary.best_window.detail",
                locale=state.locale,
                explanation=best.explanation,
                demand=format_gw(best.mean_demand_mw, locale=state.locale),
            ),
            icon="W",
            status=_card_status("green", locale=state.locale),
            provenance=_card_provenance("modelled", locale=state.locale),
        )
    else:
        metric_card(
            t("next48h.summary.best_window.label", locale=state.locale),
            t("next48h.summary.best_window.unavailable_value", locale=state.locale),
            t("next48h.summary.best_window.unavailable_detail", locale=state.locale),
            icon="W",
            status=_card_status("unknown", locale=state.locale),
        )
with summary[2]:
    metric_card(
        t("next48h.summary.confidence.label", locale=state.locale),
        selected_confidence.level,
        t(
            "next48h.summary.confidence.detail",
            locale=state.locale,
            hour=local_hour_label(selected_point.timestamp, timezone_name=settings.timezone, locale=state.locale),
            detail=selected_confidence.detail,
        ),
        icon="C",
        status=_card_status(selected_confidence.level, locale=state.locale),
        provenance=_card_provenance("modelled", locale=state.locale),
    )

section_header(
    t("next48h.sections.selected_hour.kicker", locale=state.locale),
    t(
        "next48h.sections.selected_hour.title",
        locale=state.locale,
        hour=local_hour_label(selected_point.timestamp, timezone_name=settings.timezone, locale=state.locale),
    ),
)
st.markdown(
    '<div class="ep-section-copy">'
    + t(
        "next48h.terms.intro",
        locale=state.locale,
        usual=_localized_term_tooltip_html("usual_demand", locale=state.locale),
        range=_localized_term_tooltip_html("likely_range", locale=state.locale),
    )
    + "</div>",
    unsafe_allow_html=True,
)
selected_explanation = selected_hour_explanation(
    selected_point,
    selected_snapshot,
    timezone_name=settings.timezone,
    locale=state.locale,
)
positive_total = sum(driver.value_mw for driver in selected_explanation.positive_drivers)
negative_total = sum(driver.value_mw for driver in selected_explanation.negative_drivers)
selected_metrics = st.columns(4)
with selected_metrics[0]:
    metric_card(
        t("next48h.selected_metrics.usual_demand.label", locale=state.locale),
        format_gw(selected_explanation.usual_demand_mw, locale=state.locale),
        t("next48h.selected_metrics.usual_demand.detail", locale=state.locale),
        icon="U",
        provenance=_card_provenance("fallback", locale=state.locale),
    )
with selected_metrics[1]:
    metric_card(
        t("next48h.selected_metrics.positive_drivers.label", locale=state.locale),
        format_gw(positive_total, signed=True, locale=state.locale),
        t("next48h.selected_metrics.positive_drivers.detail", locale=state.locale),
        icon="+",
        status=_card_status("watch" if positive_total else "green", locale=state.locale),
        provenance=_card_provenance("modelled", locale=state.locale),
    )
with selected_metrics[2]:
    metric_card(
        t("next48h.selected_metrics.negative_drivers.label", locale=state.locale),
        format_gw(-negative_total, signed=True, locale=state.locale),
        t("next48h.selected_metrics.negative_drivers.detail", locale=state.locale),
        icon="-",
        status=_card_status("green" if negative_total else "grey", locale=state.locale),
        provenance=_card_provenance("modelled", locale=state.locale),
    )
with selected_metrics[3]:
    metric_card(
        t("next48h.selected_metrics.final_expected_demand.label", locale=state.locale),
        format_gw(selected_explanation.expected_demand_mw, locale=state.locale),
        t(
            "next48h.selected_metrics.final_expected_demand.detail",
            locale=state.locale,
            range=format_uncertainty_range(
                selected_explanation.p10_mw,
                selected_explanation.p90_mw,
                unit="GW",
                locale=state.locale,
            ),
            error=format_signed_mw(selected_explanation.reconciliation_error_mw, locale=state.locale),
        ),
        icon="=",
        status=_card_status(selected_point.pressure_label, locale=state.locale),
        provenance=_card_provenance("modelled", locale=state.locale),
    )

explanation_card(
    t("next48h.selected_explanation.title", locale=state.locale),
    selected_explanation.text,
    label=t("next48h.selected_explanation.label", locale=state.locale),
    status=_card_status("info", locale=state.locale),
    provenance=_card_provenance("modelled", locale=state.locale),
)

# Build a single weather card that replaces the placeholder "Weather
# disagreement" factor: hourly forecast when available, otherwise live
# current weather, otherwise the bundled snapshot. Loaded once so the
# fallback chain is evaluated before the render loop reaches the slot.
weather_forecast_frame = load_live_weather_forecast()
forecast_weather_payload = weather_for_hour(weather_forecast_frame, selected_point.timestamp)
live_weather_payload = load_live_current_weather() if forecast_weather_payload is None else None
weather_summary = current_weather_summary(
    forecast_weather_payload or live_weather_payload,
    snapshot.weather,
)
weather_headline, weather_detail = _weather_copy(weather_summary, locale=state.locale)
weather_location = weather_summary.get("location") or "Paris"
weather_source = weather_summary.get("source") or "Open-Meteo"
weather_base_detail = weather_detail
if forecast_weather_payload is not None:
    weather_label = t(
        "next48h.weather.forecast_label",
        locale=state.locale,
        hour=local_hour_label(selected_point.timestamp, timezone_name=settings.timezone, locale=state.locale),
    )
    weather_detail = t(
        "next48h.weather.forecast_detail",
        locale=state.locale,
        detail=weather_base_detail,
        location=weather_location,
        source=weather_source,
    )
    weather_provenance = "modelled"
elif live_weather_payload is not None:
    weather_label = t("next48h.weather.live_label", locale=state.locale)
    weather_detail = t(
        "next48h.weather.live_detail",
        locale=state.locale,
        detail=weather_base_detail,
        location=weather_location,
        source=weather_source,
    )
    weather_provenance = "observed"
else:
    weather_label = t("next48h.weather.bundled_label", locale=state.locale)
    weather_provenance = "fallback"

section_header(
    t("next48h.sections.confidence.kicker", locale=state.locale),
    t("next48h.sections.confidence.title", locale=state.locale),
)
factor_cols = st.columns(3)
weather_factor_name = t("next48h.confidence.factors.weather_disagreement.name", locale=state.locale)
for index, factor in enumerate(selected_confidence.factors):
    with factor_cols[index % len(factor_cols)]:
        if factor.name == weather_factor_name:
            # Reuse the placeholder slot for the real weather context so the
            # section shows a single, informative weather card.
            explanation_card(
                weather_headline,
                weather_detail,
                label=weather_label,
                status=_card_status("info", locale=state.locale),
                provenance=_card_provenance(weather_provenance, locale=state.locale),
            )
        else:
            explanation_card(
                factor.value,
                factor.detail,
                label=factor.name,
                status=_card_status(factor.status, locale=state.locale),
                provenance=_card_provenance("modelled" if factor.status != "unknown" else "unavailable", locale=state.locale),
            )

section_header(
    t("next48h.sections.regional.kicker", locale=state.locale),
    t(
        "next48h.sections.regional.title",
        locale=state.locale,
        hour=local_hour_label(selected_point.timestamp, timezone_name=settings.timezone, locale=state.locale),
    ),
    t("next48h.sections.regional.copy", locale=state.locale),
)
future_regional = future_regional_map_frame(selected_snapshot, timezone_name=settings.timezone, locale=state.locale)
if not has_meaningful_anomaly_signal(future_regional):
    future_regional = projected_future_regional_map_frame(
        regional,
        selected_point,
        timezone_name=settings.timezone,
        locale=state.locale,
    )
left, right = responsive_columns("map-detail")
with left:
    if future_regional.empty:
        error_state(
            t("next48h.errors.regional_title", locale=state.locale),
            t("next48h.errors.regional_body", locale=state.locale),
        )
    else:
        regional_event = st.plotly_chart(
            regional_anomaly_choropleth(
                future_regional,
                context["regions_geojson"],
                demand_label_title=t("next48h.chart.forecast_demand", locale=state.locale),
                source_label_title=t("next48h.tables.forecast.route", locale=state.locale),
            ),
            key="next48_future_regional_map",
            width="stretch",
            on_select="rerun",
            selection_mode="points",
        )
        selected_from_map = selected_location(regional_event)
        if selected_from_map and selected_from_map != state.selected_region:
            state = with_updates(state, selected_region=selected_from_map)
            persist_app_state(state)
            st.rerun()
with right:
    if future_regional.empty:
        explanation_card(
            t("next48h.regional.unavailable_title", locale=state.locale),
            t("next48h.regional.unavailable_detail", locale=state.locale),
            label=t("next48h.regional.card_label", locale=state.locale),
            status=_card_status("unknown", locale=state.locale),
            provenance=_card_provenance("unavailable", locale=state.locale),
        )
    else:
        region_codes = future_regional["region_code"].astype(str).tolist()
        selected_region_code = state.selected_region if state.selected_region in region_codes else region_codes[0]
        region_label = st.selectbox(
            t("next48h.controls.region_detail", locale=state.locale),
            region_codes,
            index=region_codes.index(selected_region_code),
            format_func=lambda code: REGION_NAMES.get(code, code),
        )
        if region_label != state.selected_region:
            state = with_updates(state, selected_region=region_label)
            persist_app_state(state)
        selected_region = future_regional.loc[future_regional["region_code"].astype(str).eq(region_label)].iloc[0]
        explanation_card(
            str(selected_region["region_display"]),
            t(
                "next48h.regional.detail",
                locale=state.locale,
                forecast=selected_region["demand_label"],
                usual=selected_region["usual_label"],
                difference=selected_region["difference_label"],
            ),
            label=str(selected_region["freshness_label"]),
            status=_card_status("info", locale=state.locale),
            provenance=_card_provenance("modelled", locale=state.locale),
        )
        explanation_card(
            t("next48h.regional.interpretation_title", locale=state.locale),
            str(selected_region["note"]),
            label=t("next48h.regional.interpretation_label", locale=state.locale),
            status=_card_status("grey", locale=state.locale),
            provenance=_card_provenance("modelled", locale=state.locale),
        )

section_header(
    t("next48h.sections.generation_balance.kicker", locale=state.locale),
    t(
        "next48h.sections.generation_balance.title",
        locale=state.locale,
        hour=local_hour_label(selected_point.timestamp, timezone_name=settings.timezone, locale=state.locale),
    ),
)
balance = selected_snapshot.modelled_national_balance_context if selected_snapshot else None
availability = selected_snapshot.generation_availability_context if selected_snapshot else None
mix = selected_snapshot.generation_mix_estimate if selected_snapshot else None
official = selected_snapshot.official_signal_context if selected_snapshot else None
carbon = selected_snapshot.carbon_estimate if selected_snapshot else None
exchange = selected_snapshot.exchange_estimate if selected_snapshot else None
fallback_generation_total = sum(float(value or 0.0) for value in snapshot.generation_by_source.values())
fallback_available = float(snapshot.availability.get("available_mw", 0.0) or 0.0)
fallback_margin = fallback_available - float(selected_point.p50)
latest_energy = energy.sort_values("timestamp").iloc[-1]
fallback_carbon = pd.to_numeric(latest_energy.get("co2_intensity_g_per_kwh"), errors="coerce")
fallback_exchange = float(snapshot.availability.get("imports_mw", 0.0) or 0.0)

context_cols = st.columns(3)
with context_cols[0]:
    metric_card(
        t("next48h.generation.mix.label", locale=state.locale),
        format_gw(mix.total.value if mix else fallback_generation_total, locale=state.locale),
        t(
            "next48h.generation.mix.model_detail" if mix else "next48h.generation.mix.fallback_detail",
            locale=state.locale,
        ),
        icon="G",
        status=_card_status("info", locale=state.locale),
        provenance=_card_provenance("modelled" if mix else "fallback", locale=state.locale),
    )
with context_cols[1]:
    metric_card(
        t("next48h.generation.availability.label", locale=state.locale),
        format_mw(availability.announced_unavailable.value if availability else 0.0, locale=state.locale),
        (
            t(
                "next48h.generation.availability.model_detail",
                locale=state.locale,
                output=format_mw(availability.nuclear.value.value, locale=state.locale),
            )
            if availability
            else t("next48h.generation.availability.fallback_detail", locale=state.locale)
        ),
        icon="A",
        status=_card_status("watch" if availability and (availability.announced_unavailable.value or 0) > 0 else "green", locale=state.locale),
        provenance=_card_provenance("modelled" if availability else "fallback", locale=state.locale),
    )
with context_cols[2]:
    balance_status = _balance_status(selected_snapshot) if selected_snapshot else selected_point.pressure_label
    metric_card(
        t("next48h.generation.balance.label", locale=state.locale),
        _status_text(balance_status, locale=state.locale),
        (
            t(
                "next48h.generation.balance.model_detail",
                locale=state.locale,
                score=format_percentage(balance.pressure_ratio.value, locale=state.locale),
                margin=format_signed_mw(balance.supply_margin.value, locale=state.locale),
            )
            if balance
            else t(
                "next48h.generation.balance.fallback_detail",
                locale=state.locale,
                margin=format_signed_mw(fallback_margin, locale=state.locale),
            )
        ),
        icon="B",
        status=_card_status(balance_status, locale=state.locale),
        provenance=_card_provenance("modelled" if balance else "fallback", locale=state.locale),
    )
context_cols_2 = st.columns(3)
with context_cols_2[0]:
    official_status = official.status.value if official else "unknown"
    metric_card(
        t("next48h.generation.official.label", locale=state.locale),
        (official.label if official and state.locale == "en" else _status_text(official_status, locale=state.locale))
        if official
        else t("next48h.generation.official.unavailable_value", locale=state.locale),
        official.detail
        if official and official.detail and state.locale == "en"
        else t("next48h.generation.official.unavailable_detail", locale=state.locale),
        icon="O",
        status=_card_status(official_status, locale=state.locale),
        provenance=_card_provenance("official" if official else "unavailable", locale=state.locale),
    )
with context_cols_2[1]:
    metric_card(
        t("next48h.generation.carbon.label", locale=state.locale),
        format_carbon(
            carbon.intensity.value if carbon else (None if pd.isna(fallback_carbon) else float(fallback_carbon)),
            locale=state.locale,
        ),
        t("next48h.generation.carbon.model_detail" if carbon else "next48h.generation.carbon.fallback_detail", locale=state.locale),
        icon="CO2",
        status=_card_status("info", locale=state.locale),
        provenance=_card_provenance("fallback" if carbon is None or carbon.source.is_fallback else "modelled", locale=state.locale),
    )
with context_cols_2[2]:
    metric_card(
        t("next48h.generation.exchange.label", locale=state.locale),
        format_signed_mw(exchange.net_imports.value if exchange else fallback_exchange, locale=state.locale),
        t(
            "next48h.generation.exchange.model_detail" if exchange else "next48h.generation.exchange.fallback_detail",
            locale=state.locale,
        ),
        icon="X",
        status=_card_status("info", locale=state.locale),
        provenance=_card_provenance("modelled" if exchange else "fallback", locale=state.locale),
    )
with st.expander(t("next48h.expanders.estimated_generation_mix", locale=state.locale), expanded=False):
    mix_rows = generation_mix_rows(selected_snapshot, locale=state.locale)
    if mix_rows.empty:
        component_column = t("next48h.tables.generation.component", locale=state.locale)
        estimate_column = t("next48h.tables.generation.estimate", locale=state.locale)
        provenance_column = t("next48h.tables.generation.provenance", locale=state.locale)
        formula_column = t("next48h.tables.generation.formula", locale=state.locale)
        mix_rows = pd.DataFrame(
            [
                {
                    component_column: label,
                    estimate_column: format_mw(value, locale=state.locale),
                    provenance_column: t("next48h.tables.generation.fallback_provenance", locale=state.locale),
                    formula_column: t("next48h.tables.generation.fallback_formula", locale=state.locale),
                }
                for label, value in snapshot.generation_by_source.items()
            ]
        )
    st.dataframe(mix_rows, width="stretch", hide_index=True)

with st.expander(t("next48h.expanders.how_calculated.label", locale=state.locale), expanded=False):
    st.write(t("next48h.expanders.how_calculated.paragraph_1", locale=state.locale))
    st.write(t("next48h.expanders.how_calculated.paragraph_2", locale=state.locale))
    st.write(t("next48h.expanders.how_calculated.paragraph_3", locale=state.locale))

with st.expander(t("next48h.expanders.reliability_model_card.label", locale=state.locale), expanded=False):
    source_counts = Counter(point.source for point in points)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    t("next48h.tables.reliability.forecast_route", locale=state.locale): _source_text(source, locale=state.locale),
                    t("next48h.tables.reliability.hours", locale=state.locale): count,
                }
                for source, count in source_counts.items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.write(selected_confidence.detail)
    if selected_snapshot and selected_snapshot.unsupported_physical_behaviours:
        st.write(t("next48h.expanders.reliability_model_card.limitations_label", locale=state.locale))
        for limitation in selected_snapshot.unsupported_physical_behaviours:
            st.write(f"- {limitation}")

with st.expander(t("next48h.expanders.data_sources.label", locale=state.locale), expanded=False):
    st.write(t("next48h.expanders.data_sources.national", locale=state.locale, source=context["national_source"]))
    st.write(t("next48h.expanders.data_sources.regional", locale=state.locale, source=context["regional_source"]))
    st.write(
        t(
            "next48h.expanders.data_sources.map",
            locale=state.locale,
            source=context.get("geo_source", t("next48h.provenance_items.unavailable", locale=state.locale)),
        )
    )
    st.write(
        t(
            "next48h.expanders.data_sources.selected_hour",
            locale=state.locale,
            timestamp=format_timestamp(selected_point.timestamp, timezone_name=settings.timezone, locale=state.locale),
        )
    )
    if app_context.twin is not None and aligned_twin is None:
        st.write(t("next48h.expanders.data_sources.unaligned_twin", locale=state.locale))
    if selected_snapshot is not None:
        st.write(
            t(
                "next48h.expanders.data_sources.snapshot_updated",
                locale=state.locale,
                timestamp=format_timestamp(selected_snapshot.update_time, timezone_name=settings.timezone, locale=state.locale),
            )
        )
    unavailable = aligned_twin.unavailable_fields if aligned_twin else []
    if unavailable:
        st.write(t("next48h.expanders.data_sources.optional_sources", locale=state.locale))
        for item in unavailable:
            st.write(f"- {item.field}: {item.reason}")

with st.expander(t("next48h.expanders.about.label", locale=state.locale), expanded=False):
    st.write(t("next48h.expanders.about.paragraph_1", locale=state.locale))
    st.write(t("next48h.expanders.about.paragraph_2", locale=state.locale))

with st.expander(t("next48h.expanders.provenance.label", locale=state.locale), expanded=False):
    provenance_items = (
        (t("next48h.provenance_items.national", locale=state.locale), context["national_source"]),
        (t("next48h.provenance_items.regional", locale=state.locale), context["regional_source"]),
        (
            t("next48h.provenance_items.forecast_run", locale=state.locale),
            app_context.forecast_run_id
            if aligned_twin and app_context.forecast_run_id
            else t("next48h.provenance_items.aligned_legacy_forecast", locale=state.locale),
        ),
        (
            t("next48h.provenance_items.selected_timestamp", locale=state.locale),
            format_timestamp(selected_point.timestamp, timezone_name=settings.timezone, locale=state.locale),
        ),
    )
    for label, detail in provenance_items:
        st.markdown(f"**{html.escape(label)}**: {html.escape(str(detail))}")
