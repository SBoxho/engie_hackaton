from __future__ import annotations

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
    term_tooltip_html,
)
from app.components.public import (
    about_project_drawer,
    forecast_chart,
    provenance_drawer,
    render_public_header,
    selected_location,
)
from app.components.regional_map import has_meaningful_anomaly_signal, regional_anomaly_choropleth
from app.formatting import (
    format_carbon,
    format_gw,
    format_mw,
    format_percentage,
    format_signed_mw,
    format_timestamp,
    format_uncertainty_range,
)
from app.next48h_view import (
    BEST_WINDOW_OBJECTIVES,
    choose_best_window,
    confidence_summary,
    enrich_forecast_frame_with_twin,
    forecast_display_table,
    forecast_points_from_twin,
    future_regional_map_frame,
    generation_mix_rows,
    peak_forecast_row,
    projected_future_regional_map_frame,
    selected_forecast_point,
    selected_hour_explanation,
    selected_timestamp_from_chart_event,
    selected_twin_snapshot,
    twin_aligned_to_reference,
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


def _local_hour_label(timestamp: pd.Timestamp) -> str:
    return f"{timestamp.tz_convert(settings.timezone):%a %d %b %H:%M}"


def _window_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    start_local = start.tz_convert(settings.timezone)
    end_local = end.tz_convert(settings.timezone)
    if start_local.date() == end_local.date():
        return f"{start_local:%a %H:%M}-{end_local:%H:%M}"
    return f"{start_local:%a %H:%M}-{end_local:%a %H:%M}"


def _time_delta_text(selected: pd.Timestamp, reference: pd.Timestamp) -> str:
    hours = int(round((selected - reference).total_seconds() / 3600.0))
    if hours == 0:
        return "Selected hour is the expected peak."
    if hours > 0:
        return f"Selected hour is {hours}h after the expected peak."
    return f"Selected hour is {abs(hours)}h before the expected peak."


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
        return "Unknown"
    balance = snapshot.modelled_national_balance_context or snapshot.national.balance_context
    return balance.status.value.title() if balance else "Unknown"


state = read_app_state(default_mode=DomainMode.REPLAY)
app_context = load_typed_public_context(state)
context = app_context.legacy
energy: pd.DataFrame = context["energy"]
if energy.empty:
    error_state("Forecast context unavailable", "The public dashboard could not load a typed or legacy forecast context.")
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
    error_state("Forecast unavailable", "No forecast points are available for the next 48 hours.")
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
selected_confidence = confidence_summary(selected_point, selected_snapshot)

render_public_header(
    "NEXT 48H",
    "Demand range, better usage windows, confidence, and future regional context for the next two days.",
    state.mode.value.upper(),
)
render_context_bar(state, twin=aligned_twin, current_state=app_context.current_state, timezone_name=settings.timezone)
render_trust_notices(
    notice
    for notice in notices_from_contracts(aligned_twin, app_context.current_state)
    if notice.title not in {"Demo fixture mode", "Optional source unavailable"}
)

section_header(
    "Forecast",
    "48-hour demand outlook",
    "The selected timestamp from this chart is reused by the explanation, confidence, regional, generation, and balance panels.",
)
viz_note(
    "Actuals, P50 forecast, likely range, and usual-demand baseline",
    "The uncertainty ribbon is the p10-p90 likely range. It is not treated as a positive or negative pressure driver.",
    source="RTE eco2mix, usual-demand baseline, and typed twin context",
)
chart_event = st.plotly_chart(
    forecast_chart(
        energy,
        forecast,
        selected_timestamp=selected_point.timestamp,
        now_timestamp=snapshot.as_of,
        timezone_name=settings.timezone,
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

hour_labels = [f"{_local_hour_label(point.timestamp)} - {point.pressure_label}" for point in points]
selected_label = st.selectbox("Forecast hour", hour_labels, index=selected_index)
selected_label_index = hour_labels.index(selected_label)
if selected_label_index != selected_index:
    persist_app_state(with_updates(state, selected_timestamp=points[selected_label_index].timestamp.to_pydatetime()))
    st.rerun()

nav_cols = st.columns([1, 1, 3], gap="small")
with nav_cols[0]:
    if st.button("Previous hour", disabled=selected_index <= 0, width="stretch"):
        _select_hour(selected_index - 1, points)
with nav_cols[1]:
    if st.button("Next hour", disabled=selected_index >= len(points) - 1, width="stretch"):
        _select_hour(selected_index + 1, points)
with nav_cols[2]:
    st.caption(f"Selected forecast timestamp: {format_timestamp(selected_point.timestamp, timezone_name=settings.timezone)}")

with st.expander("Forecast data", expanded=False):
    st.dataframe(forecast_display_table(forecast, timezone_name=settings.timezone), width="stretch", hide_index=True)

objective_label = st.radio(
    "Best-window objective",
    list(BEST_WINDOW_OBJECTIVES.values()),
    index=2,
    horizontal=True,
)
objective_key = {value: key for key, value in BEST_WINDOW_OBJECTIVES.items()}[objective_label]
best = choose_best_window(forecast, objective_key)
st.caption("These objectives may produce different times because demand balance context and carbon context are separate signals.")

summary = st.columns(3)
with summary[0]:
    if peak is not None:
        peak_time = pd.Timestamp(peak["timestamp"])
        metric_card(
            "Expected peak",
            f"{peak_time.tz_convert(settings.timezone):%a %H:%M}",
            f"Range {format_uncertainty_range(peak['p10'], peak['p90'])}. {_time_delta_text(selected_point.timestamp, peak_time)}",
            icon="P",
            status=str(peak.get("pressure_label", "unknown")),
            provenance="modelled",
        )
    else:
        metric_card("Expected peak", "Unavailable", "No peak could be calculated.", icon="P", status="unknown", provenance="unavailable")
with summary[1]:
    if best is not None:
        metric_card(
            "Best usage window",
            _window_label(best.start, best.end),
            f"{best.explanation} Average demand {format_gw(best.mean_demand_mw)}.",
            icon="W",
            status="green",
            provenance="modelled",
        )
    else:
        metric_card("Best usage window", "Unavailable", "No eligible forecast window is available.", icon="W", status="unknown")
with summary[2]:
    metric_card(
        "Confidence",
        selected_confidence.level,
        f"Selected hour {_local_hour_label(selected_point.timestamp)}. {selected_confidence.detail}",
        icon="C",
        status=selected_confidence.level,
        provenance="modelled",
    )

section_header("Selected Hour", f"Why {_local_hour_label(selected_point.timestamp)} is high or low")
st.markdown(
    f'<div class="ep-section-copy">Terms: {term_tooltip_html("usual demand")} '
    f'and {term_tooltip_html("likely range")}.</div>',
    unsafe_allow_html=True,
)
selected_explanation = selected_hour_explanation(selected_point, selected_snapshot, timezone_name=settings.timezone)
positive_total = sum(driver.value_mw for driver in selected_explanation.positive_drivers)
negative_total = sum(driver.value_mw for driver in selected_explanation.negative_drivers)
selected_metrics = st.columns(4)
with selected_metrics[0]:
    metric_card("Usual demand", format_gw(selected_explanation.usual_demand_mw), "Comparable-history baseline.", icon="U", provenance="fallback")
with selected_metrics[1]:
    metric_card(
        "Positive drivers",
        format_gw(positive_total, signed=True),
        "Demand-side lift above usual for this selected hour.",
        icon="+",
        status="watch" if positive_total else "green",
        provenance="modelled",
    )
with selected_metrics[2]:
    metric_card(
        "Negative drivers",
        format_gw(-negative_total, signed=True),
        "Demand-side easing below usual for this selected hour.",
        icon="-",
        status="green" if negative_total else "grey",
        provenance="modelled",
    )
with selected_metrics[3]:
    metric_card(
        "Final expected demand",
        format_gw(selected_explanation.expected_demand_mw),
        f"Range {format_uncertainty_range(selected_explanation.p10_mw, selected_explanation.p90_mw, unit='GW')}; reconciliation error {format_signed_mw(selected_explanation.reconciliation_error_mw)}.",
        icon="=",
        status=selected_point.pressure_label,
        provenance="modelled",
    )

explanation_card(
    "Deterministic explanation",
    selected_explanation.text,
    label="Numerically reconciled",
    status="info",
    provenance="modelled",
)

section_header("Confidence", "What supports this forecast?")
factor_cols = st.columns(3)
for index, factor in enumerate(selected_confidence.factors):
    with factor_cols[index % len(factor_cols)]:
        explanation_card(
            factor.value,
            factor.detail,
            label=factor.name,
            status=factor.status,
            provenance="modelled" if factor.status != "unknown" else "unavailable",
        )

section_header(
    "Future Regional Map",
    f"Regional demand anomaly at {_local_hour_label(selected_point.timestamp)}",
    "This is a selected-hour forecast allocation. It is not current regional context and not a regional adequacy status.",
)
future_regional = future_regional_map_frame(selected_snapshot, timezone_name=settings.timezone)
if not has_meaningful_anomaly_signal(future_regional):
    future_regional = projected_future_regional_map_frame(regional, selected_point, timezone_name=settings.timezone)
left, right = responsive_columns("map-detail")
with left:
    if future_regional.empty:
        error_state("Regional forecast unavailable", "The selected future hour has no regional demand allocation.")
    else:
        regional_event = st.plotly_chart(
            regional_anomaly_choropleth(
                future_regional,
                context["regions_geojson"],
                demand_label_title="Forecast demand",
                source_label_title="Forecast source",
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
            "Future regional context unavailable",
            "No selected-hour regional forecast was returned by the typed twin.",
            label="Regional forecast",
            status="unknown",
            provenance="unavailable",
        )
    else:
        region_codes = future_regional["region_code"].astype(str).tolist()
        selected_region_code = state.selected_region if state.selected_region in region_codes else region_codes[0]
        region_label = st.selectbox(
            "Region detail",
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
            f"Forecast demand {selected_region['demand_label']} versus usual {selected_region['usual_label']} "
            f"({selected_region['difference_label']}).",
            label=str(selected_region["freshness_label"]),
            status="info",
            provenance="modelled",
        )
        explanation_card(
            "Regional interpretation",
            str(selected_region["note"]),
            label="Demand allocation only",
            status="grey",
            provenance="modelled",
        )

section_header("Generation And Balance", f"Selected-hour context at {_local_hour_label(selected_point.timestamp)}")
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
        "Estimated generation mix",
        format_gw(mix.total.value if mix else fallback_generation_total),
        (
            "Nuclear, wind, solar, and residual flexible sources/imports for the selected future hour."
            if mix
            else "Latest generation mix carried as an explicit fallback assumption for the selected future demand."
        ),
        icon="G",
        status="info",
        provenance="modelled" if mix else "fallback",
    )
with context_cols[1]:
    metric_card(
        "Availability context",
        format_mw(availability.announced_unavailable.value if availability else 0.0),
        (
            f"Nuclear expected output {format_mw(availability.nuclear.value.value)}."
            if availability
            else "No selected-hour unavailability feed is aligned; latest availability is fallback context only."
        ),
        icon="A",
        status="watch" if availability and (availability.announced_unavailable.value or 0) > 0 else "green",
        provenance="modelled" if availability else "fallback",
    )
with context_cols[2]:
    metric_card(
        "Balance context",
        _balance_status(selected_snapshot) if selected_snapshot else selected_point.pressure_label,
        (
            f"Score {format_percentage(balance.pressure_ratio.value)}; "
            f"supply margin {format_signed_mw(balance.supply_margin.value)}."
            if balance
            else f"Static availability fallback; supply margin {format_signed_mw(fallback_margin)}."
        ),
        icon="B",
        status=_balance_status(selected_snapshot) if selected_snapshot else selected_point.pressure_label,
        provenance="modelled" if balance else "fallback",
    )
context_cols_2 = st.columns(3)
with context_cols_2[0]:
    metric_card(
        "Official signal",
        official.label if official else "Future EcoWatt unavailable",
        official.detail if official and official.detail else "No aligned official signal is shown for this selected future hour.",
        icon="O",
        status=official.status.value.title() if official else "unknown",
        provenance="official" if official else "unavailable",
    )
with context_cols_2[1]:
    metric_card(
        "Carbon context",
        format_carbon(carbon.intensity.value if carbon else (None if pd.isna(fallback_carbon) else float(fallback_carbon))),
        carbon.method if carbon else "Latest carbon intensity carried as fallback context; it is not a balance input.",
        icon="CO2",
        status="info",
        provenance="fallback" if carbon is None or carbon.source.is_fallback else "modelled",
    )
with context_cols_2[2]:
    metric_card(
        "Exchange estimate",
        format_signed_mw(exchange.net_imports.value if exchange else fallback_exchange),
        (
            "Positive means net imports; negative means net exports."
            if exchange
            else "Latest exchange context carried as a fallback assumption."
        ),
        icon="X",
        status="info",
        provenance="modelled" if exchange else "fallback",
    )
with st.expander("Estimated generation mix", expanded=False):
    mix_rows = generation_mix_rows(selected_snapshot)
    if mix_rows.empty:
        mix_rows = pd.DataFrame(
            [
                {
                    "Component": label,
                    "Estimate": format_mw(value),
                    "Provenance": "latest snapshot fallback",
                    "Formula": "Latest observed/replay mix carried as fallback context.",
                }
                for label, value in snapshot.generation_by_source.items()
            ]
        )
    st.dataframe(mix_rows, width="stretch", hide_index=True)

with st.expander("How calculated", expanded=False):
    st.write(
        "The page uses the typed electricity-system twin only when its forecast origin aligns with the page's latest "
        "actual or demo timestamp. Otherwise it uses the aligned 48-hour forecast and labels projected regional, "
        "generation, balance, and carbon context as fallback."
    )
    st.write(
        "Demand is shown as p50 with a p10-p90 likely range and a usual-demand baseline. The selected-hour "
        "explanation reconciles p50 as usual demand plus positive drivers minus negative drivers."
    )
    st.write(
        "The best-window selector evaluates rolling three-hour windows. Balance, carbon, and combined objectives can "
        "choose different times because they optimize different quantities."
    )

with st.expander("Reliability and model card", expanded=False):
    source_counts = Counter(point.source for point in points)
    st.dataframe(
        pd.DataFrame([{"Forecast route": source, "Hours": count} for source, count in source_counts.items()]),
        width="stretch",
        hide_index=True,
    )
    st.write(selected_confidence.detail)
    if selected_snapshot and selected_snapshot.unsupported_physical_behaviours:
        st.write("Known model limitations:")
        for limitation in selected_snapshot.unsupported_physical_behaviours:
            st.write(f"- {limitation}")

with st.expander("Data sources and freshness", expanded=False):
    st.write(f"National electricity: {context['national_source']}")
    st.write(f"Regional electricity: {context['regional_source']}")
    st.write(f"Map geometry: {context.get('geo_source', 'Unavailable')}")
    st.write(f"Selected forecast hour: {format_timestamp(selected_point.timestamp, timezone_name=settings.timezone)}")
    if app_context.twin is not None and aligned_twin is None:
        st.write(
            "Typed twin snapshots were available but not aligned to this page's latest forecast origin, "
            "so selected-hour context is shown with explicit fallback labels."
        )
    if selected_snapshot is not None:
        st.write(f"Selected twin snapshot updated: {format_timestamp(selected_snapshot.update_time, timezone_name=settings.timezone)}")
    unavailable = aligned_twin.unavailable_fields if aligned_twin else []
    if unavailable:
        st.write("Unavailable optional sources:")
        for item in unavailable:
            st.write(f"- {item.field}: {item.reason}")

about_project_drawer()
provenance_drawer(
    ("National electricity", context["national_source"]),
    ("Regional electricity", context["regional_source"]),
    ("Forecast run", app_context.forecast_run_id if aligned_twin and app_context.forecast_run_id else "Aligned legacy forecast"),
    ("Selected timestamp", format_timestamp(selected_point.timestamp, timezone_name=settings.timezone)),
)
