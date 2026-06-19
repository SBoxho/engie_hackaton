from __future__ import annotations

import pandas as pd
import streamlit as st

from app.api_client import load_typed_public_context
from app.components.cards import section_header, viz_note
from app.components.foundation import (
    error_state,
    notices_from_contracts,
    render_context_bar,
    render_trust_notices,
    responsive_columns,
    term_tooltip_html,
)
from app.components.now_dashboard import (
    build_current_state_map_frame,
    build_hero_summary,
    fallback_map_frame,
    generation_mix_figure,
    render_carbon_context,
    render_driver_cards,
    render_hero_summary,
    render_forecast_point_ribbon,
    render_region_selector,
    render_status_rows,
    render_selected_forecast_context,
    render_selected_region_panel,
    selected_forecast_point,
    selected_twin_snapshot,
)
from app.components.public import about_project_drawer, provenance_drawer, render_public_header, selected_location
from app.components.regional_map import has_meaningful_anomaly_signal, regional_anomaly_choropleth
from app.formatting import format_gw
from app.state import mode_for_page, persist_app_state, read_app_state, select_timestamp_from_options, with_updates
from app.view_models import add_regional_anomalies, build_forecast_points, build_grid_snapshot, ui_mode
from src.contracts.energy_twin import DomainMode
from src.config import settings


state = read_app_state(default_mode=DomainMode.REPLAY)
app_context = load_typed_public_context(state, hours=0, include_current_state=True)
context = app_context.legacy
energy: pd.DataFrame = context["energy"]
if energy.empty:
    error_state("Energy context unavailable", "The public dashboard could not load current electricity context.")
    st.stop()

regional = add_regional_anomalies(
    context["regional"],
    context["regional_history"],
    timezone=settings.timezone,
)
snapshot = build_grid_snapshot(
    energy,
    mode=ui_mode(context["mode"]),
    regional_state=regional,
    weather=context["weather"],
    ecowatt=context["ecowatt"],
    source_label=context["national_source"],
    timezone=settings.timezone,
)

forecast_points = build_forecast_points(
    energy,
    model_payload=context["model_payload"],
    horizon_hours=12,
    timezone=settings.timezone,
)
next12 = forecast_points[:12]
default_selected_time = (
    next12[0].timestamp.to_pydatetime()
    if next12
    else snapshot.as_of.to_pydatetime()
)
selected_options = [point.timestamp for point in next12] or [pd.Timestamp(default_selected_time)]
selected_index = select_timestamp_from_options(selected_options, state.selected_timestamp)
state = with_updates(
    state,
    mode=mode_for_page(replay=app_context.is_replay, page="now"),
    selected_timestamp=selected_options[selected_index].to_pydatetime(),
    selected_forecast_run=app_context.forecast_run_id,
)
selected_snapshot = selected_twin_snapshot(app_context.twin, state.selected_timestamp)
selected_point = selected_forecast_point(forecast_points, state.selected_timestamp)
persist_app_state(state)

hero = build_hero_summary(
    app_context.current_state,
    app_context.twin,
    snapshot,
    timezone_name=settings.timezone,
    forecast_points=forecast_points,
)
render_public_header("NOW", "", state.mode.value.upper())
render_hero_summary(hero)
render_context_bar(state, twin=app_context.twin, current_state=app_context.current_state, timezone_name=settings.timezone)
render_trust_notices(
    notice
    for notice in notices_from_contracts(app_context.twin, app_context.current_state)
    if notice.title not in {"Demo fixture mode", "Partial data"}
)
render_status_rows(app_context.current_state, selected_snapshot, timezone_name=settings.timezone)

section_header(
    "Next 12 Hours",
    "Selectable demand ribbon",
    "Pick an hour to update the selected-hour forecast context.",
)
new_selected_time = render_forecast_point_ribbon(
    forecast_points,
    state.selected_timestamp,
    timezone_name=settings.timezone,
)
if new_selected_time is not None and pd.Timestamp(new_selected_time) != pd.Timestamp(state.selected_timestamp):
    persist_app_state(with_updates(state, selected_timestamp=new_selected_time))
    st.rerun()
render_selected_forecast_context(selected_point, timezone_name=settings.timezone)

section_header(
    "Regions",
    "Demand versus usual",
    "Map color compares each region with its usual demand for a comparable context. Grey regions are unavailable, not zero.",
)
map_frame = build_current_state_map_frame(app_context.current_state)
if not has_meaningful_anomaly_signal(map_frame):
    map_frame = fallback_map_frame(
        regional,
        freshness_label=str(snapshot.freshness.get("label", "Freshness unavailable")),
        source_label=str(context.get("regional_source", "Regional source unavailable")),
    )

left, right = responsive_columns("map-detail")
with left:
    event = st.plotly_chart(
        regional_anomaly_choropleth(
            map_frame,
            context["regions_geojson"],
        ),
        key="now_regional_anomaly_map",
        width="stretch",
        on_select="rerun",
        selection_mode="points",
    )
    selected_from_map = selected_location(event)
    if selected_from_map and selected_from_map != state.selected_region:
        persist_app_state(with_updates(state, selected_region=selected_from_map))
        st.rerun()

with right:
    selected_region_code = render_region_selector(state.selected_region)
    if selected_region_code != state.selected_region:
        persist_app_state(with_updates(state, selected_region=selected_region_code))
        st.rerun()
    render_selected_region_panel(app_context.current_state, map_frame, state.selected_region)

section_header("Drivers", "What is driving the signal?")
st.markdown(
    f'<div class="ep-section-copy">Terms: {term_tooltip_html("usual demand")} '
    f'and {term_tooltip_html("local generation")}.</div>',
    unsafe_allow_html=True,
)
render_driver_cards(app_context.current_state, selected_snapshot, snapshot)

section_header("Generation Mix", "Generation, demand, exchange, and carbon")
viz_note(
    "Current generation mix",
    "Major sources are labelled directly. Small sources are grouped into Other when the chart would become cluttered.",
    source="RTE eCO2mix and typed current-state API",
)
if app_context.current_state is not None:
    national = app_context.current_state.national_context
    st.plotly_chart(
        generation_mix_figure(
            national.generation_mix,
            demand_mw=national.demand.current.value,
            net_imports_mw=national.net_imports.value,
        ),
        width="stretch",
    )
else:
    st.plotly_chart(generation_mix_figure(None, demand_mw=None, net_imports_mw=None), width="stretch")
render_carbon_context(app_context.current_state, selected_snapshot)

about_project_drawer()
provenance_drawer(
    ("National electricity", context["national_source"]),
    ("Regional electricity", context["regional_source"]),
    ("Map geometry", context["geo_source"]),
    ("Current demand", format_gw(app_context.current_state.national_context.demand.current.value) if app_context.current_state else "Unavailable"),
    ("Mode", state.mode.value),
)
