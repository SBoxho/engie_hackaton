from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from app.api_client import load_typed_public_context
from app.components.cards import section_header, viz_note
from app.components.foundation import (
    error_state,
    notices_from_contracts,
    render_context_bar,
    responsive_columns,
)
from app.components.now_dashboard import (
    build_current_state_map_frame,
    build_hero_summary,
    current_weather_summary,
    fallback_map_frame,
    generation_mix_figure,
    render_carbon_context,
    render_driver_cards,
    render_hero_summary,
    render_forecast_point_ribbon,
    render_region_selector,
    render_selected_forecast_context,
    render_selected_region_panel,
    selected_forecast_point,
    selected_twin_snapshot,
)
from app.data_loader import load_live_current_weather
from app.components.public import render_public_header, selected_location
from app.components.regional_map import has_meaningful_anomaly_signal, regional_anomaly_choropleth
from app.formatting import format_gw, format_timestamp
from app.i18n import mode_label, nav_label, t
from app.state import mode_for_page, persist_app_state, read_app_state, select_timestamp_from_options, with_updates
from app.view_models import add_regional_anomalies, build_forecast_points, build_grid_snapshot, ui_mode
from src.contracts.energy_twin import DomainMode
from src.config import settings


def _term_tooltip_html(term_key: str, locale: str) -> str:
    label = t(f"now.terms.{term_key}.label", locale=locale)
    definition = t(f"now.terms.{term_key}.definition", locale=locale)
    return (
        f'<span class="ep-term" tabindex="0" role="note" '
        f'aria-label="{html.escape(label)}: {html.escape(definition)}">'
        f'<abbr title="{html.escape(definition)}">{html.escape(label)}</abbr>'
        f'<span class="ep-term-popover" aria-hidden="true">{html.escape(definition)}</span></span>'
    )


def _fallback_freshness_label(snapshot, locale: str) -> str:
    timestamp = format_timestamp(snapshot.as_of, timezone_name=settings.timezone, locale=locale)
    if snapshot.mode == "REPLAY":
        return t("now.freshness.historical_sample_at", locale=locale, timestamp=timestamp)
    return t("now.freshness.updated_at", locale=locale, timestamp=timestamp)


def _render_about_drawer(locale: str) -> None:
    with st.expander(t("now.about.title", locale=locale), expanded=False):
        st.write(t("now.about.body_1", locale=locale))
        st.write(t("now.about.body_2", locale=locale))


def _render_provenance_drawer(*items: tuple[str, str], locale: str) -> None:
    with st.expander(t("now.provenance.title", locale=locale), expanded=False):
        for label, detail in items:
            st.markdown(f"**{html.escape(label)}**: {html.escape(detail)}")


def _render_now_trust_notices(notices, locale: str) -> None:
    markup = "".join(_trust_notice_html(notice.kind, *_localized_notice_text(notice, locale), locale=locale) for notice in notices)
    if markup:
        st.markdown(f'<div class="ep-trust-stack">{markup}</div>', unsafe_allow_html=True)


def _localized_notice_text(notice, locale: str) -> tuple[str, str]:
    key = {
        "Typed twin contract unavailable": "typed_twin_contract_unavailable",
        "Delayed source data": "delayed_source_data",
        "Using last-known-good data": "using_last_known_good",
        "Optional source unavailable": "optional_source_unavailable",
        "Stale data": "stale_data",
    }.get(str(notice.title))
    if key is None:
        return str(notice.title), str(notice.body)
    return (
        t(f"now.notices.{key}.title", locale=locale),
        t(f"now.notices.{key}.body", locale=locale),
    )


def _trust_notice_html(kind: str, title: str, body: str, *, locale: str) -> str:
    role = "alert" if kind == "error" else "status"
    label = t(f"now.notices.kind.{kind}", locale=locale, default=str(kind).title())
    return (
        f'<div class="ep-trust-state ep-trust-{html.escape(kind)}" role="{role}" '
        f'aria-label="{html.escape(title)}: {html.escape(body)}">'
        f'<div class="ep-trust-label">{html.escape(label)}</div>'
        f'<div><div class="ep-trust-title">{html.escape(title)}</div>'
        f'<div class="ep-trust-body">{html.escape(body)}</div></div></div>'
    )


def _localized_unavailable(value: str, locale: str) -> str:
    return t("shared.format.unavailable", locale=locale) if value == "Unavailable" else value


state = read_app_state(default_mode=DomainMode.REPLAY)
app_context = load_typed_public_context(state, hours=0, include_current_state=True)
context = app_context.legacy
energy: pd.DataFrame = context["energy"]
if energy.empty:
    error_state(
        t("now.states.energy_unavailable_title", locale=state.locale),
        t("now.states.energy_unavailable_body", locale=state.locale),
    )
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
    locale=state.locale,
)
live_weather = current_weather_summary(load_live_current_weather(), snapshot.weather, locale=state.locale)
render_public_header(nav_label("now", locale=state.locale), "", state.mode.value.upper())
render_hero_summary(hero, locale=state.locale)
render_context_bar(
    state,
    twin=app_context.twin,
    current_state=app_context.current_state,
    timezone_name=settings.timezone,
    weather=live_weather,
    hide_replay_badge=True,
)
_render_now_trust_notices(
    (
        notice
        for notice in notices_from_contracts(app_context.twin, app_context.current_state)
        if notice.title not in {"Demo fixture mode", "Partial data"}
    ),
    state.locale,
)

section_header(
    t("now.forecast.section_kicker", locale=state.locale),
    t("now.forecast.section_title", locale=state.locale),
    t("now.forecast.section_copy", locale=state.locale),
)
new_selected_time = render_forecast_point_ribbon(
    forecast_points,
    state.selected_timestamp,
    timezone_name=settings.timezone,
    locale=state.locale,
)
if new_selected_time is not None and pd.Timestamp(new_selected_time) != pd.Timestamp(state.selected_timestamp):
    persist_app_state(with_updates(state, selected_timestamp=new_selected_time))
    st.rerun()
render_selected_forecast_context(selected_point, timezone_name=settings.timezone, locale=state.locale)

section_header(
    t("now.regions.section_kicker", locale=state.locale),
    t("now.regions.section_title", locale=state.locale),
    t("now.regions.section_copy", locale=state.locale),
)
map_frame = build_current_state_map_frame(app_context.current_state, locale=state.locale)
if not has_meaningful_anomaly_signal(map_frame):
    map_frame = fallback_map_frame(
        regional,
        freshness_label=_fallback_freshness_label(snapshot, state.locale),
        source_label=str(context.get("regional_source", t("now.states.regional_source_unavailable", locale=state.locale))),
        locale=state.locale,
    )

left, right = responsive_columns("map-detail")
with left:
    event = st.plotly_chart(
        regional_anomaly_choropleth(
            map_frame,
            context["regions_geojson"],
            demand_label_title=t("now.regions.map_observed_demand", locale=state.locale),
            usual_demand_label_title=t("now.regions.map_usual_demand", locale=state.locale),
            difference_label_title=t("now.regions.map_difference", locale=state.locale),
            freshness_label_title=t("now.regions.map_freshness", locale=state.locale),
            source_label_title=t("now.regions.map_source", locale=state.locale),
            colorbar_title=t("now.regions.map_colorbar", locale=state.locale),
            anomaly_label_title=t("now.regions.map_demand_anomaly", locale=state.locale),
            reason_label_title=t("now.regions.map_reason", locale=state.locale),
            unavailable_label=t("now.regions.map_unavailable", locale=state.locale),
            available_trace_name=t("now.regions.map_available_name", locale=state.locale),
            unavailable_trace_name=t("now.regions.map_unavailable_name", locale=state.locale),
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
    selected_region_code = render_region_selector(state.selected_region, locale=state.locale)
    if selected_region_code != state.selected_region:
        persist_app_state(with_updates(state, selected_region=selected_region_code))
        st.rerun()
    render_selected_region_panel(
        app_context.current_state,
        map_frame,
        state.selected_region,
        hide_replay_badge=True,
        locale=state.locale,
    )

section_header(t("now.drivers.section_kicker", locale=state.locale), t("now.drivers.section_title", locale=state.locale))
st.markdown(
    f'<div class="ep-section-copy">{t("now.drivers.terms_html", locale=state.locale, usual_demand=_term_tooltip_html("usual_demand", state.locale), local_generation=_term_tooltip_html("local_generation", state.locale))}</div>',
    unsafe_allow_html=True,
)
render_driver_cards(
    app_context.current_state,
    selected_snapshot,
    snapshot,
    weather_override=live_weather,
    hide_replay_badge=True,
    locale=state.locale,
)

section_header(t("now.generation_mix.section_kicker", locale=state.locale), t("now.generation_mix.section_title", locale=state.locale))
viz_note(
    t("now.generation_mix.viz_title", locale=state.locale),
    t("now.generation_mix.viz_detail", locale=state.locale),
    source=t("now.generation_mix.source", locale=state.locale),
)
if app_context.current_state is not None:
    national = app_context.current_state.national_context
    st.plotly_chart(
        generation_mix_figure(
            national.generation_mix,
            demand_mw=national.demand.current.value,
            net_imports_mw=national.net_imports.value,
            locale=state.locale,
        ),
        width="stretch",
    )
else:
    st.plotly_chart(generation_mix_figure(None, demand_mw=None, net_imports_mw=None, locale=state.locale), width="stretch")
render_carbon_context(app_context.current_state, selected_snapshot, hide_replay_badge=True, locale=state.locale)

_render_about_drawer(state.locale)
_render_provenance_drawer(
    (t("now.provenance.national_electricity", locale=state.locale), context["national_source"]),
    (t("now.provenance.regional_electricity", locale=state.locale), context["regional_source"]),
    (t("now.provenance.map_geometry", locale=state.locale), context["geo_source"]),
    (
        t("now.provenance.current_demand", locale=state.locale),
        _localized_unavailable(format_gw(app_context.current_state.national_context.demand.current.value, locale=state.locale), state.locale)
        if app_context.current_state
        else t("shared.format.unavailable", locale=state.locale),
    ),
    (t("now.provenance.mode", locale=state.locale), mode_label(state.mode, locale=state.locale)),
    locale=state.locale,
)
