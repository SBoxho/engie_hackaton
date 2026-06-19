from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from app.api_client import load_typed_public_context
from app.components.cards import explanation_card, metric_card, section_header, viz_note
from app.components.foundation import (
    TrustNotice,
    error_state,
    loading_state,
    notices_from_contracts,
    render_context_bar,
    render_trust_notices,
)
from app.components.public import about_project_drawer, provenance_drawer, render_public_header
from app.formatting import (
    format_mw,
    format_number,
    format_signed_mw,
    format_temperature,
    format_timestamp,
)
from app.generated.energy_twin_client import EnergyTwinApiClient, ScenarioRunQuery
from app.i18n import nav_label, t
from app.state import mode_for_page, persist_app_state, read_app_state, with_updates
from app.view_models import add_regional_anomalies, build_forecast_points
from app.what_if_view import (
    SCENARIO_PRESETS,
    ScenarioControls,
    baseline_scenario_chart,
    build_scenario_request,
    card_status_display,
    causal_chain_steps,
    chain_markup,
    changed_region_table,
    closest_scenario_timestamp,
    demand_delta_chart,
    event_timeline_chart,
    format_carbon_range,
    format_score_delta,
    format_signed_range,
    format_watch_high_delta,
    generation_response_chart,
    magnitude_label_from_request,
    normalize_scenario_key,
    regional_delta_choropleth,
    regional_delta_frame,
    restore_scenario_controls,
    scenario_preset_display,
    scenario_frames,
    scenario_query_params,
    scenario_summary,
    scenario_window_label,
    selected_hour_row,
    selected_timestamp_from_chart_event,
    status_display_label,
    translate_api_text,
    validate_controls,
)
from src.contracts.energy_twin import DomainMode
from src.config import settings
from src.data_sources.rte_eco2mix_regional import REGION_NAMES


WINDOW_OPTIONS = [
    ("17:00", "21:00"),
    ("18:00", "22:00"),
    ("19:00", "23:00"),
    ("00:00", "04:00"),
    ("01:00", "05:00"),
    ("02:00", "06:00"),
]


def _default_magnitude_for(scenario_type: str) -> float:
    if scenario_type == "ev_charging_shift":
        return 100_000.0
    if scenario_type == "generation_unavailability":
        return 1.3
    return 3.0


def _default_duration_for(scenario_type: str) -> int:
    return 12 if scenario_type == "ev_charging_shift" else 6


def _baseline_origin(app_context: object, legacy_points: list) -> pd.Timestamp:
    twin = getattr(app_context, "twin", None)
    if twin is not None:
        return pd.Timestamp(twin.from_time).tz_convert("UTC").floor("h")
    if legacy_points:
        return (legacy_points[0].timestamp - pd.Timedelta(hours=1)).floor("h")
    energy = getattr(app_context, "energy", pd.DataFrame())
    if isinstance(energy, pd.DataFrame) and not energy.empty and "timestamp" in energy:
        return pd.to_datetime(energy["timestamp"], utc=True, errors="coerce").max().floor("h")
    return pd.Timestamp.now(tz="UTC").floor("h")


def _sync_query_params(controls: ScenarioControls) -> None:
    for key, value in scenario_query_params(controls).items():
        if st.query_params.get(key) != value:
            st.query_params[key] = value


def _window_option_label(window: tuple[str, str]) -> str:
    return f"{window[0]}-{window[1]}"


def _scenario_result_notices(result: dict, *, base_notices: list[TrustNotice], locale: str) -> list[TrustNotice]:
    notices = list(base_notices)
    snapshot_count = int(result.get("data_versions", {}).get("baseline_snapshot_count") or 0)
    if snapshot_count and snapshot_count < 49:
        notices.append(
            TrustNotice(
                "partial",
                t("what_if.notices.partial_horizon.title", locale=locale),
                t("what_if.notices.partial_horizon.body", locale=locale, count=snapshot_count),
            )
        )
    regional_deltas = result.get("regional_deltas") or {}
    if regional_deltas and not regional_deltas.get("supported", True):
        notices.append(
            TrustNotice(
                "partial",
                t("what_if.notices.regional_unavailable.title", locale=locale),
                translate_api_text(
                    regional_deltas.get("reason") or "This scenario type has no regional demand allocation in v1.",
                    locale=locale,
                ),
            )
        )
    return notices


def _selected_hour_controls(result: dict, selected_timestamp: pd.Timestamp, *, locale: str) -> pd.Timestamp:
    _, scenario = scenario_frames(result)
    if scenario.empty:
        return selected_timestamp
    labels = [
        t(
            "what_if.selected_hour.option",
            locale=locale,
            time=format_timestamp(row.timestamp, timezone_name=settings.timezone, include_date=False, locale=locale),
            status=status_display_label(row.balance_status, locale=locale),
            delta=format_signed_mw(row.demand_delta_mw, locale=locale),
        )
        for row in scenario.itertuples(index=False)
    ]
    distances = (scenario["timestamp"] - selected_timestamp).abs()
    selected_index = int(distances.idxmin())
    selected_label = st.selectbox(t("what_if.selected_hour.label", locale=locale), labels, index=selected_index)
    new_index = labels.index(selected_label)
    nav_cols = st.columns([1, 1, 3], gap="small")
    with nav_cols[0]:
        previous_clicked = st.button(t("what_if.selected_hour.previous", locale=locale), disabled=new_index <= 0, width="stretch")
    with nav_cols[1]:
        next_clicked = st.button(t("what_if.selected_hour.next", locale=locale), disabled=new_index >= len(labels) - 1, width="stretch")
    with nav_cols[2]:
        st.caption(t("what_if.selected_hour.caption", locale=locale, label=labels[new_index]))
    if previous_clicked:
        new_index = max(0, new_index - 1)
    if next_clicked:
        new_index = min(len(labels) - 1, new_index + 1)
    return pd.Timestamp(scenario.iloc[new_index]["timestamp"])


state = read_app_state(default_mode=DomainMode.REPLAY)
locale = state.locale
app_context = load_typed_public_context(state)
context = app_context.legacy
energy: pd.DataFrame = context["energy"]
if energy.empty:
    error_state(
        t("what_if.errors.context_unavailable.title", locale=locale),
        t("what_if.errors.context_unavailable.body", locale=locale),
    )
    st.stop()

regional = add_regional_anomalies(context["regional"], context["regional_history"], timezone=settings.timezone)
legacy_points = build_forecast_points(
    energy,
    model_payload=context["model_payload"],
    horizon_hours=48,
    timezone=settings.timezone,
)
origin = _baseline_origin(app_context, legacy_points)
control_defaults = restore_scenario_controls(
    st.query_params,
    default_scenario=state.selected_scenario,
    default_region=state.selected_region,
)

state = with_updates(
    state,
    mode=mode_for_page(replay=app_context.is_replay, page="what_if"),
    selected_scenario=control_defaults.scenario_type,
    selected_region=control_defaults.region,
    selected_timestamp=(
        origin + pd.Timedelta(hours=control_defaults.start_offset_hours)
        if state.selected_timestamp is None
        or abs(pd.Timestamp(state.selected_timestamp).tz_convert("UTC") - origin) > pd.Timedelta(days=7)
        else pd.Timestamp(state.selected_timestamp).tz_convert("UTC")
    ).to_pydatetime(),
    selected_forecast_run=app_context.forecast_run_id,
)
persist_app_state(state)

render_public_header(
    nav_label("what_if", locale=state.locale),
    t("what_if.header.subtitle", locale=locale),
    state.mode.value.upper(),
)
render_context_bar(
    state,
    twin=app_context.twin,
    current_state=app_context.current_state,
    timezone_name=settings.timezone,
    hide_replay_badge=True,
)
render_trust_notices(
    notice
    for notice in notices_from_contracts(app_context.twin, app_context.current_state)
    if notice.title not in {"Demo fixture mode", "Optional source unavailable"}
)

section_header(
    t("what_if.builder.kicker", locale=locale),
    t("what_if.builder.title", locale=locale),
    t("what_if.builder.copy", locale=locale),
)

preset_labels = [scenario_preset_display(key, locale=locale)["label"] for key in SCENARIO_PRESETS]
label_to_key = {scenario_preset_display(key, locale=locale)["label"]: key for key in SCENARIO_PRESETS}
selected_label = st.selectbox(
    t("what_if.builder.preset", locale=locale),
    preset_labels,
    index=list(SCENARIO_PRESETS).index(control_defaults.scenario_type),
)
scenario_type = label_to_key[selected_label]
preset = scenario_preset_display(scenario_type, locale=locale)
scenario_changed = scenario_type != control_defaults.scenario_type
initial_magnitude = _default_magnitude_for(scenario_type) if scenario_changed else control_defaults.magnitude

builder_cols = st.columns([1, 1, 1], gap="large")
with builder_cols[0]:
    scope_options = [("national", t("what_if.builder.scope.national", locale=locale))]
    if scenario_type != "generation_unavailability":
        scope_options.append(("regional", t("what_if.builder.scope.regional", locale=locale)))
    default_scope_type = "regional" if control_defaults.scope_type == "regional" and any(key == "regional" for key, _ in scope_options) else "national"
    scope_labels = [label for _, label in scope_options]
    scope_label_to_type = {label: key for key, label in scope_options}
    default_scope_label = dict(scope_options)[default_scope_type]
    scope_label = st.selectbox(
        t("what_if.builder.geographic_scope", locale=locale),
        scope_labels,
        index=scope_labels.index(default_scope_label),
    )
    scope_type = scope_label_to_type[scope_label]
    region_codes = regional["region_code"].astype(str).tolist() if not regional.empty and "region_code" in regional else list(REGION_NAMES)
    region_labels = {
        str(row.region_code): str(getattr(row, "region_display", REGION_NAMES.get(str(row.region_code), row.region_code)))
        for row in regional.itertuples(index=False)
    } if not regional.empty and "region_code" in regional else REGION_NAMES
    selected_region = control_defaults.region if control_defaults.region in region_codes else state.selected_region
    if selected_region not in region_codes:
        selected_region = region_codes[0]
    region = st.selectbox(
        t("what_if.builder.region", locale=locale),
        region_codes,
        index=region_codes.index(selected_region),
        format_func=lambda code: region_labels.get(str(code), str(code)),
        disabled=scope_type != "regional",
    )
with builder_cols[1]:
    if scenario_type == "cold_snap":
        magnitude = float(
            st.slider(
                t("what_if.builder.magnitude", locale=locale),
                min_value=0.5,
                max_value=8.0,
                value=min(max(float(initial_magnitude), 0.5), 8.0),
                step=0.5,
            )
        )
        st.caption(t("what_if.builder.cold_caption", locale=locale, value=format_temperature(magnitude, locale=locale)))
        asset_name = control_defaults.asset_name
        ev_energy_kwh = control_defaults.ev_energy_kwh
        ev_participation = control_defaults.ev_participation
        original_window = control_defaults.original_window
        target_window = control_defaults.target_window
    elif scenario_type == "generation_unavailability":
        magnitude = float(
            st.slider(
                t("what_if.builder.magnitude", locale=locale),
                min_value=0.2,
                max_value=6.0,
                value=min(max(float(initial_magnitude), 0.2), 6.0),
                step=0.1,
            )
        )
        st.caption(t("what_if.builder.generation_caption", locale=locale, value=f"{format_number(magnitude, decimals=1, locale=locale)} GW"))
        asset_name = st.text_input(t("what_if.builder.asset_label", locale=locale), value=control_defaults.asset_name, max_chars=40)
        ev_energy_kwh = control_defaults.ev_energy_kwh
        ev_participation = control_defaults.ev_participation
        original_window = control_defaults.original_window
        target_window = control_defaults.target_window
    else:
        magnitude = float(
            st.number_input(
                t("what_if.builder.participation", locale=locale),
                min_value=10_000,
                max_value=500_000,
                value=int(min(max(float(initial_magnitude), 10_000), 500_000)),
                step=10_000,
            )
        )
        st.caption(t("what_if.builder.ev_shifted_caption", locale=locale, value=format_number(int(magnitude), locale=locale)))
        asset_name = control_defaults.asset_name
        ev_energy_kwh = float(
            st.number_input(
                t("what_if.builder.energy_per_ev", locale=locale),
                min_value=2.0,
                max_value=30.0,
                value=min(max(float(control_defaults.ev_energy_kwh), 2.0), 30.0),
                step=1.0,
            )
        )
        ev_participation = float(
            st.slider(
                t("what_if.builder.participation_rate", locale=locale),
                min_value=0.05,
                max_value=1.0,
                value=min(max(float(control_defaults.ev_participation), 0.05), 1.0),
                step=0.05,
            )
        )
        source_options = [_window_option_label(option) for option in WINDOW_OPTIONS]
        source_default = _window_option_label(control_defaults.original_window)
        source_label = st.selectbox(
            t("what_if.builder.source_window", locale=locale),
            source_options,
            index=source_options.index(source_default) if source_default in source_options else source_options.index("18:00-22:00"),
        )
        target_default = _window_option_label(control_defaults.target_window)
        target_label = st.selectbox(
            t("what_if.builder.target_window", locale=locale),
            source_options,
            index=source_options.index(target_default) if target_default in source_options else source_options.index("01:00-05:00"),
        )
        original_window = tuple(source_label.split("-", 1))  # type: ignore[assignment]
        target_window = tuple(target_label.split("-", 1))  # type: ignore[assignment]
with builder_cols[2]:
    st.metric(t("what_if.builder.concept_affected", locale=locale), preset["concept"])
    st.caption(preset["detail"])

default_window = (
    min(max(control_defaults.start_offset_hours, 0), 47),
    min(
        max(
            control_defaults.start_offset_hours + _default_duration_for(scenario_type)
            if scenario_changed
            else control_defaults.end_offset_hours,
            1,
        ),
        48,
    ),
)
if default_window[1] <= default_window[0]:
    default_window = (default_window[0], min(default_window[0] + 1, 48))
window = st.slider(
    t("what_if.builder.scenario_window", locale=locale),
    min_value=0,
    max_value=48,
    value=default_window,
    step=1,
)
start_offset, end_offset = int(window[0]), int(window[1])
timeline_cols = st.columns(3)
with timeline_cols[0]:
    st.caption(
        t(
            "what_if.builder.start_time",
            locale=locale,
            value=format_timestamp(origin + pd.Timedelta(hours=start_offset), timezone_name=settings.timezone, locale=locale),
        )
    )
with timeline_cols[1]:
    st.caption(
        t(
            "what_if.builder.end_time",
            locale=locale,
            value=format_timestamp(origin + pd.Timedelta(hours=end_offset), timezone_name=settings.timezone, locale=locale),
        )
    )
with timeline_cols[2]:
    st.caption(
        t(
            "what_if.builder.duration",
            locale=locale,
            value=format_number(max(end_offset - start_offset, 0), locale=locale),
        )
    )

controls = ScenarioControls(
    scenario_type=normalize_scenario_key(scenario_type),
    scope_type=scope_type,
    region=str(region),
    magnitude=float(magnitude),
    start_offset_hours=start_offset,
    end_offset_hours=end_offset,
    asset_name=asset_name,
    ev_energy_kwh=float(ev_energy_kwh),
    ev_participation=float(ev_participation),
    original_window=(str(original_window[0]), str(original_window[1])),
    target_window=(str(target_window[0]), str(target_window[1])),
)
_sync_query_params(controls)

state = with_updates(
    state,
    selected_scenario=controls.scenario_type,
    selected_region=controls.region,
    selected_timestamp=state.selected_timestamp or (origin + pd.Timedelta(hours=controls.start_offset_hours)).to_pydatetime(),
)
persist_app_state(state)

validation_error = validate_controls(controls, locale=locale)
if validation_error:
    error_state(t("what_if.errors.invalid_range.title", locale=locale), validation_error)
    st.stop()

request = build_scenario_request(controls, baseline_from_time=origin, timezone_name=settings.timezone)
placeholder = st.empty()
try:
    with placeholder.container():
        loading_state(t("what_if.loading.title", locale=locale), t("what_if.loading.body", locale=locale))
    with st.spinner(t("what_if.loading.spinner", locale=locale)):
        result = EnergyTwinApiClient().run_scenario(ScenarioRunQuery(request=request, use_cache=True))
    placeholder.empty()
except (OSError, TypeError, ValueError, KeyError) as exc:
    placeholder.empty()
    error_state(t("what_if.errors.api_error.title", locale=locale), t("what_if.errors.api_error.body", locale=locale, error=exc))
    st.stop()

render_trust_notices(
    _scenario_result_notices(
        result,
        base_notices=[],
        locale=locale,
    )
)

selected_timestamp = closest_scenario_timestamp(result, state.selected_timestamp or request["start_time"])
if selected_timestamp is None:
    selected_timestamp = pd.Timestamp(request["start_time"]).tz_convert("UTC")
if state.selected_timestamp is None or pd.Timestamp(state.selected_timestamp).tz_convert("UTC") != selected_timestamp:
    state = with_updates(state, selected_timestamp=selected_timestamp.to_pydatetime())
    persist_app_state(state)

summary = scenario_summary(result)

section_header(t("what_if.delta.kicker", locale=locale), t("what_if.delta.title", locale=locale))
metric_cols = st.columns(3)
with metric_cols[0]:
    metric_card(
        t("what_if.metrics.peak_demand_change.label", locale=locale),
        format_signed_mw(summary.peak_demand_delta_mw, locale=locale),
        t("what_if.metrics.peak_demand_change.detail", locale=locale),
        icon="P",
        status=card_status_display("watch" if summary.peak_demand_delta_mw > 0 else "green", locale=locale),
    )
with metric_cols[1]:
    metric_card(
        t("what_if.metrics.minimum_balance_change.label", locale=locale),
        format_score_delta(summary.min_balance_score_delta, locale=locale),
        t("what_if.metrics.minimum_balance_change.detail", locale=locale),
        icon="B",
        status=card_status_display("watch" if summary.min_balance_score_delta > 0 else "green", locale=locale),
    )
with metric_cols[2]:
    metric_card(
        t("what_if.metrics.watch_high_hours.label", locale=locale),
        format_watch_high_delta(summary.watch_high_hour_delta, locale=locale),
        t("what_if.metrics.watch_high_hours.detail", locale=locale, count=summary.changed_watch_high_hours),
        icon="!",
        status=card_status_display("watch" if summary.watch_high_hour_delta > 0 else "green", locale=locale),
    )
range_cols = st.columns(2)
with range_cols[0]:
    metric_card(
        t("what_if.metrics.import_export_range.label", locale=locale),
        format_signed_range(summary.import_export_delta_mwh_range, unit="MWh", locale=locale),
        t("what_if.metrics.import_export_range.detail", locale=locale),
        icon="I/E",
        status=card_status_display("watch" if summary.import_export_delta_mwh_range[1] > 0 else "green", locale=locale),
    )
with range_cols[1]:
    metric_card(
        t("what_if.metrics.carbon_range.label", locale=locale),
        format_carbon_range(summary.carbon_delta_tonnes_range, locale=locale),
        t("what_if.metrics.carbon_range.detail", locale=locale),
        icon="CO2",
        status=card_status_display("watch" if summary.carbon_delta_tonnes_range[1] > 0 else "green", locale=locale),
    )

st.html(chain_markup(causal_chain_steps(result, locale=locale)))

section_header(t("what_if.timeline.kicker", locale=locale), t("what_if.timeline.title", locale=locale))
viz_note(
    t("what_if.timeline.note_title", locale=locale),
    t(
        "what_if.timeline.note_detail",
        locale=locale,
        magnitude=magnitude_label_from_request(result, locale=locale),
        window=scenario_window_label(result, timezone_name=settings.timezone, locale=locale),
    ),
    source=t("what_if.sources.scenario_api", locale=locale),
)
st.plotly_chart(
    event_timeline_chart(result, selected_timestamp=selected_timestamp, timezone_name=settings.timezone, locale=locale),
    width="stretch",
    config={"displayModeBar": False},
)

section_header(t("what_if.baseline.kicker", locale=locale), t("what_if.baseline.title", locale=locale))
chart_event = st.plotly_chart(
    baseline_scenario_chart(result, selected_timestamp=selected_timestamp, timezone_name=settings.timezone, locale=locale),
    key="what_if_baseline_scenario_chart",
    width="stretch",
    on_select="rerun",
    selection_mode="points",
)
chart_timestamp = selected_timestamp_from_chart_event(chart_event)
if chart_timestamp is not None and chart_timestamp != selected_timestamp:
    persist_app_state(with_updates(state, selected_timestamp=chart_timestamp.to_pydatetime()))
    st.rerun()

selected_from_controls = _selected_hour_controls(result, selected_timestamp, locale=locale)
if selected_from_controls != selected_timestamp:
    persist_app_state(with_updates(state, selected_timestamp=selected_from_controls.to_pydatetime()))
    st.rerun()

selected = selected_hour_row(result, selected_timestamp)
if selected:
    baseline_row = selected["baseline"]
    scenario_row = selected["scenario"]
    selected_cols = st.columns(4)
    with selected_cols[0]:
        metric_card(
            t("what_if.selected_metrics.baseline_p50.label", locale=locale),
            format_mw(baseline_row.get("demand_mw"), locale=locale),
            status_display_label(baseline_row.get("balance_status", "unknown"), locale=locale),
            icon="B",
        )
    with selected_cols[1]:
        metric_card(
            t("what_if.selected_metrics.scenario_p50.label", locale=locale),
            format_mw(scenario_row.get("demand_mw"), locale=locale),
            status_display_label(scenario_row.get("balance_status", "unknown"), locale=locale),
            icon="S",
        )
    with selected_cols[2]:
        metric_card(
            t("what_if.selected_metrics.demand_delta.label", locale=locale),
            format_signed_mw(scenario_row.get("demand_delta_mw"), locale=locale),
            t("what_if.selected_metrics.demand_delta.detail", locale=locale),
            icon="+/-",
            status=card_status_display("watch" if float(scenario_row.get("demand_delta_mw") or 0) > 0 else "green", locale=locale),
        )
    with selected_cols[3]:
        metric_card(
            t("what_if.selected_metrics.residual_unresolved.label", locale=locale),
            format_signed_mw(scenario_row.get("unresolved_residual_mw"), locale=locale),
            t("what_if.selected_metrics.residual_unresolved.detail", locale=locale),
            icon="R",
            status=card_status_display("watch" if float(scenario_row.get("unresolved_residual_mw") or 0) > 0 else "green", locale=locale),
        )

viz_note(
    t("what_if.demand_delta.note_title", locale=locale),
    t("what_if.demand_delta.note_detail", locale=locale),
    source=t("what_if.sources.scenario_api", locale=locale),
)
st.plotly_chart(demand_delta_chart(result, locale=locale), width="stretch", config={"displayModeBar": False})

section_header(t("what_if.generation.kicker", locale=locale), t("what_if.generation.title", locale=locale))
viz_note(
    t("what_if.generation.note_title", locale=locale),
    translate_api_text(result.get("estimated_generation_response_range", {}).get("method", "Scenario response estimate."), locale=locale),
    source=t("what_if.sources.scenario_api_estimate", locale=locale),
)
st.plotly_chart(generation_response_chart(result, locale=locale), width="stretch", config={"displayModeBar": False})
with st.expander(t("what_if.generation.methods_expander", locale=locale), expanded=False):
    st.write(translate_api_text(result.get("estimated_generation_response_range", {}).get("method"), locale=locale))
    st.write(translate_api_text(result.get("estimated_import_export_delta", {}).get("method"), locale=locale))
    st.write(translate_api_text(result.get("estimated_carbon_range", {}).get("method"), locale=locale))

section_header(t("what_if.regional.kicker", locale=locale), t("what_if.regional.title", locale=locale))
regional_delta = regional_delta_frame(regional, result, locale=locale)
regional_supported = bool((result.get("regional_deltas") or {}).get("supported", True))
if not regional_supported:
    explanation_card(
        t("what_if.regional.no_allocation.title", locale=locale),
        translate_api_text((result.get("regional_deltas") or {}).get("reason", "This scenario changes national supply context only."), locale=locale),
        label=t("what_if.regional.no_allocation.label", locale=locale),
    )
st.plotly_chart(
    regional_delta_choropleth(regional_delta, context["regions_geojson"], locale=locale),
    width="stretch",
    key="what_if_delta_map",
    config={"displayModeBar": False},
)
changed_table = changed_region_table(regional_delta, locale=locale)
if changed_table.empty:
    st.info(t("what_if.regional.no_changed_regions", locale=locale))
else:
    st.dataframe(changed_table, width="stretch", hide_index=True)

with st.expander(t("what_if.assumptions.expander", locale=locale), expanded=False):
    st.markdown(f"**{t('what_if.assumptions.sensitivity_heading', locale=locale)}**")
    st.write(t("what_if.assumptions.sensitivity_body", locale=locale))
    st.markdown(f"**{t('what_if.assumptions.api_heading', locale=locale)}**")
    for assumption in result.get("assumptions") or []:
        st.write(translate_api_text(assumption, locale=locale))
    st.markdown(f"**{t('what_if.assumptions.limitations_heading', locale=locale)}**")
    for caveat in result.get("caveats") or []:
        st.write(translate_api_text(caveat, locale=locale))
    st.markdown(f"**{t('what_if.assumptions.versions_heading', locale=locale)}**")
    versions = [
        (t("what_if.versions.result_id", locale=locale), result.get("result_id")),
        (t("what_if.versions.baseline_run", locale=locale), result.get("baseline_forecast_run_id")),
        (t("what_if.versions.generated_at", locale=locale), result.get("generated_at")),
    ]
    versions.extend(
        (t("what_if.versions.model", locale=locale, name=key), value)
        for key, value in (result.get("model_versions") or {}).items()
    )
    versions.extend(
        (t("what_if.versions.data", locale=locale, name=key), value)
        for key, value in (result.get("data_versions") or {}).items()
        if key != "baseline_snapshot_ids"
    )
    for label, value in versions:
        st.write(f"{label}: {value}")

about_project_drawer(locale=locale)
provenance_drawer(
    (t("what_if.provenance.scenario_api_result", locale=locale), str(result.get("result_id"))),
    (t("what_if.provenance.baseline_forecast_run", locale=locale), str(result.get("baseline_forecast_run_id"))),
    (t("what_if.provenance.scenario_preset", locale=locale), html.unescape(selected_label)),
    (t("what_if.provenance.regional_electricity", locale=locale), context["regional_source"]),
    locale=locale,
)
