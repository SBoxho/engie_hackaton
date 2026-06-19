from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from app.components.cards import explanation_card, metric_card, section_header, status_badge, viz_note
from app.components.charts import MIX_COLUMNS
from app.components.layout import apply_theme
from app.components.regional_map import regional_demand_choropleth
from app.data_loader import REGIONAL_CONTEXT_CACHE_VERSION
from app.view_models import add_regional_anomalies, synthesize_regional_history
from src.contracts.status_thresholds import score_status, status_label
from src.config import settings
from src.demo_mode import external_api_enabled, mode_badge_color
from src.data_sources.rte_eco2mix_regional import (
    RegionalEco2MixError,
    demo_regional_snapshot,
    fallback_region_geojson,
    fetch_regional_eco2mix,
    load_cached_regional_eco2mix,
    load_region_geojson,
    prepare_regional_snapshot,
    source_attribution,
)

apply_theme()


@st.cache_data(ttl=900, show_spinner=False)
def load_regional_data(
    hours: int,
    cache_version: int = REGIONAL_CONTEXT_CACHE_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    _ = cache_version
    if settings.is_demo_mode and not external_api_enabled():
        snapshot = demo_regional_snapshot()
        return snapshot, synthesize_regional_history(snapshot, timezone=settings.timezone), "Historical regional sample", True
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    try:
        raw = fetch_regional_eco2mix(start=start, end=end)
        snapshot = prepare_regional_snapshot(raw)
        return snapshot, synthesize_regional_history(snapshot, timezone=settings.timezone), "Official regional RTE eco2mix, refreshed from ODRE", False
    except (RegionalEco2MixError, OSError):
        try:
            raw = load_cached_regional_eco2mix()
            snapshot = prepare_regional_snapshot(raw)
            return snapshot, synthesize_regional_history(snapshot, timezone=settings.timezone), "Official regional RTE eco2mix, cached snapshot", False
        except (RegionalEco2MixError, FileNotFoundError, ValueError, OSError):
            snapshot = demo_regional_snapshot()
            return snapshot, synthesize_regional_history(snapshot, timezone=settings.timezone), "Historical regional sample", True


@st.cache_data(ttl=86400, show_spinner=False)
def load_regions() -> tuple[dict, str, bool]:
    if settings.is_demo_mode and not external_api_enabled():
        return fallback_region_geojson(), "Bundled simplified France region boundaries", True
    try:
        return load_region_geojson(), "Official French administrative regions via API Geo", False
    except (RegionalEco2MixError, AttributeError, TypeError, ValueError):
        return fallback_region_geojson(), "Bundled simplified France region boundaries", True


def format_mw(value: float) -> str:
    return f"{value:,.0f} MW"


def selected_location(event: object) -> str | None:
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    points = getattr(selection, "points", None) if selection is not None else None
    if points is None and isinstance(event, dict):
        points = event.get("selection", {}).get("points", [])
    if not points:
        return None
    point = points[0]
    if isinstance(point, dict):
        value = point.get("location")
        if value:
            return str(value)
        customdata = point.get("customdata") or []
        if customdata:
            return str(customdata[0])
    return None


def main_source(row: pd.Series) -> tuple[str, float]:
    values = {
        label: float(row.get(column, 0) or 0)
        for column, label in MIX_COLUMNS.items()
    }
    label = max(values, key=values.get)
    return label, values[label]


def mix_sentence(row: pd.Series) -> str:
    lead, value = main_source(row)
    renewable = float(row.get("renewable_share", 0) or 0)
    fossil = float(row.get("fossil_share", 0) or 0)
    return (
        f"{lead} is the largest measured source at {format_mw(value)}. "
        f"Renewables are {renewable:.0%} of local measured production; fossil output is {fossil:.0%}."
    )


def interpretation(row: pd.Series, frame: pd.DataFrame) -> tuple[str, str]:
    anomaly = float(row.get("demand_anomaly_pct", 0) or 0)
    label = str(row.get("demand_anomaly_label", "Normal"))
    renewable = float(row.get("renewable_share", 0) or 0)
    co2 = float(row.get("co2_intensity_g_per_kwh", 0) or 0)
    if anomaly >= 0.15:
        title = "Demand is above comparable history"
        detail = f"This region is {label.lower()} for the same season, day type, and local hour."
    elif anomaly <= -0.10:
        title = "Demand is below comparable history"
        detail = f"This region is {label.lower()} for the same season, day type, and local hour."
    else:
        title = "Demand is close to usual"
        detail = "Comparable regional history puts this snapshot in the normal range."
    if renewable >= 0.45 and co2 <= 45:
        detail = f"{detail} Renewable output is strong and the CO2 signal is relatively low."
    elif co2 >= 70:
        detail = f"{detail} The CO2 signal is worth watching in this snapshot."
    return title, detail


st.markdown('<div class="ep-eyebrow">Regional electricity map</div>', unsafe_allow_html=True)
st.markdown('<div class="ep-hero">Live grid detail</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ep-subtitle">Regional demand anomaly and local generation mix from RTE eco2mix, with historical fallback context.</div>',
    unsafe_allow_html=True,
)

regional, regional_history, data_status, demo_data = load_regional_data(settings.history_hours)
regional = add_regional_anomalies(regional, regional_history, timezone=settings.timezone)
regions_geojson, geo_status, demo_geo = load_regions()
status_badge(settings.app_mode_label, mode_badge_color())
status_badge(data_status, "grey" if demo_data else "blue")
status_badge(geo_status, "grey" if demo_geo else "blue")

latest_ts = regional["timestamp"].max()
local_ts = latest_ts.tz_convert(settings.timezone) if latest_ts.tzinfo else latest_ts
peak_region = regional.loc[regional["consumption_mw"].idxmax()]
renewable_region = regional.loc[regional["renewable_share"].idxmax()]

section_header(
    "Map",
    "Regional demand anomaly",
    "Color shows each region's demand against comparable history for the same season, day type, and local hour.",
)

left, right = st.columns([1.65, 1], gap="large")
with left:
    viz_note(
        "Regional demand map",
        "This map compares regional demand with comparable history. Use labels and hover details to discuss demand, local generation, and renewable share without relying only on color.",
        source="RTE / ODRE + data.gouv.fr",
    )
    event = st.plotly_chart(
        regional_demand_choropleth(regional, regions_geojson),
        key="regional_demand_map",
        width="stretch",
        on_select="rerun",
        selection_mode="points",
    )
    event_code = selected_location(event)
    if event_code in set(regional["region_code"]):
        st.session_state["selected_region_code"] = event_code

with right:
    codes = regional["region_code"].tolist()
    stored_code = st.session_state.get("selected_region_code")
    if stored_code not in codes:
        stored_code = str(peak_region["region_code"])
        st.session_state["selected_region_code"] = stored_code
    labels = regional.set_index("region_code")["region_display"].to_dict()
    selected_code = st.selectbox(
        "Selected region",
        codes,
        index=codes.index(stored_code),
        format_func=lambda code: labels.get(code, code),
    )
    st.session_state["selected_region_code"] = selected_code
    selected = regional.loc[regional["region_code"] == selected_code].iloc[0]
    title, detail = interpretation(selected, regional)

    metric_card(
        "Demand",
        format_mw(float(selected["consumption_mw"])),
        str(selected["demand_anomaly_label"]),
        icon="Demand",
    )
    metric_card(
        "Local generation",
        format_mw(float(selected["total_production_mw"])),
        mix_sentence(selected),
        icon="Mix",
    )
    explanation_card(
        title,
        detail,
        label=str(selected["region_display"]),
        status=status_label(score_status(float(selected.get("demand_anomaly_score", 0.5) or 0.5))),
    )

section_header("Snapshot", "Regional highlights")
cols = st.columns(4)
with cols[0]:
    metric_card("Last update", f"{local_ts:%H:%M}", f"{local_ts:%d %b %Y}, Europe/Paris.", icon="Now")
with cols[1]:
    metric_card(
        "Peak demand",
        str(peak_region["region_display"]),
        format_mw(float(peak_region["consumption_mw"])),
        icon="Peak",
    )
with cols[2]:
    metric_card(
        "Renewable leader",
        str(renewable_region["region_display"]),
        f"{float(renewable_region['renewable_share']):.0%} renewable share.",
        icon="RES",
    )
with cols[3]:
    metric_card(
        "Coverage",
        f"{len(regional)} regions",
        "Live or fallback regional records joined to region geometry.",
        icon="Map",
    )

with st.expander("Regional values", expanded=False):
    display = regional[
        [
            "region_display",
            "consumption_mw",
            "total_production_mw",
            "renewable_share",
            "co2_intensity_g_per_kwh",
            "demand_anomaly_label",
            "demand_anomaly_percentile",
        ]
    ].rename(
        columns={
            "region_display": "Region",
            "consumption_mw": "Demand MW",
            "total_production_mw": "Production MW",
            "renewable_share": "Renewable share",
            "co2_intensity_g_per_kwh": "CO2 g/kWh",
            "demand_anomaly_label": "Demand vs usual",
            "demand_anomaly_percentile": "Demand percentile",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)

sources = source_attribution()
st.caption(
    "Sources: regional RTE eco2mix via "
    f"[ODRE]({sources['regional_eco2mix']}); French administrative region geometry via "
    f"[data.gouv.fr/API Geo]({sources['regional_geojson']}). "
    "Offline fallback boundaries use simplified IGN/INSEE-derived GeoJSON from "
    f"[france-geojson]({sources['regional_geojson_fallback']}). "
    "If regional electricity data is unavailable, this page uses a historical sample snapshot."
)
