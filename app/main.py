from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.charts import consumption_chart, mix_donut, production_area_chart
from app.components.cards import (
    driver_card,
    explanation_card,
    hackathon_footer,
    horizon_forecast_card,
    message_box,
    metric_card,
    section_header,
    source_badges,
    status_badge,
    story_steps,
    viz_note,
)
from app.components.data_quality import render_data_quality
from app.components.deployment_health import render_deployment_health
from app.components.energy_weather import (
    build_energy_weather_timeline,
    energy_weather_heatmap,
    summarize_energy_weather,
)
from app.components.layout import apply_theme
from app.components.mood_explanation import render_mood_explanation
from app.components.weather_context import render_weather_context
from src.artifact_contract import demo_blocking_message, validate_demo_bundle
from src.config import settings
from src.data_processing.clean_energy_mix import clean_energy_mix
from src.data_processing.features import add_time_features
from src.data_processing.storage import PartitionedParquetStore
from src.data_processing.weather_features import join_energy_weather
from src.data_sources.ecowatt import load_ecowatt_window, source_attribution, status_at
from src.data_sources.rte_eco2mix import Eco2MixError, fetch_eco2mix, load_cached_eco2mix
from src.demo_mode import (
    demo_ecowatt,
    demo_energy,
    demo_model_evaluation,
    demo_mood_artifact,
    demo_weather,
    external_api_enabled,
    mode_badge_color,
)
from src.models.mood_calibration import FIXED_THRESHOLDS, classify_mood

st.set_page_config(page_title="Energy Pulse France", page_icon=":zap:", layout="wide")
apply_theme()

if settings.is_demo_mode:
    demo_checks = validate_demo_bundle(log=True)
    blocking = demo_blocking_message(demo_checks)
    if blocking:
        st.error("Required demo artifacts are not ready.")
        st.caption(blocking)
        st.code("python -m scripts.export_demo_bundle")
        st.stop()


@st.cache_data(ttl=900, show_spinner=False)
def load_data(hours: int) -> tuple[pd.DataFrame, str]:
    """Prefer a fresh official observation, then explicit local fallbacks."""
    if settings.is_demo_mode and not external_api_enabled():
        demo = demo_energy()
        if not demo.empty:
            return demo, "Demo energy sample"
        return pd.DataFrame(), "Demo energy sample unavailable"

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    try:
        raw = fetch_eco2mix(start=start, end=end)
        clean = add_time_features(clean_energy_mix(raw), settings.timezone)
        PartitionedParquetStore(settings.energy_store_dir).upsert(clean)
        return clean, "Official RTE eco2mix, refreshed from ODRE"
    except (Eco2MixError, OSError):
        stored = PartitionedParquetStore(settings.energy_store_dir).read(start=start, end=end)
        if not stored.empty:
            return stored, "Official RTE eco2mix, local partitioned snapshot"
        raw = load_cached_eco2mix()
        return clean_energy_mix(raw), "Official RTE eco2mix, cached raw snapshot"


@st.cache_data(ttl=900, show_spinner=False)
def load_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if settings.is_demo_mode and not external_api_enabled():
        return demo_weather(start, end)
    if not settings.weather_features_path.exists():
        return pd.DataFrame()
    weather = pd.read_parquet(settings.weather_features_path)
    return weather.loc[weather["timestamp"].between(start, end)].copy()


@st.cache_data(ttl=900, show_spinner=False)
def load_ecowatt(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    if settings.is_demo_mode and not external_api_enabled():
        return demo_ecowatt(start, end)
    try:
        return load_ecowatt_window(start, end, timezone_name=settings.timezone)
    except (OSError, ValueError) as exc:
        return pd.DataFrame(), f"EcoWatt unavailable: {exc}"


@st.cache_data(ttl=900, show_spinner=False)
def load_model_evaluation() -> dict[str, Any]:
    if settings.is_demo_mode:
        return demo_model_evaluation()
    path = settings.processed_dir / "demand_model" / "evaluation.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def mood_artifact() -> tuple[dict[str, Any], str]:
    if settings.is_demo_mode:
        artifact = demo_mood_artifact()
        if artifact:
            return artifact, "demo calibration"
    if settings.mood_artifact_path.exists():
        try:
            payload = json.loads(settings.mood_artifact_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload, "calibrated"
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {
        "timezone": settings.timezone,
        "min_sample": 1,
        "segments": [],
        "fixed_thresholds": FIXED_THRESHOLDS,
        "precedence": ["Carbon-heavy", "Tense", "Renewable-rich", "Calm"],
        "source": {"name": "Explicit fixed-threshold fallback"},
        "generated_at": None,
    }, "fixed fallback"


def format_mw(value: float) -> str:
    return f"{value:,.0f} MW"


def weather_summary(weather: pd.DataFrame) -> tuple[str, str, dict[str, float] | None]:
    if weather.empty:
        return "Not cached", "Weather context is available after running the weather pipeline.", None
    latest = weather.sort_values("timestamp").iloc[-1]
    temp = float(latest.get("weather_temperature_c", 0))
    wind = float(latest.get("weather_wind_speed_kmh", 0))
    cloud = float(latest.get("weather_cloud_cover_pct", 0))
    if temp <= 5:
        headline = "Cold lift"
        detail = f"Cold weather can lift heating demand. Latest national proxy: {temp:.1f} C."
    elif temp >= 27:
        headline = "Heat lift"
        detail = f"Hot weather can lift cooling demand. Latest national proxy: {temp:.1f} C."
    elif wind >= 35:
        headline = "Windy"
        detail = f"Wind is noticeable at {wind:.0f} km/h, with cloud cover near {cloud:.0f}%."
    else:
        headline = "Mild"
        detail = f"Weather pressure looks moderate: {temp:.1f} C, wind {wind:.0f} km/h."
    return headline, detail, {"temp": temp, "wind": wind, "cloud": cloud}


def compact_mood_reason(reason: str) -> str:
    if "CO" in reason and "fossil" in reason:
        return "CO2 or fossil output is above its calibrated upper range."
    if "Consumption" in reason:
        return "Demand is above its calibrated high-demand range."
    if "Renewable" in reason:
        return "Renewables are high and CO2 intensity is low."
    return reason


def confidence_parts(text: str) -> tuple[str, str, str]:
    if ":" in text:
        label, detail = text.split(":", 1)
        label = label.strip()
        detail = detail.strip()
    else:
        label, detail = "Unknown", text
    status = "Watch" if label.lower().startswith(("low", "unknown")) else "Comfortable"
    return label, detail, status


def ecowatt_card_parts(signal: dict[str, Any], source_status: str) -> tuple[str, str, str]:
    status = str(signal.get("ecowatt_status", "unknown"))
    label = str(signal.get("ecowatt_label", "Unknown"))
    source = str(signal.get("ecowatt_source", source_status))
    if status == "green":
        detail = "EcoWatt is the official electricity weather signal. Current signal is normal."
    elif status == "orange":
        detail = "EcoWatt is the official electricity weather signal. The grid is tense."
    elif status == "red":
        detail = "EcoWatt is the official electricity weather signal. The grid is very tense."
    else:
        detail = "EcoWatt is the official electricity weather signal. No current signal is available."
    if source and source != "Unavailable":
        detail = f"{detail} Source: {source}."
    return label, detail, status


def demand_pressure(value: float, quantiles: pd.Series) -> tuple[str, str, float]:
    q25 = float(quantiles.loc[0.25])
    q60 = float(quantiles.loc[0.60])
    q85 = float(quantiles.loc[0.85])
    if value >= q85:
        return "High", "#ef4444", 1.0
    if value >= q60:
        return "Elevated", "#f59e0b", 0.72
    if value <= q25:
        return "Light", "#0284c7", 0.28
    return "Normal", "#10b981", 0.5


def build_pressure_forecast(data: pd.DataFrame, latest_ts: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    frame = data[["timestamp", "consumption_mw"]].dropna().sort_values("timestamp").copy()
    quantiles = frame["consumption_mw"].quantile([0.25, 0.60, 0.85])
    start = latest_ts.floor("h") + pd.Timedelta(hours=1)
    targets = pd.DataFrame({"target": pd.date_range(start=start, periods=24, freq="h")})
    targets["reference_timestamp"] = targets["target"] - pd.Timedelta(hours=24)
    reference = frame.rename(columns={"timestamp": "reference_timestamp"})
    forecast = pd.merge_asof(
        targets.sort_values("reference_timestamp"),
        reference,
        on="reference_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=45),
    ).sort_values("target")
    source = "Yesterday's same-hour demand, translated into pressure bands."
    if forecast["consumption_mw"].isna().any():
        recent = frame.tail(24).reset_index(drop=True)
        forecast["consumption_mw"] = forecast["consumption_mw"].fillna(
            pd.Series(recent["consumption_mw"].tolist() * 2).iloc[: len(forecast)].to_numpy()
        )
        source = "Recent demand pattern, translated into pressure bands."
    pressure = forecast["consumption_mw"].apply(lambda value: demand_pressure(float(value), quantiles))
    forecast[["pressure", "color", "height"]] = pd.DataFrame(pressure.tolist(), index=forecast.index)
    return forecast, source


def pressure_timeline(forecast: pd.DataFrame) -> go.Figure:
    labels = forecast["target"].dt.tz_convert(settings.timezone).dt.strftime("%H:%M")
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=forecast["height"],
            marker_color=forecast["color"],
            customdata=forecast[["pressure", "consumption_mw"]],
            hovertemplate="<b>%{x}</b><br>Pressure: %{customdata[0]}<br>Reference: %{customdata[1]:,.0f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None,
        yaxis=dict(visible=False, range=[0, 1.08]),
        hovermode="x",
        showlegend=False,
    )
    return fig


def render_forecast_cards(forecast: pd.DataFrame) -> None:
    local = forecast.copy()
    local["local_time"] = local["target"].dt.tz_convert(settings.timezone)
    local["hour_label"] = local["local_time"].dt.strftime("%H:%M")
    first = local.iloc[0]
    first_six = local.head(6)
    peak = local.loc[local["height"].idxmax()]
    easy = local.loc[local["height"].idxmin()]

    cols = st.columns(4)
    with cols[0]:
        horizon_forecast_card(
            "Next hour",
            str(first["pressure"]),
            f"{first['hour_label']} local time, based on nearby recent demand.",
            status=str(first["pressure"]),
        )
    with cols[1]:
        six_peak = first_six.loc[first_six["height"].idxmax()]
        horizon_forecast_card(
            "Next 6 hours",
            str(six_peak["pressure"]),
            f"Highest pressure in this window appears around {six_peak['hour_label']}.",
            status=str(six_peak["pressure"]),
        )
    with cols[2]:
        horizon_forecast_card(
            "Peak watch",
            str(peak["pressure"]),
            f"The strongest pressure signal appears around {peak['hour_label']}.",
            status=str(peak["pressure"]),
        )
    with cols[3]:
        horizon_forecast_card(
            "Softer window",
            str(easy["pressure"]),
            f"The lightest pressure signal appears around {easy['hour_label']}.",
            status=str(easy["pressure"]),
        )


def render_forecast_check(payload: dict[str, Any]) -> None:
    comparisons = pd.DataFrame(payload.get("baseline_comparison", []))
    if comparisons.empty:
        message_box(
            "Forecast check unavailable",
            "Run the forecast evaluation to compare the app outlook with simple reference rules.",
            kind="info",
        )
        return

    comparisons = comparisons.sort_values("horizon_hours")
    wins = comparisons.loc[comparisons["improvement_vs_strongest_baseline_percent"] > 0]
    misses = comparisons.loc[comparisons["improvement_vs_strongest_baseline_percent"] <= 0]
    win_text = ", ".join(f"{int(row.horizon_hours)}h" for row in wins.itertuples()) or "none yet"
    miss_text = ", ".join(f"{int(row.horizon_hours)}h" for row in misses.itertuples()) or "none"
    if not wins.empty and misses.empty:
        status = "Good"
    elif not wins.empty:
        status = "Watch"
    else:
        status = "Needs work"
    explanation_card(
        "Forecast check",
        f"Stronger than simple reference rules at: {win_text}. Needs more work at: {miss_text}.",
        label="Reliability",
        status=status,
    )


def render_how_it_works() -> None:
    with st.expander("How it works: data, model, explainability, limitations", expanded=False):
        st.markdown(
            """
            <div class="ep-how-grid">
              <div class="ep-how-item">
                <div class="ep-label">Data</div>
                <div class="ep-title">Official grid signals plus context</div>
                <div class="ep-detail">RTE/ODRE electricity observations anchor the app. Open-Meteo weather features, EcoWatt status, public catalog metadata, and transparent appliance assumptions add context.</div>
              </div>
              <div class="ep-how-item">
                <div class="ep-label">Model</div>
                <div class="ep-title">Measured patterns before black boxes</div>
                <div class="ep-detail">The 24-hour outlook uses the best available signal: fresh model points, RTE forecast fields when present, or recent same-hour behavior. Every fallback is shown.</div>
              </div>
              <div class="ep-how-item">
                <div class="ep-label">Explainability</div>
                <div class="ep-title">Status labels explain the action</div>
                <div class="ep-detail">Each hour is labeled Comfortable, Watch, Tense, or Low-carbon opportunity. Hover details show demand, CO2 intensity, EcoWatt, and the source used.</div>
              </div>
              <div class="ep-how-item">
                <div class="ep-label">Limitations</div>
                <div class="ep-title">Decision support, not operations</div>
                <div class="ep-detail">This is a hackathon demo and public education tool. It is not an RTE operational forecast, tariff signal, or verified carbon accounting method.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_why_it_matters() -> None:
    st.markdown(
        """
        <div class="ep-why-grid">
          <div class="ep-why-item">
            <div class="ep-label">Demand peaks</div>
            <div class="ep-title">Small timing choices add up</div>
            <div class="ep-detail">Even flexible household loads matter when many people move them away from evening pressure.</div>
          </div>
          <div class="ep-why-item">
            <div class="ep-label">Grid tension</div>
            <div class="ep-title">Not every hour has the same risk</div>
            <div class="ep-detail">EcoWatt and demand pressure make tense hours visible before the user chooses an action.</div>
          </div>
          <div class="ep-why-item">
            <div class="ep-label">Low-carbon timing</div>
            <div class="ep-title">Cleaner hours should be legible</div>
            <div class="ep-detail">The app separates demand pressure from CO2 intensity so the best shift window is easier to explain.</div>
          </div>
          <div class="ep-why-item">
            <div class="ep-label">Citizen action</div>
            <div class="ep-title">From signal to behavior</div>
            <div class="ep-detail">The simulator turns a grid signal into a concrete choice: when to run flexible appliances.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    with st.spinner("Connecting to the French grid data feed..."):
        data, source_status = load_data(settings.history_hours)
except (Eco2MixError, FileNotFoundError, ValueError) as exc:
    st.error(f"No official energy data is available yet: {exc}")
    st.code("python -m scripts.update_data --hours 72")
    st.stop()

if data.empty:
    st.error("No demo energy sample is available. Export the demo bundle before deployment.")
    st.code("python -m scripts.export_demo_bundle")
    st.stop()

data = data.sort_values("timestamp")
latest = data.iloc[-1]
artifact, calibration_status = mood_artifact()
mood = classify_mood(latest.to_dict(), artifact)
local_time = latest["timestamp"].tz_convert(settings.timezone)
weather = load_weather(data["timestamp"].min(), data["timestamp"].max())
weather_headline, weather_detail, weather_values = weather_summary(weather)
model_payload = load_model_evaluation()
ecowatt_start = latest["timestamp"].floor("h") - pd.Timedelta(hours=1)
ecowatt_end = latest["timestamp"].floor("h") + pd.Timedelta(hours=25)
ecowatt, ecowatt_source_status = load_ecowatt(ecowatt_start, ecowatt_end)
current_ecowatt = status_at(ecowatt, latest["timestamp"])
ecowatt_label, ecowatt_detail, ecowatt_status = ecowatt_card_parts(
    current_ecowatt, ecowatt_source_status
)
energy_weather = build_energy_weather_timeline(
    data,
    latest_ts=latest["timestamp"],
    model_payload=model_payload,
    mood_artifact=artifact,
    ecowatt=ecowatt,
    timezone=settings.timezone,
)
energy_summary = summarize_energy_weather(energy_weather.timeline, energy_weather.metadata)
confidence_label, confidence_detail, confidence_status = confidence_parts(energy_summary["confidence"])
shift_status = "Unknown" if energy_summary["shift"].startswith("None visible") else "Low-carbon opportunity"
avoid_status = "Comfortable" if energy_summary["avoid"].startswith("None visible") else "Tense"

render_deployment_health(
    data=data,
    source_status=source_status,
    weather=weather,
    ecowatt=ecowatt,
    model_payload=model_payload,
    calibration_status=calibration_status,
)

st.markdown('<div class="ep-eyebrow">France electricity weather</div>', unsafe_allow_html=True)
st.markdown('<div class="ep-hero">Energy Pulse France</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ep-subtitle">A three-minute jury story: observe the live grid, understand what is moving demand, and act by shifting flexible use to better hours.</div>',
    unsafe_allow_html=True,
)
status_badge(source_status, "blue")
status_badge(settings.app_mode_label, mode_badge_color())
status_badge(ecowatt_source_status, "blue" if not ecowatt.empty else "grey")
if settings.is_demo_mode and ecowatt.empty:
    message_box("EcoWatt status", ecowatt_source_status, kind="warning")
source_badges(
    [
        ("RTE / ODRÉ", "eco2mix + EcoWatt"),
        ("Open-Meteo", "weather context"),
        ("ADEME", "appliance assumptions"),
        ("data.gouv.fr", "public catalog"),
    ]
)
if settings.is_demo_mode:
    message_box(
        "Demo data mode is active",
        "This public deployment is using the committed demo_data bundle and does not call external APIs unless DEMO_ALLOW_EXTERNAL_API=1 is set.",
        kind="info",
    )

section_header("Demo narrative", "Observe, understand, act")
story_steps(
    [
        (
            "Observe",
            "What is the grid doing?",
            "Start with live demand, CO2 intensity, EcoWatt, weather, and the current grid mood.",
        ),
        (
            "Understand",
            "Why is demand changing?",
            "Connect the pulse to weather, recent demand pressure, generation mix, and calibrated thresholds.",
        ),
        (
            "Act",
            "When should we shift usage?",
            "Use the next 24 hours to spot easier and lower-carbon windows for flexible appliances.",
        ),
    ]
)

render_how_it_works()

section_header(
    "Why it matters",
    "From grid tension to citizen action",
    "The demo links national electricity signals to choices that a household, campus, or city can understand quickly.",
)
render_why_it_matters()

section_header(
    "Observe",
    "Current grid pulse",
    "These cards answer the first jury question: what is happening on the French electricity system right now?",
)
cols = st.columns(6)
with cols[0]:
    metric_card("Demand", format_mw(float(latest["consumption_mw"])), "How much power France is using now.", icon="kW")
with cols[1]:
    metric_card(
        "CO2 intensity",
        f"{float(latest['co2_intensity_g_per_kwh']):,.0f} g/kWh",
        "Carbon signal from the official source.",
        icon="CO2",
    )
with cols[2]:
    metric_card(
        "Grid mood",
        str(mood["mood"]),
        compact_mood_reason(str(mood["reason"])),
        icon="Pulse",
        status=str(mood["mood"]),
    )
with cols[3]:
    metric_card("EcoWatt", ecowatt_label, ecowatt_detail, icon="EW", status=ecowatt_status)
with cols[4]:
    metric_card("Weather influence", weather_headline, weather_detail, icon="Met")
with cols[5]:
    metric_card("Last update", f"{local_time:%H:%M}", f"{local_time:%d %b %Y}, Europe/Paris.", icon="Now")

section_header(
    "Act",
    "24h Energy Weather",
    "A public-friendly outlook for normal use, flexible shifting, and careful hours. Text labels are shown on the chart so the meaning is not color-only.",
)
summary_cols = st.columns(3)
with summary_cols[0]:
    horizon_forecast_card(
        "Best hours to shift usage",
        energy_summary["shift"],
        "Prefer these for flexible appliances when practical.",
        status=shift_status,
    )
with summary_cols[1]:
    horizon_forecast_card(
        "Hours to avoid",
        energy_summary["avoid"],
        "These show the strongest watch or tense signal.",
        status=avoid_status,
    )
with summary_cols[2]:
    horizon_forecast_card(
        "Model confidence / uncertainty",
        confidence_label,
        confidence_detail,
        status=confidence_status,
    )
viz_note(
    "24-hour energy weather map",
    "Read across the hours: the app row combines demand and CO2 context; the EcoWatt row shows the official electricity-weather signal when available.",
    source="RTE / ODRE + model context",
)
st.plotly_chart(energy_weather_heatmap(energy_weather.timeline), width="stretch")
message_box(
    "How to read this",
    "Comfortable means normal use; Low-carbon opportunity is a better shift window; "
    "Watch and Tense mean it is worth avoiding flexible demand where practical. "
    "The EcoWatt row is the official electricity weather signal when public data is available. "
    "This is a demo outlook, not an RTE operational forecast.",
    kind="info",
)
with st.expander("Advanced values behind the 24h Energy Weather", expanded=False):
    viz_note(
        "Technical values behind each hour",
        "This table exposes the exact demand, CO2, threshold, model, and source fields used to create the public labels.",
        source="Traceability",
    )
    advanced_columns = [
        "target",
        "status",
        "ecowatt_status",
        "ecowatt_label",
        "ecowatt_source",
        "demand_signal_mw",
        "demand_source",
        "model_predicted_mw",
        "rte_forecast_mw",
        "reference_consumption_mw",
        "co2_intensity_g_per_kwh",
        "co2_source",
        "threshold_source",
        "demand_high_threshold_mw",
        "co2_low_threshold",
        "co2_high_threshold",
    ]
    available_columns = [column for column in advanced_columns if column in energy_weather.timeline]
    st.dataframe(energy_weather.timeline[available_columns], width="stretch", hide_index=True)
    st.json(energy_weather.metadata)

section_header(
    "Understand",
    "What is moving the pulse?",
    "These drivers explain the live mood in plain language before anyone has to inspect the raw data.",
)
history_quantiles = data["consumption_mw"].quantile([0.25, 0.60, 0.85])
current_pressure, _, _ = demand_pressure(float(latest["consumption_mw"]), history_quantiles)
renewable_share = float(latest.get("renewable_share", 0))
co2_intensity = float(latest.get("co2_intensity_g_per_kwh", 0))
driver_cols = st.columns(4)
with driver_cols[0]:
    driver_card(
        "Use",
        f"Demand is {current_pressure.lower()}",
        f"Current use is {format_mw(float(latest['consumption_mw']))}, compared with the recent range.",
    )
with driver_cols[1]:
    driver_card("Met", weather_headline, weather_detail)
with driver_cols[2]:
    driver_card(
        "Mix",
        "Clean supply is visible",
        f"Renewables are contributing {renewable_share:.1%} of measured domestic generation.",
    )
with driver_cols[3]:
    driver_card(
        "CO2",
        "Carbon signal stays explicit",
        f"The latest source intensity is {co2_intensity:,.0f} g/kWh, shown separately from demand.",
    )

section_header("Action", "What can I do?")
explanation_card(
    "Try shifting flexible demand away from high-pressure hours.",
    "Choose an appliance, move it on the timeline, and compare the pressure signal once the simulator is connected.",
    label="Demand shifting",
    icon="Tune",
)
st.page_link("pages/7_demand_shifting_simulator.py", label="Open demand-shifting simulator", icon=":material/tune:")

section_header("Trust", "Forecast check")
render_forecast_check(model_payload)

with st.expander("Advanced / Data Science", expanded=False):
    st.write("Deep-dive pages for the technical jury and for continuing development.")
    link_cols = st.columns(3)
    with link_cols[0]:
        st.page_link("pages/1_live_grid.py", label="Live grid detail")
        st.page_link("pages/2_forecast.py", label="Forecast workspace")
    with link_cols[1]:
        st.page_link("pages/3_explainability.py", label="Explainability")
        st.page_link("pages/4_historical.py", label="Historical grid")
    with link_cols[2]:
        st.page_link("pages/5_baselines.py", label="Demand baselines")
        st.page_link("pages/6_demand_model.py", label="Demand model")

    st.divider()
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Demand pulse detail")
        viz_note(
            "Recent demand trend",
            "This line shows how national consumption has moved over the recent window, making peaks and recovery periods visible.",
            source="RTE / ODRE",
        )
        st.plotly_chart(consumption_chart(data), width="stretch")
        st.subheader("What powers France")
        viz_note(
            "Generation mix over time",
            "The stacked area chart shows which production sources are contributing as demand changes.",
            source="RTE / ODRE",
        )
        st.plotly_chart(production_area_chart(data), width="stretch")
    with right:
        st.subheader("Latest energy mix")
        viz_note(
            "Current production share",
            "The donut summarizes the latest measured mix so the CO2 signal can be discussed separately from demand.",
            source="RTE / ODRE",
        )
        st.plotly_chart(mix_donut(latest), width="stretch")

    if weather.empty:
        st.info("Weather context is not cached yet. Run `python -m scripts.fetch_weather --start ... --end ...`.")
    else:
        render_weather_context(weather)
        joined = join_energy_weather(data[["timestamp", "consumption_mw"]], weather)
        overlap = joined[["consumption_mw", "weather_temperature_c"]].dropna()
        if len(overlap) >= 4:
            st.caption(
                "Recent demand/temperature correlation over aligned observations: "
                f"{overlap.corr().iloc[0, 1]:+.2f}. This is descriptive, not causal."
            )

    render_mood_explanation(mood, artifact)
    st.caption(f"Mood thresholds: {calibration_status}.")
    with st.expander("Data quality and freshness", expanded=False):
        render_data_quality(data)

sources = source_attribution()
st.caption(
    "Data sources: RTE eco2mix via ODRE; EcoWatt public history via "
    f"[ODRE]({sources['current_history']}) / [data.gouv.fr]({sources['data_gouv']}); "
    f"legacy EcoWatt via [ODRE]({sources['legacy_history']}); optional live EcoWatt via "
    f"[RTE API]({sources['rte_live_api']})."
)
hackathon_footer(
    project="Energy Pulse France",
    team="ENGIE hackathon team",
    detail="Streamlit demo for electricity-weather awareness, explainable demand context, and practical load shifting.",
)
