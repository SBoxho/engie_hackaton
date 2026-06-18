"""Explainable forecast cockpit page."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.cards import (
    driver_card,
    explanation_card,
    horizon_forecast_card,
    message_box,
    metric_card,
    section_header,
    viz_note,
)
from app.components.charts import dark_chart_layout
from app.components.energy_weather import build_energy_weather_timeline, energy_weather_heatmap, summarize_energy_weather
from app.components.layout import apply_theme
from src.config import settings
from src.data_processing.clean_energy_mix import clean_energy_mix
from src.data_processing.features import add_time_features
from src.data_processing.storage import PartitionedParquetStore
from src.data_sources.ecowatt import load_cached_ecowatt
from src.data_sources.rte_eco2mix import load_cached_eco2mix
from src.demo_mode import demo_ecowatt, demo_energy, demo_model_evaluation, demo_mood_artifact

st.set_page_config(page_title="Forecast Cockpit", page_icon=":material/insights:", layout="wide")
apply_theme()

BASELINE_LABELS = {
    "persistence": "Persistence",
    "day_naive": "Previous day",
    "week_naive": "Previous week",
    "rte_forecast": "RTE J/J-1 forecast",
}


def _format_mw(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return "n/a" if pd.isna(numeric) else f"{float(numeric):,.0f} MW"


def _format_pct(value: Any, *, signed: bool = False) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):+,.1f}%" if signed else f"{float(numeric):,.1f}%"


@st.cache_data(ttl=900, show_spinner=False)
def load_grid_context() -> tuple[pd.DataFrame, str]:
    if settings.is_demo_mode:
        frame = demo_energy()
        return frame.sort_values("timestamp"), "Demo energy sample"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    try:
        store_frame = PartitionedParquetStore(settings.energy_store_dir).read(start=start, end=end)
        if not store_frame.empty:
            return store_frame.sort_values("timestamp"), "Local processed eco2mix snapshot"
    except (OSError, ValueError):
        pass
    try:
        raw = load_cached_eco2mix()
        clean = add_time_features(clean_energy_mix(raw), settings.timezone)
        return clean.sort_values("timestamp"), "Cached raw eco2mix snapshot"
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame(), "Grid context unavailable"


@st.cache_data(ttl=900, show_spinner=False)
def load_ecowatt_context(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    if settings.is_demo_mode:
        return demo_ecowatt(start, end)
    try:
        ecowatt = load_cached_ecowatt(timezone_name=settings.timezone)
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame(), "EcoWatt unavailable offline"
    if ecowatt.empty or "timestamp" not in ecowatt:
        return pd.DataFrame(), "EcoWatt unavailable offline"
    return ecowatt.loc[ecowatt["timestamp"].between(start, end)].copy(), "Cached EcoWatt snapshot"


@st.cache_data(ttl=900, show_spinner=False)
def load_model_payload() -> dict[str, Any]:
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


def load_mood_artifact() -> dict[str, Any] | None:
    if settings.is_demo_mode:
        return demo_mood_artifact() or None
    if not settings.mood_artifact_path.exists():
        return None
    try:
        return json.loads(settings.mood_artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def normalise_predictions(payload: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(payload.get("predictions", []))
    if frame.empty:
        return frame
    for column in ("origin_timestamp", "target_timestamp"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame.dropna(subset=["target_timestamp"]).sort_values("target_timestamp")


def comparison_for(comparisons: pd.DataFrame, horizon: Any) -> pd.Series | None:
    if comparisons.empty or "horizon_hours" not in comparisons:
        return None
    rows = comparisons.loc[comparisons["horizon_hours"].eq(horizon)]
    return None if rows.empty else rows.iloc[0]


def strongest_baseline_column(predictions: pd.DataFrame, comparison: pd.Series | None) -> tuple[str | None, str]:
    candidates: list[str] = []
    if comparison is not None and pd.notna(comparison.get("strongest_baseline")):
        candidates.append(f"{comparison.get('strongest_baseline')}_predicted_mw")
    candidates.extend(["rte_forecast_predicted_mw", "day_naive_predicted_mw", "week_naive_predicted_mw", "persistence_predicted_mw"])
    for column in candidates:
        if column in predictions and predictions[column].notna().any():
            key = column.replace("_predicted_mw", "")
            return column, BASELINE_LABELS.get(key, key.replace("_", " ").title())
    return None, "Baseline"


def forecast_chart(rows: pd.DataFrame, baseline_column: str | None, baseline_label: str) -> go.Figure:
    fig = go.Figure()
    if {"model_interval_lower_mw", "model_interval_upper_mw"}.issubset(rows.columns):
        fig.add_trace(go.Scatter(x=rows["target_timestamp"], y=rows["model_interval_upper_mw"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rows["target_timestamp"], y=rows["model_interval_lower_mw"], mode="lines", fill="tonexty", fillcolor="rgba(59,130,246,.18)", line=dict(width=0), name="Model uncertainty"))
    fig.add_trace(go.Scatter(x=rows["target_timestamp"], y=rows["target_mw"], mode="lines", name="Actual demand", line=dict(color="#f8fafc", width=3)))
    fig.add_trace(go.Scatter(x=rows["target_timestamp"], y=rows["model_predicted_mw"], mode="lines", name="AI forecast", line=dict(color="#38bdf8", width=3)))
    if baseline_column:
        fig.add_trace(go.Scatter(x=rows["target_timestamp"], y=rows[baseline_column], mode="lines", name=baseline_label, line=dict(color="#fbbf24", width=2, dash="dot")))
    fig.update_layout(**dark_chart_layout(height=420, xaxis_title=None, yaxis_title="Demand (MW)", hovermode="x unified", legend_title=None, margin=dict(l=10, r=10, t=20, b=10)))
    return fig


def error_comparison_chart(rows: pd.DataFrame, baseline_column: str | None, baseline_label: str) -> go.Figure:
    model_mae = (rows["target_mw"] - rows["model_predicted_mw"]).abs().mean()
    names = ["AI forecast"]
    values = [model_mae]
    colors = ["#38bdf8"]
    if baseline_column:
        names.append(baseline_label)
        values.append((rows["target_mw"] - rows[baseline_column]).abs().mean())
        colors.append("#fbbf24")
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors, text=[_format_mw(v) for v in values], textposition="outside"))
    fig.update_layout(**dark_chart_layout(height=300, yaxis_title="Recent MAE", xaxis_title=None, showlegend=False, margin=dict(l=10, r=10, t=20, b=10)))
    return fig


def stress_window(timeline: pd.DataFrame) -> pd.Series | None:
    if timeline.empty:
        return None
    ranked = timeline.assign(_rank=timeline["status"].map({"Tense": 4, "Watch": 3, "Comfortable": 2, "Low-carbon opportunity": 1, "Unknown": 0}).fillna(0))
    ranked = ranked.sort_values(["_rank", "demand_signal_mw"], ascending=[False, False])
    return None if ranked.empty else ranked.iloc[0]


def render_driver_cards(row: pd.Series) -> None:
    cards = row.get("explanation_cards") if "explanation_cards" in row else None
    if isinstance(cards, list) and cards:
        cols = st.columns(min(4, len(cards)))
        for card, col in zip(cards[:4], cols):
            with col:
                explanation_card(str(card.get("title", "Forecast driver")), str(card.get("detail", "")), label=str(card.get("family_label", "Driver")), icon=str(card.get("icon", "AI")), status="green" if card.get("direction") == "up" else "blue")
        return
    cols = st.columns(3)
    with cols[0]:
        driver_card("AI", "Model drivers unavailable", "This evaluation artifact does not include local explanation cards for the selected point.")
    with cols[1]:
        driver_card("Δ", "Compare against baseline", "Use the dotted baseline line to separate model signal from simple repeat-pattern forecasts.")
    with cols[2]:
        driver_card("⏱", "Horizon matters", "Select another horizon to see whether errors widen or improve over longer look-ahead windows.")


payload = load_model_payload()
predictions = normalise_predictions(payload)
metrics = pd.DataFrame(payload.get("metrics", []))
comparisons = pd.DataFrame(payload.get("baseline_comparison", []))
grid, grid_source = load_grid_context()

st.title("Explainable Forecast Cockpit")
st.markdown('<div class="ep-subtitle">AI-centred demand forecasting with baseline honesty, pressure windows, and plain-language drivers.</div>', unsafe_allow_html=True)
st.caption(settings.app_mode_label)
if settings.is_demo_mode:
    message_box(
        "Judge mode: demo data",
        "This page is using the committed demo bundle for deterministic forecasting, with live API calls disabled unless DEMO_ALLOW_EXTERNAL_API=1.",
        kind="info",
    )

if predictions.empty or "model_predicted_mw" not in predictions or "target_mw" not in predictions:
    message_box("Forecast artifact missing", "Run the demand-model pipeline or export the demo bundle to unlock the cockpit. The page is stable, but no model rows are available yet.", kind="warning")
    st.code("python -m scripts.build_features && python -m scripts.train_demand_model && python -m scripts.evaluate_demand_model")
    st.stop()

horizons = sorted(predictions["horizon_hours"].dropna().unique()) if "horizon_hours" in predictions else [None]
default_idx = 0
horizon = st.sidebar.selectbox("Forecast horizon", horizons, index=default_idx, format_func=lambda value: f"{int(value)} hours" if pd.notna(value) else "All")
selected = predictions.loc[predictions["horizon_hours"].eq(horizon)].copy() if horizon is not None else predictions.copy()
selected = selected.sort_values("target_timestamp").tail(7 * 96)
comparison = comparison_for(comparisons, horizon)
baseline_column, baseline_label = strongest_baseline_column(selected, comparison)
latest = selected.sort_values("target_timestamp").iloc[-1]

start = grid["timestamp"].min() if not grid.empty and "timestamp" in grid else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
end = grid["timestamp"].max() if not grid.empty and "timestamp" in grid else pd.Timestamp.now(tz="UTC")
ecowatt, ecowatt_source = load_ecowatt_context(start, end + pd.Timedelta(days=2))
energy_weather = build_energy_weather_timeline(grid, latest_ts=end, model_payload=payload, mood_artifact=load_mood_artifact(), ecowatt=ecowatt, timezone=settings.timezone)
summary = summarize_energy_weather(energy_weather.timeline, energy_weather.metadata)
stress = stress_window(energy_weather.timeline)

improvement = comparison.get("improvement_vs_strongest_baseline_percent") if comparison is not None else None
badge = "beats baseline" if pd.notna(improvement) and float(improvement) > 0 else "experimental horizon"

hero = st.columns([1.1, 1.1, 1.2, 1.2])
with hero[0]:
    metric_card("Latest AI forecast", _format_mw(latest.get("model_predicted_mw")), f"{int(horizon)}h horizon for {latest['target_timestamp'].strftime('%d %b %H:%M UTC')}", icon="AI", status=badge)
with hero[1]:
    metric_card("Actual demand", _format_mw(latest.get("target_mw")), "Most recent evaluated target in the artifact.", icon="MW")
with hero[2]:
    metric_card("Baseline edge", _format_pct(improvement, signed=True), f"Against {baseline_label} on the chronological test split.", icon="Δ", status=badge)
with hero[3]:
    if stress is not None:
        metric_card("Next stress window", str(stress.get("status", "Unknown")), f"Around {stress.get('hour_label', '--:--')} · {_format_mw(stress.get('demand_signal_mw'))} demand signal.", icon="⚡", status=str(stress.get("status", "Unknown")))
    else:
        metric_card("Next stress window", "Unknown", "No 24-hour pressure timeline is available.", icon="⚡", status="unknown")

section_header("Forecast", "Actuals versus AI forecast", "The top chart keeps the jury story simple: actual demand, AI forecast, uncertainty band, and the strongest available simple reference.")
st.plotly_chart(forecast_chart(selected, baseline_column, baseline_label), width="stretch")

left, right = st.columns([1.2, 1])
with left:
    viz_note("Model versus baseline", "Lower error is better. This mini backtest uses the visible rows for the selected horizon.", source="Demand model evaluation")
    st.plotly_chart(error_comparison_chart(selected, baseline_column, baseline_label), width="stretch")
with right:
    section_header("Pressure", "Next grid stress period", "The timeline uses fresh model points where possible, then RTE forecast or recent same-hour demand as graceful fallbacks.")
    if stress is not None:
        horizon_forecast_card("Peak pressure watch", str(stress.get("status", "Unknown")), f"{stress.get('hour_label', '--:--')} local · source: {stress.get('demand_source', 'Unavailable')}", status=str(stress.get("status", "Unknown")))
    st.plotly_chart(energy_weather_heatmap(energy_weather.timeline), width="stretch")
    st.caption(f"Shift-friendly windows: {summary['shift']} · Avoid if practical: {summary['avoid']}")

section_header("Explainability", "Top drivers in human language", payload.get("explanation_disclaimer", "Explanations are approximate and model-derived; they are not causal explanations."))
render_driver_cards(latest)

section_header("Honesty", "Experimental model status", "A polished forecast is only useful if its limits are visible.")
model_mae = comparison.get("model_mae_mw") if comparison is not None else None
base_mae = comparison.get("strongest_baseline_mae_mw") if comparison is not None else None
honesty_cols = st.columns(4)
honesty_cols[0].metric("Model MAE", _format_mw(model_mae))
honesty_cols[1].metric(f"{baseline_label} MAE", _format_mw(base_mae))
honesty_cols[2].metric("Improvement", _format_pct(improvement, signed=True))
honesty_cols[3].metric("Data source", grid_source)
message_box("Not an operational forecast", "This is an experimental hackathon model trained and evaluated on historical public artefacts. Use it to explain demand pressure and model behaviour, not as an RTE operational instruction, tariff signal, or guaranteed carbon forecast.", kind="warning")

with st.expander("Raw evidence: selected predictions, metrics, and timeline"):
    tabs = st.tabs(["Predictions", "Metrics", "Baseline comparison", "Pressure timeline"])
    tabs[0].dataframe(selected.tail(200), width="stretch", hide_index=True)
    tabs[1].dataframe(metrics.loc[metrics.get("horizon_hours", pd.Series(dtype=float)).eq(horizon)] if not metrics.empty and "horizon_hours" in metrics else metrics, width="stretch", hide_index=True)
    tabs[2].dataframe(comparisons, width="stretch", hide_index=True)
    tabs[3].dataframe(energy_weather.timeline, width="stretch", hide_index=True)
    st.caption(f"Grid: {grid_source} · EcoWatt: {ecowatt_source} · Generated: {payload.get('generated_at', 'unknown')}")
