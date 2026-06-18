from __future__ import annotations

import html
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
    message_box,
    metric_card,
    section_header,
    status_badge_html,
    viz_note,
)
from app.components.charts import dark_chart_layout
from app.components.energy_weather import build_energy_weather_timeline
from app.components.layout import apply_theme
from src.config import settings
from src.data_processing.clean_energy_mix import clean_energy_mix
from src.data_processing.features import add_time_features
from src.data_processing.storage import PartitionedParquetStore
from src.demo_mode import (
    demo_ecowatt,
    demo_energy,
    demo_model_evaluation,
    demo_mood_artifact,
)
from src.data_sources.ecowatt import load_cached_ecowatt
from src.data_sources.rte_eco2mix import load_cached_eco2mix
from src.models.load_shift_simulator import (
    PRESSURE_COLORS,
    PRESSURE_POINTS,
    ShiftAction,
    ShiftScore,
    build_demo_timeline,
    compute_shift_score,
    load_actions,
    load_assumption_config,
    row_for_local_hour,
)

st.set_page_config(
    page_title="Flatten the Peak", page_icon=":material/emoji_events:", layout="wide"
)
apply_theme()


def _format_hour(hour: int) -> str:
    return f"{hour:02d}:00"


def _format_mw(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "Unavailable"
    return f"{float(numeric):,.0f} MW"


def _format_co2(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "Unavailable"
    return f"{float(numeric):,.0f} g/kWh"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mood_artifact() -> dict[str, Any] | None:
    if settings.is_demo_mode:
        return demo_mood_artifact() or None
    return _load_json(settings.mood_artifact_path) or None


def _model_payload() -> dict[str, Any] | None:
    if settings.is_demo_mode:
        return demo_model_evaluation() or None
    path = settings.processed_dir / "demand_model" / "evaluation.json"
    return _load_json(path) or None


@st.cache_data(ttl=900, show_spinner=False)
def load_local_energy() -> tuple[pd.DataFrame, str]:
    """Load local grid context without making a network request."""
    if settings.is_demo_mode:
        energy = demo_energy()
        if not energy.empty:
            return energy.sort_values("timestamp"), "Demo energy sample"
        return pd.DataFrame(), "Offline demo grid profile"

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    store = PartitionedParquetStore(settings.energy_store_dir)
    try:
        recent = store.read(start=start, end=end)
        if not recent.empty:
            return recent.sort_values("timestamp"), "Local processed eco2mix snapshot"
        stored = store.read()
        if not stored.empty:
            return (
                stored.sort_values("timestamp").tail(7 * 24 * 4),
                "Local processed eco2mix snapshot (historical)",
            )
    except (OSError, ValueError):
        pass

    try:
        raw = load_cached_eco2mix()
        clean = add_time_features(clean_energy_mix(raw), settings.timezone)
        return clean.sort_values("timestamp"), "Cached raw eco2mix snapshot"
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame(), "Offline demo grid profile"


@st.cache_data(ttl=900, show_spinner=False)
def load_local_ecowatt(
    start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, str]:
    if settings.is_demo_mode:
        return demo_ecowatt(start, end)
    try:
        ecowatt = load_cached_ecowatt(timezone_name=settings.timezone)
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame(), "EcoWatt unavailable offline"
    if ecowatt.empty:
        return ecowatt, "EcoWatt unavailable offline"
    frame = ecowatt.loc[ecowatt["timestamp"].between(start, end)].copy()
    return frame, (
        "Cached EcoWatt snapshot"
        if not frame.empty
        else "EcoWatt unavailable for this window"
    )


def build_grid_context() -> tuple[pd.DataFrame, str, str]:
    energy, energy_source = load_local_energy()
    if energy.empty:
        return (
            build_demo_timeline(timezone=settings.timezone),
            energy_source,
            "Demo assumptions",
        )

    latest_ts = pd.to_datetime(energy["timestamp"], utc=True).max()
    start = latest_ts.floor("h") - pd.Timedelta(hours=1)
    end = latest_ts.floor("h") + pd.Timedelta(hours=25)
    ecowatt, ecowatt_source = load_local_ecowatt(start, end)
    result = build_energy_weather_timeline(
        energy,
        latest_ts=latest_ts,
        model_payload=_model_payload(),
        mood_artifact=_mood_artifact(),
        ecowatt=ecowatt,
        timezone=settings.timezone,
    )
    return result.timeline, energy_source, ecowatt_source


def comparison_card(title: str, row: pd.Series, *, role: str) -> None:
    status = str(row.get("status", "Unknown"))
    ecowatt = str(row.get("ecowatt_label", "Unavailable"))
    st.markdown(
        f"""
        <div class="ep-horizon-card ep-border-{_status_class(status)}">
          <div class="ep-card-row">
            <div class="ep-label">{html.escape(role)}</div>
            {status_badge_html(status, status)}
          </div>
          <div class="ep-value">{html.escape(title)}</div>
          <div class="ep-detail">
            Demand pressure: {html.escape(_format_mw(row.get("demand_signal_mw")))}<br>
            CO2 intensity: {html.escape(_format_co2(row.get("co2_intensity_g_per_kwh")))}<br>
            EcoWatt: {html.escape(ecowatt)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_class(status: str) -> str:
    lookup = {
        "Comfortable": "green",
        "Low-carbon opportunity": "blue",
        "Watch": "yellow",
        "Tense": "red",
        "Unknown": "grey",
    }
    return lookup.get(status, "grey")


def badge_for_score(score: ShiftScore) -> tuple[str, str, str]:
    if score.total_points >= 100 or score.peak_avoidance_bonus >= 35:
        return "Peak Hero", "green", "You moved meaningful load out of a stressed hour."
    if score.total_points >= 45 or score.peak_avoidance_bonus > 0:
        return (
            "Grid Helper",
            "blue",
            "Good move: the intervention eased the daily pressure curve.",
        )
    if score.co2_delta_kg > 0:
        return (
            "Carbon Saver",
            "blue",
            "The grid was not very stressed, but the new timing is cleaner.",
        )
    return (
        "Try Again",
        "yellow",
        "Look for a tense/watch hour and shift into a calmer or lower-carbon window.",
    )


def score_card(score: ShiftScore) -> None:
    badge, badge_status, badge_detail = badge_for_score(score)
    pressure_drop = max(
        PRESSURE_POINTS.get(score.original_pressure, 0)
        - PRESSURE_POINTS.get(score.shifted_pressure, 0),
        0,
    )
    st.markdown(
        f"""
        <div class="ep-explanation-card">
          <div class="ep-card-row">
            <div class="ep-label">Challenge score</div>
            {status_badge_html(badge, badge_status)}
          </div>
          <div class="ep-value">{score.total_points:,} pts</div>
          <div class="ep-detail">
            {html.escape(badge_detail)}<br>
            Pressure reduction: {pressure_drop} level(s)<br>
            Grid relief: {score.grid_relief_points:,} · Low-carbon: {score.low_carbon_bonus:,} · Peak: {score.peak_avoidance_bonus:,}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def timeline_chart(
    timeline: pd.DataFrame, original_hour: int, shifted_hour: int
) -> go.Figure:
    frame = timeline.copy()
    frame["target"] = pd.to_datetime(frame["target"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["target"]).sort_values("target")
    frame["local_hour"] = frame["target"].dt.tz_convert(settings.timezone).dt.hour
    if "hour_label" not in frame:
        frame["hour_label"] = (
            frame["target"].dt.tz_convert(settings.timezone).dt.strftime("%H:%M")
        )
    frame["pressure_points"] = frame["status"].map(PRESSURE_POINTS).fillna(0)
    frame["color"] = (
        frame["status"].map(PRESSURE_COLORS).fillna(PRESSURE_COLORS["Unknown"])
    )
    frame["selected"] = ""
    frame.loc[frame["local_hour"].eq(original_hour), "selected"] = "Before"
    frame.loc[frame["local_hour"].eq(shifted_hour), "selected"] = "After"

    fig = go.Figure()
    fig.add_bar(
        x=frame["hour_label"],
        y=frame["pressure_points"],
        marker_color=frame["color"],
        customdata=frame[
            ["status", "demand_signal_mw", "co2_intensity_g_per_kwh", "selected"]
        ],
        hovertemplate=(
            "<b>%{x}</b><br>Status: %{customdata[0]}<br>"
            "Demand: %{customdata[1]:,.0f} MW<br>CO2: %{customdata[2]:,.0f} g/kWh"
            "<br>%{customdata[3]}<extra></extra>"
        ),
    )
    selected = frame.loc[frame["local_hour"].isin([original_hour, shifted_hour])]
    fig.add_scatter(
        x=selected["hour_label"],
        y=selected["pressure_points"] + 0.28,
        mode="markers+text",
        marker=dict(size=15, color="#f8fafc", line=dict(color="#0f766e", width=2)),
        text=selected["selected"],
        textposition="top center",
        hoverinfo="skip",
    )
    fig.update_layout(
        **dark_chart_layout(
            height=280,
            margin=dict(l=10, r=10, t=12, b=10),
            xaxis_title=None,
            yaxis=dict(
                title=None,
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["?", "OK", "Watch", "Tense"],
                range=[0, 3.7],
            ),
            showlegend=False,
        )
    )
    return fig


def before_after_chart(score: ShiftScore) -> go.Figure:
    before = PRESSURE_POINTS.get(score.original_pressure, 0)
    after = PRESSURE_POINTS.get(score.shifted_pressure, 0)
    fig = go.Figure()
    fig.add_bar(
        x=["Baseline", "Your shift"],
        y=[before, after],
        marker_color=[
            PRESSURE_COLORS.get(score.original_pressure, "#9ca3af"),
            PRESSURE_COLORS.get(score.shifted_pressure, "#9ca3af"),
        ],
        text=[score.original_pressure, score.shifted_pressure],
        textposition="outside",
        hovertemplate="%{x}<br>Pressure: %{y}<extra></extra>",
    )
    fig.update_layout(
        **dark_chart_layout(
            height=260,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title=None,
            yaxis=dict(
                title="Pressure level",
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["?", "OK", "Watch", "Tense"],
                range=[0, 3.7],
            ),
            showlegend=False,
        )
    )
    return fig


def outcome_message(score: ShiftScore) -> tuple[str, str, str]:
    badge, _, _ = badge_for_score(score)
    pressure_drop = max(
        PRESSURE_POINTS.get(score.original_pressure, 0)
        - PRESSURE_POINTS.get(score.shifted_pressure, 0),
        0,
    )
    if pressure_drop > 0:
        body = f"{badge}: you moved {score.energy_mwh:.2f} MWh from {score.original_pressure.lower()} to {score.shifted_pressure.lower()}, cutting the visible pressure by {pressure_drop} level(s)."
        kind = "info"
    else:
        body = f"{badge}: this move shifts {score.energy_mwh:.2f} MWh, but it does not reduce the pressure level. Try moving out of a Watch or Tense hour."
        kind = "warning"
    if score.co2_delta_kg > 0:
        body += f" The simplified carbon signal improves by about {score.co2_delta_kg:.0f} kg CO2."
    elif score.co2_delta_kg < 0:
        body += " The shifted hour appears more carbon-intensive, so the carbon effect is negative in this model."
    else:
        body += " Carbon impact is neutral or unavailable with the current inputs."
    return "Instant verdict", body, kind


def render_action_assumption(action: ShiftAction) -> None:
    label = "Placeholder" if action.placeholder else "Fallback"
    status = "yellow" if action.placeholder else "blue"
    st.markdown(
        f"""
        <div class="ep-driver-card">
          <div class="ep-icon">{html.escape(action.icon)}</div>
          <div class="ep-card-row">
            <div class="ep-label">{html.escape(label)}</div>
            {status_badge_html(action.source_label, status)}
          </div>
          <div class="ep-title">{html.escape(action.label)}</div>
          <div class="ep-detail">
            {action.energy_kwh_per_event:.1f} kWh per household event over about {action.duration_hours} h.<br>
            {html.escape(action.source_detail)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


actions = load_actions()
assumption_config = load_assumption_config()
timeline, grid_source, ecowatt_source = build_grid_context()

section_header(
    "Can you flatten the peak?",
    "A 60-second load-shifting challenge",
    "Pick one flexible action, move it away from the daily peak, and see whether your choice lowers demand pressure and carbon intensity in the available grid signal.",
)
st.caption(settings.app_mode_label)
if settings.is_demo_mode:
    message_box(
        "Judge mode: demo data",
        "The simulator uses the bundled grid profile, forecast artifact, EcoWatt sample, and visible appliance assumptions for a repeatable live presentation.",
        kind="info",
    )
message_box(
    "Educational challenge — not an operational dispatch tool",
    "Scores are designed for learning in a live demo. Appliance energy values are visible assumptions, and optional forecast, EcoWatt, or CO2 signals gracefully fall back to demo or unavailable states.",
    kind="info",
)

control, explainer = st.columns([1.05, 1])
with control:
    st.subheader("Make your move")
    action_labels = {action.label: action_id for action_id, action in actions.items()}
    selected_label = st.selectbox(
        "1) Flexible action",
        list(action_labels),
        index=(
            list(action_labels).index("Dishwasher")
            if "Dishwasher" in action_labels
            else 0
        ),
        help="One simple choice keeps the demo fast; detailed assumptions remain visible below.",
    )
    selected_action = actions[action_labels[selected_label]]
    ambition = st.radio(
        "2) Participation level",
        ["Neighbourhood · 1,000 homes", "Town · 10,000 homes", "City · 50,000 homes"],
        horizontal=False,
    )
    households = {
        "Neighbourhood · 1,000 homes": 1_000,
        "Town · 10,000 homes": 10_000,
        "City · 50,000 homes": 50_000,
    }[ambition]
    original_hour = st.slider(
        "3) Peak hour to flatten", min_value=0, max_value=23, value=19, format="%02d:00"
    )
    shifted_hour = st.select_slider(
        "4) Move it to",
        options=list(range(24)),
        value=3,
        format_func=_format_hour,
        help="Try a night valley or low-carbon daytime window.",
    )

with explainer:
    render_action_assumption(selected_action)
    st.caption(f"Grid context: {grid_source}. EcoWatt context: {ecowatt_source}.")

original_row = row_for_local_hour(timeline, original_hour, timezone=settings.timezone)
shifted_row = row_for_local_hour(timeline, shifted_hour, timezone=settings.timezone)
score = compute_shift_score(selected_action, households, original_row, shifted_row)
pressure_drop = max(
    PRESSURE_POINTS.get(score.original_pressure, 0)
    - PRESSURE_POINTS.get(score.shifted_pressure, 0),
    0,
)

section_header(
    "Scoreboard",
    "Before vs after outcomes",
    "Success should be obvious at a glance: lower peak pressure, a clear point score, and a plain-language verdict.",
)
metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card(
        "Peak reduction",
        f"{score.energy_mwh:.2f} MWh",
        "Flexible energy removed from the selected baseline hour.",
        icon="Peak",
    )
with metric_cols[1]:
    metric_card(
        "Pressure reduction",
        f"{pressure_drop} level(s)",
        f"{score.original_pressure} → {score.shifted_pressure}",
        icon="Grid",
    )
with metric_cols[2]:
    co2_value = (
        f"{score.co2_delta_kg:.0f} kg" if score.co2_delta_kg else "Unavailable/neutral"
    )
    metric_card(
        "Carbon effect",
        co2_value,
        "Positive means the shifted hour has lower CO2 intensity.",
        icon="CO2",
    )
with metric_cols[3]:
    badge, badge_status, _ = badge_for_score(score)
    metric_card(
        "Badge",
        badge,
        f"{score.total_points:,} challenge points",
        icon="Award",
        status=badge_status,
    )

before, after, total = st.columns([1, 1, 0.9])
with before:
    comparison_card(_format_hour(original_hour), original_row, role="Baseline")
with after:
    comparison_card(
        _format_hour(shifted_hour), shifted_row, role="Your adjusted scenario"
    )
with total:
    score_card(score)

verdict_title, verdict_body, verdict_kind = outcome_message(score)
message_box(verdict_title, verdict_body, kind=verdict_kind)

chart_cols = st.columns([0.8, 1.2])
with chart_cols[0]:
    viz_note(
        "Challenge snapshot",
        "Side-by-side pressure levels make success or failure instant.",
        source="Educational score",
    )
    st.plotly_chart(before_after_chart(score), width="stretch")
with chart_cols[1]:
    viz_note(
        "24-hour pressure map",
        "Bars show the educational pressure score for each hour. Markers show your baseline and shifted timing.",
        source="Simulator context",
    )
    st.plotly_chart(
        timeline_chart(timeline, original_hour, shifted_hour), width="stretch"
    )

with st.expander("Transparent educational assumptions and scoring", expanded=False):
    st.write(assumption_config["disclaimer"])
    for note in assumption_config.get("source_notes", []):
        st.caption(note)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "action": action.label,
                    "kWh per household event": action.energy_kwh_per_event,
                    "duration hours": action.duration_hours,
                    "source": action.source_label,
                    "placeholder": action.placeholder,
                    "detail": action.source_detail,
                }
                for action in actions.values()
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.write(
        "Scoring combines approximate shifted MWh, pressure reduction between the baseline and shifted hours, "
        "and any lower-carbon timing signal. Points and badges are game mechanics for learning, not billing, dispatch, "
        "settlement, comfort, or verified carbon accounting."
    )

st.link_button("Back to Energy Pulse France", "/")
