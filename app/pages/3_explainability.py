"""Forecast explainability page backed by grouped-ablation artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.cards import explanation_card, message_box, metric_card, section_header, viz_note
from app.components.charts import dark_chart_layout
from app.components.layout import apply_theme
from src.artifact_contract import ArtifactSpec, validate_artifact
from src.config import settings
from src.demo_mode import demo_model_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION = PROJECT_ROOT / "data" / "processed" / "demand_model" / "evaluation.json"
BASELINE_LABELS = {
    "persistence": "Persistence",
    "day_naive": "Previous day",
    "week_naive": "Previous week",
    "rte_forecast": "RTE J/J-1 forecast",
}
FAMILY_ORDER = ["weather", "calendar", "recent_demand", "weekly_pattern", "data_quality"]
FAMILY_LABELS = {
    "weather": "Weather",
    "calendar": "Calendar",
    "recent_demand": "Lagged demand",
    "weekly_pattern": "Seasonality",
    "data_quality": "Data freshness",
}
HONESTY_COPY = (
    "These are approximate model-behaviour explanations from grouped ablation, not causal proof. "
    "They show how the trained model changes when one feature group is replaced by a typical reference value."
)


@st.cache_data(ttl=900, show_spinner=False)
def load_evaluation(path: str, demo_mode: bool) -> dict[str, Any]:
    if demo_mode:
        return demo_model_evaluation()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact root must be a JSON object")
    return payload


def _to_frame(records: Any) -> pd.DataFrame:
    frame = pd.DataFrame(records or [])
    for column in ("origin_timestamp", "target_timestamp"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame.dropna(subset=["target_timestamp"]) if "target_timestamp" in frame else frame


def _format_mw(value: Any, *, signed: bool = False) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):+,.0f} MW" if signed else f"{float(numeric):,.0f} MW"


def _format_pct(value: Any, *, signed: bool = False) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):+,.1f}%" if signed else f"{float(numeric):,.1f}%"


def _metric_row(metrics: pd.DataFrame, horizon: Any, model: str = "demand_hgb") -> pd.Series:
    if metrics.empty or not {"horizon_hours", "model"}.issubset(metrics.columns):
        return pd.Series(dtype=object)
    rows = metrics.loc[metrics["horizon_hours"].eq(horizon) & metrics["model"].eq(model)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _comparison_row(comparisons: pd.DataFrame, horizon: Any) -> pd.Series:
    if comparisons.empty or "horizon_hours" not in comparisons:
        return pd.Series(dtype=object)
    rows = comparisons.loc[comparisons["horizon_hours"].eq(horizon)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _family_contributions(row: pd.Series) -> pd.DataFrame:
    cards = row.get("explanation_cards")
    if not isinstance(cards, list):
        return pd.DataFrame(columns=["family", "label", "delta_mw", "direction"])
    rows = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        family = str(card.get("family", "other"))
        rows.append(
            {
                "family": family,
                "label": FAMILY_LABELS.get(family, str(card.get("family_label", family.title()))),
                "delta_mw": pd.to_numeric(card.get("delta_mw"), errors="coerce"),
                "direction": card.get("direction"),
                "title": card.get("title"),
                "detail": card.get("detail"),
                "icon": card.get("icon"),
            }
        )
    frame = pd.DataFrame(rows).dropna(subset=["delta_mw"])
    if frame.empty:
        return frame
    frame["order"] = frame["family"].map({name: idx for idx, name in enumerate(FAMILY_ORDER)}).fillna(99)
    return frame.sort_values(["order", "delta_mw"], ascending=[True, False])


def _aggregate_importance(rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, row in rows.iterrows():
        family = _family_contributions(row)
        if not family.empty:
            parts.append(family[["family", "label", "delta_mw"]])
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    return (
        combined.assign(abs_delta_mw=combined["delta_mw"].abs())
        .groupby(["family", "label"], as_index=False)
        .agg(mean_delta_mw=("delta_mw", "mean"), mean_abs_delta_mw=("abs_delta_mw", "mean"), samples=("delta_mw", "count"))
        .sort_values("mean_abs_delta_mw", ascending=False)
    )


def _contribution_chart(frame: pd.DataFrame) -> go.Figure:
    colors = ["#22c55e" if value >= 0 else "#38bdf8" for value in frame["delta_mw"]]
    fig = go.Figure(go.Bar(x=frame["label"], y=frame["delta_mw"], marker_color=colors, text=[_format_mw(v, signed=True) for v in frame["delta_mw"]], textposition="outside"))
    fig.update_layout(**dark_chart_layout(height=330, xaxis_title=None, yaxis_title="Ablation delta (MW)", showlegend=False, margin=dict(l=10, r=10, t=20, b=10)))
    return fig


def _importance_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=frame["label"], y=frame["mean_abs_delta_mw"], marker_color="#a78bfa", text=[_format_mw(v) for v in frame["mean_abs_delta_mw"]], textposition="outside"))
    fig.update_layout(**dark_chart_layout(height=300, xaxis_title=None, yaxis_title="Mean absolute delta (MW)", showlegend=False, margin=dict(l=10, r=10, t=20, b=10)))
    return fig


def _plain_language_summary(row: pd.Series, family: pd.DataFrame) -> str:
    forecast = row.get("model_predicted_mw")
    actual = row.get("target_mw")
    if family.empty:
        return "The forecast is shown, but this artifact does not include reliable local driver cards for this point."
    top = family.reindex(family["delta_mw"].abs().sort_values(ascending=False).index).head(2)
    high = top.loc[top["delta_mw"] > 0, "label"].tolist()
    low = top.loc[top["delta_mw"] < 0, "label"].tolist()
    parts = []
    if high:
        parts.append(f"higher mainly because of {', '.join(high)}")
    if low:
        parts.append(f"lower because {', '.join(low)} offset the forecast")
    driver_text = "; ".join(parts) if parts else "close to its typical reference pattern"
    actual_text = ""
    if pd.notna(pd.to_numeric(actual, errors="coerce")):
        actual_text = f" The observed value for this evaluated target was {_format_mw(actual)}, so the model error is {_format_mw(float(actual) - float(forecast), signed=True)}."
    return f"The model forecast is {_format_mw(forecast)} and looks {driver_text}.{actual_text}"


def _status_for_delta(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "grey"
    return "green" if numeric >= 0 else "blue"


st.set_page_config(page_title="Forecast Explainability", page_icon=":material/psychology:", layout="wide")
apply_theme()

st.title("Forecast explainability")
st.markdown('<div class="ep-subtitle">Grouped ablation explanations for recent demand forecasts, with plain-language drivers and technical evidence.</div>', unsafe_allow_html=True)
st.caption(settings.app_mode_label)

default_path = settings.demo_model_evaluation_path if settings.is_demo_mode else DEFAULT_EVALUATION
artifact_path = st.sidebar.text_input("Evaluation artifact", str(default_path), disabled=settings.is_demo_mode)
check = validate_artifact(ArtifactSpec("demand_evaluation", "Demand model evaluation", Path(artifact_path), "json", True, required_keys=("predictions", "metrics")))
if not settings.is_demo_mode and not check.ok:
    message_box("Explanation artifact unavailable", f"Demand-model evaluation is {check.status}: {check.detail}", kind="warning")
    st.code("python -m scripts.build_features && python -m scripts.train_demand_model && python -m scripts.evaluate_demand_model")
    st.stop()

try:
    payload = load_evaluation(artifact_path, settings.is_demo_mode)
except (OSError, ValueError, json.JSONDecodeError) as exc:
    message_box("Explanation artifact unreadable", f"The page is stable, but no trustworthy explanation can be shown yet: {exc}", kind="warning")
    st.stop()

predictions = _to_frame(payload.get("predictions"))
metrics = pd.DataFrame(payload.get("metrics", []))
comparisons = pd.DataFrame(payload.get("baseline_comparison", []))
if predictions.empty or metrics.empty:
    message_box("No evaluated forecasts", "The evaluation artifact does not contain model predictions and metrics yet.", kind="warning")
    st.stop()

horizons = sorted(predictions["horizon_hours"].dropna().unique())
horizon = st.sidebar.selectbox("Forecast horizon", horizons, format_func=lambda value: f"{int(value)} hours")
window_size = st.sidebar.slider("Recent window for aggregate importance", min_value=12, max_value=96, value=48, step=12, help="Uses already-computed explanation cards; no SHAP or expensive retraining runs in the app.")
selected = predictions.loc[predictions["horizon_hours"].eq(horizon)].sort_values("target_timestamp")
recent = selected.tail(window_size)
latest = selected.iloc[-1]
family = _family_contributions(latest)
aggregate = _aggregate_importance(recent)
metric = _metric_row(metrics, horizon)
comparison = _comparison_row(comparisons, horizon)

latest_target = predictions["target_timestamp"].max()
data_audit = payload.get("data_audit", {}) if isinstance(payload.get("data_audit"), dict) else {}
periods = payload.get("training_periods", {}).get(str(int(horizon)), {}) if isinstance(payload.get("training_periods"), dict) else {}
strongest_key = comparison.get("strongest_baseline") if not comparison.empty else None
strongest_label = BASELINE_LABELS.get(str(strongest_key), str(strongest_key or "Baseline"))

section_header("What happened?", "Why is demand high or low?", HONESTY_COPY)
message_box("Plain-language answer", _plain_language_summary(latest, family))

cols = st.columns(4)
cols[0].metric("Forecast", _format_mw(latest.get("model_predicted_mw")), help="Selected latest evaluated target for this horizon.")
cols[1].metric("Actual demand", _format_mw(latest.get("target_mw")))
cols[2].metric("Model MAE", _format_mw(metric.get("mae_mw")))
cols[3].metric("Baseline edge", _format_pct(comparison.get("improvement_vs_strongest_baseline_percent"), signed=True), help=f"Compared with {strongest_label} on the same chronological test rows.")

fresh = st.columns(3)
with fresh[0]:
    metric_card("Latest evaluated target", latest_target.strftime("%d %b %Y %H:%M UTC"), "The newest target row included in the explanation artifact.", icon="Now")
with fresh[1]:
    metric_card("Artifact generated", str(payload.get("generated_at", "unknown")), "When the evaluation JSON was written.", icon="JSON")
with fresh[2]:
    metric_card("Training coverage", f"{data_audit.get('start_utc', 'unknown')} → {data_audit.get('end_utc', 'unknown')}", "Historical source window used by the pipeline.", icon="Data")

section_header("Grouped drivers", "Ablation contribution by model signal family", "Positive bars lifted the forecast versus a typical reference; negative bars lowered it. The method is cached in the artifact and lightweight in demo mode.")
if family.empty:
    message_box("Grouped explanation unavailable", str(latest.get("explanation_error") or "No explanation cards were stored for this prediction."), kind="warning")
else:
    st.plotly_chart(_contribution_chart(family), width="stretch")
    card_cols = st.columns(min(4, len(family)))
    for (_, row), col in zip(family.iterrows(), card_cols):
        with col:
            explanation_card(str(row.get("title") or row["label"]), str(row.get("detail") or "Grouped ablation contribution."), label=str(row["label"]), icon=str(row.get("icon") or "AI"), status=_status_for_delta(row["delta_mw"]))

section_header("Technical evidence", "Recent-window importance and horizon metrics", "This section is for judges who want to audit whether the explanation is consistent with the current evaluation window.")
left, right = st.columns([1.1, 1])
with left:
    viz_note("Safe additional importance", "Mean absolute grouped-ablation delta over recent evaluated predictions. It reuses stored cards instead of running a heavy SHAP computation.", source="Demand model evaluation")
    if aggregate.empty:
        st.caption("No recent-window grouped contributions are available.")
    else:
        st.plotly_chart(_importance_chart(aggregate.head(6)), width="stretch")
with right:
    st.markdown("**Horizon-aware metrics**")
    metric_rows = metrics.loc[metrics.get("horizon_hours", pd.Series(dtype=float)).eq(horizon)].copy() if "horizon_hours" in metrics else metrics.copy()
    if not metric_rows.empty and "model" in metric_rows:
        metric_rows["model"] = metric_rows["model"].map(BASELINE_LABELS).fillna(metric_rows["model"])
    st.dataframe(metric_rows, width="stretch", hide_index=True)

with st.expander("Raw explanation details for experts", expanded=False):
    st.markdown("**Selected prediction**")
    st.dataframe(pd.DataFrame([latest]), width="stretch", hide_index=True)
    st.markdown("**Grouped contributions**")
    st.dataframe(family.drop(columns=["order"], errors="ignore"), width="stretch", hide_index=True)
    technical = latest.get("technical_contributions")
    if isinstance(technical, list) and technical:
        st.markdown("**Top raw feature ablations**")
        st.dataframe(pd.DataFrame(technical), width="stretch", hide_index=True)
    else:
        st.caption("Raw feature-level contribution rows are not available for this prediction.")
    if periods:
        st.markdown("**Selected horizon train/test split**")
        st.json(periods)

message_box("Limits", "The model knows historical demand, calendar, lagged-demand and weather-derived features present in the artifact. It infers associations learned during backtesting. It does not know future policy changes, outages, behavioural shocks, or causal ground truth, so these explanations should support discussion rather than operational decisions.", kind="warning")
