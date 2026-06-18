"""Experimental demand model dashboard page."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.cards import explanation_card, horizon_forecast_card, message_box, section_header, viz_note
from app.components.layout import apply_theme
from src.artifact_contract import ArtifactSpec, validate_artifact
from src.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION = PROJECT_ROOT / "data" / "processed" / "demand_model" / "evaluation.json"
BASELINE_LABELS = {
    "persistence": "Persistence",
    "day_naive": "Previous day",
    "week_naive": "Previous week",
    "rte_forecast": "RTE J/J-1 forecast",
}
SEASON_LABELS = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
EXPLANATION_FALLBACK = (
    "This artifact was created before local explanations were available, or the explanation "
    "step could not run for these rows."
)


@st.cache_data(show_spinner=False)
def load_evaluation(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact root must be a JSON object")
    return payload


def format_mw(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.0f} MW"


def format_percent(value: float | int | None, *, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{float(value):{sign}.1f}%"


def format_fraction_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def format_interval(row: pd.Series) -> str:
    lower = row.get("model_interval_lower_mw")
    upper = row.get("model_interval_upper_mw")
    if lower is None or upper is None or pd.isna(lower) or pd.isna(upper):
        return "uncertainty band unavailable"
    return f"{format_mw(lower)} to {format_mw(upper)}"


def comparison_for_horizon(comparisons: pd.DataFrame, horizon: int | float) -> pd.Series | None:
    if comparisons.empty or "horizon_hours" not in comparisons:
        return None
    rows = comparisons.loc[comparisons["horizon_hours"].eq(horizon)]
    return None if rows.empty else rows.iloc[0]


def trust_badge(row: pd.Series | None) -> str:
    if row is None:
        return "Experimental horizon"
    label = row.get("reliability_badge")
    if isinstance(label, str) and label:
        return label
    improvement = row.get("improvement_vs_strongest_baseline_percent")
    beats = bool(improvement is not None and pd.notna(improvement) and float(improvement) > 0)
    return "Model edge detected" if beats else "Experimental horizon"


def render_forecast_cards(predictions: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    section_header(
        "Forecasts",
        "Point forecasts with uncertainty",
        "Every horizon remains visible, including horizons where the model has not beaten the strongest baseline.",
    )
    horizons = sorted(predictions["horizon_hours"].dropna().unique())
    columns = st.columns(min(4, max(1, len(horizons))))
    for index, horizon in enumerate(horizons):
        rows = predictions.loc[predictions["horizon_hours"].eq(horizon)].sort_values("target_timestamp")
        if rows.empty:
            continue
        latest = rows.iloc[-1]
        comparison = comparison_for_horizon(comparisons, horizon)
        badge = trust_badge(comparison)
        beats = badge == "Model edge detected"
        baseline_text = "beats strongest baseline" if beats else "does not beat strongest baseline"
        with columns[index % len(columns)]:
            horizon_forecast_card(
                f"{int(horizon)}h horizon",
                format_mw(latest.get("model_predicted_mw")),
                f"Band: {format_interval(latest)}. {baseline_text}.",
                status=badge,
            )


def selected_horizon_chart(rows: pd.DataFrame, baseline_column: str, baseline_label: str) -> go.Figure:
    figure = go.Figure()
    has_interval = {"model_interval_lower_mw", "model_interval_upper_mw"}.issubset(rows.columns)
    if has_interval:
        figure.add_trace(
            go.Scatter(
                x=rows["target_timestamp"],
                y=rows["model_interval_upper_mw"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Upper interval",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=rows["target_timestamp"],
                y=rows["model_interval_lower_mw"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(147, 197, 253, 0.18)",
                line=dict(width=0),
                hovertemplate="Interval lower: %{y:,.0f} MW<extra></extra>",
                name="Uncertainty band",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=rows["target_timestamp"],
            y=rows["target_mw"],
            mode="lines",
            name="Actual demand",
            line=dict(color="#f8fafc", width=2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rows["target_timestamp"],
            y=rows["model_predicted_mw"],
            mode="lines",
            name="Model",
            line=dict(color="#93c5fd", width=2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rows["target_timestamp"],
            y=rows[baseline_column],
            mode="lines",
            name=baseline_label,
            line=dict(color="#fbbf24", width=1.6, dash="dot"),
        )
    )
    figure.update_layout(xaxis_title=None, yaxis_title="MW", legend_title=None, hovermode="x unified")
    return figure


def render_trust_panel(
    comparisons: pd.DataFrame,
    metrics: pd.DataFrame,
    payload: dict,
    horizon: int | float,
    data_audit: dict,
) -> None:
    section_header(
        "Trust panel",
        "Model honesty by horizon",
        "Positive improvement means the model beat the strongest eligible baseline on the same chronological test rows.",
    )
    comparison = comparison_for_horizon(comparisons, horizon)
    model_metric = metrics.loc[
        metrics["horizon_hours"].eq(horizon) & metrics["model"].eq("demand_hgb")
    ]
    model_metric_row = model_metric.iloc[0] if not model_metric.empty else pd.Series(dtype=object)
    columns = st.columns(4)
    if comparison is not None:
        strongest = comparison.get("strongest_baseline") or "none"
        columns[0].metric("Model MAE", format_mw(comparison.get("model_mae_mw")))
        columns[1].metric(
            f"{BASELINE_LABELS.get(strongest, strongest)} MAE",
            format_mw(comparison.get("strongest_baseline_mae_mw")),
        )
        columns[2].metric(
            "Improvement",
            format_percent(comparison.get("improvement_vs_strongest_baseline_percent"), signed=True),
        )
    else:
        columns[0].metric("Model MAE", "n/a")
        columns[1].metric("Baseline MAE", "n/a")
        columns[2].metric("Improvement", "n/a")
    columns[3].metric(
        "Data coverage",
        format_fraction_percent(model_metric_row.get("coverage", None)),
        help="Usable model/target pairs in the selected chronological test window.",
    )
    st.caption(
        f"Last trained: {payload.get('model_generated_at') or 'unknown'} | "
        f"training coverage: {data_audit.get('start_utc', 'unknown')} to {data_audit.get('end_utc', 'unknown')}"
    )
    periods = payload.get("training_periods", {}).get(str(int(horizon)), {})
    if periods:
        st.caption(
            "Selected horizon split: "
            f"train {periods.get('train_start_utc', 'unknown')} to {periods.get('train_end_utc', 'unknown')} | "
            f"test {periods.get('test_start_utc', 'unknown')} to {periods.get('test_end_utc', 'unknown')}"
        )
    if comparisons.empty:
        return
    view = comparisons.copy()
    view["horizon_hours"] = view["horizon_hours"].map(lambda value: f"{int(value)}h")
    view["strongest_baseline"] = view["strongest_baseline"].map(BASELINE_LABELS).fillna(view["strongest_baseline"])
    rename = {
        "horizon_hours": "Horizon",
        "reliability_badge": "Badge",
        "model_beats_strongest_baseline": "Beats baseline",
        "model_mae_mw": "Model MAE",
        "strongest_baseline": "Strongest baseline",
        "strongest_baseline_mae_mw": "Baseline MAE",
        "improvement_vs_strongest_baseline_percent": "Improvement %",
        "prediction_interval_coverage": "Interval coverage",
        "prediction_interval_mean_width_mw": "Mean band width MW",
    }
    display_columns = [column for column in rename if column in view]
    st.dataframe(view[display_columns].rename(columns=rename), width="stretch", hide_index=True)


def contribution_status(direction: str | None) -> str:
    if direction == "up":
        return "green"
    if direction == "down":
        return "blue"
    return "grey"


def render_explained_forecasts(rows: pd.DataFrame, payload: dict) -> None:
    section_header(
        "Explainability",
        "Recent forecast drivers",
        payload.get(
            "explanation_disclaimer",
            "Explanations are approximate and model-derived; they are not causal explanations.",
        ),
    )
    if "explanation_cards" not in rows:
        message_box("Explanation unavailable", EXPLANATION_FALLBACK, kind="warning")
        return

    explained = rows.sort_values("target_timestamp").tail(3).iloc[::-1]
    for _, row in explained.iterrows():
        with st.container(border=True):
            target = row["target_timestamp"].strftime("%Y-%m-%d %H:%M UTC")
            st.markdown(f"**{target} · {int(row['horizon_hours'])}h horizon**")
            forecast_cols = st.columns(3)
            forecast_cols[0].metric("Model forecast", format_mw(row.get("model_predicted_mw")))
            forecast_cols[1].metric("Actual demand", format_mw(row.get("target_mw")))
            error = row.get("target_mw") - row.get("model_predicted_mw")
            forecast_cols[2].metric("Forecast error", format_mw(error))

            cards = row.get("explanation_cards")
            if not isinstance(cards, list) or not cards:
                message_box("Explanation unavailable", row.get("explanation_error") or EXPLANATION_FALLBACK, kind="warning")
            else:
                card_cols = st.columns(min(4, len(cards)))
                for card, column in zip(cards, card_cols):
                    with column:
                        explanation_card(
                            str(card.get("title", "Forecast driver")),
                            str(card.get("detail", "")),
                            label=str(card.get("family_label", "Driver")),
                            icon=str(card.get("icon", "")),
                            status=contribution_status(card.get("direction")),
                        )

            with st.expander("Technical raw feature contributions"):
                contributions = row.get("technical_contributions")
                if isinstance(contributions, list) and contributions:
                    technical = pd.DataFrame(contributions)
                    technical = technical.rename(
                        columns={
                            "feature": "Raw feature",
                            "family_label": "Family",
                            "direction": "Direction",
                            "delta_mw": "Delta MW",
                        }
                    )
                    st.dataframe(
                        technical[["Raw feature", "Family", "Direction", "Delta MW"]],
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("No raw contribution detail is available for this forecast.")


apply_theme()

st.title("Demand model")
st.caption(settings.app_mode_label)
st.caption(
    "Experimental weather-aware demand model. This is not an RTE operational forecast "
    "and should only be read as a backtested research artifact."
)

default_evaluation = settings.demo_model_evaluation_path if settings.is_demo_mode else DEFAULT_EVALUATION
artifact_path = st.text_input("Evaluation artifact", str(default_evaluation))
check = validate_artifact(
    ArtifactSpec(
        "demand_evaluation",
        "Demand model evaluation",
        Path(artifact_path),
        "json",
        True,
        required_keys=("predictions", "metrics"),
    )
)
try:
    payload = load_evaluation(artifact_path) if check.ok else {}
except (OSError, ValueError, json.JSONDecodeError) as exc:
    if settings.is_demo_mode:
        st.info("The bundled demand-model demo artifact is not available.")
        st.code("python -m scripts.export_demo_bundle")
    else:
        st.info("Run the demand model pipeline before opening this page.")
        st.code(
            "python -m scripts.build_features\n"
            "python -m scripts.train_demand_model\n"
            "python -m scripts.evaluate_demand_model"
        )
    st.caption(f"Artifact unavailable: {exc}")
    st.stop()
if not check.ok:
    st.warning(f"Demand-model artifact is {check.status}: {check.detail}")
    st.stop()

predictions = pd.DataFrame(payload.get("predictions", []))
metrics = pd.DataFrame(payload.get("metrics", []))
comparisons = pd.DataFrame(payload.get("baseline_comparison", []))
if predictions.empty or metrics.empty:
    st.warning("The evaluation artifact contains no model predictions.")
    st.stop()

for column in ("origin_timestamp", "target_timestamp"):
    predictions[column] = pd.to_datetime(predictions[column], utc=True)

data_audit = payload.get("data_audit", {})
weather_audit = data_audit.get("weather") or {}
latest_target = predictions["target_timestamp"].max()
st.write(
    f"Artifact generated: **{payload.get('generated_at', 'unknown')}** · "
    f"latest evaluated target: **{latest_target.strftime('%Y-%m-%d %H:%M UTC')}**"
)
st.write(
    f"Training source coverage: **{data_audit.get('start_utc', 'unknown')} → "
    f"{data_audit.get('end_utc', 'unknown')}** · "
    f"weather overlap: **{weather_audit.get('overlap_fraction_of_energy_timestamps', 0):.1%}**"
)

render_forecast_cards(predictions, comparisons)

left, right = st.columns(2)
horizon = left.selectbox(
    "Forecast horizon",
    sorted(predictions["horizon_hours"].unique()),
    format_func=lambda value: f"{int(value)} hours",
)
available_comparison = comparisons.loc[comparisons["horizon_hours"].eq(horizon)]
if not available_comparison.empty:
    row = available_comparison.iloc[0]
    strongest = row.get("strongest_baseline") or "none"
    improvement = row.get("improvement_vs_strongest_baseline_percent")
    right.metric(
        "Improvement vs strongest baseline",
        "n/a" if pd.isna(improvement) else f"{improvement:+.1f}%",
        help=f"Strongest eligible baseline: {BASELINE_LABELS.get(strongest, strongest)}",
    )

selected = predictions.loc[predictions["horizon_hours"].eq(horizon)].copy()
selected = selected.sort_values("target_timestamp").tail(7 * 96)
baseline_column = None
if not available_comparison.empty and pd.notna(available_comparison.iloc[0].get("strongest_baseline")):
    baseline_column = f"{available_comparison.iloc[0]['strongest_baseline']}_predicted_mw"
if baseline_column not in selected:
    baseline_column = "persistence_predicted_mw"

baseline_label = BASELINE_LABELS.get(baseline_column.replace("_predicted_mw", ""), "Baseline")
figure = selected_horizon_chart(selected, baseline_column, baseline_label)
viz_note(
    "Forecast versus reality",
    "This chart compares the model, actual demand, uncertainty band, and strongest simple baseline for the selected horizon.",
    source="Demand model evaluation",
)
st.plotly_chart(figure, width="stretch")

render_trust_panel(comparisons, metrics, payload, horizon, data_audit)

render_explained_forecasts(selected, payload)

metric_view = metrics.loc[metrics["horizon_hours"].eq(horizon)].copy()
metric_view["model"] = metric_view["model"].map(BASELINE_LABELS).fillna(metric_view["model"])
st.dataframe(metric_view, width="stretch", hide_index=True)

periods = payload.get("training_periods", {}).get(str(int(horizon)), {})
with st.expander("Training and evaluation period"):
    st.json(periods)

segments = pd.DataFrame(payload.get("segment_metrics", []))
with st.expander("Segment performance"):
    if segments.empty:
        st.caption("No hour or season segment has enough samples for a stable summary.")
    else:
        view = segments.loc[segments["horizon_hours"].eq(horizon)].copy()
        if "segment_value" in view:
            view["segment_value"] = view["segment_value"].astype(str)
            is_season = view["segment"].eq("target_season")
            season_values = pd.to_numeric(view.loc[is_season, "segment_value"], errors="coerce")
            view.loc[is_season, "segment_value"] = season_values.map(SEASON_LABELS).fillna(
                view.loc[is_season, "segment_value"]
            )
        st.dataframe(view, width="stretch", hide_index=True)

with st.expander("Data audit"):
    st.json(data_audit)
