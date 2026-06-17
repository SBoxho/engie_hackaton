"""Interactive seasonal-naive baseline backtest results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.layout import apply_theme


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = PROJECT_ROOT / "data" / "processed" / "baseline_backtest.json"
LABELS = {
    "persistence": "Persistence (last observed value)",
    "day_naive": "Day-naive (same time one day earlier)",
    "week_naive": "Week-naive (same time one week earlier)",
}


@st.cache_data(show_spinner=False)
def load_artifact(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


apply_theme()

st.title("Demand baselines")
st.caption(
    "These are simple rule-based reference forecasts—not AI or machine-learning models. "
    "They show the minimum performance a future model should beat."
)

artifact_path = st.text_input("Backtest artifact", str(DEFAULT_ARTIFACT))
try:
    payload = load_artifact(artifact_path)
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.info("Run `python -m scripts.backtest_baselines` first.")
    st.caption(f"Artifact unavailable: {exc}")
    st.stop()

predictions = pd.DataFrame(payload.get("predictions", []))
metrics = pd.DataFrame(payload.get("metrics", []))
if predictions.empty or metrics.empty:
    st.warning("The artifact contains no backtest results.")
    st.stop()

for column in ("origin", "target", "source_timestamp"):
    predictions[column] = pd.to_datetime(predictions[column], utc=True)

left, right = st.columns(2)
horizon = left.selectbox(
    "Forecast horizon", sorted(metrics["horizon_hours"].unique()), format_func=lambda x: f"{x} hours"
)
available_baselines = metrics.loc[metrics["horizon_hours"].eq(horizon), "baseline"].tolist()
baseline = right.selectbox("Reference rule", available_baselines, format_func=lambda x: LABELS.get(x, x))

selected = predictions.loc[
    predictions["horizon_hours"].eq(horizon) & predictions["baseline"].eq(baseline)
].copy()
valid = selected.dropna(subset=["actual_mw", "predicted_mw"])
metric = metrics.loc[
    metrics["horizon_hours"].eq(horizon) & metrics["baseline"].eq(baseline)
].iloc[0]

if valid.empty:
    st.warning("No aligned actual/prediction pairs exist for this selection.")
else:
    period_start = valid["target"].min().strftime("%Y-%m-%d %H:%M UTC")
    period_end = valid["target"].max().strftime("%Y-%m-%d %H:%M UTC")
    st.write(f"Evaluation period: **{period_start} → {period_end}**")
    columns = st.columns(4)
    columns[0].metric("MAE", f"{metric['mae_mw']:,.0f} MW")
    columns[1].metric("RMSE", f"{metric['rmse_mw']:,.0f} MW")
    columns[2].metric("sMAPE", f"{metric['smape_percent']:.2f}%")
    columns[3].metric(
        "Coverage",
        f"{metric['coverage']:.1%}",
        help=f"{int(metric['sample_count']):,} usable pairs out of "
        f"{int(metric['available_target_count']):,} available targets.",
    )

    chart_data = valid[["target", "actual_mw", "predicted_mw"]].rename(
        columns={"actual_mw": "Actual demand", "predicted_mw": "Baseline forecast"}
    )
    chart_data = chart_data.melt("target", var_name="series", value_name="MW")
    figure = px.line(chart_data, x="target", y="MW", color="series")
    figure.update_layout(xaxis_title=None, legend_title=None, hovermode="x unified")
    st.plotly_chart(figure, width="stretch")

st.caption(
    f"Exact 15-minute target alignment · {int(metric['sample_count']):,} samples · "
    f"{int(metric['missing_target_count']):,} missing targets · "
    f"{int(metric['missing_prediction_count']):,} unavailable lagged predictions"
)

with st.expander("All metrics"):
    display = metrics.copy()
    display["baseline"] = display["baseline"].map(LABELS).fillna(display["baseline"])
    st.dataframe(display, width="stretch", hide_index=True)
