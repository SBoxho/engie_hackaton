from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.components.regional_map import has_meaningful_anomaly_signal, regional_anomaly_choropleth
from app.components.public import forecast_chart
from app.next48h_view import (
    choose_best_window,
    confidence_summary,
    future_regional_map_frame,
    projected_future_regional_map_frame,
    selected_hour_explanation,
)
from app.view_models import ForecastPointView, add_regional_anomalies, synthesize_regional_history
from src.data_sources.rte_eco2mix_regional import demo_regional_snapshot, fallback_region_geojson
from tests.test_twin_api import NOW, service


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def _point(
    *,
    timestamp: str = "2026-06-18T18:00:00Z",
    p10: float = 49_000,
    p50: float = 52_000,
    p90: float = 55_000,
    usual: float = 50_000,
    horizon: int = 6,
) -> ForecastPointView:
    return ForecastPointView(
        timestamp=pd.Timestamp(timestamp),
        p10=p10,
        p50=p50,
        p90=p90,
        source="Usual-demand baseline fallback",
        pressure_label="Watch",
        drivers=[],
        backtest_error=1_200,
        horizon_hours=horizon,
        usual_demand_mw=usual,
    )


def test_selected_hour_explanation_numerically_reconciles_to_forecast() -> None:
    explanation = selected_hour_explanation(_point())

    assert explanation.reconciled_demand_mw == pytest.approx(explanation.expected_demand_mw)
    assert explanation.reconciliation_error_mw == pytest.approx(0)
    assert sum(driver.value_mw for driver in explanation.positive_drivers) == pytest.approx(2_000)
    assert not explanation.negative_drivers
    assert "uncertainty is shown only as the likely range" in explanation.text


def test_confidence_summary_uses_range_horizon_fallback_and_recent_error() -> None:
    summary = confidence_summary(_point(p10=46_000, p50=52_000, p90=58_000, horizon=30))

    names = {factor.name for factor in summary.factors}
    assert {"Forecast horizon", "Interval width", "Weather disagreement", "Fallback sources", "Recent model performance"} == names
    assert summary.level in {"Low", "Medium"}
    assert any(factor.status == "watch" for factor in summary.factors)


def test_best_window_objectives_can_select_different_times() -> None:
    start = pd.Timestamp("2026-06-18T00:00:00Z")
    forecast = pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(hours=hour) for hour in range(6)],
            "p10": [45_000] * 6,
            "p50": [50_000, 50_200, 52_000, 52_100, 51_900, 51_700],
            "p90": [55_000] * 6,
            "pressure_label": ["Normal"] * 6,
            "balance_score": [0.08, 0.10, 0.82, 0.84, 0.80, 0.78],
            "carbon_g_per_kwh": [92, 90, 25, 24, 26, 28],
        }
    )

    balance = choose_best_window(forecast, "lowest_balance", window_hours=2)
    carbon = choose_best_window(forecast, "lowest_carbon", window_hours=2)
    combined = choose_best_window(forecast, "combined", window_hours=2)

    assert balance is not None and carbon is not None and combined is not None
    assert balance.start == start
    assert carbon.start == start + pd.Timedelta(hours=2)
    assert balance.start != carbon.start
    assert combined.objective == "combined"


def test_future_regional_map_frame_is_labelled_as_selected_future_state() -> None:
    snapshot = service().get_twin(from_timestamp=NOW, hours=2, region="11").snapshots[2]
    frame = future_regional_map_frame(snapshot)

    assert not frame.empty
    assert frame["freshness_label"].str.contains("Forecast hour").all()
    assert frame["source_label"].eq("Future regional demand forecast").all()
    assert not frame["source_label"].str.lower().str.contains("current|observed").any()
    assert frame["demand_anomaly_pct"].notna().any()
    assert frame.loc[frame["demand_anomaly_pct"].isna(), "availability_flag"].eq(False).all()

    figure = regional_anomaly_choropleth(frame, fallback_region_geojson())
    available = next(trace for trace in figure.data if trace.name == "Available regional anomaly")
    assert pd.Series(available.z).notna().all()


def test_sparse_future_regional_map_falls_back_to_projected_all_region_signal() -> None:
    snapshot = service().get_twin(from_timestamp=NOW, hours=2, region="11").snapshots[2]
    typed_frame = future_regional_map_frame(snapshot)
    current = demo_regional_snapshot()
    regional = add_regional_anomalies(current, synthesize_regional_history(current), timezone="Europe/Paris")
    projected = projected_future_regional_map_frame(regional, _point())

    assert not has_meaningful_anomaly_signal(typed_frame)
    assert has_meaningful_anomaly_signal(projected)
    assert projected["region_code"].nunique() == 13


def test_primary_forecast_chart_contains_required_layers() -> None:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-18T00:00:00Z", periods=4, freq="h"),
            "consumption_mw": [49_000, 49_400, 50_100, 50_700],
        }
    )
    forecast = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-18T04:00:00Z", periods=4, freq="h"),
            "p10": [49_000, 49_500, 50_000, 50_400],
            "p50": [51_000, 51_500, 52_000, 52_400],
            "p90": [53_000, 53_500, 54_000, 54_400],
            "usual_demand_mw": [50_000, 50_300, 50_600, 50_900],
            "pressure_label": ["Normal", "Watch", "High", "Normal"],
            "source": ["fixture"] * 4,
        }
    )

    figure = forecast_chart(
        history,
        forecast,
        selected_timestamp=pd.Timestamp("2026-06-18T05:00:00Z"),
        now_timestamp=pd.Timestamp("2026-06-18T03:00:00Z"),
    )
    trace_names = {trace.name for trace in figure.data}

    assert {"Actual demand", "Uncertainty interval", "Usual-demand baseline", "Forecast demand", "Selected hour"} <= trace_names
    assert figure.layout.shapes
    assert any(getattr(trace, "selectedpoints", None) for trace in figure.data if trace.name == "Forecast demand")


def test_next48h_page_selecting_time_updates_dependent_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("DEMO_ALLOW_EXTERNAL_API", "0")
    app = AppTest.from_file(APP_MAIN, default_timeout=20)
    app.switch_page("pages/next_48h.py")
    app.run(timeout=45)

    assert not app.exception
    assert app.selectbox

    selected_option = app.selectbox[0].options[min(5, len(app.selectbox[0].options) - 1)]
    selected_label = selected_option.split(" - ")[0]
    app.selectbox[0].select(selected_option)
    app.run(timeout=45)

    assert not app.exception
    rendered = "\n".join(str(item.value) for item in app.markdown)
    assert rendered.count(selected_label) >= 3
    assert "Deterministic explanation" in rendered
    assert "Future Regional Map" in rendered
    assert "not current regional context" in rendered
    assert "Generation And Balance" in rendered
    assert any(button.label == "Previous hour" for button in app.button)
    assert any(button.label == "Next hour" for button in app.button)
