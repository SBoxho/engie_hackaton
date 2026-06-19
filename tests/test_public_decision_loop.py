from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.generated.energy_twin_client import CurrentStateQuery, EnergyTwinApiClient, ScenarioRunQuery, TwinQuery
from app.components.regional_map import regional_demand_choropleth
from app.data_loader import load_national_energy, load_public_context
from app.view_models import add_regional_anomalies, build_forecast_points, synthesize_regional_history
from src.api.forecast_explanations import ForecastExplanationApiService
from src.config import settings
from src.data_sources.rte_eco2mix_regional import (
    demo_regional_snapshot,
    fallback_department_geojson,
    fallback_region_geojson,
)

APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


@pytest.mark.parametrize(
    ("current_mw", "expected"),
    [
        (88.0, "12% below usual"),
        (104.0, "Normal"),
        (118.0, "18% above usual"),
    ],
)
def test_regional_anomaly_uses_season_day_type_and_local_hour(current_mw: float, expected: str) -> None:
    current = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-05T17:00:00Z"),
                "region_code": "11",
                "region_display": "Ile-de-France",
                "consumption_mw": current_mw,
                "renewable_share": 0.2,
                "total_production_mw": 500.0,
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(f"2026-01-{day:02d}T17:00:00Z"),
                "region_code": "11",
                "region_display": "Ile-de-France",
                "consumption_mw": 100.0,
                "renewable_share": 0.2,
                "total_production_mw": 500.0,
            }
            for day in (5, 12, 19, 26)
        ]
    )

    result = add_regional_anomalies(current, history, timezone="Europe/Paris")

    assert result.loc[0, "demand_anomaly_label"] == expected
    assert result.loc[0, "comparable_count"] == 4


def test_regional_map_defaults_to_regions_without_department_outlines() -> None:
    regional = demo_regional_snapshot()
    region_geojson = fallback_region_geojson()
    department_geojson = fallback_department_geojson()

    assert len(region_geojson["features"]) == 13

    figure = regional_demand_choropleth(regional, region_geojson, department_geojson)
    assert [trace.type for trace in figure.data] == ["choropleth"]

    figure_with_departments = regional_demand_choropleth(
        regional,
        region_geojson,
        department_geojson,
        department_metrics_available=True,
    )
    assert "scattergeo" in [trace.type for trace in figure_with_departments.data]


def test_replay_regional_context_produces_varied_map_shading() -> None:
    regional = demo_regional_snapshot()
    history = synthesize_regional_history(regional, timezone="Europe/Paris")

    shaded = add_regional_anomalies(regional, history, timezone="Europe/Paris")

    assert shaded["demand_anomaly_score"].nunique() > 6
    assert shaded["demand_anomaly_score"].min() < 0.2
    assert shaded["demand_anomaly_score"].max() > 0.8


def test_replay_next_48h_uses_presentation_anchored_demo_window() -> None:
    load_national_energy.clear()

    context = load_public_context()
    points = build_forecast_points(
        context["energy"],
        model_payload=context["model_payload"],
        horizon_hours=48,
        timezone=settings.timezone,
    )

    assert points

    latest = pd.to_datetime(context["energy"]["timestamp"].max(), utc=True)
    first_forecast = points[0].timestamp
    last_forecast = points[-1].timestamp
    manifest = json.loads((settings.demo_dir / "manifest.json").read_text(encoding="utf-8"))
    source_end = pd.to_datetime(manifest["window_end_utc"], utc=True)
    anchor_end = pd.to_datetime(settings.demo_anchor_end_utc, utc=True)
    expected_end = anchor_end if anchor_end > source_end else source_end
    local_forecast_dates = {
        point.timestamp.tz_convert(settings.timezone).normalize()
        for point in points
    }

    assert latest == expected_end
    assert first_forecast == latest.floor("h") + pd.Timedelta(hours=1)
    assert last_forecast <= latest.floor("h") + pd.Timedelta(hours=48)
    assert min(local_forecast_dates) >= latest.tz_convert(settings.timezone).normalize()
    assert max(local_forecast_dates) <= latest.tz_convert(settings.timezone).normalize() + pd.Timedelta(days=2)


@pytest.mark.parametrize(
    ("page_path", "expected_text"),
    [
        (None, "En direct"),
        ("pages/next_48h.py", "Prochaines 48 h"),
        ("pages/what_if.py", "Et si ?"),
        ("pages/technical_lab.py", "Labo technique"),
        ("pages/1_live_grid.py", "Live grid detail"),
    ],
)
def test_streamlit_public_pages_smoke(monkeypatch: pytest.MonkeyPatch, page_path: str | None, expected_text: str) -> None:
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("DEMO_ALLOW_EXTERNAL_API", "0")
    app = AppTest.from_file(APP_MAIN, default_timeout=20)
    if page_path is not None:
        app.switch_page(page_path)

    app.run(timeout=30)

    assert not app.exception
    rendered = "\n".join(str(item.value) for item in app.markdown)
    assert expected_text in rendered


def test_complete_judge_journey_runs_without_live_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("DEMO_ALLOW_EXTERNAL_API", "0")
    client = EnergyTwinApiClient()

    now = client.get_current_state(CurrentStateQuery(region="11"))
    assert now.national_context.demand.current.value is not None
    assert now.national_context.demand.difference_vs_usual_pct.value is not None
    assert now.operating_state.value == "historical_replay"

    twin = client.get_twin(TwinQuery(hours=48, region="11"))
    forecast_hours = twin.snapshots[1:]
    peak = max(forecast_hours, key=lambda snapshot: snapshot.demand_forecast.p50.value or 0.0)
    assert peak.demand_forecast.confidence.confidence.value in {"low", "medium", "high"}
    assert peak.modelled_national_balance_context is not None

    forecast = ForecastExplanationApiService().create_forecast(hours=48)
    explained_peak = max(forecast["points"], key=lambda point: point.get("expected_demand_gw") or 0.0)
    assert explained_peak["confidence_level"]
    assert explained_peak["concept_contributions"]
    assert explained_peak["confidence_reasons"]
    assert explained_peak["route"] in {"model", "baseline_fallback"}
    if explained_peak["route"] == "baseline_fallback":
        assert "fallback" in explained_peak["explanation"].lower()
    else:
        assert explained_peak["top_positive_concept_drivers"] or explained_peak["top_negative_concept_drivers"]

    scenario = client.run_scenario(
        ScenarioRunQuery(
            request={
                "scenario_type": "cold_snap",
                "magnitude": {"temperature_delta_c": -4.0},
                "scope": "national",
                "start_time": pd.Timestamp(peak.event_time).isoformat().replace("+00:00", "Z"),
                "duration_hours": 4,
                "baseline_from_time": pd.Timestamp(twin.from_time).isoformat().replace("+00:00", "Z"),
                "hours": 48,
                "assumptions": {"source": "judge journey test", "heating": "directional"},
            },
            use_cache=False,
        )
    )

    assert scenario["demand_delta"]["series"]
    assert "changed_watch_or_high_hours" in scenario
    assert scenario["estimated_import_export_delta"]["net_import_delta_mwh_range"]
    assert scenario["estimated_carbon_range"]["total_tonnes_co2_delta_range"]
    assert scenario["regional_deltas"]["regions"]
