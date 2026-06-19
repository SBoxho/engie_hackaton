from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.what_if_view import (
    baseline_scenario_chart,
    build_scenario_request,
    causal_chain_steps,
    changed_region_table,
    demand_delta_chart,
    regional_delta_choropleth,
    regional_delta_frame,
    restore_scenario_controls,
    scenario_frames,
    scenario_summary,
)
from src.api.scenarios import ScenarioService
from src.data_sources.rte_eco2mix_regional import fallback_region_geojson
from tests.test_twin_api import NOW, service as twin_fixture_service


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def scenario_service() -> ScenarioService:
    return ScenarioService(
        twin_service=twin_fixture_service(),
        now=lambda: NOW + timedelta(minutes=7),
        cache_enabled=False,
    )


def cold_request(**updates: Any) -> dict[str, Any]:
    request = {
        "scenario_type": "cold_snap",
        "magnitude": {"temperature_delta_c": -4.0},
        "scope": "national",
        "start_time": "2026-06-18T14:00:00Z",
        "duration_hours": 4,
        "baseline_from_time": "2026-06-18T12:00:00Z",
        "hours": 24,
        "assumptions": {"source": "what_if_test"},
    }
    request.update(updates)
    return request


def outage_request(**updates: Any) -> dict[str, Any]:
    request = {
        "scenario_type": "generation_unavailability",
        "magnitude": {"unavailable_capacity_mw": 1_300.0, "asset_name": "Unit A"},
        "scope": "national",
        "start_time": "2026-06-18T14:00:00Z",
        "duration_hours": 4,
        "baseline_from_time": "2026-06-18T12:00:00Z",
        "hours": 24,
        "assumptions": {"source": "what_if_test"},
    }
    request.update(updates)
    return request


def ev_request(**updates: Any) -> dict[str, Any]:
    request = {
        "scenario_type": "ev_charging_shift",
        "magnitude": {
            "vehicles": 100_000,
            "average_energy_per_vehicle_kwh": 8,
            "participation_rate": 0.5,
            "original_charging_window": {"start": "18:00", "end": "22:00"},
            "target_charging_window": {"start": "01:00", "end": "05:00"},
        },
        "scope": "national",
        "start_time": "2026-06-18T16:00:00Z",
        "end_time": "2026-06-19T05:00:00Z",
        "baseline_from_time": "2026-06-18T12:00:00Z",
        "hours": 30,
        "timezone": "Europe/Paris",
        "assumptions": {"source": "what_if_test"},
    }
    request.update(updates)
    return request


@pytest.mark.parametrize(
    ("scenario_request", "expected_chain_text"),
    [
        (cold_request(), "Heating demand rises"),
        (outage_request(), "Available generation falls"),
        (ev_request(), "Overnight demand rises"),
    ],
)
def test_causal_chain_changes_with_scenario_type(scenario_request: dict[str, Any], expected_chain_text: str) -> None:
    result = scenario_service().run(scenario_request, use_cache=False)

    chain_text = " ".join(title for title, _ in causal_chain_steps(result))

    assert expected_chain_text in chain_text


def test_delta_summary_synchronizes_with_api_series() -> None:
    result = scenario_service().run(ev_request(), use_cache=False)
    baseline, scenario = scenario_frames(result)
    summary = scenario_summary(result)

    baseline_watch = baseline["balance_status"].astype(str).str.lower().isin({"watch", "high"}).sum()
    scenario_watch = scenario["balance_status"].astype(str).str.lower().isin({"watch", "high"}).sum()
    expected_min_score_delta = scenario["balance_score"].min() - baseline["balance_score"].min()

    assert summary.watch_high_hour_delta == int(scenario_watch - baseline_watch)
    assert summary.min_balance_score_delta == pytest.approx(expected_min_score_delta)
    assert summary.peak_demand_delta_mw == pytest.approx(result["peak_demand_delta_mw"])

    delta_figure = demand_delta_chart(result)
    demand_delta_values = list(delta_figure.data[0].y)
    assert min(demand_delta_values) < 0
    assert max(demand_delta_values) > 0


def test_baseline_scenario_chart_contains_required_layers() -> None:
    result = scenario_service().run(cold_request(), use_cache=False)

    figure = baseline_scenario_chart(result, selected_timestamp="2026-06-18T15:00:00Z")
    names = {trace.name for trace in figure.data}

    assert {"Baseline uncertainty", "Baseline P50", "Delta area", "Scenario P50", "Selected hour"} <= names
    assert figure.layout.shapes


def test_regional_delta_frame_lists_exactly_changed_regions() -> None:
    result = scenario_service().run(cold_request(scope={"type": "regional", "regions": ["11"]}), use_cache=False)
    baseline, _ = scenario_frames(result)
    regional_context = pd.DataFrame(baseline.iloc[0]["regional_demand_context"])
    regional_context = regional_context.rename(columns={"region_name": "region_display", "demand_mw": "consumption_mw"})

    frame = regional_delta_frame(regional_context, result)
    table = changed_region_table(frame)

    assert frame["changed"].sum() == 1
    assert table["Region"].tolist() == ["Ile-de-France"]


def test_regional_delta_map_uses_geo_choropleth_and_peak_delta_z_values() -> None:
    result = scenario_service().run(cold_request(scope={"type": "regional", "regions": ["11"]}), use_cache=False)
    baseline, _ = scenario_frames(result)
    regional_context = pd.DataFrame(baseline.iloc[0]["regional_demand_context"])
    regional_context = regional_context.rename(columns={"region_name": "region_display", "demand_mw": "consumption_mw"})
    frame = regional_delta_frame(regional_context, result)

    figure = regional_delta_choropleth(frame, fallback_region_geojson())
    changed = next(trace for trace in figure.data if trace.name == "Changed regional demand delta")
    unchanged = next(trace for trace in figure.data if trace.name == "Unchanged regional demand delta")

    assert {trace.type for trace in figure.data} == {"choropleth"}
    assert changed.featureidkey == "properties.code"
    assert list(changed.locations) == ["11"]
    assert list(changed.z) == pytest.approx([1910.769230769231])
    assert changed.zmin == pytest.approx(-1910.769230769231)
    assert changed.zmid == 0
    assert changed.zmax == pytest.approx(1910.769230769231)
    assert changed.colorbar.title.text == "Peak delta"
    assert changed.customdata[0][0] == "Ile-de-France"
    assert changed.customdata[0][1] == pytest.approx(1910.769230769231)
    assert changed.customdata[0][2] == pytest.approx(7393.846153846154)
    assert changed.customdata[0][3] is True
    assert "84" in list(unchanged.locations)


def test_unsupported_regional_delta_map_labels_regions_unavailable() -> None:
    result = scenario_service().run(outage_request(), use_cache=False)
    frame = regional_delta_frame(pd.DataFrame(), result)

    figure = regional_delta_choropleth(frame, fallback_region_geojson())

    assert [trace.type for trace in figure.data] == ["choropleth"]
    assert figure.data[0].name == "Unavailable regional demand delta"
    assert set(figure.data[0].locations) == set(frame["region_code"])
    assert "Demand delta: Unavailable" in figure.data[0].hovertemplate


def test_url_restoration_builds_scenario_api_request_window() -> None:
    controls = restore_scenario_controls(
        {
            "scenario": ["ev_charging_shift"],
            "scope": ["regional"],
            "region": ["11"],
            "mag": ["120000"],
            "start": ["16"],
            "duration": ["13"],
            "ev_kwh": ["9"],
            "participation": ["0.6"],
            "source_window": ["18:00-22:00"],
            "target_window": ["01:00-05:00"],
        },
        default_region="11",
    )
    request = build_scenario_request(
        controls,
        baseline_from_time=pd.Timestamp("2026-06-18T12:00:00Z"),
        timezone_name="Europe/Paris",
    )

    assert request["scenario_type"] == "ev_charging_shift"
    assert request["scope"] == {"type": "regional", "regions": ["11"]}
    assert request["start_time"] == "2026-06-19T04:00:00Z"
    assert request["end_time"] == "2026-06-19T17:00:00Z"
    assert request["magnitude"]["vehicles"] == 120_000
    assert request["magnitude"]["average_energy_per_vehicle_kwh"] == 9
    assert request["magnitude"]["participation_rate"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("scenario", "expected_label"),
    [
        ("cold_snap", "Colder weather"),
        ("generation_unavailability", "Generation unavailable"),
        ("ev_charging_shift", "EV charging shift"),
    ],
)
def test_what_if_page_runs_each_scenario_type(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_label: str,
) -> None:
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("DEMO_ALLOW_EXTERNAL_API", "0")
    app = AppTest.from_file(APP_MAIN, default_timeout=25)
    app.query_params.update({"scenario": [scenario], "start": ["16"], "duration": ["8"], "mag": ["3"]})
    if scenario == "generation_unavailability":
        app.query_params["mag"] = ["1.3"]
    if scenario == "ev_charging_shift":
        app.query_params.update({"mag": ["100000"], "ev_kwh": ["8"], "participation": ["0.5"]})
    app.switch_page("pages/what_if.py")
    app.run(timeout=75)

    rendered = "\n".join(str(item.value) for item in app.markdown)

    assert not app.exception
    assert app.selectbox[0].value == expected_label
    assert "Delta First" in rendered
    assert "Scenario API" in rendered


def test_what_if_timeline_editing_updates_url_and_delta_view(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MODE", "demo")
    monkeypatch.setenv("DEMO_ALLOW_EXTERNAL_API", "0")
    app = AppTest.from_file(APP_MAIN, default_timeout=25)
    app.switch_page("pages/what_if.py")
    app.run(timeout=75)

    app.selectbox[0].select("EV charging shift")
    app.run(timeout=75)
    timeline_slider = next(slider for slider in app.slider if slider.label == "Scenario window")
    timeline_slider.set_value((16, 28))
    app.run(timeout=75)

    rendered = "\n".join(str(item.value) for item in app.markdown)

    assert not app.exception
    assert app.query_params["scenario"] == ["ev_charging_shift"]
    assert app.query_params["start"] == ["16"]
    assert app.query_params["duration"] == ["12"]
    assert "Hourly demand delta" in rendered
    assert "Negative bars reduce demand" in rendered
